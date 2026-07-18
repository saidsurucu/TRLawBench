# -*- coding: utf-8 -*-
"""
TRLawBench Benchmark 3 — Güncel Gemma 4 12B Unified (Modal/vLLM) + Yargı PRO MCP retry

Modal'da vLLM ile servis edilen güncel gemma-4-12B-it'nin (yeni canonical chat
template, 2026-07-09) plain koşusunda YANLIŞ kalan soruları, Yargı PRO MCP +
Gemma yargipro sistem promptu (prompts/gemma-4-12b-yargipro-system.md) ile yeniden sorar.

- Model : google/gemma-4-12B-it @ Modal vLLM (tool-call-parser gemma4)
- MCP   : https://yargi.betaspacestudio.com/mcp — statik bearer (.env: YARGI_MCP_TOKEN)
- Jüri  : YOK — sadece exact match; jüri sonradan benchmark3_inkling_judge_merge.py
          benzeri akışla eklenir
- Kaynak: --source ile verilen plain detail JSON'unun yanlışları

Kullanım:
  uv run python -u benchmark3_gemma12b_modal_yargipro.py --source results/benchmark3_detail_google_gemma-4-12B-it_YYYYMMDD_HHMMSS.json
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from datetime import datetime

from openai import AsyncOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# --------------------------------------------------
JSON_PATH = "benchmark3_questions.json"
RESULTS_DIR = "results"
MODEL_SLUG = "google/gemma-4-12B-it"
MODEL_LABEL = "gemma-4-12b-modal-yargipro"
SYSTEM_PROMPT_PATH = "prompts/gemma-4-12b-yargipro-system.md"
CHECKPOINT = os.path.join(RESULTS_DIR, f"_{MODEL_LABEL}_checkpoint.json")

MODAL_BASE = "https://saidsrc--gemma4-12b-vllm-serve.modal.run/v1"
MODAL_KEY = "trlawbench-gemma4"

MCP_URL = "https://yargi.betaspacestudio.com/mcp"
MCP_TOKEN = os.getenv("YARGI_MCP_TOKEN", "")

MAX_ROUNDS = 15            # maks model turu (her tur birden çok tool çağırabilir)
FORCE_ANSWER_AT = 12       # bu tura gelince araçlar kaldırılır, cevap yazmaya zorlanır
ANSWER_MAX_TOKENS = 8000
QUESTION_TIMEOUT = 900
TOOL_RESULT_CAP = 8000     # tek tool sonucu karakter sınırı
TOOL_CALL_TIMEOUT = 150

# Prompttaki stratejiyle uyumlu çekirdek araçlar. agentic_legal_deep_research ve
# meta araçlar (guide/login/install) bilinçli olarak dışarıda.
ALLOWED_TOOLS = {
    "mevzuat_ara", "mevzuat_icinde_ara", "mevzuat_getir",
    "ictihat_ara", "aym_ictihat_ara", "semantik_ictihat_ara", "ictihat_getir",
    "kurum_karari_ara", "kurum_karari_getir",
}

SYSTEM_PROMPT_ASK = open(SYSTEM_PROMPT_PATH, encoding="utf-8").read()


def extract_answer(text: str) -> str | None:
    m = re.search(r"CEVAP:\s*([A-Ea-e])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:cevap|answer|doğru\s*(?:seçenek|cevap))[:\s]*([A-Ea-e])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        if len(last) == 1 and last.upper() in "ABCDE":
            return last.upper()
    return None


def to_openai_tools(mcp_tools) -> list[dict]:
    out = []
    for t in mcp_tools:
        if t.name not in ALLOWED_TOOLS:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": (t.description or "")[:1024],
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        })
    return out


def mcp_result_text(result) -> str:
    parts = []
    for c in getattr(result, "content", []) or []:
        txt = getattr(c, "text", None)
        if txt:
            parts.append(txt)
    text = "\n".join(parts).strip() or "[boş sonuç]"
    if len(text) > TOOL_RESULT_CAP:
        text = text[:TOOL_RESULT_CAP] + "\n...[kısaltıldı]"
    return text


class UsageMeter:
    """Model API çağrılarının token + süre sayacı (tek soru için)."""

    def __init__(self):
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.api_time = 0.0       # istemci tarafı: istek gönder → yanıt al (ağ dahil)
        self.gen_time = 0.0       # vLLM time_info vermez, 0 kalır → api_time'a düşer
        self.rounds = 0

    def add(self, resp, dur: float):
        self.rounds += 1
        self.api_time += dur
        u = getattr(resp, "usage", None)
        if u:
            self.completion_tokens += getattr(u, "completion_tokens", 0) or 0
            self.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0

    @property
    def tok_per_s(self) -> float:
        base = self.gen_time if self.gen_time > 0 else self.api_time
        return self.completion_tokens / base if base > 0 else 0.0


async def chat(oai: AsyncOpenAI, meter: UsageMeter, **kwargs):
    """Retry'lı chat.completions çağrısı; usage ve süreyi meter'a işler."""
    for attempt in range(5):
        t0 = time.time()
        try:
            resp = await oai.chat.completions.create(model=MODEL_SLUG, **kwargs)
            if resp.choices:
                meter.add(resp, time.time() - t0)
                return resp
        except Exception as e:
            if attempt == 4:
                raise
            print(f"      api hata ({type(e).__name__}: {str(e)[:120]}), yeniden...", flush=True)
        await asyncio.sleep(3 * (attempt + 1))
    return None


async def force_answer_letter(oai: AsyncOpenAI, meter: UsageMeter, messages: list, prior_text: str) -> str:
    """Araştırma bitti ama 'CEVAP: X' üretilmedi → tek satır harf iste."""
    msgs = messages + [{"role": "user", "content":
        "Yukarıdaki araştırmaya göre kesin kararını ver. SADECE şu formatta tek satır yaz: "
        "'CEVAP: X' (X = A, B, C, D veya E). Başka hiçbir şey yazma."}]
    try:
        resp = await chat(oai, meter, messages=msgs, max_tokens=100)
        if resp and resp.choices:
            c = (resp.choices[0].message.content or "").strip()
            if c and extract_answer(c):
                print(f"      → zorunlu cevap: {extract_answer(c)}", flush=True)
                return (prior_text + "\n\n" + c).strip() if prior_text else c
    except Exception:
        pass
    return prior_text


FORCE_TOOL_ROUNDS = 0  # main() --force-first ile ayarlanır


async def run_question(oai: AsyncOpenAI, meter: UsageMeter, prompt: str) -> tuple[str, list[str]]:
    """Tek soru için MCP session aç + manuel agentic döngü."""
    tools_used = []
    async with streamablehttp_client(MCP_URL, headers={"Authorization": f"Bearer {MCP_TOKEN}"}) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools_list = (await session.list_tools()).tools
            oai_tools = to_openai_tools(tools_list)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_ASK},
                {"role": "user", "content": prompt},
            ]
            last_text = ""
            for _round in range(MAX_ROUNDS):
                force_answer = _round >= FORCE_ANSWER_AT
                if force_answer:
                    messages.append({"role": "user", "content":
                        "Yeterince araştırdın. ARTIK ARAÇ KULLANMA. Topladığın bilgiye dayanarak "
                        "seçenekleri değerlendir ve cevabını 'CEVAP: X' ile bitir."})
                resp = await chat(
                    oai, meter, messages=messages,
                    tools=None if force_answer else oai_tools,
                    tool_choice=None if force_answer else "auto",
                    max_tokens=ANSWER_MAX_TOKENS,
                )
                if resp is None or not resp.choices:
                    return (last_text or "").strip() or "[EMPTY_CHOICES]", tools_used
                msg = resp.choices[0].message
                if msg.content:
                    last_text = msg.content

                if not msg.tool_calls:
                    final = (last_text or "").strip()
                    # 0-araç sendromu önlemi: ilk FORCE_TOOL_ROUNDS turda araçsız cevap kabul edilmez
                    if len(tools_used) == 0 and not force_answer and _round < FORCE_TOOL_ROUNDS:
                        print("      (araçsız cevap reddedildi → araç zorlaması)", flush=True)
                        messages.append({"role": "assistant", "content": msg.content or ""})
                        messages.append({"role": "user", "content":
                            "DUR. Cevabını henüz kabul etmiyorum. Ezberden cevap verme riskin yüksek. "
                            "ÖNCE MCP araçlarıyla doğrula: ilgili kanunu mevzuat_ara ile bul, maddeyi "
                            "mevzuat_getir ile OKU, gerekiyorsa semantik_ictihat_ara ile içtihat tara. "
                            "Araç sonuçlarını gördükten sonra cevabını 'CEVAP: X' ile bitir."})
                        continue
                    if not final and not tools_used and not force_answer and _round < MAX_ROUNDS - 2:
                        print("      (boş yanıt → araç kullanmaya yönlendiriliyor)", flush=True)
                        messages.append({"role": "user", "content":
                            "Boş yanıt verdin. ÖNCE MCP araçlarıyla (mevzuat_ara, semantik_ictihat_ara, "
                            "mevzuat_getir vb.) araştır, SONRA cevabını 'CEVAP: X' ile bitir."})
                        continue
                    if extract_answer(final) is None:
                        final = await force_answer_letter(oai, meter, messages, final)
                    return final, tools_used

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                })
                for tc in msg.tool_calls:
                    tools_used.append(tc.function.name)
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    tt = time.time()
                    try:
                        res = await asyncio.wait_for(
                            session.call_tool(tc.function.name, args), timeout=TOOL_CALL_TIMEOUT)
                        content = mcp_result_text(res)
                        print(f"      · {tc.function.name} ({time.time()-tt:.0f}s, {len(content)}c)", flush=True)
                    except asyncio.TimeoutError:
                        content = f"[ARAÇ TIMEOUT: {tc.function.name} >{TOOL_CALL_TIMEOUT}s]"
                        print(f"      · {tc.function.name} TIMEOUT >{TOOL_CALL_TIMEOUT}s", flush=True)
                    except Exception as e:
                        content = f"[ARAÇ HATASI: {type(e).__name__}: {str(e)[:150]}]"
                        print(f"      · {tc.function.name} HATA {type(e).__name__}", flush=True)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

            final = (last_text or "").strip()
            if extract_answer(final) is None:
                final = await force_answer_letter(oai, meter, messages, final)
            return final or "[MAX_ROUNDS]", tools_used


def unwrap_exc(e) -> str:
    excs = getattr(e, "exceptions", None)
    if excs:
        return " | ".join(unwrap_exc(x) for x in excs)
    return f"{type(e).__name__}: {str(e)[:200]}"


async def ask_one(oai: AsyncOpenAI, prompt: str, retries: int = 2) -> tuple[str, float, list[str], UsageMeter]:
    for attempt in range(retries + 1):
        meter = UsageMeter()
        t0 = time.time()
        try:
            text, tools = await asyncio.wait_for(run_question(oai, meter, prompt), timeout=QUESTION_TIMEOUT)
            if text and extract_answer(text) is None and attempt < retries:
                print(f"    cevap çıkmadı ({time.time()-t0:.0f}s), yeniden...", flush=True)
                continue
            return text or "[EMPTY_RESPONSE]", time.time() - t0, tools, meter
        except asyncio.TimeoutError:
            if attempt == retries:
                return "[TIMEOUT]", time.time() - t0, [], meter
            print("    timeout, yeniden...", flush=True)
        except Exception as e:
            real = unwrap_exc(e)
            if attempt == retries:
                return f"[AGENT_ERROR: {real[:180]}]", time.time() - t0, [], meter
            print(f"    hata ({real[:160]}), {10*(attempt+1)}s sonra yeniden...", flush=True)
            await asyncio.sleep(10 * (attempt + 1))
    return "[MAX_RETRIES]", 0.0, [], UsageMeter()


def build_ask_prompt(q: dict) -> str:
    opts_upper = {k.upper(): v for k, v in q.get("options", {}).items()}
    valid = sorted(k for k in opts_upper if "A" <= k <= "E")
    opts_text = "\n".join(f"{k}: {opts_upper[k]}" for k in valid)
    return (
        f"Soru:\n{q.get('question','')}\n\n"
        f"Seçenekler:\n{opts_text}\n\n"
        f"Bu soruyu MCP araçlarıyla araştır, ilgili mevzuat/içtihatı oku, doğru cevabı gerekçesiyle belirt."
    )


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        try:
            return {int(k): v for k, v in json.load(open(CHECKPOINT, encoding="utf-8")).items()}
        except Exception:
            return {}
    return {}


def save_checkpoint(cp: dict):
    tmp = CHECKPOINT + ".tmp"
    json.dump({str(k): v for k, v in cp.items()}, open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    os.replace(tmp, CHECKPOINT)


def print_report(records: list[dict]):
    print(f"\n{'='*100}")
    print(f"  GÜNCEL GEMMA 4 12B (MODAL/vLLM) + YARGI PRO — SONUÇ RAPORU")
    print(f"{'='*100}")
    hdr = f"{'ID':>4} {'Kategori':<24} {'Doğru':>5} {'Model':>5} {'Sonuç':>7} {'Token':>7} {'API sn':>7} {'Duvar sn':>8} {'tok/sn':>8} {'Araç':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        sonuc = "DOĞRU" if r["dogru"] else "YANLIŞ"
        print(f"{r['id']:>4} {r['category'][:24]:<24} {r['correct_answer']:>5} {str(r['model_answer']):>5} "
              f"{sonuc:>7} {r['completion_tokens']:>7} {r['api_time_s']:>7.1f} {r['wall_time_s']:>8.1f} "
              f"{r['tok_per_s']:>8.1f} {r['tool_calls']:>5}")

    n = len(records)
    dogru = sum(1 for r in records if r["dogru"])
    tot_tok = sum(r["completion_tokens"] for r in records)
    tot_api = sum(r["api_time_s"] for r in records)
    tot_wall = sum(r["wall_time_s"] for r in records)
    tot_tools = sum(r["tool_calls"] for r in records)
    micro_api_tps = tot_tok / tot_api if tot_api > 0 else 0.0
    macro_tps = sum(r["tok_per_s"] for r in records) / n if n else 0.0

    print("-" * len(hdr))
    print(f"\n  DOĞRU / YANLIŞ     : {dogru} doğru, {n - dogru} yanlış  ({dogru}/{n} = %{100*dogru/n:.1f})")
    print(f"  Toplam token       : {tot_tok:,} completion token ({tot_tools} araç çağrısı)")
    print(f"  Toplam süre        : {tot_wall:.1f} sn duvar | {tot_api:.1f} sn API (ağ dahil)")
    print(f"  Hız (API micro)    : {micro_api_tps:.1f} tok/sn")
    print(f"  Hız (macro ort.)   : {macro_tps:.1f} tok/sn")
    print(f"  Soru başına ort.   : {tot_tok/n:.0f} token, {tot_api/n:.1f} sn API, {tot_wall/n:.1f} sn duvar")

    return {
        "dogru": dogru, "yanlis": n - dogru, "toplam": n,
        "dogru_pct": round(100 * dogru / n, 1) if n else 0,
        "toplam_completion_token": tot_tok,
        "toplam_api_sn": round(tot_api, 1),
        "toplam_duvar_sn": round(tot_wall, 1),
        "toplam_arac_cagrisi": tot_tools,
        "hiz_api_micro_tok_sn": round(micro_api_tps, 1),
        "hiz_macro_tok_sn": round(macro_tps, 1),
        "ort_token_per_soru": round(tot_tok / n, 0) if n else 0,
        "ort_api_sn_per_soru": round(tot_api / n, 1) if n else 0,
        "ort_duvar_sn_per_soru": round(tot_wall / n, 1) if n else 0,
    }


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="Plain koşunun benchmark3_detail_*.json dosyası (yanlışlar buradan alınır)")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--ids", default="",
                    help="Virgülle ayrılmış id listesi (source'un yanlışları yerine bunlar sorulur)")
    ap.add_argument("--force-first", type=int, default=0,
                    help="İlk N turda tool_choice=required (0-araç sendromu önlemi)")
    ap.add_argument("--label-suffix", default="",
                    help="Checkpoint/çıktı dosya adına eklenecek ek (ör: pass2)")
    args = ap.parse_args()

    global FORCE_TOOL_ROUNDS, CHECKPOINT
    FORCE_TOOL_ROUNDS = args.force_first
    if args.label_suffix:
        CHECKPOINT = os.path.join(RESULTS_DIR, f"_{MODEL_LABEL}-{args.label_suffix}_checkpoint.json")

    if not MCP_TOKEN:
        print("YARGI_MCP_TOKEN gerekli (.env)!"); return

    detail = json.load(open(args.source, encoding="utf-8"))
    if args.ids:
        wrong_ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        wrong_ids = [r["id"] for r in detail["results"] if not r["exact_match"]]
    print(f"Kaynak detail: {args.source}")
    print(f"Yanlış soru sayısı: {len(wrong_ids)} → {wrong_ids}")
    print(f"Sistem promptu: {SYSTEM_PROMPT_PATH} ({len(SYSTEM_PROMPT_ASK)} karakter)")

    all_q = {q.get("general_id", q.get("id")): q for q in json.load(open(JSON_PATH, encoding="utf-8"))}

    cp = {} if args.fresh else load_checkpoint()
    if cp:
        done = [i for i in wrong_ids if i in cp and cp[i].get("model_answer") is not None]
        print(f"Checkpoint'ten {len(done)} soru tamam, atlanacak: {done}")

    oai = AsyncOpenAI(base_url=MODAL_BASE, api_key=MODAL_KEY, timeout=300)
    print(f"Test : {MODEL_SLUG} @ Modal vLLM + Yargı PRO MCP (jürisiz, exact match)\n")

    for i, qid in enumerate(wrong_ids, 1):
        if qid in cp and cp[qid].get("model_answer") is not None:
            continue
        q = all_q.get(qid)
        if q is None:
            print(f"[{i}/{len(wrong_ids)}] id={qid} soru bulunamadı, atlandı"); continue

        print(f"[{i}/{len(wrong_ids)}] id={qid} | {q.get('category','?')} | doğru={q.get('correct_answer')}")
        prompt = build_ask_prompt(q)
        text, wall, tools, meter = await ask_one(oai, prompt)
        ans = extract_answer(text)
        dogru = (ans is not None and str(q.get("correct_answer", "")).upper().startswith(ans))

        rec = {
            "id": qid,
            "category": q.get("category", ""),
            "correct_answer": q.get("correct_answer", ""),
            "model_answer": ans,
            "dogru": dogru,
            "completion_tokens": meter.completion_tokens,
            "prompt_tokens": meter.prompt_tokens,
            "api_time_s": round(meter.api_time, 2),
            "wall_time_s": round(wall, 2),
            "tok_per_s": round(meter.tok_per_s, 1),
            "rounds": meter.rounds,
            "tool_calls": len(tools),
            "tools_used": dict(Counter(tools)),
            "model_response": text,
        }
        cp[qid] = rec
        save_checkpoint(cp)
        print(f"    → cevap={ans} ({'DOĞRU' if dogru else 'YANLIŞ'}) | "
              f"{meter.completion_tokens} tok | api {meter.api_time:.1f}s | duvar {wall:.1f}s | "
              f"{meter.tok_per_s:.1f} tok/sn | {len(tools)} araç\n", flush=True)

    records = [cp[qid] for qid in wrong_ids if qid in cp and cp[qid].get("model_answer") is not None]
    summary_stats = print_report(records)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "test_model": f"{MODEL_SLUG} (Modal vLLM, güncel chat template)",
        "mcp": "yargi-pro (betaspacestudio)",
        "system_prompt": SYSTEM_PROMPT_PATH,
        "kaynak_detail": args.source,
        "judge": None,
        "summary": summary_stats,
        "results": records,
    }
    suffix = f"-{args.label_suffix}" if args.label_suffix else ""
    out_path = os.path.join(RESULTS_DIR, f"benchmark3_modal_{MODEL_LABEL}{suffix}_{ts}.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nKaydedildi: {out_path}")


if __name__ == "__main__":
    asyncio.run(main_async())
