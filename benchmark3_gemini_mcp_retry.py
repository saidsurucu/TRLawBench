# -*- coding: utf-8 -*-
"""
TRLawBench Benchmark 3 — Gemini 3.5 Flash + Yargı MCP Pro retry (yanlış sorular)

Plain Gemini 3.5 Flash (OpenRouter) run'ında YANLIŞ cevaplanan soruları,
pydantic-ai + MCPServerStreamableHTTP ile Yargı MCP Pro'ya bağlanarak yeniden sorar.

- Model: google/gemini-3.5-flash (OpenRouter, pydantic-ai)
- MCP: https://yargi.betaspacestudio.com/mcp — STATİK BEARER TOKEN (.env: YARGI_MCP_TOKEN)
- Judge: Gemini 3.1 Pro (OpenRouter)
- Checkpoint/resume: her sorudan sonra diske yazılır; çökerse kaldığı yerden devam.

Çıktılar:
- benchmark3_detail/summary_google_gemini-3.5-flash-mcp-retry_*  → sadece retry edilenler
- benchmark3_detail/summary_google_gemini-3.5-flash-mcp_*        → 100 soruluk birleşik (web UI)

Kullanım:
  uv run python -u benchmark3_gemini_mcp_retry.py
  uv run python -u benchmark3_gemini_mcp_retry.py --fresh
"""

import argparse
import asyncio
import glob
import json
import os
import time
from collections import defaultdict, Counter

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.usage import UsageLimits

from benchmark3_fable5 import (
    SYSTEM_PROMPT_JUDGE, JUDGE_MODEL,
    call_judge, extract_answer, extract_score,
)

# --------------------------------------------------
JSON_PATH = "benchmark3_questions.json"
RESULTS_DIR = "results"
CHECKPOINT = os.path.join(RESULTS_DIR, "_gemini_mcp_checkpoint.json")

MODEL_SLUG = "google/gemini-3.5-flash"
MODEL_LABEL = "gemini-3.5-flash-mcp"
PLAIN_GLOB = "benchmark3_detail_google_gemini-3.5-flash_*.json"

MAX_TOOL_CALLS = 100      # 3.1 Pro patolojik araç döngüsüne giriyor; yüksek limit
QUESTION_TIMEOUT = 900     # tek soru için üst sınır (s)
MCP_READ_TIMEOUT = 180
ASK_RETRIES = 2            # agent hatası / cevapsız için yeniden deneme

MCP_URL = "https://yargi.betaspacestudio.com/mcp"
MCP_TOKEN = os.getenv("YARGI_MCP_TOKEN", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# --base-url / --api-key-env ile değiştirilebilir (ör. native Google API)
API_BASE = OPENROUTER_BASE
API_KEY_ENV = "OPENROUTER_API_KEY"
JUDGE_MODEL_OVERRIDE = None

SYSTEM_PROMPT_ASK = """Sen Türkiye Cumhuriyeti hukukuna göre hukuk sorularını yanıtlayan uzman bir hukuk profesörüsün.

Elinde Yargı MCP Pro araçları var (mevzuat + içtihat). MUTLAKA araçları kullanarak araştır:

## ARAÇLAR
- yargi_search_mevzuat: Mevzuat/kanun arama (kanun adı veya numarası).
- yargi_search_within_mevzuat: Belirli bir mevzuatın İÇİNDE madde/keyword arama.
- yargi_get_mevzuat_document: Mevzuat maddesinin/metninin tamamı.
- yargi_search_bedesten_unified: Yargıtay/Danıştay/yerel mahkeme/AYM kararı KEYWORD arama.
- yargi_search_bedesten_semantic: İçtihatta ANLAM TABANLI (semantik) arama — olayı doğal cümleyle yaz.
- yargi_get_bedesten_document_markdown: Kararın TAM METNİNİ getir (documentId ile).

## STRATEJİ (her adımı SIRAYLA ve EKSİKSİZ uygula)
1. Sorudaki olayı ve hukuk dalını tespit et.
2. İlgili kanunu/maddeyi yargi_search_mevzuat + yargi_search_within_mevzuat ile bul,
   yargi_get_mevzuat_document ile TAM METNİNİ oku.
3. İÇTİHAT ARAŞTIRMASI — bu adım ZORUNLUDUR, her soruda yap:
   a) yargi_search_bedesten_semantic ile EN AZ BİR semantik arama yap (olayı doğal bir
      cümleyle, hukuki kavramlarla betimle).
   b) Ayrıca yargi_search_bedesten_unified ile keyword araması yap.
   c) En alakalı kararı yargi_get_bedesten_document_markdown ile aç, gerekçesini oku.
   Sadece keyword/mevzuat araması YETERSİZDİR; semantik içtihat aramasını atlama.
4. "Hangisi yanlıştır / bağdaşmaz" gibi negatif soruları dikkatle yorumla.
5. Seçenekleri tek tek, bulduğun mevzuat metni VE içtihat gerekçelerine dayanarak değerlendir.

Araç kullanmadan cevap VERME. Cevabının SONUNDA mutlaka "CEVAP: X" formatında
(X = A, B, C, D veya E) doğru şıkkı belirt."""


def build_agent() -> Agent:
    model = OpenAIChatModel(
        MODEL_SLUG,
        provider=OpenAIProvider(base_url=API_BASE, api_key=os.getenv(API_KEY_ENV, "")),
    )
    mcp = MCPServerStreamableHTTP(
        MCP_URL,
        headers={"Authorization": f"Bearer {MCP_TOKEN}"},
        tool_prefix="yargi",
        timeout=30,
        read_timeout=MCP_READ_TIMEOUT,
        max_retries=5,   # GPT 5.5 malformed tool-args üretebiliyor; hatayı modele geri ver, düzeltsin
    )
    return Agent(model, toolsets=[mcp], system_prompt=SYSTEM_PROMPT_ASK)


def count_tool_calls(result) -> list[str]:
    names = []
    for m in result.all_messages():
        for p in getattr(m, "parts", []):
            if type(p).__name__ == "ToolCallPart" and getattr(p, "tool_name", None):
                names.append(p.tool_name)
    return names


async def ask_one(agent: Agent, prompt: str) -> tuple[str, float, list[str]]:
    no_answer_left = 1
    for attempt in range(ASK_RETRIES + 1):
        t0 = time.time()
        try:
            async with agent:
                result = await asyncio.wait_for(
                    agent.run(prompt, usage_limits=UsageLimits(request_limit=MAX_TOOL_CALLS + 2)),
                    timeout=QUESTION_TIMEOUT,
                )
            dur = time.time() - t0
            text = (result.output or "").strip()
            tools = count_tool_calls(result)

            if text and extract_answer(text) is None and no_answer_left > 0 and attempt < ASK_RETRIES:
                no_answer_left -= 1
                print(f"    cevap çıkmadı ({dur:.0f}s), yeniden deneniyor...", flush=True)
                continue
            return text or "[EMPTY_RESPONSE]", dur, tools

        except asyncio.TimeoutError:
            if attempt == ASK_RETRIES:
                return "[TIMEOUT]", time.time() - t0, []
            print(f"    timeout ({QUESTION_TIMEOUT}s), yeniden deneniyor...", flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:150]}"
            if attempt == ASK_RETRIES:
                return f"[AGENT_ERROR: {err}]", time.time() - t0, []
            print(f"    agent hata ({err}), {10*(attempt+1)}s sonra yeniden...", flush=True)
            await asyncio.sleep(10 * (attempt + 1))
    return "[MAX_RETRIES]", 0.0, []


def build_ask_prompt(q: dict) -> tuple[str, str]:
    opts_upper = {k.upper(): v for k, v in q.get("options", {}).items()}
    valid = sorted(k for k in opts_upper if "A" <= k <= "E")
    opts_text = "\n".join(f"{k}: {opts_upper[k]}" for k in valid)
    prompt = (
        f"Soru:\n{q.get('question','')}\n\n"
        f"Seçenekler:\n{opts_text}\n\n"
        f"Bu soruyu MCP araçlarıyla araştır, ilgili mevzuat/içtihatı oku, doğru cevabı gerekçesiyle belirt."
    )
    return prompt, opts_text


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


async def main_async():
    global MODEL_SLUG, MODEL_LABEL, CHECKPOINT, PLAIN_GLOB, API_BASE, API_KEY_ENV, JUDGE_MODEL_OVERRIDE, JUDGE_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--model", default=MODEL_SLUG, help="OpenRouter model slug (örn google/gemini-3.1-pro-preview)")
    ap.add_argument("--label", default=MODEL_LABEL, help="Çıktı etiketi (örn gemini-3.1-pro-mcp)")
    ap.add_argument("--base-url", default=None, help="Model+jüri için özel base URL (örn native Google API)")
    ap.add_argument("--api-key-env", default=None, help="Model+jüri API key env var (örn GEMINI_API_KEY)")
    ap.add_argument("--judge-model", default=None, help="Jüri model adı override")
    args = ap.parse_args()

    MODEL_SLUG = args.model
    MODEL_LABEL = args.label
    CHECKPOINT = os.path.join(RESULTS_DIR, f"_{MODEL_LABEL}_checkpoint.json")
    if args.base_url:
        API_BASE = args.base_url
    if args.api_key_env:
        API_KEY_ENV = args.api_key_env
    if args.judge_model:
        JUDGE_MODEL_OVERRIDE = args.judge_model

    if not MCP_TOKEN:
        print("YARGI_MCP_TOKEN env var gerekli (.env)!"); return
    if not os.getenv(API_KEY_ENV):
        print(f"{API_KEY_ENV} gerekli!"); return

    plain_path = args.plain
    if not plain_path:
        cands = sorted(glob.glob(os.path.join(RESULTS_DIR, PLAIN_GLOB)))
        cands = [c for c in cands if "ckpt" not in c]
        if not cands:
            print("Plain Gemini detail bulunamadı, --plain ile ver."); return
        plain_path = cands[-1]
    print(f"Plain detail: {plain_path}")
    plain = json.load(open(plain_path, encoding="utf-8"))

    wrong = [r for r in plain["results"] if not r["exact_match"]]
    wrong_ids = [r["id"] for r in wrong]
    print(f"Yanlış soru sayısı: {len(wrong_ids)} → {wrong_ids}")

    all_q = {q.get("general_id", q.get("id")): q for q in json.load(open(JSON_PATH, encoding="utf-8"))}

    cp = {} if args.fresh else load_checkpoint()
    if cp:
        done = [i for i in wrong_ids if i in cp and cp[i].get("model_answer") is not None]
        print(f"Checkpoint'ten {len(done)} soru zaten tamam, atlanacak: {done}")

    if JUDGE_MODEL_OVERRIDE:
        JUDGE_MODEL = JUDGE_MODEL_OVERRIDE
    judge_client = OpenAI(base_url=API_BASE, api_key=os.getenv(API_KEY_ENV), timeout=600)
    agent = build_agent()
    print(f"Test : {MODEL_SLUG} + Yargı MCP Pro (OpenRouter, pydantic-ai)")
    print(f"Judge: {JUDGE_MODEL}\n")

    error_codes = ("[EMPTY_RESPONSE]", "[TIMEOUT]", "[AGENT_ERROR", "[MAX_RETRIES]")

    for i, qid in enumerate(wrong_ids, 1):
        if qid in cp and cp[qid].get("model_answer") is not None:
            r = cp[qid]
            print(f"[{i}/{len(wrong_ids)}] ID {qid}: checkpoint'ten ({r['model_answer']}, "
                  f"{'✓' if r['exact_match'] else '✗'})", flush=True)
            continue

        q = all_q.get(qid)
        plain_r = next(r for r in wrong if r["id"] == qid)
        correct = plain_r["correct_answer"]
        category = plain_r.get("category", "")
        qname = plain_r.get("question_name", "")
        if not q:
            print(f"[{i}/{len(wrong_ids)}] ID {qid}: soru JSON'da yok, atlandı."); continue

        print(f"[{i}/{len(wrong_ids)}] ID {qid} — {category} (plain: {plain_r['model_answer']}, doğru: {correct})", flush=True)
        prompt, opts_text = build_ask_prompt(q)
        resp, dur, tools = await ask_one(agent, prompt)

        if any(resp.startswith(c) for c in error_codes):
            print(f"    HATA: {resp[:80]}", flush=True)
            cp[qid] = {
                "id": qid, "category": category, "question_name": qname,
                "correct_answer": correct, "model_answer": None,
                "model_response": resp[:200], "judge_response": "",
                "score": 0, "ask_duration": round(dur, 2), "judge_duration": 0,
                "exact_match": False, "tool_calls": [], "tool_count": 0,
            }
            save_checkpoint(cp); continue

        ans = extract_answer(resp)
        exact = ans == correct
        tc = Counter(tools)
        print(f"    MCP: {ans or '?'} (doğru: {correct}) {'✓ DÜZELDİ' if exact else '✗'} "
              f"({dur:.0f}s, {len(tools)} araç: {dict(tc)})", flush=True)

        judge_prompt = (
            f"## Soru\n{q.get('question','')}\n\n"
            f"## Seçenekler\n{opts_text}\n\n"
            f"## Doğru Cevap: {correct}\n\n"
            f"## Doğru Açıklama\n{q.get('solution','')}\n\n"
            f"## Öğrencinin Cevabı\n{resp}\n\n"
            f"Öğrencinin cevabını doğru açıklama ile karşılaştırarak 10 üzerinden puanla."
        )
        judge_resp, judge_dur = call_judge(judge_client, SYSTEM_PROMPT_JUDGE, judge_prompt, JUDGE_MODEL)
        score = extract_score(judge_resp)
        if score is None:
            score = 10 if exact else 0

        cp[qid] = {
            "id": qid, "category": category, "question_name": qname,
            "correct_answer": correct, "model_answer": ans,
            "model_response": resp, "judge_response": judge_resp,
            "score": score, "ask_duration": round(dur, 2),
            "judge_duration": round(judge_dur, 2), "exact_match": exact,
            "tool_calls": tools, "tool_count": len(tools),
        }
        save_checkpoint(cp)
        print(f"    Puan: {score}/10  [checkpoint kaydedildi]", flush=True)

    save_outputs(plain, cp, wrong_ids)


def _summarize(test_model, results):
    topic = defaultdict(lambda: {"tested": 0, "correct": 0, "total_score": 0})
    for r in results:
        t = r.get("question_name", "")
        topic[t]["tested"] += 1
        if r["exact_match"]:
            topic[t]["correct"] += 1
        topic[t]["total_score"] += r.get("score", 0) or 0
    scores = [r["score"] for r in results if r.get("score") is not None]
    exact = sum(1 for r in results if r["exact_match"])
    tested = len(results)
    return {
        "test_model": test_model, "judge_model": JUDGE_MODEL,
        "total_questions": tested, "tested": tested, "exact_match": exact,
        "exact_match_pct": round(exact / tested * 100, 2) if tested else 0,
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "topic_stats": {
            t: {"tested": s["tested"], "correct": s["correct"],
                "avg_score": round(s["total_score"] / s["tested"], 2) if s["tested"] else 0}
            for t, s in topic.items()
        },
    }


def _write(label, detail_obj, summary_obj, ts):
    safe = f"google_{label}"
    dp = os.path.join(RESULTS_DIR, f"benchmark3_detail_{safe}_{ts}.json")
    sp = os.path.join(RESULTS_DIR, f"benchmark3_summary_{safe}_{ts}.json")
    json.dump(detail_obj, open(dp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(summary_obj, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  Detay : {dp}")
    print(f"  Özet  : {sp}")


def save_outputs(plain, cp, wrong_ids):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    retry_list = [cp[i] for i in wrong_ids if i in cp]
    fixed = sum(1 for r in retry_list if r["exact_match"])

    print("\n--- RETRY (sadece yanlışlar) ---")
    retry_detail = {
        "test_model": f"google/{MODEL_LABEL}-retry", "judge_model": JUDGE_MODEL,
        "total_questions": len(retry_list), "tested": len(retry_list), "results": retry_list,
    }
    _write(f"{MODEL_LABEL}-retry", retry_detail, _summarize(f"google/{MODEL_LABEL}-retry", retry_list), ts)

    merged = []
    for r in plain["results"]:
        if r["id"] in cp:
            rr = dict(cp[r["id"]]); rr["sira"] = r.get("sira"); merged.append(rr)
        else:
            merged.append(r)
    print("\n--- BİRLEŞİK (100 soru) ---")
    merged_detail = {
        "test_model": f"google/{MODEL_LABEL}", "judge_model": JUDGE_MODEL,
        "total_questions": len(merged), "tested": len(merged), "results": merged,
    }
    msum = _summarize(f"google/{MODEL_LABEL}", merged)
    _write(MODEL_LABEL, merged_detail, msum, ts)

    plain_exact = sum(1 for r in plain["results"] if r["exact_match"])
    print(f"\n{'='*60}")
    print(f"  GEMINI 3.5 FLASH + YARGI MCP PRO — RETRY SONUCU")
    print(f"{'='*60}")
    print(f"Yanlış soru          : {len(wrong_ids)}")
    print(f"MCP ile düzelen       : {fixed}/{len(retry_list)}")
    print(f"Plain exact match     : {plain_exact}/100 (%{plain_exact})")
    print(f"MCP sonrası (birleşik): {msum['exact_match']}/100 (%{msum['exact_match_pct']})")
    print(f"Birleşik ort. puan    : {msum['avg_score']}/10")
    print("\nSoru bazında:")
    for qid in wrong_ids:
        if qid in cp:
            r = cp[qid]
            plain_r = next(x for x in plain["results"] if x["id"] == qid)
            mark = "✓ DÜZELDİ" if r["exact_match"] else "✗"
            print(f"  ID {qid:>3}: plain={plain_r['model_answer']} → mcp={r['model_answer']} "
                  f"(doğru={r['correct_answer']}) {mark} [{r['tool_count']} araç]")


if __name__ == "__main__":
    asyncio.run(main_async())
