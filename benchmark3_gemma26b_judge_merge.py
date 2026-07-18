# -*- coding: utf-8 -*-
"""
Güncel Gemma 4 26B (Modal) Yargı PRO retry cevaplarını (pass1 + pass2 birleşik)
SONRADAN jüriye puanlatır ve plain run ile birleştirir.

Girdi : results/benchmark3_detail_google_gemma-4-26B-A4B-it_20260718_182524.json (plain, jürili)
        results/benchmark3_modal_gemma-4-26b-modal-yargipro_20260718_184928.json (pass1, jürisiz)
        results/benchmark3_modal_gemma-4-26b-modal-yargipro-pass2_20260718_190747.json (pass2, jürisiz)
Birleşim kuralı: pass1 doğruysa pass1 kaydı; değilse pass2 kaydı (pass2 sadece pass1
yanlışlarına koşuldu, araç zorlamalı).
Çıktı : benchmark3_{detail,summary}_google_gemma-4-26B-A4B-it-mcp-retry_{ts}.json (46 retry, jürili)
        benchmark3_{detail,summary}_google_gemma-4-26B-A4B-it-mcp_{ts}.json (100 birleşik)
"""

import json
import os
import re
import time
from collections import defaultdict

from openai import OpenAI, RateLimitError, APIError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

RESULTS_DIR = "results"
JSON_PATH = "benchmark3_questions.json"
PLAIN_PATH = "results/benchmark3_detail_google_gemma-4-26B-A4B-it_20260718_182524.json"
PASS1_PATH = "results/benchmark3_modal_gemma-4-26b-modal-yargipro_20260718_184928.json"
PASS2_PATH = "results/benchmark3_modal_gemma-4-26b-modal-yargipro-pass2_20260718_190747.json"
CHECKPOINT = os.path.join(RESULTS_DIR, "_gemma26b_judge_checkpoint.json")
JUDGE_MODEL = "google/gemini-3.1-pro-preview"
SAFE_MODEL = "google_gemma-4-26B-A4B-it"
MODEL_NAME = "google/gemma-4-26B-A4B-it"

SYSTEM_PROMPT_JUDGE = """Sen bir hukuk eğitmenisin. Bir öğrencinin hukuk sorusuna verdiği cevabı değerlendireceksin.

Sana şunlar verilecek:
- Soru
- Doğru cevap ve açıklaması
- Öğrencinin cevabı

Puanlama kriterleri (10 üzerinden):
- 10: Doğru şık seçilmiş VE gerekçe tam doğru
- 9: Doğru şık seçilmiş, gerekçede küçük eksiklik var
- 8: Doğru şık seçilmiş ama gerekçe yetersiz veya kısmen hatalı
- 7: Yanlış şık ama muhakeme yolu büyük ölçüde doğru, doğru cevaba çok yakın
- 6: Yanlış şık ama konuyu anlıyor, ilgili kanun/içtihadı biliyor
- 5: Yanlış şık, konuyu kısmen anlıyor ama önemli hatalar var
- 4: Yanlış şık, ilgili konuyu biliyor ama yanlış uyguluyor
- 3: Yanlış şık, konudan sapıyor ama bazı doğru bilgiler içeriyor
- 2: Yanlış şık, ciddi kavram hataları var
- 1: Tamamen yanlış veya alakasız cevap
- 0: Cevap yok veya anlaşılmaz

ÖNEMLİ: Cevabının EN SON SATIRI mutlaka ve yalnızca şu formatta olmalı:
PUAN: X
(X = 0-10 arası tam sayı, başka bir şey yazma)"""


def extract_score(text: str) -> int | None:
    m = re.search(r"PUAN:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return min(int(m.group(1)), 10)
    m = re.search(r"(\d+)\s*/\s*10", text)
    if m:
        return min(int(m.group(1)), 10)
    return None


def call_judge(client, prompt, max_retries=4, base_wait=5):
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                          {"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=8192)
            if resp and resp.choices:
                raw = (resp.choices[0].message.content or "").strip()
                if raw:
                    return raw, time.time() - t0
        except (RateLimitError, APIError) as e:
            if attempt == max_retries:
                return f"[API_ERROR: {type(e).__name__}]", time.time() - t0
            wait = base_wait * (2 ** attempt)
            print(f"    judge hata ({type(e).__name__}), {wait}s...", flush=True)
            time.sleep(wait)
        except Exception as e:
            return f"[UNEXPECTED_ERROR: {type(e).__name__}]", time.time() - t0
    return "[MAX_RETRIES]", 0.0


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
    return {"test_model": test_model, "judge_model": JUDGE_MODEL, "total_questions": tested,
            "tested": tested, "exact_match": exact,
            "exact_match_pct": round(exact / tested * 100, 2) if tested else 0,
            "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "topic_stats": {t: {"tested": s["tested"], "correct": s["correct"],
                                "avg_score": round(s["total_score"] / s["tested"], 2) if s["tested"] else 0}
                            for t, s in topic.items()}}


def _write(label, detail_obj, summary_obj, ts):
    safe = f"{SAFE_MODEL}{label}"
    dp = os.path.join(RESULTS_DIR, f"benchmark3_detail_{safe}_{ts}.json")
    sp = os.path.join(RESULTS_DIR, f"benchmark3_summary_{safe}_{ts}.json")
    json.dump(detail_obj, open(dp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(summary_obj, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  Detay : {dp}")
    print(f"  Özet  : {sp}")


def main():
    plain = json.load(open(PLAIN_PATH, encoding="utf-8"))
    pass1 = json.load(open(PASS1_PATH, encoding="utf-8"))
    pass2 = json.load(open(PASS2_PATH, encoding="utf-8"))
    all_q = {q.get("general_id", q.get("id")): q for q in json.load(open(JSON_PATH, encoding="utf-8"))}
    plain_by_id = {r["id"]: r for r in plain["results"]}

    # pass1 + pass2 birleşimi: pass1 doğruysa pass1, değilse pass2 (varsa)
    p2_by_id = {r["id"]: r for r in pass2["results"]}
    combined = []
    for r in pass1["results"]:
        if r["dogru"] or r["id"] not in p2_by_id:
            rec = dict(r); rec["retry_pass"] = 1
        else:
            rec = dict(p2_by_id[r["id"]]); rec["retry_pass"] = 2
        combined.append(rec)
    n_ok = sum(1 for r in combined if r["dogru"])
    print(f"Birleşik retry: {n_ok}/{len(combined)} doğru "
          f"(pass1: {sum(1 for r in combined if r['retry_pass']==1 and r['dogru'])}, "
          f"pass2: {sum(1 for r in combined if r['retry_pass']==2 and r['dogru'])})")

    cp = {}
    if os.path.exists(CHECKPOINT):
        cp = {int(k): v for k, v in json.load(open(CHECKPOINT, encoding="utf-8")).items()}
        print(f"Checkpoint: {len(cp)} soru puanlanmış.")

    judge_client = OpenAI(base_url="https://openrouter.ai/api/v1",
                          api_key=os.getenv("OPENROUTER_API_KEY"))

    print(f"{len(combined)} retry cevabı jüriye puanlatılıyor ({JUDGE_MODEL})...\n")

    for i, rr in enumerate(combined, 1):
        qid = rr["id"]
        if qid in cp:
            continue
        q = all_q[qid]
        opts_upper = {k.upper(): v for k, v in q.get("options", {}).items()}
        valid = sorted(k for k in opts_upper if "A" <= k <= "E")
        opts_text = "\n".join(f"{k}: {opts_upper[k]}" for k in valid)

        judge_prompt = (
            f"## Soru\n{q.get('question','')}\n\n"
            f"## Seçenekler\n{opts_text}\n\n"
            f"## Doğru Cevap: {rr['correct_answer']}\n\n"
            f"## Doğru Açıklama\n{q.get('solution','')}\n\n"
            f"## Öğrencinin Cevabı\n{rr['model_response']}\n\n"
            f"Öğrencinin cevabını doğru açıklama ile karşılaştırarak 10 üzerinden puanla."
        )
        judge_resp, judge_dur = call_judge(judge_client, judge_prompt)
        score = extract_score(judge_resp)
        if score is None:
            score = 10 if rr["dogru"] else 0
            print(f"  [{i}] id={qid} puan çıkarılamadı, varsayılan {score}", flush=True)

        prec = plain_by_id[qid]
        cp[qid] = {
            "sira": prec.get("sira"), "id": qid, "category": rr["category"],
            "question_name": prec.get("question_name", ""),
            "correct_answer": rr["correct_answer"], "model_answer": rr["model_answer"],
            "model_response": rr["model_response"], "judge_response": judge_resp,
            "score": score, "ask_duration": rr.get("wall_time_s", 0),
            "judge_duration": round(judge_dur, 2), "exact_match": rr["dogru"],
            "tool_calls": rr.get("tool_calls", 0), "tools_used": rr.get("tools_used", {}),
            "retry_pass": rr["retry_pass"],
        }
        json.dump({str(k): v for k, v in cp.items()},
                  open(CHECKPOINT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  [{i}/{len(combined)}] id={qid} {'✓' if rr['dogru'] else '✗'} puan={score} ({judge_dur:.0f}s)", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    retry_list = [cp[r["id"]] for r in combined]

    print(f"\n--- RETRY ({len(retry_list)} yanlış, jürili) ---")
    _write("-mcp-retry",
           {"test_model": f"{MODEL_NAME}-mcp-retry", "judge_model": JUDGE_MODEL,
            "total_questions": len(retry_list), "tested": len(retry_list), "results": retry_list},
           _summarize(f"{MODEL_NAME}-mcp-retry", retry_list), ts)

    merged = []
    for r in plain["results"]:
        merged.append(cp[r["id"]] if r["id"] in cp else r)
    print("\n--- BİRLEŞİK (100 soru) ---")
    msum = _summarize(f"{MODEL_NAME}-mcp", merged)
    _write("-mcp",
           {"test_model": f"{MODEL_NAME}-mcp", "judge_model": JUDGE_MODEL,
            "total_questions": len(merged), "tested": len(merged), "results": merged}, msum, ts)

    print(f"\n{'='*60}\n  GÜNCEL GEMMA 4 26B + YARGI PRO — BİRLEŞİK SONUÇ\n{'='*60}")
    print(f"MCP sonrası exact match: {msum['exact_match']}/100 (%{msum['exact_match_pct']})")
    print(f"MCP sonrası ort. puan  : {msum['avg_score']}/10")
    os.remove(CHECKPOINT)
    print("Checkpoint temizlendi.")


if __name__ == "__main__":
    main()
