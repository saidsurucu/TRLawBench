# -*- coding: utf-8 -*-
"""
TRLawBench Benchmark 3 — Claude Fable 5 via Anthropic API
Test: anthropic SDK (native, streaming, adaptive thinking) — Judge: Gemini 3.1 Pro via OpenRouter

Fable 5 API notları (claude-api skill'inden):
- temperature/top_p/top_k KALDIRILDI → gönderilirse 400
- Adaptive thinking only; Fable 5'te thinking:{disabled} bile 400 → ya adaptive ya hiç gönderme
- max_tokens yüksekken (>16K) non-streaming SDK timeout guard'a takılır → streaming şart
"""

import json
import time
import re
import os
from collections import defaultdict

import anthropic
from openai import OpenAI, RateLimitError, APIError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------
JSON_PATH = "benchmark3_questions.json"
RESULTS_DIR = "results"

TEST_MODEL = "claude-fable-5"
TEST_EFFORT = "high"          # low | medium | high | xhigh | max
MODEL_LABEL = "fable-5-thinking"   # sonuç dosyası adı için
MAX_TOKENS = 32000            # adaptive thinking + açık uçlu cevap için bol pay

JUDGE_MODEL = "google/gemini-3.1-pro-preview"

SYSTEM_PROMPT_ASK = """Sen Türkiye Cumhuriyeti hukukuna göre hukuk sorularını yanıtlayan uzman bir hukuk profesörüsün.

Sana bir hukuk sorusu ve seçenekler verilecek. Şunları yap:
1. Soruyu analiz et
2. İlgili kanun maddelerini, içtihat kararlarını veya hukuk doktrinini açıkla
3. Seçenekleri tek tek değerlendir
4. Doğru cevabı gerekçesiyle birlikte belirt

Cevabının SONUNDA mutlaka "CEVAP: X" formatında (X = A, B, C, D veya E) doğru şıkkı belirt."""

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


# --------------------------------------------------
# Anthropic API call (test model) — streaming + adaptive thinking
# --------------------------------------------------
def call_anthropic(client: anthropic.Anthropic, system: str, prompt: str,
                   max_retries: int = 4, base_wait: int = 10) -> tuple[str, float]:
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            # Streaming: max_tokens yüksek + adaptive thinking → timeout guard'ı aşmak için şart.
            # temperature GÖNDERME (Fable 5'te 400 verir).
            with client.messages.stream(
                model=TEST_MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                thinking={"type": "adaptive"},
                output_config={"effort": TEST_EFFORT},
            ) as stream:
                message = stream.get_final_message()

            duration = time.time() - t0
            text = ""
            for block in message.content:
                if block.type == "text":
                    text += block.text
            return text.strip() or "[EMPTY_RESPONSE]", duration

        except anthropic.RateLimitError:
            if attempt == max_retries:
                return "[RATE_LIMIT_ERROR]", time.time() - t0
            wait = base_wait * (2 ** attempt)
            print(f"    Anthropic rate limit, {wait}s bekleniyor...")
            time.sleep(wait)

        except anthropic.APIError as e:
            if attempt == max_retries:
                return f"[API_ERROR: {e}]", time.time() - t0
            wait = base_wait * (2 ** attempt)
            print(f"    Anthropic API error, {wait}s bekleniyor: {e}")
            time.sleep(wait)

        except Exception as e:
            return f"[UNEXPECTED_ERROR: {type(e).__name__}: {e}]", time.time() - t0

    return "[MAX_RETRIES_REACHED]", time.time() - t0


# --------------------------------------------------
# OpenRouter API call (judge model)
# --------------------------------------------------
def call_judge(client: OpenAI, system: str, prompt: str, model: str,
               max_retries: int = 4, base_wait: int = 5) -> tuple[str, float]:
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=8192,
            )
            duration = time.time() - t0
            if not response or not response.choices:
                return "[EMPTY_CHOICES]", duration
            raw = (response.choices[0].message.content or "").strip()
            return raw or "[EMPTY_RESPONSE]", duration

        except RateLimitError:
            if attempt == max_retries:
                return "[RATE_LIMIT_ERROR]", time.time() - t0
            wait = base_wait * (2 ** attempt)
            print(f"    Judge rate limit, {wait}s bekleniyor...")
            time.sleep(wait)

        except APIError as e:
            code = getattr(e, "status_code", None)
            if attempt == max_retries:
                return f"[API_ERROR_{code}]", time.time() - t0
            wait = base_wait * (2 ** attempt)
            print(f"    Judge APIError ({code}), {wait}s bekleniyor...")
            time.sleep(wait)

        except Exception as e:
            return f"[UNEXPECTED_ERROR: {type(e).__name__}]", time.time() - t0

    return "[MAX_RETRIES_REACHED]", time.time() - t0


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


def extract_score(text: str) -> int | None:
    m = re.search(r"PUAN:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return min(int(m.group(1)), 10)
    m = re.search(r"(\d+)\s*/\s*10", text)
    if m:
        return min(int(m.group(1)), 10)
    return None


# --------------------------------------------------
def run_benchmark(test_client, judge_client, questions):
    results = []
    topic_stats = defaultdict(lambda: {"tested": 0, "total_score": 0, "correct": 0, "api_err": 0})
    error_codes = ("[EMPTY_RESPONSE]", "[MAX_RETRIES_REACHED]", "[RATE_LIMIT_ERROR]",
                   "[AUTH_ERROR]", "[API_ERROR", "[UNEXPECTED_ERROR]", "[EMPTY_CHOICES]")

    for i, q in enumerate(questions, 1):
        qname = q.get("question_name", "")
        qtext = q.get("question", "")
        options = q.get("options", {})
        correct_answer = q.get("correct_answer", "").strip().upper()
        solution = q.get("solution", "")
        qid = q.get("general_id", q.get("id", i))
        category = q.get("category", "")

        print(f"\n[{i}/{len(questions)}] ID:{qid} — {category}", flush=True)

        if not qtext or not options or not correct_answer:
            print("  Geçersiz, atlandı.", flush=True)
            continue

        topic_stats[qname]["tested"] += 1

        opts_upper = {k.upper(): v for k, v in options.items()}
        valid_opts = sorted(k for k in opts_upper if "A" <= k <= "E")
        opts_text = "\n".join(f"{k}: {opts_upper[k]}" for k in valid_opts)

        ask_prompt = (
            f"Soru:\n{qtext}\n\n"
            f"Seçenekler:\n{opts_text}\n\n"
            f"Bu soruyu analiz et, ilgili hukuk kurallarını açıkla ve doğru cevabı gerekçesiyle birlikte belirt."
        )

        model_response, ask_duration = call_anthropic(test_client, SYSTEM_PROMPT_ASK, ask_prompt)

        is_error = any(model_response.startswith(c) for c in error_codes)
        if is_error:
            topic_stats[qname]["api_err"] += 1
            print(f"  HATA: {model_response[:80]}", flush=True)
            results.append({
                "sira": i, "id": qid, "category": category, "question_name": qname,
                "correct_answer": correct_answer, "model_answer": None,
                "model_response": model_response[:200], "judge_response": "",
                "score": 0, "ask_duration": ask_duration, "judge_duration": 0,
                "exact_match": False,
            })
            continue

        model_answer = extract_answer(model_response)
        exact = model_answer == correct_answer
        if exact:
            topic_stats[qname]["correct"] += 1

        print(f"  Model: {model_answer or '?'} (doğru: {correct_answer}) {'✓' if exact else '✗'} ({ask_duration:.1f}s)", flush=True)

        # Judge
        judge_prompt = (
            f"## Soru\n{qtext}\n\n"
            f"## Seçenekler\n{opts_text}\n\n"
            f"## Doğru Cevap: {correct_answer}\n\n"
            f"## Doğru Açıklama\n{solution}\n\n"
            f"## Öğrencinin Cevabı\n{model_response}\n\n"
            f"Öğrencinin cevabını doğru açıklama ile karşılaştırarak 10 üzerinden puanla."
        )

        judge_response, judge_duration = call_judge(judge_client, SYSTEM_PROMPT_JUDGE, judge_prompt, JUDGE_MODEL)

        score = extract_score(judge_response)
        if score is None:
            score = 10 if exact else 0
            print(f"  Jüri puan çıkarılamadı, varsayılan: {score}", flush=True)

        topic_stats[qname]["total_score"] += score
        print(f"  Puan: {score}/10 (jüri: {judge_duration:.1f}s)", flush=True)

        results.append({
            "sira": i, "id": qid, "category": category, "question_name": qname,
            "correct_answer": correct_answer, "model_answer": model_answer,
            "model_response": model_response, "judge_response": judge_response,
            "score": score, "ask_duration": round(ask_duration, 2),
            "judge_duration": round(judge_duration, 2), "exact_match": exact,
        })

    return {
        "test_model": f"anthropic/{MODEL_LABEL}",
        "judge_model": JUDGE_MODEL,
        "total_questions": len(questions),
        "tested": sum(s["tested"] for s in topic_stats.values()),
        "results": results,
        "topic_stats": dict(topic_stats),
    }


def save_results(r: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_model = r["test_model"].replace("/", "_").replace(":", "_")
    ts = time.strftime("%Y%m%d_%H%M%S")

    detail_path = os.path.join(RESULTS_DIR, f"benchmark3_detail_{safe_model}_{ts}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"\nDetay: {detail_path}")

    results = r["results"]
    scores = [x["score"] for x in results if x["score"] is not None]
    exact_matches = sum(1 for x in results if x["exact_match"])
    tested = r["tested"]
    summary = {
        "test_model": r["test_model"],
        "judge_model": r["judge_model"],
        "total_questions": r["total_questions"],
        "tested": tested,
        "exact_match": exact_matches,
        "exact_match_pct": round(exact_matches / tested * 100, 2) if tested else 0,
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "topic_stats": {
            t: {
                "tested": s["tested"],
                "correct": s["correct"],
                "avg_score": round(s["total_score"] / s["tested"], 2) if s["tested"] > 0 else 0,
            }
            for t, s in r["topic_stats"].items()
        },
    }
    spath = os.path.join(RESULTS_DIR, f"benchmark3_summary_{safe_model}_{ts}.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Özet: {spath}")

    # Report
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n{'='*65}")
    print(f"  BENCHMARK 3 — {r['test_model']}")
    print(f"{'='*65}")
    print(f"Exact Match : {exact_matches}/{tested} ({exact_matches/tested*100:.1f}%)" if tested else "")
    print(f"Ort. Puan   : {avg_score:.2f}/10")
    for topic, st in sorted(r["topic_stats"].items()):
        if st["tested"] > 0:
            avg = st["total_score"] / st["tested"]
            print(f"  {topic}: {avg:.1f}/10 (exact: {st['correct']}/{st['tested']})")


def main():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    if not anthropic_key:
        print("ANTHROPIC_API_KEY env var gerekli!")
        return
    if not openrouter_key:
        print("OPENROUTER_API_KEY env var gerekli!")
        return

    test_client = anthropic.Anthropic(api_key=anthropic_key)
    judge_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

    print(f"Test : {TEST_MODEL} (Anthropic native API, streaming, adaptive thinking, effort={TEST_EFFORT})")
    print(f"Judge: {JUDGE_MODEL} (OpenRouter)")

    with open(JSON_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"{len(questions)} soru yüklendi.\n")

    result = run_benchmark(test_client, judge_client, questions)
    save_results(result)
    print("\nTamamlandı.")


if __name__ == "__main__":
    main()
