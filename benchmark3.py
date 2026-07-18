# -*- coding: utf-8 -*-
"""
TRLawBench Benchmark 3 - Açık Uçlu Değerlendirme
Aşama 1: Model soruyu şıklarla birlikte açıklamalı cevaplar
Aşama 2: Gemini 3.1 Pro cevabı doğru çözümle karşılaştırıp 10 üzerinden puanlar
"""

import json
import time
import re
import os
import csv
import argparse
import traceback
from collections import defaultdict

try:
    from openai import OpenAI, RateLimitError, APIError, AuthenticationError
except ImportError:
    print("Hata: 'openai' kütüphanesi bulunamadı.")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------
# Sabitler
# --------------------------------------------------
JSON_PATH = "benchmark3_questions.json"
RESULTS_DIR = "results"

# Aşama 1: Modele soruyu sor
SYSTEM_PROMPT_ASK = """Sen Türkiye Cumhuriyeti hukukuna göre hukuk sorularını yanıtlayan uzman bir hukuk profesörüsün.

Sana bir hukuk sorusu ve seçenekler verilecek. Şunları yap:
1. Soruyu analiz et
2. İlgili kanun maddelerini, içtihat kararlarını veya hukuk doktrinini açıkla
3. Seçenekleri tek tek değerlendir
4. Doğru cevabı gerekçesiyle birlikte belirt

Cevabının SONUNDA mutlaka "CEVAP: X" formatında (X = A, B, C, D veya E) doğru şıkkı belirt."""

# Aşama 2: Gemini değerlendirmesi
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

ERROR_CODES = (
    "[EMPTY_RESPONSE]", "[MAX_RETRIES_REACHED]", "[RATE_LIMIT_ERROR]",
    "[AUTH_ERROR]", "[API_ERROR", "[UNEXPECTED_ERROR]", "[EMPTY_CHOICES]",
)


# --------------------------------------------------
# API Çağrısı
# --------------------------------------------------
def call_model(client: OpenAI, system: str, prompt: str, model: str,
               max_retries: int = 4, base_wait: int = 5, verbose: bool = True,
               enable_thinking: bool = False, max_tokens: int = 8192,
               openrouter_reasoning: bool = False,
               reasoning_effort: str | None = None) -> tuple[str, float]:
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            extra_body: dict = {}
            if enable_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}
            if openrouter_reasoning:
                reasoning_cfg: dict = {"enabled": True}
                if reasoning_effort:
                    reasoning_cfg["effort"] = reasoning_effort
                extra_body["reasoning"] = reasoning_cfg
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
            duration = time.time() - t0
            if not response or not response.choices:
                if attempt == max_retries:
                    return "[EMPTY_CHOICES]", duration
                wait = base_wait * (2 ** attempt)
                if verbose:
                    print(f"    Boş choices, {wait}s bekleniyor...")
                time.sleep(wait)
                continue
            raw = (response.choices[0].message.content or "").strip()
            if raw:
                return raw, duration
            # Boş içerik (reasoning modelde sağlayıcı ara sıra boş döner) → retry
            if attempt == max_retries:
                return "[EMPTY_RESPONSE]", duration
            wait = base_wait * (2 ** attempt)
            if verbose:
                print(f"    Boş cevap, {wait}s bekleniyor...")
            time.sleep(wait)

        except RateLimitError:
            if attempt == max_retries:
                return "[RATE_LIMIT_ERROR]", time.time() - t0
            wait = base_wait * (2 ** attempt)
            if verbose:
                print(f"    Rate limit, {wait}s bekleniyor...")
            time.sleep(wait)

        except AuthenticationError as e:
            return "[AUTH_ERROR]", time.time() - t0

        except APIError as e:
            code = getattr(e, "status_code", None) or type(e).__name__
            if attempt == max_retries:
                return f"[API_ERROR_{code}]", time.time() - t0
            wait = base_wait * (2 ** attempt)
            if verbose:
                print(f"    APIError ({code}), {wait}s bekleniyor...")
            time.sleep(wait)

        except Exception as e:
            return f"[UNEXPECTED_ERROR: {type(e).__name__}]", time.time() - t0

    return "[MAX_RETRIES_REACHED]", time.time() - t0


def extract_answer(text: str) -> str | None:
    """Cevaptan şık harfini çıkar."""
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
    """Değerlendirmeden puanı çıkar."""
    m = re.search(r"PUAN:\s*(\d+)", text, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        return min(score, 10)
    m = re.search(r"(\d+)\s*/\s*10", text)
    if m:
        return min(int(m.group(1)), 10)
    return None


# --------------------------------------------------
# Ana Benchmark
# --------------------------------------------------
def run_benchmark(
    test_client: OpenAI,
    judge_client: OpenAI,
    questions: list[dict],
    test_model: str,
    judge_model: str,
    verbose: bool = True,
    skip_first: int = 0,
    enable_thinking: bool = False,
    max_tokens: int = 8192,
    openrouter_reasoning: bool = False,
    reasoning_effort: str | None = None,
) -> dict:

    results = []
    topic_stats = defaultdict(lambda: {"tested": 0, "total_score": 0, "correct": 0, "api_err": 0})

    for i, q in enumerate(questions, 1):
        if i <= skip_first:
            continue

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

        # Seçenekleri hazırla
        opts_upper = {k.upper(): v for k, v in options.items()}
        valid_opts = sorted(k for k in opts_upper if "A" <= k <= "E")
        opts_text = "\n".join(f"{k}: {opts_upper[k]}" for k in valid_opts)

        # --- AŞAMA 1: Modele sor ---
        ask_prompt = (
            f"Soru:\n{qtext}\n\n"
            f"Seçenekler:\n{opts_text}\n\n"
            f"Bu soruyu analiz et, ilgili hukuk kurallarını açıkla ve doğru cevabı gerekçesiyle birlikte belirt."
        )

        model_response, ask_duration = call_model(test_client, SYSTEM_PROMPT_ASK, ask_prompt, test_model, verbose=verbose, enable_thinking=enable_thinking, max_tokens=max_tokens, openrouter_reasoning=openrouter_reasoning, reasoning_effort=reasoning_effort)

        is_error = any(model_response.startswith(c) for c in ERROR_CODES)
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

        if verbose:
            print(f"  Model: {model_answer or '?'} (doğru: {correct_answer}) {'✓' if exact else '✗'} ({ask_duration:.1f}s)", flush=True)
            if model_response and len(model_response) > 100:
                print(f"  Cevap özet: {model_response[:150]}...", flush=True)

        # --- AŞAMA 2: Gemini değerlendir ---
        judge_prompt = (
            f"## Soru\n{qtext}\n\n"
            f"## Seçenekler\n{opts_text}\n\n"
            f"## Doğru Cevap: {correct_answer}\n\n"
            f"## Doğru Açıklama\n{solution}\n\n"
            f"## Öğrencinin Cevabı\n{model_response}\n\n"
            f"Öğrencinin cevabını doğru açıklama ile karşılaştırarak 10 üzerinden puanla."
        )

        judge_response, judge_duration = call_model(judge_client, SYSTEM_PROMPT_JUDGE, judge_prompt, judge_model, verbose=False)

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
        "test_model": test_model,
        "judge_model": judge_model,
        "total_questions": len(questions),
        "tested": sum(s["tested"] for s in topic_stats.values()),
        "results": results,
        "topic_stats": dict(topic_stats),
    }


# --------------------------------------------------
# Rapor ve Kayıt
# --------------------------------------------------
def print_report(r: dict):
    tested = r["tested"]
    results = r["results"]
    scores = [x["score"] for x in results if x["score"] is not None]
    exact_matches = sum(1 for x in results if x["exact_match"])
    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"\n{'='*65}")
    print(f"  BENCHMARK 3 SONUÇLARI")
    print(f"{'='*65}")
    print(f"Test Model  : {r['test_model']}")
    print(f"Jüri Model  : {r['judge_model']}")
    print(f"Toplam Soru : {r['total_questions']}")
    print(f"Test Edilen : {tested}")
    print(f"Exact Match : {exact_matches}/{tested} ({exact_matches/tested*100:.1f}%)" if tested else "")
    print(f"Ort. Puan   : {avg_score:.2f}/10")
    print(f"{'='*65}")

    # Puan dağılımı
    dist = defaultdict(int)
    for s in scores:
        dist[s] += 1
    print("\nPuan Dağılımı:")
    for p in range(10, -1, -1):
        if dist[p]:
            bar = "█" * dist[p]
            print(f"  {p:>2}: {bar} ({dist[p]})")

    # Konu bazlı
    print(f"\nKonu Bazlı:")
    for topic, st in sorted(r["topic_stats"].items()):
        if st["tested"] > 0:
            avg = st["total_score"] / st["tested"]
            em = st["correct"]
            print(f"  {topic}: {avg:.1f}/10 (exact: {em}/{st['tested']})")


def save_results(r: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_model = r["test_model"].replace("/", "_").replace(":", "_")
    ts = time.strftime("%Y%m%d_%H%M%S")

    # TSV
    tsv_path = os.path.join(RESULTS_DIR, f"benchmark3_{safe_model}_{ts}.tsv")
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["Sıra", "ID", "Kategori", "Doğru", "Model", "Exact", "Puan", "Süre(sn)"])
        for x in r["results"]:
            w.writerow([
                x["sira"], x["id"], x["category"], x["correct_answer"],
                x["model_answer"] or "", "✓" if x["exact_match"] else "✗",
                x["score"], x["ask_duration"],
            ])
    print(f"\nTSV: {tsv_path}")

    # Detaylı JSON
    detail_path = os.path.join(RESULTS_DIR, f"benchmark3_detail_{safe_model}_{ts}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"Detay: {detail_path}")

    # Özet JSON
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


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TRLawBench Benchmark 3 - Açık Uçlu Değerlendirme")
    parser.add_argument("--model", type=str, required=True, help="Test edilecek model")
    parser.add_argument("--judge", type=str, default="google/gemini-2.5-pro-preview-05-06",
                        help="Jüri model (varsayılan: Gemini 2.5 Pro)")
    parser.add_argument("--model-base-url", type=str, default=None,
                        help="Test modeli base URL (varsayılan: OpenRouter)")
    parser.add_argument("--model-api-key-env", type=str, default="OPENROUTER_API_KEY",
                        help="Test modeli API key env var")
    parser.add_argument("--judge-base-url", type=str, default=None,
                        help="Jüri modeli base URL (varsayılan: OpenRouter)")
    parser.add_argument("--judge-api-key-env", type=str, default="OPENROUTER_API_KEY",
                        help="Jüri modeli API key env var")
    parser.add_argument("--data", type=str, default=JSON_PATH)
    parser.add_argument("--skip", type=int, default=0, help="İlk N soruyu atla")
    parser.add_argument("--limit", type=int, default=0, help="Sadece ilk N soruyu sor (0=hepsi)")
    parser.add_argument("--only-ids", type=str, default="",
                        help="Virgülle ayrılmış sıra numaraları (1-indexed). Ör: 17,26,36")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Model max_tokens (thinking'de daha yüksek gerekebilir)")
    parser.add_argument("--think", action="store_true",
                        help="Gemma 4 thinking mode (chat_template_kwargs={'enable_thinking': True})")
    parser.add_argument("--reasoning", action="store_true",
                        help="OpenRouter reasoning mode (extra_body={'reasoning': {'enabled': True}})")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="OpenRouter reasoning effort seviyesi (low/medium/high)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Test modeli client
    test_api_key = os.getenv(args.model_api_key_env, "no-key")
    test_base = args.model_base_url or "https://openrouter.ai/api/v1"
    test_client = OpenAI(base_url=test_base, api_key=test_api_key)
    print(f"Test model: {args.model} @ {test_base}")

    # Jüri modeli client
    judge_api_key = os.getenv(args.judge_api_key_env, "no-key")
    judge_base = args.judge_base_url or "https://openrouter.ai/api/v1"
    judge_client = OpenAI(base_url=judge_base, api_key=judge_api_key)
    print(f"Jüri model: {args.judge} @ {judge_base}")

    # Soruları yükle
    with open(args.data, encoding="utf-8") as f:
        questions = json.load(f)
    if args.only_ids:
        wanted = {int(x.strip()) for x in args.only_ids.split(",") if x.strip()}
        questions = [q for i, q in enumerate(questions, 1) if i in wanted]
        print(f"Sadece {len(questions)} seçili soru sorulacak (ID'ler: {sorted(wanted)})")
    elif args.limit > 0:
        questions = questions[:args.limit]
    print(f"{len(questions)} soru yüklendi.")

    if args.think:
        print("Thinking mode: AÇIK (chat_template_kwargs={'enable_thinking': True})")
    if args.reasoning:
        eff = f" (effort={args.reasoning_effort})" if args.reasoning_effort else ""
        print(f"Reasoning mode: AÇIK (OpenRouter extra_body){eff}")
    print(f"max_tokens: {args.max_tokens}")

    # Benchmark
    result = run_benchmark(
        test_client=test_client,
        judge_client=judge_client,
        questions=questions,
        test_model=args.model,
        judge_model=args.judge,
        verbose=not args.quiet,
        skip_first=args.skip,
        enable_thinking=args.think,
        max_tokens=args.max_tokens,
        openrouter_reasoning=args.reasoning,
        reasoning_effort=args.reasoning_effort,
    )

    print_report(result)
    save_results(result)
    print("\nTamamlandı.")


if __name__ == "__main__":
    main()
