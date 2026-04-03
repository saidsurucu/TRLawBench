# -*- coding: utf-8 -*-
"""
TRLawBench - Türk Hukuku LLM Benchmark Script
OpenRouter API üzerinden çeşitli modelleri test eder.
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
    print("Hata: 'openai' kütüphanesi bulunamadı. Yüklemek için: pip install openai")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv yoksa ortam değişkenlerinden okur

# --------------------------------------------------
# Sabitler
# --------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
JSON_PATH = os.path.join(DATA_DIR, "osym_legal_questions.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SYSTEM_INSTRUCTION = (
    "Sen Türkiye Cumhuriyeti hukukuna göre Türkçe hukuk sorularını yanıtlayan bir asistansın. "
    "Sadece verilen şıklardan doğru olanının harfini (A, B, C, D veya E) yaz. "
    "Cevap olarak başka hiçbir şey yazma."
)

ERROR_CODES = (
    "[EMPTY_RESPONSE]", "[MAX_RETRIES_REACHED]", "[RATE_LIMIT_ERROR]",
    "[AUTH_ERROR]", "[API_ERROR", "[UNEXPECTED_ERROR]", "[EMPTY_CHOICES]",
)


# --------------------------------------------------
# API İstemcisi
# --------------------------------------------------
def create_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# --------------------------------------------------
# Model Çağrısı
# --------------------------------------------------
def ask_model(
    client: OpenAI,
    prompt: str,
    model_name: str,
    enable_reasoning: bool = False,
    verbose: bool = True,
    max_retries: int = 4,
    base_wait: int = 5,
) -> tuple[str, float | None, dict | None]:
    """OpenRouter'a istek gönderir. (yanıt, süre, info) döndürür."""

    if verbose:
        print(f"  Prompt (ilk 200): {prompt[:200]}...")

    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            extra_body = {}
            if enable_reasoning:
                extra_body["reasoning"] = {"enabled": True}

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": SYSTEM_INSTRUCTION + "\n\n" + prompt}],
                extra_body=extra_body if extra_body else None,
            )
            duration = time.time() - t0

            if not response or not response.choices:
                return "[EMPTY_CHOICES]", duration, None

            message = response.choices[0].message
            raw_text = (message.content or "").strip()

            info = {
                "reasoning_content": None,
                "reasoning_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            # Reasoning içeriğini al
            if hasattr(message, "reasoning_details") and message.reasoning_details:
                if isinstance(message.reasoning_details, list):
                    parts = []
                    for d in message.reasoning_details:
                        c = getattr(d, "content", None) or (d.get("content") if isinstance(d, dict) else None)
                        if c:
                            parts.append(c)
                    info["reasoning_content"] = "\n".join(parts) if parts else None
                elif hasattr(message.reasoning_details, "content"):
                    info["reasoning_content"] = message.reasoning_details.content
            if hasattr(message, "reasoning") and message.reasoning:
                info["reasoning_content"] = message.reasoning

            # Token kullanımı
            if hasattr(response, "usage") and response.usage:
                info["prompt_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
                info["completion_tokens"] = getattr(response.usage, "completion_tokens", 0) or 0
                info["total_tokens"] = getattr(response.usage, "total_tokens", 0) or 0
                if hasattr(response.usage, "reasoning_tokens"):
                    info["reasoning_tokens"] = response.usage.reasoning_tokens or 0
                elif hasattr(response.usage, "completion_tokens_details"):
                    det = response.usage.completion_tokens_details
                    if hasattr(det, "reasoning_tokens"):
                        info["reasoning_tokens"] = det.reasoning_tokens or 0

            if verbose:
                print(f"  Yanıt: '{raw_text}' ({duration:.2f}s)")

            return raw_text or "[EMPTY_RESPONSE]", duration, info

        except RateLimitError:
            if attempt == max_retries:
                return "[RATE_LIMIT_ERROR]", None, None
            wait = base_wait * (2 ** attempt)
            print(f"  Rate limit, {wait}s bekleniyor... (deneme {attempt+1})")
            time.sleep(wait)

        except AuthenticationError as e:
            print(f"  Kimlik doğrulama hatası: {e}")
            return "[AUTH_ERROR]", None, None

        except APIError as e:
            if attempt == max_retries:
                return f"[API_ERROR_{e.status_code}]", None, None
            wait = base_wait * (2 ** attempt)
            print(f"  API hatası ({e.status_code}), {wait}s bekleniyor...")
            time.sleep(wait)

        except Exception as e:
            print(f"  Beklenmedik hata: {type(e).__name__}: {e}")
            traceback.print_exc()
            return "[UNEXPECTED_ERROR]", None, None

    return "[MAX_RETRIES_REACHED]", None, None


# --------------------------------------------------
# Cevap Temizleme
# --------------------------------------------------
def clean_answer(ans: str | None) -> str | None:
    if not ans:
        return None
    if any(ans.startswith(c) for c in ERROR_CODES):
        return None

    # Tek harf A-E
    m = re.search(r"^\s*([A-Ea-e])\s*$", ans)
    if m:
        return m.group(1).upper()

    # "Cevap: X" formatı
    m = re.search(r"(?:cevap|answer)[:\s]*([A-Ea-e])", ans, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Metinde geçen ilk tek harf
    stripped = ans.strip().upper()
    if len(stripped) == 1 and "A" <= stripped <= "E":
        return stripped

    return None


# --------------------------------------------------
# Ana Test Döngüsü
# --------------------------------------------------
def run_benchmark(
    client: OpenAI,
    questions: list[dict],
    model_name: str,
    enable_reasoning: bool = False,
    verbose: bool = True,
) -> dict:
    """Benchmark'ı çalıştırır ve sonuçları döndürür."""

    correct = 0
    wrong = 0
    skipped = 0
    api_err = 0
    details = []
    tsv_rows = []
    response_times = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    reasoning_contents = []
    topic_stats = defaultdict(lambda: {"tested": 0, "correct": 0, "api_err": 0})

    t_start = time.time()

    for i, q in enumerate(questions, start=1):
        qname = q.get("question_name", "Bilinmeyen")
        qtext = q.get("question", "")
        options = q.get("options", {})
        truth_raw = q.get("answer")
        qid = q.get("id", i)

        print(f"\n[{i}/{len(questions)}] ID:{qid} — {qname}")

        # Validasyon
        if not qtext or not options or not truth_raw or not isinstance(truth_raw, str):
            print("  Geçersiz format, atlandı.")
            skipped += 1
            continue

        truth = truth_raw.strip().upper()
        opts_upper = {k.upper(): v for k, v in options.items()}
        if truth not in opts_upper:
            print("  Doğru cevap seçeneklerde yok, atlandı.")
            skipped += 1
            continue

        valid_opts = sorted(k for k in opts_upper if "A" <= k <= "E")
        if not valid_opts:
            print("  Geçerli seçenek yok, atlandı.")
            skipped += 1
            continue

        topic_stats[qname]["tested"] += 1

        opts_text = "\n".join(f"{k}: {opts_upper[k]}" for k in valid_opts)
        prompt = (
            f"Soru:\n{qtext}\n\n"
            f"Seçenekler:\n{opts_text}\n\n"
            f"Türk Hukukuna göre doğru seçenek hangisidir? Sadece seçeneğin harfini (A, B, C, D veya E) yaz."
        )

        try:
            raw, duration, info = ask_model(
                client, prompt, model_name, enable_reasoning, verbose
            )
        except Exception as e:
            api_err += 1
            wrong += 1
            topic_stats[qname]["api_err"] += 1
            details.append({"index": i, "id": qid, "question_name": qname, "hata": str(e)})
            tsv_rows.append([i, qid, qname, truth, "[PYTHON_ERROR]", "HATA", "N/A", "0"])
            if isinstance(e, AuthenticationError):
                break
            time.sleep(5)
            continue

        if duration is not None:
            response_times.append(duration)
        if info:
            total_prompt_tokens += info.get("prompt_tokens", 0)
            total_completion_tokens += info.get("completion_tokens", 0)
            total_reasoning_tokens += info.get("reasoning_tokens", 0)
            if info.get("reasoning_content"):
                reasoning_contents.append({
                    "question_id": qid,
                    "question_name": qname,
                    "reasoning": info["reasoning_content"],
                })

        is_error = any(raw.startswith(c) for c in ERROR_CODES)
        if is_error:
            api_err += 1
            wrong += 1
            topic_stats[qname]["api_err"] += 1
            details.append({"index": i, "id": qid, "question_name": qname, "hata": raw})
            tsv_rows.append([i, qid, qname, truth, raw, "HATA", f"{duration:.3f}" if duration else "N/A", "0"])
            continue

        cleaned = clean_answer(raw)
        print(f"  Temiz: {cleaned} — Doğru: {truth}", end=" ")

        if cleaned == truth:
            correct += 1
            topic_stats[qname]["correct"] += 1
            print("✓")
            result_str = "DOĞRU"
            model_out = cleaned
        else:
            wrong += 1
            details.append({
                "index": i, "id": qid, "question_name": qname,
                "correct": truth, "model_raw": raw, "model_cleaned": cleaned,
            })
            print("✗")
            result_str = "YANLIŞ"
            model_out = cleaned if cleaned else f"[RAW:{raw[:50]}]"

        r_tok = str(info.get("reasoning_tokens", 0)) if info else "0"
        tsv_rows.append([
            i, qid, qname, truth, model_out, result_str,
            f"{duration:.3f}" if duration else "N/A", r_tok,
        ])

    total_time = time.time() - t_start
    tested = len(questions) - skipped

    return {
        "model": model_name,
        "reasoning": enable_reasoning,
        "total_questions": len(questions),
        "tested": tested,
        "correct": correct,
        "wrong": wrong,
        "api_err": api_err,
        "skipped": skipped,
        "total_time": total_time,
        "response_times": response_times,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "topic_stats": dict(topic_stats),
        "details": details,
        "tsv_rows": tsv_rows,
        "reasoning_contents": reasoning_contents,
    }


# --------------------------------------------------
# Rapor ve Kayıt
# --------------------------------------------------
def print_report(r: dict):
    tested = r["tested"]
    print("\n" + "=" * 65)
    print(" " * 22 + "GENEL RAPOR")
    print("=" * 65)
    print(f"Model                : {r['model']}")
    print(f"Reasoning            : {r['reasoning']}")
    print(f"Toplam Soru          : {r['total_questions']}")
    print(f"Test Edilen          : {tested}")
    print(f"Doğru                : {r['correct']}")
    print(f"Yanlış (Model)       : {r['wrong'] - r['api_err']}")
    print(f"API Hataları         : {r['api_err']}")
    print(f"Atlanan              : {r['skipped']}")
    print(f"Toplam Süre          : {r['total_time']:.2f}s")

    print("\n--- Token İstatistikleri ---")
    print(f"Prompt Tokens        : {r['total_prompt_tokens']:,}")
    print(f"Completion Tokens    : {r['total_completion_tokens']:,}")
    print(f"Reasoning Tokens     : {r['total_reasoning_tokens']:,}")

    if r["response_times"]:
        avg = sum(r["response_times"]) / len(r["response_times"])
        print(f"Ort. Yanıt Süresi    : {avg:.3f}s")

    if tested > 0:
        acc = (r["correct"] / tested) * 100
        print(f"\n*** BAŞARI ORANI: {acc:.2f}% ***")
    print("=" * 65)

    # Konu bazlı
    print("\n" + "=" * 20 + " KONU BAZLI " + "=" * 20)
    for topic, st in sorted(r["topic_stats"].items()):
        if st["tested"] > 0:
            a = (st["correct"] / st["tested"]) * 100
            print(f"  {topic}: {a:.1f}% ({st['correct']}/{st['tested']})")
    print("=" * 52)

    # Hata detayları (ilk 20)
    if r["details"]:
        print("\n--- Hatalar/Yanlışlar (ilk 20) ---")
        for d in r["details"][:20]:
            qid = d.get("id", d["index"])
            if "hata" in d:
                print(f"  [HATA] ID {qid}: {d['hata']}")
            else:
                print(f"  [YANLIŞ] ID {qid}: Model={d.get('model_cleaned','N/A')}, Doğru={d.get('correct')}")


def save_results(r: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_model = r["model"].replace("/", "_").replace(":", "_")
    ts = time.strftime("%Y%m%d_%H%M%S")

    # TSV
    if r["tsv_rows"]:
        tsv_path = os.path.join(RESULTS_DIR, f"results_{safe_model}_{ts}.tsv")
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            w.writerow(["Sıra", "ID", "Konu", "Doğru", "Model", "Sonuç", "Süre(sn)", "Reasoning Tokens"])
            w.writerows(r["tsv_rows"])
        print(f"\nTSV kaydedildi: {tsv_path}")

    # Reasoning
    if r["reasoning_contents"] and r["reasoning"]:
        rpath = os.path.join(RESULTS_DIR, f"reasoning_{safe_model}_{ts}.json")
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(r["reasoning_contents"], f, ensure_ascii=False, indent=2)
        print(f"Reasoning kaydedildi: {rpath}")

    # Özet JSON
    summary = {
        "model": r["model"],
        "reasoning": r["reasoning"],
        "total_questions": r["total_questions"],
        "tested": r["tested"],
        "correct": r["correct"],
        "wrong": r["wrong"],
        "api_errors": r["api_err"],
        "skipped": r["skipped"],
        "accuracy": round((r["correct"] / r["tested"] * 100), 2) if r["tested"] > 0 else 0,
        "total_time_seconds": round(r["total_time"], 2),
        "avg_response_time": round(sum(r["response_times"]) / len(r["response_times"]), 3) if r["response_times"] else None,
        "tokens": {
            "prompt": r["total_prompt_tokens"],
            "completion": r["total_completion_tokens"],
            "reasoning": r["total_reasoning_tokens"],
        },
        "topic_stats": {
            t: {"accuracy": round(s["correct"] / s["tested"] * 100, 1) if s["tested"] > 0 else 0, **s}
            for t, s in r["topic_stats"].items()
        },
    }
    spath = os.path.join(RESULTS_DIR, f"summary_{safe_model}_{ts}.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Özet kaydedildi: {spath}")


# --------------------------------------------------
# CLI
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TRLawBench - Türk Hukuku LLM Benchmark")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                        help="OpenRouter model adı (varsayılan: google/gemma-3-27b-it)")
    parser.add_argument("--reasoning", action="store_true",
                        help="Reasoning modunu etkinleştir")
    parser.add_argument("--quiet", action="store_true",
                        help="Detaylı çıktıyı kapat")
    parser.add_argument("--data", type=str, default=JSON_PATH,
                        help="Soru dosyası yolu")
    parser.add_argument("--api-key-env", type=str, default="OPENROUTER_API_KEY",
                        help="API anahtarı ortam değişkeni adı (varsayılan: OPENROUTER_API_KEY)")
    args = parser.parse_args()

    # API anahtarı
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"Hata: '{args.api_key_env}' ortam değişkeni tanımlanmamış.")
        print(f"Kullanım: export {args.api_key_env}=sk-or-...")
        exit(1)
    print(f"API anahtarı alındı (Son 4: ...{api_key[-4:]})")

    # Soruları yükle
    print(f"Sorular yükleniyor: {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"{len(questions)} soru yüklendi.")

    # İstemci
    client = create_client(api_key)

    # Test
    print(f"\n{'='*65}")
    print(f"  Model: {args.model}")
    print(f"  Reasoning: {args.reasoning}")
    print(f"  Soru sayısı: {len(questions)}")
    print(f"{'='*65}\n")

    results = run_benchmark(
        client=client,
        questions=questions,
        model_name=args.model,
        enable_reasoning=args.reasoning,
        verbose=not args.quiet,
    )

    print_report(results)
    save_results(results)
    print("\nTest tamamlandı.")


if __name__ == "__main__":
    main()
