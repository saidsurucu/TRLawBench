# -*- coding: utf-8 -*-
"""
TRLawBench Benchmark 3 — paralel sürüm.

benchmark3.py ile AYNI prompt / parametre / çıktı formatı; tek fark soruların
bir thread havuzunda eşzamanlı sorulması (openai SDK sync client thread-safe).
Her soru için: model cevabı → jüri puanı (soru içinde sıralı, sorular arası paralel).

Ek olarak checkpoint tutar: kesilirse aynı komutla kaldığı yerden devam eder.

Kullanım:
  uv run python -u benchmark3_parallel.py --model x-ai/grok-4.5 \
      --judge google/gemini-3.1-pro-preview --max-tokens 16000 --concurrency 10
"""

import argparse
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from benchmark3 import (
    JSON_PATH, RESULTS_DIR,
    SYSTEM_PROMPT_ASK, SYSTEM_PROMPT_JUDGE, ERROR_CODES,
    call_model, extract_answer, extract_score,
    print_report, save_results,
)

_print_lock = threading.Lock()
_cp_lock = threading.Lock()

# benchmark3.ERROR_CODES bazı kodları kapalı parantezle listeliyor ("[UNEXPECTED_ERROR]")
# ama call_model "[UNEXPECTED_ERROR: JSONDecodeError]" döndürüyor → startswith tutmuyor,
# hata YANLIŞ olarak sayılıyordu. Burada prefix olarak eşleştir.
ERROR_PREFIXES = tuple(c.rstrip("]") for c in ERROR_CODES)


def is_error(resp: str) -> bool:
    return resp.startswith(ERROR_PREFIXES)


def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def checkpoint_path(model: str) -> str:
    safe = model.replace("/", "_").replace(":", "_")
    return os.path.join(RESULTS_DIR, f"_{safe}_parallel_checkpoint.json")


def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return {int(k): v for k, v in json.load(f).items()}
        except Exception:
            return {}
    return {}


def save_checkpoint(path: str, cp: dict):
    with _cp_lock:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in cp.items()}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


def build_opts_text(options: dict) -> tuple[str, list[str]]:
    opts_upper = {k.upper(): v for k, v in options.items()}
    valid = sorted(k for k in opts_upper if "A" <= k <= "E")
    return "\n".join(f"{k}: {opts_upper[k]}" for k in valid), valid


def do_question(i: int, q: dict, total: int, test_client: OpenAI, judge_client: OpenAI,
                test_model: str, judge_model: str, max_tokens: int,
                openrouter_reasoning: bool, reasoning_effort: str | None,
                enable_thinking: bool = False) -> dict | None:
    qname = q.get("question_name", "")
    qtext = q.get("question", "")
    options = q.get("options", {})
    correct = q.get("correct_answer", "").strip().upper()
    solution = q.get("solution", "")
    qid = q.get("general_id", q.get("id", i))
    category = q.get("category", "")

    if not qtext or not options or not correct:
        log(f"[{i}/{total}] ID:{qid} — geçersiz, atlandı.")
        return None

    opts_text, _ = build_opts_text(options)

    ask_prompt = (
        f"Soru:\n{qtext}\n\n"
        f"Seçenekler:\n{opts_text}\n\n"
        f"Bu soruyu analiz et, ilgili hukuk kurallarını açıkla ve doğru cevabı gerekçesiyle birlikte belirt."
    )
    # call_model kendi içinde retry yapıyor ama JSONDecodeError gibi generic
    # exception'larda anında pes ediyor → dıştan bir tur daha dene.
    for attempt in range(3):
        model_response, ask_duration = call_model(
            test_client, SYSTEM_PROMPT_ASK, ask_prompt, test_model,
            verbose=False, max_tokens=max_tokens, enable_thinking=enable_thinking,
            openrouter_reasoning=openrouter_reasoning, reasoning_effort=reasoning_effort,
        )
        if not is_error(model_response):
            break
        if attempt < 2:
            log(f"[{i}/{total}] ID:{qid} — {model_response[:40]}, yeniden deneniyor...")
            time.sleep(10 * (attempt + 1))

    if is_error(model_response):
        log(f"[{i}/{total}] ID:{qid} — HATA: {model_response[:60]}")
        return {
            "sira": i, "id": qid, "category": category, "question_name": qname,
            "correct_answer": correct, "model_answer": None,
            "model_response": model_response[:200], "judge_response": "",
            "score": 0, "ask_duration": round(ask_duration, 2), "judge_duration": 0,
            "exact_match": False,
        }

    model_answer = extract_answer(model_response)
    exact = model_answer == correct

    judge_prompt = (
        f"## Soru\n{qtext}\n\n"
        f"## Seçenekler\n{opts_text}\n\n"
        f"## Doğru Cevap: {correct}\n\n"
        f"## Doğru Açıklama\n{solution}\n\n"
        f"## Öğrencinin Cevabı\n{model_response}\n\n"
        f"Öğrencinin cevabını doğru açıklama ile karşılaştırarak 10 üzerinden puanla."
    )
    judge_response, judge_duration = call_model(
        judge_client, SYSTEM_PROMPT_JUDGE, judge_prompt, judge_model, verbose=False
    )
    score = extract_score(judge_response)
    if score is None:
        score = 10 if exact else 0

    log(f"[{i}/{total}] ID:{qid} — {category}: {model_answer or '?'} "
        f"(doğru {correct}) {'✓' if exact else '✗'} puan {score}/10 ({ask_duration:.0f}s)")

    return {
        "sira": i, "id": qid, "category": category, "question_name": qname,
        "correct_answer": correct, "model_answer": model_answer,
        "model_response": model_response, "judge_response": judge_response,
        "score": score, "ask_duration": round(ask_duration, 2),
        "judge_duration": round(judge_duration, 2), "exact_match": exact,
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark 3 — paralel")
    ap.add_argument("--model", required=True)
    ap.add_argument("--judge", default="google/gemini-3.1-pro-preview")
    ap.add_argument("--data", default=JSON_PATH)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    ap.add_argument("--fresh", action="store_true", help="checkpoint'i yok say")
    ap.add_argument("--ids", default="", help="Sadece bu sıra numaralarını işle (virgüllü)")
    ap.add_argument("--force", action="store_true", help="Checkpoint'te olsa bile yeniden sor")
    ap.add_argument("--model-base-url", default=None, help="Test modeli için özel base URL (ör. Modal vLLM)")
    ap.add_argument("--model-api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--think", action="store_true", help="Gemma 4 thinking mode")
    args = ap.parse_args()
    target_ids = {int(x) for x in args.ids.split(",") if x.strip()}

    or_key = os.getenv("OPENROUTER_API_KEY", "no-key")
    or_base = "https://openrouter.ai/api/v1"
    test_base = args.model_base_url or or_base
    test_key = os.getenv(args.model_api_key_env, "no-key")
    test_client = OpenAI(base_url=test_base, api_key=test_key, timeout=600)
    judge_client = OpenAI(base_url=or_base, api_key=or_key)

    with open(args.data, encoding="utf-8") as f:
        questions = json.load(f)
    total = len(questions)

    cp_path = checkpoint_path(args.model)
    cp = {} if args.fresh else load_checkpoint(cp_path)

    print(f"Test model : {args.model} @ {test_base}")
    if args.think:
        print("Thinking mode: AÇIK")
    print(f"Jüri model : {args.judge}")
    print(f"{total} soru | eşzamanlılık: {args.concurrency} | max_tokens: {args.max_tokens}")
    if cp:
        print(f"Checkpoint: {len(cp)} soru hazır, atlanacak.")
    print()

    t0 = time.time()
    todo = [
        (i, q) for i, q in enumerate(questions, 1)
        if (not target_ids or i in target_ids)          # --ids ile sınırla
        and (args.force or i not in cp)                  # --force ile checkpoint'i ez
    ]
    if target_ids:
        print(f"Sadece {len(todo)} hedef soru işlenecek: {sorted(i for i,_ in todo)}"
              f"{' [FORCE]' if args.force else ''}\n")

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(do_question, i, q, total, test_client, judge_client,
                      args.model, args.judge, args.max_tokens,
                      args.reasoning, args.reasoning_effort, args.think): i
            for i, q in todo
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                log(f"[{i}/{total}] beklenmeyen hata: {type(e).__name__}: {e}")
                continue
            # API hatası olan soruyu checkpoint'e yazma → sonraki çalıştırmada tekrar sorulur
            if r is not None and r["model_answer"] is not None:
                cp[i] = r
                save_checkpoint(cp_path, cp)
            elif r is not None:
                log(f"[{i}/{total}] kalıcı hata, checkpoint'e yazılmadı.")

    results = [cp[i] for i in sorted(cp)]
    topic_stats = defaultdict(lambda: {"tested": 0, "total_score": 0, "correct": 0, "api_err": 0})
    for r in results:
        st = topic_stats[r["question_name"]]
        st["tested"] += 1
        st["total_score"] += r["score"] or 0
        if r["exact_match"]:
            st["correct"] += 1
        if r["model_answer"] is None:
            st["api_err"] += 1

    out = {
        "test_model": args.model,
        "judge_model": args.judge,
        "total_questions": total,
        "tested": sum(s["tested"] for s in topic_stats.values()),
        "results": results,
        "topic_stats": dict(topic_stats),
    }

    print(f"\nToplam süre: {(time.time()-t0)/60:.1f} dk")
    print_report(out)
    save_results(out)
    print("\nTamamlandı.")


if __name__ == "__main__":
    main()
