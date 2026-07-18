# -*- coding: utf-8 -*-
"""
Gemma 4 12B Unified it (bf16) + MTP drafter (speculative decoding) — Modal + vLLM

google/gemma-4-12B-it: encoder'sız "unified" multimodal mimari (11.95B dense, 256K ctx).
google/gemma-4-12B-it-assistant: MTP draft modeli — vLLM 0.25.1 bunu otomatik olarak
gemma4_mtp speculative yöntemine çevirir (çıktı kalitesi birebir aynı, decode hızlanır).

Deploy : modal deploy modal_gemma4_12b_vllm.py
Endpoint: https://saidsrc--gemma4-12b-vllm-serve.modal.run/v1
API key : trlawbench-gemma4 (istemci tarafında OpenAI api_key olarak ver)
"""

import modal

APP_NAME = "gemma4-12b-vllm"
MODEL_NAME = "google/gemma-4-12B-it"
DRAFT_MODEL = "google/gemma-4-12B-it-assistant"
USE_SPECULATIVE = False  # vLLM 0.25.1 gemma4_mtp CUDA graph capture bug'ı (pinned memory) → kapalı
VLLM_PORT = 8000
API_KEY = "trlawbench-gemma4"

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "vllm",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

app = modal.App(APP_NAME)

hf_cache_vol = modal.Volume.from_name("gemma4-12b-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("gemma4-12b-vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu=["H100", "H200", "A100-80GB"],
    scaledown_window=15 * 60,
    timeout=24 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=45 * 60)
def serve():
    import json
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--api-key", API_KEY,
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "gemma4",
        "--reasoning-parser", "gemma4",
        "--limit-mm-per-prompt", '{"image": 0, "audio": 0}',
    ]
    if USE_SPECULATIVE:
        cmd += ["--speculative-config",
                json.dumps({"model": DRAFT_MODEL, "num_speculative_tokens": 3})]
    print("Başlatılıyor:", " ".join(cmd))
    subprocess.Popen(cmd)
