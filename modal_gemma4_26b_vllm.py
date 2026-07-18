# -*- coding: utf-8 -*-
"""
Güncel Gemma 4 26B-A4B-it (bf16, yeni canonical chat template 2026-07-09) — Modal + vLLM

HF repo'daki chat_template.jinja 2026-07-09'da düzeltildi (tool-calling döngüleri,
turn kapanışları, thinking sıralaması). Ağırlıklar değişmedi. Bu deploy modelin
KENDİ (güncel) şablonunu kullanır — vLLM'in örnek şablonu bilinçli olarak verilmez.

Deploy : modal deploy modal_gemma4_26b_vllm.py
Endpoint: https://saidsrc--gemma4-26b-vllm-serve.modal.run/v1
API key : trlawbench-gemma4 (istemci tarafında OpenAI api_key olarak ver)
"""

import modal

APP_NAME = "gemma4-26b-vllm"
MODEL_NAME = "google/gemma-4-26B-A4B-it"
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

hf_cache_vol = modal.Volume.from_name("gemma4-26b-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("gemma4-26b-vllm-cache", create_if_missing=True)


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
        "--async-scheduling",
    ]
    print("Başlatılıyor:", " ".join(cmd))
    subprocess.Popen(cmd)
