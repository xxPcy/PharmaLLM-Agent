from __future__ import annotations

import os
import time
from typing import Any

import gradio as gr
from openai import OpenAI


API_BASE = os.getenv("PHARMA_LLM_API_BASE", "http://127.0.0.1:8000/v1")
MODEL_NAME = os.getenv("PHARMA_LLM_MODEL", "qwen3-8b-pharma-sft")


def ask_model(question: str) -> tuple[str, str]:
    if not question.strip():
        return "", "Latency: 0.000s | Generated tokens: 0"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=API_BASE)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a medical QA assistant. Provide cautious, evidence-aware answers.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    latency = time.perf_counter() - started
    content = response.choices[0].message.content or ""
    usage: Any = getattr(response, "usage", None)
    generated_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    return content, f"Latency: {latency:.3f}s | Generated tokens: {generated_tokens}"


with gr.Blocks(title="PharmaLLM") as demo:
    gr.Markdown("# PharmaLLM")
    gr.Markdown(
        f"Model: {MODEL_NAME} | Backend: vLLM | GPU: 4x NVIDIA A800 80GB"
    )
    question = gr.Textbox(label="Medical question", lines=5)
    answer = gr.Textbox(label="Model answer", lines=12)
    metrics = gr.Textbox(label="Runtime metrics", interactive=False)
    submit = gr.Button("Ask")
    submit.click(ask_model, inputs=question, outputs=[answer, metrics])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

