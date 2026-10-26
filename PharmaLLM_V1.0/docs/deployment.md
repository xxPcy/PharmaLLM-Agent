# Deployment

The vLLM service is started with:

```bash
MODEL_PATH=outputs/qwen3-8b-pharma-sft bash scripts/serve_vllm.sh
```

The Gradio app calls the OpenAI-compatible API and does not load model weights
inside the web process.

