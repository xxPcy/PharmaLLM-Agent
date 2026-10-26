# Architecture

PharmaLLM follows a staged pipeline:

1. Normalize public medical datasets into one JSONL schema.
2. Validate, clean, deduplicate, and remove benchmark contamination.
3. Build 80k SFT training samples plus validation samples.
4. Fine-tune Qwen3-8B with DeepSpeed ZeRO-3 on 4x A800 GPUs.
5. Evaluate base and SFT models on CMB-Exam and MedQA-ZH.
6. Quantize with AWQ and deploy with vLLM tensor parallelism.
7. Benchmark latency, throughput, memory, and prefix caching.

