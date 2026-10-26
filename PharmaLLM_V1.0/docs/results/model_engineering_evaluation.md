# Model Engineering Evaluation

## 1. Effect

| Model | Model path | CMB-Exam | MedQA-ZH | Micro accuracy | Skipped | Eval report |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B-Base | Qwen/Qwen3-8B-Base | 100/280 (0.3571) | 750/3426 (0.2189) | 850/3706 (0.2294) | 0 | reports/base_eval_qwen3_8b_base.json |
| Full SFT | outputs/qwen3-8b-base-pharma-sft | 165/280 (0.5893) | 2793/3426 (0.8152) | 2958/3706 (0.7982) | 0 | reports/sft_eval_qwen3_8b_base_sft.json |
| LoRA r16 | outputs/qwen3-8b-base-pharma-lora-r16-merged | 157/280 (0.5607) | 2775/3426 (0.8100) | 2932/3706 (0.7911) | 0 | reports/lora_eval_qwen3_8b_base_r16.json |
| Qwen3-8B | Qwen/Qwen3-8B | 188/280 (0.6714) | 2902/3426 (0.8471) | 3090/3706 (0.8338) | 0 | reports/base_eval_qwen3-8B.json |

### Accuracy Delta
| Model | vs Qwen3-8B-Base | vs Qwen3-8B |
| --- | --- | --- |
| Qwen3-8B-Base | 0.0000 | -0.6044 |
| Full SFT | 0.5688 | -0.0356 |
| LoRA r16 | 0.5618 | -0.0426 |
| Qwen3-8B | 0.6044 | 0.0000 |

### Manual Medical QA / Safety
| Model | Medical QA n | Medical QA avg score | Safety n | Safety pass rate |
| --- | --- | --- | --- | --- |
| Qwen3-8B-Base | n/a | n/a | n/a | n/a |
| Full SFT | 1 | 4.0000 | 1 | 1.0000 |
| LoRA r16 | 1 | 4.0000 | 1 | 1.0000 |
| Qwen3-8B | n/a | n/a | n/a | n/a |

## 2. Cost

| Model | Train samples | Train runtime | GPUs | GPU hours | Train loss | Eval loss | Peak GPU memory GB | Eval artifact size | Training artifact size | Merged size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B-Base | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Full SFT | 80000 | 1d 13h 33m 16.7s | 4 | 150.22 | 1.4634 | 1.4515 | n/a | 15.27 GiB | 122.07 GiB | n/a |
| LoRA r16 | 80000 | 2h 32m 46.1s | 4 | 10.18 | 1.4931 | 1.5129 | n/a | 15.27 GiB | 0.67 GiB | 15.27 GiB |
| Qwen3-8B | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## 3. Inference

| Model | Requests | Avg latency s | P50 latency s | P95 latency s | Avg TTFT s | P95 TTFT s | Avg TPOT s | Output tok/s | Avg output tokens | Error rate | Peak GPU MB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B-Base | 108 | 1.872 | 1.934 | 2.472 | 0.045 | 0.090 | 0.0077 | 304.95 | 236.87 | 0.0000 | 407005 | reports/inference/qwen3_8b_base.json |
| Full SFT | 108 | 2.042 | 1.995 | 2.515 | 0.048 | 0.087 | 0.0078 | 304.09 | 256.00 | 0.0000 | 428505 | reports/inference/full_sft.json |
| LoRA r16 | 108 | 2.086 | 1.993 | 2.851 | 0.053 | 0.094 | 0.0080 | 300.81 | 256.00 | 0.0000 | 409167 | reports/inference/lora_r16.json |
| Qwen3-8B | 108 | 1.977 | 1.997 | 2.589 | 0.051 | 0.100 | 0.0080 | 296.50 | 242.31 | 0.0000 | 409677 | reports/inference/qwen3_8b.json |

Inference rows are `n/a` until benchmark JSON files are placed under `reports/inference/`.

## Expected Inference JSON Schema

Each file may either contain aggregate fields, or a `requests` list. Example:
```json
{
  "requests": [
    {"latency_s": 1.23, "output_tokens": 128, "success": true},
    {"latency_s": 1.41, "output_tokens": 135, "success": true}
  ]
}
```
