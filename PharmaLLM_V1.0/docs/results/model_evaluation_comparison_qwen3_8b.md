# Qwen3-8B / Qwen3-8B-Base / Pharma-SFT 评测对比

生成日期：2026-09-24

## 评测设置

| 项目 | 值 |
| --- | --- |
| CMB-Exam split | `val` |
| MedQA-ZH split | `test` |
| generation batch size | 1 |
| max new tokens | 8 |
| temperature | 0.0 |
| top_p | 1.0 |
| dtype | `bfloat16` |
| device_map | `auto` |

## 模型与参数概览

| 模型 | 本次角色 | 训练阶段 | 参数规模 | 结构摘要 | 本地结果文件 |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3-8B-Base` | 原始预训练底座 | Pretraining | 8.2B 总参数，6.95B 非 embedding 参数 | 36 层，GQA：32 Q heads / 8 KV heads，上下文 32,768 | `reports/base_eval_qwen3_8b_base.json` |
| `outputs/qwen3-8b-base-pharma-sft` | 基于 `Qwen3-8B-Base` 的医疗 SFT | Pretraining + 本项目 SFT | 继承 8.2B 架构 | 36 层，GQA：32 Q heads / 8 KV heads，上下文 32,768 | `reports/sft_eval_qwen3_8b_base_sft.json` |
| `Qwen/Qwen3-8B` | 官方后训练对照模型 | Pretraining + Post-training | 8.2B 总参数，6.95B 非 embedding 参数 | 36 层，GQA：32 Q heads / 8 KV heads，原生 32,768，上下文可用 YaRN 扩展到 131,072 | `reports/base_eval_qwen3-8B.json` |

## 选择题评测结果

| 模型 | CMB-Exam correct / total | CMB-Exam accuracy | MedQA-ZH correct / total | MedQA-ZH accuracy | Micro correct / total | Micro accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3-8B-Base` | 100 / 280 | 0.3571 | 750 / 3426 | 0.2189 | 850 / 3706 | 0.2294 |
| `outputs/qwen3-8b-base-pharma-sft` | 165 / 280 | 0.5893 | 2793 / 3426 | 0.8152 | 2958 / 3706 | 0.7982 |
| `Qwen/Qwen3-8B` | 188 / 280 | 0.6714 | 2902 / 3426 | 0.8471 | 3090 / 3706 | 0.8338 |

## 差值对比

| 对比 | CMB-Exam delta | MedQA-ZH delta | Micro delta | 结论 |
| --- | ---: | ---: | ---: | --- |
| Pharma-SFT - `Qwen3-8B-Base` | +0.2321 | +0.5963 | +0.5688 | 本项目 SFT 相比原始底座提升非常明显 |
| Pharma-SFT - `Qwen3-8B` | -0.0821 | -0.0318 | -0.0356 | 本项目 SFT 仍低于官方后训练模型 |
| `Qwen3-8B` - `Qwen3-8B-Base` | +0.3143 | +0.6282 | +0.6044 | 官方后训练带来大幅能力增益 |

## 本项目 SFT 训练摘要

| 指标 | 值 |
| --- | ---: |
| train samples | 80000 |
| eval samples | 2000 |
| epoch | 1.0 |
| train loss | 1.4634 |
| eval loss | 1.4515 |
| train runtime | 135196.6555 秒 |
| train samples / second | 0.592 |
| train steps / second | 0.018 |
| eval samples / second | 1.88 |

## 结论

1. 如果比较对象是纯预训练底座 `Qwen/Qwen3-8B-Base`，本项目医疗 SFT 是有效的：Micro accuracy 从 0.2294 提升到 0.7982。
2. 如果比较对象是官方后训练模型 `Qwen/Qwen3-8B`，本项目 SFT 还没有超过官方模型：Micro accuracy 为 0.7982 vs 0.8338。
3. `Qwen/Qwen3-8B` 强很多的主要原因不是参数更多，而是它不是裸 base；它经过了官方后训练，包括长 CoT 冷启动、推理 RL、思考/非思考模式融合，以及通用 RL。
4. 当前评测是“医学选择题 + 严格只输出选项字母”的任务，因此非常吃指令跟随、格式跟随和考试式推理能力。`Qwen3-8B-Base` 是预训练补全文本模型，在这种 chat/instruction 格式任务上天然吃亏。

## 参考来源

- Qwen3-8B model card: https://huggingface.co/Qwen/Qwen3-8B
- Qwen3-8B-Base model card: https://huggingface.co/Qwen/Qwen3-8B-Base
- Qwen3 official blog: https://qwenlm.github.io/blog/qwen3/
