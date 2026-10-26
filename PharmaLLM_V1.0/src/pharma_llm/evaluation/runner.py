from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from pharma_llm.utils.config import load_yaml_config


OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class ChoiceExample:
    benchmark: str
    example_id: str
    question: str
    choices: dict[str, str]
    answer: str | None = None
    metadata: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PharmaLLM on Chinese medical choice benchmarks.")
    parser.add_argument("--model-path", required=True, help="HF model directory or model id to evaluate.")
    parser.add_argument("--eval-config", default="configs/eval.yaml", help="Path to configs/eval.yaml.")
    parser.add_argument("--output", default=None, help="JSON report output path.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        choices=["cmb_exam", "medqa_zh"],
        help="Benchmarks to run. Defaults to all benchmarks in the eval config.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum scored/evaluated samples per benchmark. Useful for quick checkpoint checks.",
    )
    parser.add_argument("--cmb-split", default=None, choices=["train", "val", "test"], help="CMB-Exam split.")
    parser.add_argument("--batch-size", type=int, default=None, help="Generation batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Generation top-p.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype used to load the model.",
    )
    parser.add_argument("--device-map", default="auto", help='Transformers device_map, e.g. "auto" or "cuda:0".')
    return parser.parse_args()


def normalize_choice_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        letters = [str(item).strip().upper() for item in value]
        return "".join(letter for letter in letters if letter)
    normalized = unicodedata.normalize("NFKC", str(value)).upper()
    letters = re.findall(r"[A-Z]", normalized)
    return "".join(letters) if letters else None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def resolve_cmb_file(data_dir: str | Path, split: str) -> Path:
    root = Path(data_dir)
    split_file_names = {
        "train": "CMB-train-merge.json",
        "val": "CMB-val-merge.json",
        "test": "CMB-test-choice-question-merge.json",
    }
    file_name = split_file_names[split]
    candidates = [
        root / "CMB" / "CMB-Exam" / f"CMB-{split}" / file_name,
        root / "CMB-Exam" / f"CMB-{split}" / file_name,
        root / f"CMB-{split}" / file_name,
        root / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find CMB-Exam {split} file. Tried:\n  {joined}")


def load_cmb_exam(data_dir: str | Path, *, split: str = "val") -> list[ChoiceExample]:
    data_path = resolve_cmb_file(data_dir, split)
    rows = load_json(data_path)
    if not isinstance(rows, list):
        raise ValueError(f"CMB-Exam file must contain a list: {data_path}")

    examples: list[ChoiceExample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        raw_options = row.get("option")
        if not isinstance(raw_options, dict):
            continue
        choices = {str(key).upper(): str(value) for key, value in raw_options.items()}
        question = str(row.get("question", "")).strip()
        if not question or not choices:
            continue
        examples.append(
            ChoiceExample(
                benchmark="cmb_exam",
                example_id=str(row.get("id", index)),
                question=question,
                choices=choices,
                answer=normalize_choice_answer(row.get("answer")),
                metadata={
                    "split": split,
                    "exam_type": row.get("exam_type"),
                    "exam_class": row.get("exam_class"),
                    "exam_subject": row.get("exam_subject"),
                    "question_type": row.get("question_type"),
                    "source_path": str(data_path),
                },
            )
        )
    return examples


def load_medqa_zh(data_dir: str | Path) -> list[ChoiceExample]:
    root = Path(data_dir)
    data_path = root if root.is_file() else root / "medqa_zh_4op.jsonl"
    rows = load_jsonl(data_path)

    examples: list[ChoiceExample] = []
    for index, row in enumerate(rows):
        choices_raw = row.get("choices")
        if not isinstance(choices_raw, list):
            continue
        choices = {OPTION_LABELS[i]: str(choice) for i, choice in enumerate(choices_raw)}
        question = str(row.get("problem_text", "")).strip()
        if not question or not choices:
            continue
        examples.append(
            ChoiceExample(
                benchmark="medqa_zh",
                example_id=str(row.get("problem_id", index)),
                question=question,
                choices=choices,
                answer=normalize_choice_answer(row.get("answer")),
                metadata={"source_path": str(data_path), "points": row.get("points")},
            )
        )
    return examples


def select_examples(examples: list[ChoiceExample], max_samples: int | None) -> list[ChoiceExample]:
    if max_samples is None:
        return examples
    return examples[: max(0, max_samples)]


def build_prompt(example: ChoiceExample) -> list[dict[str, str]]:
    option_lines = [f"{label}. {text}" for label, text in sorted(example.choices.items())]
    option_labels = "".join(sorted(example.choices))
    user_text = (
        f"{example.question}\n\n"
        + "\n".join(option_lines)
        + f"\n\n请只回答正确选项字母，可选范围：{option_labels}。不要解释。"
    )
    return [
        {"role": "system", "content": "你是中文医学考试助手。请严格按题目要求作答。"},
        {"role": "user", "content": user_text},
    ]


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            return fallback_chatml(messages)
    except ValueError:
        return fallback_chatml(messages)


def fallback_chatml(messages: list[dict[str, str]]) -> str:
    text = ""
    for message in messages:
        text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


def extract_choice_answer(text: str, choices: dict[str, str]) -> str:
    allowed = "".join(sorted(choices))
    if not allowed:
        return ""

    normalized = unicodedata.normalize("NFKC", text).upper().strip()
    normalized = normalized.replace(" ", "")
    marker_pattern = rf"(?:正确答案|答案|选项|选择|ANSWER|ANS)[:：为是]*([{allowed}](?:[,，、/;；和及]*[{allowed}])*)"
    marker_match = re.search(marker_pattern, normalized)
    if marker_match:
        return normalize_extracted_letters(marker_match.group(1), allowed)

    leading_match = re.match(rf"^[：:。.\-—]*([{allowed}](?:[,，、/;；和及]*[{allowed}])*)", normalized)
    if leading_match:
        return normalize_extracted_letters(leading_match.group(1), allowed)

    any_match = re.search(rf"[{allowed}](?:[,，、/;；和及]*[{allowed}])*", normalized)
    if any_match:
        return normalize_extracted_letters(any_match.group(0), allowed)
    return ""


def normalize_extracted_letters(text: str, allowed: str) -> str:
    seen: list[str] = []
    for letter in re.findall(rf"[{allowed}]", text):
        if letter not in seen:
            seen.append(letter)
    return "".join(seen)


def is_correct_prediction(prediction: str, answer: str | None) -> bool | None:
    if answer is None:
        return None
    return prediction == answer


def resolve_torch_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(model_path: str | Path, *, dtype: str, device_map: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import __version__ as transformers_version

    from packaging.version import Version

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    torch_dtype = resolve_torch_dtype(dtype)
    dtype_key = "dtype" if Version(transformers_version) >= Version("4.56.0") else "torch_dtype"
    if torch_dtype != "auto":
        model_kwargs[dtype_key] = torch_dtype
    else:
        model_kwargs[dtype_key] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tokenizer


def generate_answers(
    examples: list[ChoiceExample],
    *,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    import torch

    outputs: list[str] = []
    do_sample = temperature > 0
    for start in tqdm(range(0, len(examples), batch_size), desc="Generating"):
        batch = examples[start : start + batch_size]
        prompts = [apply_chat_template(tokenizer, build_prompt(example)) for example in batch]
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
        tokenized = {key: value.to(model.device) for key, value in tokenized.items()}
        prompt_length = tokenized["input_ids"].shape[1]
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        with torch.inference_mode():
            generated = model.generate(
                **tokenized,
                **generation_kwargs,
            )
        for row_index, output_ids in enumerate(generated):
            new_tokens = output_ids[prompt_length:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def score_examples(examples: list[ChoiceExample], raw_outputs: list[str]) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    correct = 0
    scored = 0
    skipped = 0

    for example, raw_output in zip(examples, raw_outputs, strict=True):
        prediction = extract_choice_answer(raw_output, example.choices)
        is_correct = is_correct_prediction(prediction, example.answer)
        if is_correct is None:
            skipped += 1
        else:
            scored += 1
            correct += int(is_correct)
        predictions.append(
            {
                "id": example.example_id,
                "gold": example.answer,
                "prediction": prediction,
                "raw_output": raw_output,
                "correct": is_correct,
                "metadata": example.metadata or {},
            }
        )

    accuracy = correct / scored if scored else None
    return {
        "total_samples": len(examples),
        "scored_samples": scored,
        "skipped_samples": skipped,
        "correct": correct,
        "accuracy": accuracy,
        "predictions": predictions,
    }


def load_benchmark_examples(eval_cfg: dict[str, Any], benchmark: str, *, cmb_split: str | None) -> list[ChoiceExample]:
    benchmark_cfg = eval_cfg.get("benchmarks", {}).get(benchmark, {})
    if benchmark == "cmb_exam":
        split = cmb_split or str(benchmark_cfg.get("split", "val"))
        return load_cmb_exam(benchmark_cfg.get("data_dir", "data/benchmarks/cmb_exam"), split=split)
    if benchmark == "medqa_zh":
        return load_medqa_zh(benchmark_cfg.get("data_dir", "data/benchmarks/medqa_zh"))
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def write_markdown_summary(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        f"- Model: `{report['model_path']}`",
        f"- Generated at: {report['generated_at']}",
        "",
        "| Benchmark | Scored | Correct | Accuracy | Skipped |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for benchmark, result in report["results"].items():
        accuracy = result["accuracy"]
        accuracy_text = "n/a" if accuracy is None else f"{accuracy:.4f}"
        lines.append(
            f"| {benchmark} | {result['scored_samples']} | {result['correct']} | "
            f"{accuracy_text} | {result['skipped_samples']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_cfg = load_yaml_config(args.eval_config)
    generation_cfg = eval_cfg.get("generation", {})
    output_cfg = eval_cfg.get("outputs", {})

    benchmarks = args.benchmarks or list(eval_cfg.get("benchmarks", {"cmb_exam": {}, "medqa_zh": {}}))
    batch_size = int(args.batch_size or generation_cfg.get("batch_size", 1))
    max_new_tokens = int(args.max_new_tokens or generation_cfg.get("max_new_tokens", 8))
    temperature = float(args.temperature if args.temperature is not None else generation_cfg.get("temperature", 0.0))
    top_p = float(args.top_p if args.top_p is not None else generation_cfg.get("top_p", 1.0))
    output_path = Path(args.output or output_cfg.get("sft_eval", "reports/sft_eval.json"))

    model, tokenizer = load_model_and_tokenizer(args.model_path, dtype=args.dtype, device_map=args.device_map)

    report: dict[str, Any] = {
        "model_path": str(args.model_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "max_samples": args.max_samples,
        },
        "results": {},
    }

    for benchmark in benchmarks:
        examples = load_benchmark_examples(eval_cfg, benchmark, cmb_split=args.cmb_split)
        examples = select_examples(examples, args.max_samples)
        raw_outputs = generate_answers(
            examples,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        report["results"][benchmark] = score_examples(examples, raw_outputs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = Path(output_cfg.get("summary_md", "reports/evaluation_summary.md"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_summary(report, summary_path)

    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))
    for benchmark, result in report["results"].items():
        accuracy = result["accuracy"]
        accuracy_text = "n/a" if accuracy is None else f"{accuracy:.4f}"
        print(
            f"{benchmark}: scored={result['scored_samples']} "
            f"correct={result['correct']} accuracy={accuracy_text} skipped={result['skipped_samples']}"
        )
    print(f"Wrote JSON report to {output_path}")
    print(f"Wrote markdown summary to {summary_path}")


if __name__ == "__main__":
    main()
