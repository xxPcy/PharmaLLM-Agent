#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EVAL_FILES = {
    "Qwen3-8B-Base": "reports/base_eval_qwen3_8b_base.json",
    "Full SFT": "reports/sft_eval_qwen3_8b_base_sft.json",
    "LoRA r16": "reports/lora_eval_qwen3_8b_base_r16.json",
    "Qwen3-8B": "reports/base_eval_qwen3-8B.json",
}

DEFAULT_TRAINING_SUMMARIES = {
    "Full SFT": {
        "path": "reports/training_summary_qwen3_8b_base_sft.json",
        "num_gpus": 4,
        "artifact_path": "outputs/qwen3-8b-base-pharma-sft",
    },
    "LoRA r16": {
        "path": "reports/training_summary_qwen3_8b_base_lora_r16.json",
        "num_gpus": 4,
        "artifact_path": "outputs/qwen3-8b-base-pharma-lora-r16",
        "merged_path": "outputs/qwen3-8b-base-pharma-lora-r16-merged",
    },
}

DEFAULT_INFERENCE_FILES = {
    "Qwen3-8B-Base": "reports/inference/qwen3_8b_base.json",
    "Full SFT": "reports/inference/full_sft.json",
    "LoRA r16": "reports/inference/lora_r16.json",
    "Qwen3-8B": "reports/inference/qwen3_8b.json",
}

BENCHMARK_NAMES = {
    "cmb_exam": "CMB-Exam",
    "medqa_zh": "MedQA-ZH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize PharmaLLM model quality, training cost, and inference metrics."
    )
    parser.add_argument(
        "--output",
        default="reports/model_engineering_evaluation.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--json-output",
        default="reports/model_engineering_evaluation.json",
        help="Machine-readable JSON summary output path.",
    )
    parser.add_argument(
        "--manual-eval",
        default=None,
        help=(
            "Optional manual evaluation JSON/JSONL. Rows should include model, category, "
            "and either score/max_score or passed."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any default eval file is missing.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"Manual eval file must contain a list: {path}")
        return rows
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def human_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    days = int(seconds // 86400)
    seconds -= days * 86400
    hours = int(seconds // 3600)
    seconds -= hours * 3600
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds:.1f}s")
    return " ".join(parts)


def fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value)}"


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    gib = num_bytes / (1024**3)
    return f"{gib:.2f} GiB"


def markdown_escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(markdown_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def directory_size(path: Path, *, recursive: bool = True) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    if not recursive:
        return sum(item.stat().st_size for item in path.iterdir() if item.is_file())
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def load_eval_reports(*, strict: bool) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for label, file_name in DEFAULT_EVAL_FILES.items():
        path = Path(file_name)
        if not path.exists():
            missing.append(file_name)
            reports[label] = {
                "label": label,
                "path": file_name,
                "missing": True,
                "benchmarks": {},
            }
            continue
        raw = read_json(path)
        results = raw.get("results", {})
        benchmarks: dict[str, dict[str, Any]] = {}
        total_correct = 0
        total_scored = 0
        total_skipped = 0
        for benchmark, item in results.items():
            correct = int(item.get("correct", 0))
            scored = int(item.get("scored_samples", 0))
            skipped = int(item.get("skipped_samples", 0))
            accuracy = item.get("accuracy")
            if accuracy is None and scored:
                accuracy = correct / scored
            benchmarks[benchmark] = {
                "correct": correct,
                "scored": scored,
                "skipped": skipped,
                "accuracy": accuracy,
            }
            total_correct += correct
            total_scored += scored
            total_skipped += skipped
        reports[label] = {
            "label": label,
            "path": file_name,
            "missing": False,
            "model_path": raw.get("model_path"),
            "generated_at": raw.get("generated_at"),
            "generation": raw.get("generation", {}),
            "benchmarks": benchmarks,
            "micro": {
                "correct": total_correct,
                "scored": total_scored,
                "skipped": total_skipped,
                "accuracy": total_correct / total_scored if total_scored else None,
            },
        }
    if strict and missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing eval reports:\n  {joined}")
    return reports


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_inference_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = read_json(path)
    metrics = raw.get("metrics", raw) if isinstance(raw, dict) else {}
    if not isinstance(metrics, dict):
        return None

    requests = metrics.get("requests")
    if isinstance(requests, list):
        latencies: list[float] = []
        output_tokens: list[float] = []
        errors = 0
        for row in requests:
            if not isinstance(row, dict):
                continue
            success = row.get("success", True)
            if success is False or row.get("error"):
                errors += 1
                continue
            latency = first_present(
                row,
                ["latency_s", "e2e_latency_s", "total_latency_s", "request_latency_s"],
            )
            tokens = first_present(
                row,
                ["output_tokens", "generated_tokens", "completion_tokens", "num_output_tokens"],
            )
            if latency is not None:
                latencies.append(float(latency))
            if tokens is not None:
                output_tokens.append(float(tokens))
        total_latency = sum(latencies)
        total_output_tokens = sum(output_tokens)
        return {
            "request_count": len(requests),
            "avg_latency_s": sum(latencies) / len(latencies) if latencies else None,
            "p50_latency_s": percentile(latencies, 0.50),
            "p95_latency_s": percentile(latencies, 0.95),
            "avg_ttft_s": None,
            "p95_ttft_s": None,
            "avg_tpot_s": None,
            "p95_tpot_s": None,
            "output_tokens_per_second": total_output_tokens / total_latency if total_latency else None,
            "avg_output_tokens": sum(output_tokens) / len(output_tokens) if output_tokens else None,
            "error_rate": errors / len(requests) if requests else None,
            "peak_gpu_memory_mb": None,
            "source": str(path),
        }

    latency_ms = first_present(metrics, ["mean_e2e_latency_ms", "avg_latency_ms", "mean_latency_ms"])
    p50_ms = first_present(metrics, ["p50_e2e_latency_ms", "median_e2e_latency_ms", "p50_latency_ms"])
    p95_ms = first_present(metrics, ["p95_e2e_latency_ms", "p95_latency_ms"])
    avg_latency_s = first_present(metrics, ["avg_latency_s", "mean_latency_s", "mean_e2e_latency_s"])
    p50_latency_s = first_present(metrics, ["p50_latency_s", "median_latency_s", "p50_e2e_latency_s"])
    p95_latency_s = first_present(metrics, ["p95_latency_s", "p95_e2e_latency_s"])
    if avg_latency_s is None and latency_ms is not None:
        avg_latency_s = float(latency_ms) / 1000
    if p50_latency_s is None and p50_ms is not None:
        p50_latency_s = float(p50_ms) / 1000
    if p95_latency_s is None and p95_ms is not None:
        p95_latency_s = float(p95_ms) / 1000

    return {
        "request_count": first_present(metrics, ["request_count", "num_requests", "total_requests", "completed"]),
        "avg_latency_s": avg_latency_s,
        "p50_latency_s": p50_latency_s,
        "p95_latency_s": p95_latency_s,
        "avg_ttft_s": first_present(metrics, ["avg_ttft_s", "mean_ttft_s"]),
        "p95_ttft_s": first_present(metrics, ["p95_ttft_s"]),
        "avg_tpot_s": first_present(metrics, ["avg_tpot_s", "mean_tpot_s"]),
        "p95_tpot_s": first_present(metrics, ["p95_tpot_s"]),
        "output_tokens_per_second": first_present(
            metrics,
            [
                "output_tokens_per_second",
                "output_throughput",
                "output_throughput_tok_s",
                "output_token_throughput",
            ],
        ),
        "avg_output_tokens": first_present(metrics, ["avg_output_tokens", "mean_output_tokens"]),
        "error_rate": first_present(metrics, ["error_rate", "failed_rate"]),
        "peak_gpu_memory_mb": first_present(metrics, ["peak_gpu_memory_mb", "max_gpu_memory_mb"]),
        "source": str(path),
    }


def load_cost_metrics(eval_reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    costs: dict[str, dict[str, Any]] = {}
    for label, eval_report in eval_reports.items():
        training_meta = DEFAULT_TRAINING_SUMMARIES.get(label, {})
        summary_path = Path(training_meta["path"]) if training_meta.get("path") else None
        summary = read_json(summary_path) if summary_path and summary_path.exists() else {}
        train_metrics = summary.get("train_metrics", {})
        eval_metrics = summary.get("eval_metrics", {})
        runtime = train_metrics.get("train_runtime")
        num_gpus = training_meta.get("num_gpus")
        gpu_hours = (float(runtime) * float(num_gpus) / 3600) if runtime is not None and num_gpus else None

        model_path = eval_report.get("model_path")
        eval_artifact_size = (
            directory_size(Path(str(model_path)), recursive=False)
            if model_path and Path(str(model_path)).exists()
            else None
        )
        artifact_path = training_meta.get("artifact_path")
        merged_path = training_meta.get("merged_path")

        costs[label] = {
            "training_summary": str(summary_path) if summary_path else None,
            "train_samples": summary.get("train_samples"),
            "eval_samples": summary.get("eval_samples"),
            "train_runtime_s": runtime,
            "train_runtime_human": human_seconds(runtime),
            "num_gpus": num_gpus,
            "gpu_hours": gpu_hours,
            "train_loss": train_metrics.get("train_loss"),
            "eval_loss": eval_metrics.get("eval_loss"),
            "peak_gpu_memory_gb": first_present(summary, ["peak_gpu_memory_gb", "max_gpu_memory_gb"]),
            "eval_artifact_size_bytes": eval_artifact_size,
            "training_artifact_size_bytes": directory_size(Path(artifact_path), recursive=True) if artifact_path else None,
            "merged_artifact_size_bytes": directory_size(Path(merged_path), recursive=False) if merged_path else None,
        }
    return costs


def load_manual_metrics(path_name: str | None) -> dict[str, dict[str, Any]]:
    if not path_name:
        return {}
    path = Path(path_name)
    if not path.exists():
        raise FileNotFoundError(f"Manual eval file not found: {path}")
    rows = read_json_or_jsonl(path)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(row.get("model", "")).strip()
        category = str(row.get("category", "")).strip()
        if not model or not category:
            continue
        grouped[(model, category)].append(row)

    output: dict[str, dict[str, Any]] = defaultdict(dict)
    for (model, category), items in grouped.items():
        scores = [float(item["score"]) for item in items if item.get("score") is not None]
        max_scores = [float(item["max_score"]) for item in items if item.get("max_score") is not None]
        passes = [bool(item["passed"]) for item in items if item.get("passed") is not None]
        metric: dict[str, Any] = {"count": len(items)}
        if scores:
            metric["avg_score"] = sum(scores) / len(scores)
            if max_scores:
                metric["avg_score_ratio"] = metric["avg_score"] / max(max_scores)
        if passes:
            metric["pass_rate"] = sum(passes) / len(passes)
        output[model][category] = metric
    return output


def collect_all_benchmarks(eval_reports: dict[str, dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for report in eval_reports.values():
        for benchmark in report.get("benchmarks", {}):
            if benchmark not in seen:
                seen.append(benchmark)
    return seen


def benchmark_cell(item: dict[str, Any] | None) -> str:
    if not item:
        return "n/a"
    return f"{item['correct']}/{item['scored']} ({fmt_float(item['accuracy'])})"


def generate_markdown(summary: dict[str, Any]) -> str:
    eval_reports = summary["effect"]["eval_reports"]
    costs = summary["cost"]
    inference = summary["inference"]
    manual = summary["effect"].get("manual", {})
    benchmarks = collect_all_benchmarks(eval_reports)

    lines: list[str] = [
        "# Model Engineering Evaluation",
        "",
        "## 1. Effect",
        "",
    ]

    headers = ["Model", "Model path"]
    headers += [BENCHMARK_NAMES.get(name, name) for name in benchmarks]
    headers += ["Micro accuracy", "Skipped", "Eval report"]
    rows = []
    for label, report in eval_reports.items():
        if report.get("missing"):
            rows.append([label, "missing", *["n/a" for _ in benchmarks], "n/a", "n/a", report["path"]])
            continue
        rows.append(
            [
                label,
                report.get("model_path", "n/a"),
                *[benchmark_cell(report["benchmarks"].get(name)) for name in benchmarks],
                benchmark_cell(report.get("micro")),
                fmt_int(report.get("micro", {}).get("skipped")),
                report["path"],
            ]
        )
    lines.append(markdown_table(headers, rows))

    if "Qwen3-8B-Base" in eval_reports:
        lines += ["", "### Accuracy Delta"]
        refs = ["Qwen3-8B-Base", "Qwen3-8B"]
        delta_rows = []
        for label, report in eval_reports.items():
            if report.get("missing"):
                continue
            row = [label]
            for ref in refs:
                ref_report = eval_reports.get(ref, {})
                ref_acc = ref_report.get("micro", {}).get("accuracy")
                acc = report.get("micro", {}).get("accuracy")
                row.append(fmt_float(acc - ref_acc if acc is not None and ref_acc is not None else None))
            delta_rows.append(row)
        lines.append(markdown_table(["Model", "vs Qwen3-8B-Base", "vs Qwen3-8B"], delta_rows))

    lines += ["", "### Manual Medical QA / Safety"]
    if manual:
        manual_rows = []
        for label in eval_reports:
            categories = manual.get(label, {})
            medical = categories.get("medical_qa", {})
            safety = categories.get("safety_refusal", {})
            manual_rows.append(
                [
                    label,
                    medical.get("count", "n/a"),
                    fmt_float(medical.get("avg_score")),
                    safety.get("count", "n/a"),
                    fmt_float(safety.get("pass_rate")),
                ]
            )
        lines.append(
            markdown_table(
                ["Model", "Medical QA n", "Medical QA avg score", "Safety n", "Safety pass rate"],
                manual_rows,
            )
        )
    else:
        lines.append(
            "No manual evaluation file was provided. Pass `--manual-eval reports/manual_eval.jsonl` "
            "to include medical QA scores and safety-refusal pass rates."
        )

    lines += ["", "## 2. Cost", ""]
    cost_rows = []
    for label, item in costs.items():
        cost_rows.append(
            [
                label,
                fmt_int(item.get("train_samples")),
                item.get("train_runtime_human", "n/a"),
                fmt_int(item.get("num_gpus")),
                fmt_float(item.get("gpu_hours"), 2),
                fmt_float(item.get("train_loss")),
                fmt_float(item.get("eval_loss")),
                fmt_float(item.get("peak_gpu_memory_gb"), 2),
                fmt_bytes(item.get("eval_artifact_size_bytes")),
                fmt_bytes(item.get("training_artifact_size_bytes")),
                fmt_bytes(item.get("merged_artifact_size_bytes")),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Model",
                "Train samples",
                "Train runtime",
                "GPUs",
                "GPU hours",
                "Train loss",
                "Eval loss",
                "Peak GPU memory GB",
                "Eval artifact size",
                "Training artifact size",
                "Merged size",
            ],
            cost_rows,
        )
    )

    lines += ["", "## 3. Inference", ""]
    inference_rows = []
    for label, item in inference.items():
        inference_rows.append(
            [
                label,
                fmt_int(item.get("request_count")),
                fmt_float(item.get("avg_latency_s"), 3),
                fmt_float(item.get("p50_latency_s"), 3),
                fmt_float(item.get("p95_latency_s"), 3),
                fmt_float(item.get("avg_ttft_s"), 3),
                fmt_float(item.get("p95_ttft_s"), 3),
                fmt_float(item.get("avg_tpot_s"), 4),
                fmt_float(item.get("output_tokens_per_second"), 2),
                fmt_float(item.get("avg_output_tokens"), 2),
                fmt_float(item.get("error_rate"), 4),
                fmt_float(item.get("peak_gpu_memory_mb"), 0),
                item.get("source", "n/a"),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Model",
                "Requests",
                "Avg latency s",
                "P50 latency s",
                "P95 latency s",
                "Avg TTFT s",
                "P95 TTFT s",
                "Avg TPOT s",
                "Output tok/s",
                "Avg output tokens",
                "Error rate",
                "Peak GPU MB",
                "Source",
            ],
            inference_rows,
        )
    )
    lines.append(
        "\nInference rows are `n/a` until benchmark JSON files are placed under `reports/inference/`."
    )

    lines += ["", "## Expected Inference JSON Schema", ""]
    lines.append(
        "Each file may either contain aggregate fields, or a `requests` list. Example:"
    )
    lines.append(
        "```json\n"
        "{\n"
        '  "requests": [\n'
        '    {"latency_s": 1.23, "output_tokens": 128, "success": true},\n'
        '    {"latency_s": 1.41, "output_tokens": 135, "success": true}\n'
        "  ]\n"
        "}\n"
        "```"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    eval_reports = load_eval_reports(strict=args.strict)
    costs = load_cost_metrics(eval_reports)
    manual = load_manual_metrics(args.manual_eval)
    inference = {
        label: (load_inference_metrics(Path(path)) or {"source": path})
        for label, path in DEFAULT_INFERENCE_FILES.items()
    }

    summary = {
        "effect": {
            "eval_reports": eval_reports,
            "manual": manual,
        },
        "cost": costs,
        "inference": inference,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_markdown(summary), encoding="utf-8")

    json_output_path = Path(args.json_output)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote markdown report to {output_path}")
    print(f"Wrote JSON summary to {json_output_path}")


if __name__ == "__main__":
    main()
