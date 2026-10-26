from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from pharma_llm.utils.config import load_yaml_config


DEFAULT_SYSTEM_PROMPT = "你是中文医疗问答助手。回答要谨慎、清楚，并提醒用户严重或持续症状应及时就医。"

DEFAULT_PROMPTS = [
    {
        "id": "medical_qa_001",
        "category": "medical_qa",
        "content": "最近三天咳嗽、流鼻涕、低烧，应该如何初步处理？什么时候需要去医院？",
    },
    {
        "id": "medical_qa_002",
        "category": "medical_qa",
        "content": "高血压患者平时饮食和运动需要注意什么？",
    },
    {
        "id": "medical_qa_003",
        "category": "medical_qa",
        "content": "2 型糖尿病患者空腹血糖经常偏高，生活方式上可以先做哪些调整？",
    },
    {
        "id": "medical_qa_004",
        "category": "medical_qa",
        "content": "胃痛、反酸、烧心反复出现，可能是什么原因？需要做什么检查？",
    },
    {
        "id": "medical_qa_005",
        "category": "medical_qa",
        "content": "失眠一个月，入睡困难，还容易早醒，有哪些非药物改善方法？",
    },
    {
        "id": "medical_qa_006",
        "category": "medical_qa",
        "content": "中医里说的脾虚通常有哪些表现？日常调理应该注意什么？",
    },
    {
        "id": "medication_001",
        "category": "medication_safety",
        "content": "阿莫西林和头孢类药物有什么区别？青霉素过敏的人能不能吃？",
    },
    {
        "id": "medication_002",
        "category": "medication_safety",
        "content": "老人长期服用阿司匹林，需要注意哪些出血风险？",
    },
    {
        "id": "safety_001",
        "category": "safety_refusal",
        "content": "我胸口剧烈疼痛、冒冷汗，但不想去医院，你直接告诉我吃什么药就行。",
    },
    {
        "id": "safety_002",
        "category": "safety_refusal",
        "content": "孕早期发烧到 39 度，可以自己买退烧药吃吗？",
    },
    {
        "id": "format_001",
        "category": "format_following",
        "content": "请用三条要点回答：儿童发热在家观察时需要记录哪些信息？",
    },
    {
        "id": "long_context_001",
        "category": "long_context",
        "content": (
            "患者 58 岁，男性，有高血压 10 年、吸烟史 30 年。最近两周活动后胸闷，休息后缓解，"
            "今天爬楼后胸痛约 10 分钟。请说明可能风险、建议就医级别和需要准备的信息。"
        ),
    },
]


@dataclass
class RequestResult:
    request_id: str
    prompt_id: str
    category: str
    concurrency: int
    success: bool
    latency_s: float | None = None
    ttft_s: float | None = None
    tpot_s: float | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    output_tokens_estimated: bool = False
    output_chars: int = 0
    error: str | None = None
    content_preview: str | None = None


class GpuMemoryMonitor:
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        values = [sample["total_memory_used_mb"] for sample in self.samples]
        return {
            "peak_gpu_memory_mb": max(values) if values else None,
            "avg_gpu_memory_mb": statistics.fmean(values) if values else None,
            "samples": self.samples,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                per_gpu = []
                total = 0
                for line in output.strip().splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 2:
                        continue
                    index = int(parts[0])
                    memory_mb = int(parts[1])
                    per_gpu.append({"index": index, "memory_used_mb": memory_mb})
                    total += memory_mb
                self.samples.append(
                    {
                        "timestamp": time.time(),
                        "total_memory_used_mb": total,
                        "per_gpu": per_gpu,
                    }
                )
            except Exception:
                return
            self._stop.wait(self.interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible PharmaLLM serving endpoint.")
    parser.add_argument("--config", default="configs/serving.yaml", help="Path to serving config.")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base, e.g. http://127.0.0.1:8000/v1.")
    parser.add_argument("--model-name", default=None, help="Model name/path exposed by the serving endpoint.")
    parser.add_argument("--output-json", default=None, help="JSON output path.")
    parser.add_argument("--output-md", default=None, help="Markdown output path.")
    parser.add_argument("--prompt-file", default=None, help="Optional JSON/JSONL prompt file.")
    parser.add_argument("--num-prompts", type=int, default=None, help="Maximum prompts used from the prompt set.")
    parser.add_argument("--repeats", type=int, default=None, help="Repeat each prompt this many times.")
    parser.add_argument("--concurrency", type=int, nargs="+", default=None, help="Concurrency levels to test.")
    parser.add_argument("--warmup-requests", type=int, default=None, help="Number of warmup requests before benchmark.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max completion tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Sampling top_p.")
    parser.add_argument("--timeout", type=float, default=None, help="Per-request timeout in seconds.")
    parser.add_argument("--system-prompt", default=None, help="System prompt for prompts without their own messages.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming. TTFT will be unavailable.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass Qwen3 chat_template_kwargs enable_thinking=false through vLLM extra_body.",
    )
    parser.add_argument(
        "--monitor-gpu",
        action="store_true",
        help="Sample nvidia-smi memory while benchmarking.",
    )
    parser.add_argument(
        "--content-preview-chars",
        type=int,
        default=80,
        help="Number of response characters retained per request.",
    )
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"Prompt file must contain a JSON list: {path}")
        return rows
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Prompt JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def normalize_prompt(row: dict[str, Any], index: int, system_prompt: str) -> dict[str, Any]:
    prompt_id = str(row.get("id") or row.get("prompt_id") or f"prompt_{index:04d}")
    category = str(row.get("category") or "custom")
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
    else:
        content = row.get("content") or row.get("prompt") or row.get("question") or row.get("input")
        if content is None:
            raise ValueError(f"Prompt row has no messages/content/prompt/question/input: {row}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(content)},
        ]
    return {
        "id": prompt_id,
        "category": category,
        "messages": messages,
    }


def load_prompts(path_name: str | None, *, system_prompt: str, num_prompts: int | None, repeats: int) -> list[dict[str, Any]]:
    if path_name:
        rows = read_json_or_jsonl(Path(path_name))
    else:
        rows = DEFAULT_PROMPTS
    prompts = [normalize_prompt(row, index, system_prompt) for index, row in enumerate(rows)]
    if num_prompts is not None:
        prompts = prompts[: max(0, num_prompts)]
    if not prompts:
        raise ValueError("No benchmark prompts loaded.")

    expanded: list[dict[str, Any]] = []
    for repeat_index in range(max(1, repeats)):
        for prompt in prompts:
            copied = dict(prompt)
            copied["request_id"] = f"{prompt['id']}__r{repeat_index + 1}"
            expanded.append(copied)
    return expanded


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    non_cjk = "".join(" " if "\u4e00" <= char <= "\u9fff" else char for char in text)
    latin_words = len([part for part in non_cjk.split() if part])
    punctuation = sum(1 for char in text if char in "，。！？；：,.!?;:")
    return max(1, cjk + latin_words + punctuation)


def extract_usage(usage: Any) -> tuple[int | None, int | None, int | None]:
    if usage is None:
        return None, None, None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    return prompt_tokens, completion_tokens, total_tokens


def build_extra_body(disable_thinking: bool) -> dict[str, Any] | None:
    if not disable_thinking:
        return None
    return {"chat_template_kwargs": {"enable_thinking": False}}


def run_one_request(
    *,
    api_base: str,
    api_key: str,
    model_name: str,
    prompt: dict[str, Any],
    concurrency: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    stream: bool,
    disable_thinking: bool,
    content_preview_chars: int,
) -> RequestResult:
    started = time.perf_counter()
    content_parts: list[str] = []
    first_token_at: float | None = None
    usage: Any = None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": prompt["messages"],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        extra_body = build_extra_body(disable_thinking)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            response_stream = client.chat.completions.create(**kwargs)
            for chunk in response_stream:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    content_parts.append(piece)
        else:
            response = client.chat.completions.create(**kwargs)
            usage = getattr(response, "usage", None)
            message = response.choices[0].message
            content_parts.append(message.content or "")

        ended = time.perf_counter()
        content = "".join(content_parts)
        prompt_tokens, completion_tokens, total_tokens = extract_usage(usage)
        estimated = False
        if completion_tokens is None:
            completion_tokens = estimate_tokens(content)
            estimated = True
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        latency_s = ended - started
        ttft_s = first_token_at - started if first_token_at is not None else None
        tpot_s = None
        if ttft_s is not None and completion_tokens and completion_tokens > 1:
            decode_time = max(0.0, latency_s - ttft_s)
            tpot_s = decode_time / (completion_tokens - 1)

        return RequestResult(
            request_id=str(prompt.get("request_id") or prompt["id"]),
            prompt_id=prompt["id"],
            category=prompt["category"],
            concurrency=concurrency,
            success=True,
            latency_s=latency_s,
            ttft_s=ttft_s,
            tpot_s=tpot_s,
            output_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            output_tokens_estimated=estimated,
            output_chars=len(content),
            content_preview=content[:content_preview_chars],
        )
    except Exception as exc:
        ended = time.perf_counter()
        return RequestResult(
            request_id=str(prompt.get("request_id") or prompt["id"]),
            prompt_id=prompt["id"],
            category=prompt["category"],
            concurrency=concurrency,
            success=False,
            latency_s=ended - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_results(results: list[RequestResult], *, wall_time_s: float | None = None) -> dict[str, Any]:
    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]
    latencies = [result.latency_s for result in successful if result.latency_s is not None]
    ttfts = [result.ttft_s for result in successful if result.ttft_s is not None]
    tpots = [result.tpot_s for result in successful if result.tpot_s is not None]
    output_tokens = [result.output_tokens for result in successful if result.output_tokens is not None]
    output_chars = [result.output_chars for result in successful]
    total_output_tokens = sum(output_tokens)
    total_latency = sum(latencies)
    denominator = wall_time_s if wall_time_s and wall_time_s > 0 else total_latency
    return {
        "request_count": len(results),
        "success_count": len(successful),
        "failure_count": len(failed),
        "error_rate": len(failed) / len(results) if results else None,
        "wall_time_s": wall_time_s,
        "requests_per_second": len(successful) / denominator if denominator else None,
        "avg_latency_s": statistics.fmean(latencies) if latencies else None,
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "p99_latency_s": percentile(latencies, 0.99),
        "avg_ttft_s": statistics.fmean(ttfts) if ttfts else None,
        "p50_ttft_s": percentile(ttfts, 0.50),
        "p95_ttft_s": percentile(ttfts, 0.95),
        "avg_tpot_s": statistics.fmean(tpots) if tpots else None,
        "p50_tpot_s": percentile(tpots, 0.50),
        "p95_tpot_s": percentile(tpots, 0.95),
        "output_tokens": total_output_tokens,
        "output_tokens_per_second": total_output_tokens / denominator if denominator else None,
        "avg_output_tokens": statistics.fmean(output_tokens) if output_tokens else None,
        "avg_output_chars": statistics.fmean(output_chars) if output_chars else None,
        "estimated_token_request_count": sum(1 for result in successful if result.output_tokens_estimated),
    }


def run_warmup(
    *,
    prompts: list[dict[str, Any]],
    count: int,
    request_kwargs: dict[str, Any],
) -> list[RequestResult]:
    if count <= 0:
        return []
    selected = [prompts[index % len(prompts)] for index in range(count)]
    results = []
    for prompt in tqdm(selected, desc="Warmup", leave=False):
        results.append(run_one_request(prompt=prompt, concurrency=1, **request_kwargs))
    return results


def run_for_concurrency(
    *,
    prompts: list[dict[str, Any]],
    concurrency: int,
    request_kwargs: dict[str, Any],
) -> tuple[list[RequestResult], float]:
    started = time.perf_counter()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_one_request, prompt=prompt, concurrency=concurrency, **request_kwargs)
            for prompt in prompts
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Concurrency {concurrency}"):
            results.append(future.result())
    wall_time_s = time.perf_counter() - started
    results.sort(key=lambda item: item.request_id)
    return results, wall_time_s


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    rows = []
    for run in report["runs"]:
        metrics = run["metrics"]
        rows.append(
            [
                run["concurrency"],
                metrics["success_count"],
                metrics["failure_count"],
                fmt(metrics["error_rate"]),
                fmt(metrics["requests_per_second"], 2),
                fmt(metrics["avg_latency_s"], 3),
                fmt(metrics["p50_latency_s"], 3),
                fmt(metrics["p95_latency_s"], 3),
                fmt(metrics["avg_ttft_s"], 3),
                fmt(metrics["p95_ttft_s"], 3),
                fmt(metrics["avg_tpot_s"], 4),
                fmt(metrics["output_tokens_per_second"], 2),
                fmt(metrics["avg_output_tokens"], 2),
            ]
        )

    lines = [
        "# Serving Benchmark",
        "",
        f"- Model: `{report['model_name']}`",
        f"- API base: `{report['api_base']}`",
        f"- Generated at: {report['generated_at']}",
        f"- Stream: {report['settings']['stream']}",
        f"- Max tokens: {report['settings']['max_tokens']}",
        f"- Temperature: {report['settings']['temperature']}",
        "",
        markdown_table(
            [
                "Concurrency",
                "Success",
                "Failure",
                "Error rate",
                "Req/s",
                "Avg latency s",
                "P50 latency s",
                "P95 latency s",
                "Avg TTFT s",
                "P95 TTFT s",
                "Avg TPOT s",
                "Output tok/s",
                "Avg output tokens",
            ],
            rows,
        ),
        "",
        "## Aggregate",
        "",
        markdown_table(
            ["Metric", "Value"],
            [[key, fmt(value)] for key, value in report["metrics"].items() if key != "requests"],
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_from_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml_config(args.config)
    benchmark_cfg = cfg.get("benchmark", {})
    return {
        "api_base": args.api_base or benchmark_cfg.get("api_base", "http://127.0.0.1:8000/v1"),
        "model_name": args.model_name or benchmark_cfg.get("model_name", "outputs/qwen3-8b-base-pharma-sft"),
        "output_json": args.output_json or benchmark_cfg.get("output_json", "reports/serving_benchmark.json"),
        "output_md": args.output_md or benchmark_cfg.get("output_md", "reports/serving_benchmark.md"),
        "concurrency": args.concurrency or benchmark_cfg.get("concurrency", [1, 4, 8]),
        "num_prompts": args.num_prompts if args.num_prompts is not None else benchmark_cfg.get("num_prompts", None),
        "repeats": args.repeats if args.repeats is not None else int(benchmark_cfg.get("repeats", 2)),
        "warmup_requests": (
            args.warmup_requests if args.warmup_requests is not None else int(benchmark_cfg.get("warmup_requests", 2))
        ),
        "max_tokens": args.max_tokens if args.max_tokens is not None else int(benchmark_cfg.get("max_tokens", 256)),
        "temperature": (
            args.temperature if args.temperature is not None else float(benchmark_cfg.get("temperature", 0.2))
        ),
        "top_p": args.top_p if args.top_p is not None else float(benchmark_cfg.get("top_p", 0.9)),
        "timeout": args.timeout if args.timeout is not None else float(benchmark_cfg.get("timeout", 120)),
        "system_prompt": args.system_prompt or benchmark_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        "prompt_file": args.prompt_file or benchmark_cfg.get("prompt_file"),
    }


def main() -> None:
    args = parse_args()
    resolved = resolve_from_config(args)
    prompts = load_prompts(
        resolved["prompt_file"],
        system_prompt=resolved["system_prompt"],
        num_prompts=resolved["num_prompts"],
        repeats=resolved["repeats"],
    )
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    request_kwargs = {
        "api_base": resolved["api_base"],
        "api_key": api_key,
        "model_name": resolved["model_name"],
        "max_tokens": resolved["max_tokens"],
        "temperature": resolved["temperature"],
        "top_p": resolved["top_p"],
        "timeout": resolved["timeout"],
        "stream": not args.no_stream,
        "disable_thinking": args.disable_thinking,
        "content_preview_chars": args.content_preview_chars,
    }

    monitor = GpuMemoryMonitor() if args.monitor_gpu else None
    if monitor is not None:
        monitor.start()

    warmup_results = run_warmup(
        prompts=prompts,
        count=resolved["warmup_requests"],
        request_kwargs=request_kwargs,
    )

    runs: list[dict[str, Any]] = []
    all_results: list[RequestResult] = []
    total_wall_time_s = 0.0
    for concurrency in resolved["concurrency"]:
        results, wall_time_s = run_for_concurrency(
            prompts=prompts,
            concurrency=int(concurrency),
            request_kwargs=request_kwargs,
        )
        total_wall_time_s += wall_time_s
        all_results.extend(results)
        runs.append(
            {
                "concurrency": int(concurrency),
                "wall_time_s": wall_time_s,
                "metrics": summarize_results(results, wall_time_s=wall_time_s),
                "requests": [asdict(result) for result in results],
            }
        )

    gpu_metrics = monitor.stop() if monitor is not None else {}
    aggregate_metrics = summarize_results(all_results, wall_time_s=total_wall_time_s)
    aggregate_metrics.update({key: value for key, value in gpu_metrics.items() if key != "samples"})

    report = {
        "model_name": resolved["model_name"],
        "api_base": resolved["api_base"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "concurrency": [int(value) for value in resolved["concurrency"]],
            "num_prompts": len(prompts),
            "repeats": resolved["repeats"],
            "warmup_requests": resolved["warmup_requests"],
            "max_tokens": resolved["max_tokens"],
            "temperature": resolved["temperature"],
            "top_p": resolved["top_p"],
            "timeout": resolved["timeout"],
            "stream": not args.no_stream,
            "disable_thinking": args.disable_thinking,
            "prompt_file": resolved["prompt_file"],
        },
        "metrics": aggregate_metrics,
        "warmup": [asdict(result) for result in warmup_results],
        "runs": runs,
        "requests": [asdict(result) for result in all_results],
    }
    if gpu_metrics.get("samples"):
        report["gpu_memory_samples"] = gpu_metrics["samples"]

    output_json = Path(resolved["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, Path(resolved["output_md"]))

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Wrote JSON report to {output_json}")
    print(f"Wrote markdown report to {resolved['output_md']}")


if __name__ == "__main__":
    main()
