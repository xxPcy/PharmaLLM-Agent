from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from pharma_llm.data.validate import iter_jsonl, write_jsonl
from pharma_llm.utils.logging import setup_logging


LOGGER = setup_logging(__name__)
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class ExactDedupReport:
    input_count: int
    output_count: int
    duplicate_count: int
    duplicate_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExactDedupResult:
    records: list[dict[str, Any]]
    report: ExactDedupReport


def normalize_for_exact_dedup(text: str) -> str:
    """Normalize text for exact duplicate hashing."""

    normalized = unicodedata.normalize("NFKC", text)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def user_assistant_text(record: dict[str, Any]) -> str:
    """Extract user and assistant text used by the exact dedup key."""

    user_parts: list[str] = []
    assistant_parts: list[str] = []
    for message in record.get("messages", []):
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    return "\n".join(
        [
            normalize_for_exact_dedup("\n".join(user_parts)),
            normalize_for_exact_dedup("\n".join(assistant_parts)),
        ]
    )


def exact_dedup_key(record: dict[str, Any]) -> str:
    payload = user_assistant_text(record)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_deduplicate(records: Iterable[dict[str, Any]]) -> ExactDedupResult:
    """Remove exact duplicates while preserving first occurrence order."""

    seen: set[str] = set()
    unique_records: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    input_count = 0

    for record in records:
        input_count += 1
        key = exact_dedup_key(record)
        if key in seen:
            duplicate_ids.append(str(record.get("id", "")))
            continue
        seen.add(key)
        unique_records.append(record)

    report = ExactDedupReport(
        input_count=input_count,
        output_count=len(unique_records),
        duplicate_count=len(duplicate_ids),
        duplicate_ids=duplicate_ids,
    )
    return ExactDedupResult(records=unique_records, report=report)


def exact_deduplicate_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    duplicates_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> ExactDedupReport:
    """Run exact deduplication over a JSONL file."""

    records: list[dict[str, Any]] = []
    for line_number, payload, errors in iter_jsonl(input_path):
        if errors:
            raise ValueError(f"Invalid JSONL at line {line_number}: {errors}")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid JSONL at line {line_number}: record must be an object"
            )
        records.append(payload)

    result = exact_deduplicate(records)
    write_jsonl(result.records, output_path)

    if duplicates_path is not None:
        duplicate_set = set(result.report.duplicate_ids)
        duplicate_records = [
            record for record in records if str(record.get("id", "")) in duplicate_set
        ]
        write_jsonl(duplicate_records, duplicates_path)

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result.report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    LOGGER.info(
        "Exact dedup complete: input=%s output=%s duplicates=%s",
        result.report.input_count,
        result.report.output_count,
        result.report.duplicate_count,
    )
    return result.report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact deduplication.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--duplicates-output", help="Optional duplicates JSONL path.")
    parser.add_argument("--report", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exact_deduplicate_jsonl(
        args.input,
        args.output,
        duplicates_path=args.duplicates_output,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
