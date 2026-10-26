from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, cast

from pharma_llm.data.schema import normalize_training_record, validate_training_record
from pharma_llm.utils.config import load_yaml_config
from pharma_llm.utils.logging import setup_logging


LOGGER = setup_logging(__name__)


@dataclass
class JsonlValidationReport:
    input_path: str
    valid_count: int
    invalid_count: int
    invalid_ratio: float
    error_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, Any, list[str]]]:
    """Yield line numbers and JSON objects from a JSONL file."""

    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                yield line_number, None, ["line is empty"]
                continue
            try:
                yield line_number, json.loads(stripped), []
            except json.JSONDecodeError as exc:
                yield line_number, None, [f"invalid JSON: {exc.msg}"]


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write records to a JSONL file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_jsonl(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    invalid_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> JsonlValidationReport:
    """Validate a canonical JSONL file and optionally write valid/invalid rows."""

    schema_config: dict[str, Any] = {}
    if config_path is not None:
        schema_config = load_yaml_config(config_path).get("schema", {})

    legal_roles = set(schema_config.get("legal_roles", [])) or None
    require_user = bool(schema_config.get("require_user_message", True))
    require_assistant = bool(schema_config.get("require_assistant_message", True))

    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    error_counter: Counter[str] = Counter()

    for line_number, payload, parse_errors in iter_jsonl(input_path):
        errors = list(parse_errors)
        if not parse_errors:
            errors.extend(
                validate_training_record(
                    payload,
                    legal_roles=legal_roles,
                    require_user_message=require_user,
                    require_assistant_message=require_assistant,
                )
            )

        if errors:
            error_counter.update(errors)
            invalid_records.append(
                {"line": line_number, "errors": errors, "record": payload}
            )
            continue

        valid_records.append(normalize_training_record(cast(dict[str, Any], payload)))

    if output_path is not None:
        write_jsonl(valid_records, output_path)
    if invalid_path is not None:
        write_jsonl(invalid_records, invalid_path)

    total = len(valid_records) + len(invalid_records)
    report = JsonlValidationReport(
        input_path=str(input_path),
        valid_count=len(valid_records),
        invalid_count=len(invalid_records),
        invalid_ratio=(len(invalid_records) / total) if total else 0.0,
        error_counts=dict(error_counter),
    )
    LOGGER.info(
        "Validated %s: valid=%s invalid=%s",
        input_path,
        report.valid_count,
        report.invalid_count,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PharmaLLM JSONL records.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", help="Optional valid-record JSONL path.")
    parser.add_argument("--invalid-output", help="Optional invalid-record JSONL path.")
    parser.add_argument("--config", default="configs/data.yaml", help="Data config path.")
    parser.add_argument("--report", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_jsonl(
        args.input,
        output_path=args.output,
        invalid_path=args.invalid_output,
        config_path=args.config,
    )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
