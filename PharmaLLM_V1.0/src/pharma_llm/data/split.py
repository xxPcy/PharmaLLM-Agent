from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from pharma_llm.data.validate import iter_jsonl, write_jsonl
from pharma_llm.utils.config import load_yaml_config
from pharma_llm.utils.logging import setup_logging


LOGGER = setup_logging(__name__)


@dataclass
class SplitReport:
    input_count: int
    train_count: int
    validation_count: int
    train_source_counts: dict[str, int]
    validation_source_counts: dict[str, int]
    seed: int
    output_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, payload, errors in iter_jsonl(path):
        if errors:
            raise ValueError(f"Invalid JSONL at line {line_number}: {errors}")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid JSONL at line {line_number}: record must be an object"
            )
        records.append(payload)
    return records


def sample_records(
    records: list[dict[str, Any]],
    sample_size: int,
    *,
    rng: random.Random,
    label: str,
) -> list[dict[str, Any]]:
    if sample_size < 0:
        raise ValueError(f"{label} sample size must be non-negative")
    if len(records) < sample_size:
        raise ValueError(
            f"{label}: requested {sample_size} records, but only {len(records)} exist"
        )
    return rng.sample(records, sample_size)


def source_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(record.get("source", "")) for record in records))


def split_train_validation(
    records: list[dict[str, Any]],
    *,
    huatuogpt_train_samples: int,
    tcm_train_samples: int,
    validation_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SplitReport]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record.get("source", ""))].append(record)

    train_huatuogpt = sample_records(
        by_source["huatuogpt"],
        huatuogpt_train_samples,
        rng=rng,
        label="huatuogpt train",
    )
    train_tcm = sample_records(
        by_source["tcm"],
        tcm_train_samples,
        rng=rng,
        label="tcm train",
    )

    train_records = train_huatuogpt + train_tcm
    train_ids = {str(record.get("id", "")) for record in train_records}
    remaining_records = [
        record for record in records if str(record.get("id", "")) not in train_ids
    ]
    validation_records = sample_records(
        remaining_records,
        validation_samples,
        rng=rng,
        label="validation",
    )

    rng.shuffle(train_records)
    rng.shuffle(validation_records)

    report = SplitReport(
        input_count=len(records),
        train_count=len(train_records),
        validation_count=len(validation_records),
        train_source_counts=source_counts(train_records),
        validation_source_counts=source_counts(validation_records),
        seed=seed,
        output_paths={},
    )
    return train_records, validation_records, report


def split_jsonl(
    *,
    input_path: str | Path,
    train_output: str | Path,
    validation_output: str | Path,
    huatuogpt_train_samples: int,
    tcm_train_samples: int,
    validation_samples: int,
    seed: int,
    report_path: str | Path | None = None,
) -> SplitReport:
    records = load_records(input_path)
    train_records, validation_records, report = split_train_validation(
        records,
        huatuogpt_train_samples=huatuogpt_train_samples,
        tcm_train_samples=tcm_train_samples,
        validation_samples=validation_samples,
        seed=seed,
    )

    write_jsonl(train_records, train_output)
    write_jsonl(validation_records, validation_output)
    report.output_paths = {
        "train": str(train_output),
        "validation": str(validation_output),
    }

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    LOGGER.info(
        "Split complete: train=%s validation=%s",
        report.train_count,
        report.validation_count,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split deduplicated PharmaLLM JSONL into train and validation sets."
    )
    parser.add_argument("--config", default="configs/data.yaml", help="Data config.")
    parser.add_argument("--input", help="Input deduplicated JSONL path.")
    parser.add_argument("--train-output", help="Output train JSONL path.")
    parser.add_argument("--validation-output", help="Output validation JSONL path.")
    parser.add_argument("--report", default="reports/split_report.json")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--huatuogpt-train-samples", type=int)
    parser.add_argument("--tcm-train-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    return parser.parse_args()


def config_default(config: dict[str, Any], section: str, key: str, fallback: Any) -> Any:
    section_value = config.get(section, {})
    if not isinstance(section_value, Mapping):
        return fallback
    value = section_value.get(key)
    return fallback if value is None else value


def config_seed(config: dict[str, Any], fallback: int) -> int:
    value = config.get("seed", fallback)
    return int(value if value is not None else fallback)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    split_jsonl(
        input_path=args.input
        or config_default(
            config,
            "paths",
            "deduplicated",
            "data/interim/deduplicated.jsonl",
        ),
        train_output=args.train_output
        or config_default(config, "paths", "train", "data/processed/train.jsonl"),
        validation_output=args.validation_output
        or config_default(
            config,
            "paths",
            "validation",
            "data/processed/val.jsonl",
        ),
        huatuogpt_train_samples=args.huatuogpt_train_samples
        if args.huatuogpt_train_samples is not None
        else int(config_default(config, "sampling", "huatuogpt_train_samples", 60000)),
        tcm_train_samples=args.tcm_train_samples
        if args.tcm_train_samples is not None
        else int(config_default(config, "sampling", "tcm_train_samples", 20000)),
        validation_samples=args.validation_samples
        if args.validation_samples is not None
        else int(config_default(config, "sampling", "validation_samples", 2000)),
        seed=args.seed if args.seed is not None else config_seed(config, 42),
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
