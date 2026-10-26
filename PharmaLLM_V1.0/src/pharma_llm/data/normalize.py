from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

from pharma_llm.data.schema import build_training_record
from pharma_llm.data.validate import iter_jsonl
from pharma_llm.utils.config import load_yaml_config
from pharma_llm.utils.logging import setup_logging


LOGGER = setup_logging(__name__)
WHITESPACE_ONLY_LINES_RE = re.compile(r"\n{3,}")
NON_ID_CHARS_RE = re.compile(r"[^0-9A-Za-z_\-]+")
HUATUO_USER_PREFIXES = ("问：", "问:", "问题：", "问题:")
HUATUO_ASSISTANT_PREFIXES = ("答：", "答:", "回答：", "回答:")
TCM_TRANSLATION_MARKER = "traditionalTrans"


@dataclass
class NormalizationReport:
    huatuogpt_input_count: int
    huatuogpt_output_count: int
    tcm_input_count: int
    tcm_output_count: int
    merged_output_count: int
    tcm_files: list[str]
    skipped_counts: dict[str, int]
    output_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_text(value: Any) -> str:
    """Normalize line endings and trim surrounding whitespace."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return WHITESPACE_ONLY_LINES_RE.sub("\n\n", text)


def strip_prefix(text: str, prefixes: Iterable[str]) -> str:
    stripped = text.lstrip()
    for prefix in prefixes:
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].lstrip()
    return stripped


def combine_instruction_input(instruction: Any, input_text: Any) -> str:
    instruction_text = clean_text(instruction)
    input_body = clean_text(input_text)
    if instruction_text and input_body:
        return f"{instruction_text}\n\n{input_body}"
    return instruction_text or input_body


def safe_id_part(value: str) -> str:
    cleaned = NON_ID_CHARS_RE.sub("_", value).strip("_")
    return cleaned or "record"


def normalize_huatuogpt_payload(
    payload: Any,
    *,
    record_index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert one HuatuoGPT raw row into the canonical SFT schema."""

    if not isinstance(payload, dict):
        return None, "huatuogpt.record_not_object"

    data = payload.get("data")
    if not isinstance(data, list) or len(data) < 2:
        return None, "huatuogpt.data_not_pair"

    user = strip_prefix(clean_text(data[0]), HUATUO_USER_PREFIXES)
    assistant = strip_prefix(clean_text(data[1]), HUATUO_ASSISTANT_PREFIXES)
    if not user:
        return None, "huatuogpt.empty_user"
    if not assistant:
        return None, "huatuogpt.empty_assistant"

    record = build_training_record(
        record_id=f"huatuogpt_{record_index:06d}",
        source="huatuogpt",
        category="general_medical",
        user=user,
        assistant=assistant,
    )
    return record, None


def load_tcm_json_records(path: str | Path) -> list[Any]:
    """Load one TCM JSON file.

    The upstream dataset ships as JSON arrays, which are small enough to load file by
    file and keep the implementation dependency-free.
    """

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"TCM JSON file must contain a list: {json_path}")
    return payload


def normalize_tcm_payload(
    payload: Any,
    *,
    file_stem: str,
    record_index: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert one TCM instruction row into the canonical SFT schema."""

    if not isinstance(payload, dict):
        return None, "tcm.record_not_object"

    user = combine_instruction_input(payload.get("instruction"), payload.get("input"))
    assistant = clean_text(payload.get("output"))
    if not user:
        return None, "tcm.empty_user"
    if not assistant:
        return None, "tcm.empty_assistant"

    category = (
        "classical_chinese_translation"
        if TCM_TRANSLATION_MARKER in file_stem
        else "traditional_chinese_medicine"
    )
    record = build_training_record(
        record_id=f"tcm_{safe_id_part(file_stem)}_{record_index:06d}",
        source="tcm",
        category=category,
        user=user,
        assistant=assistant,
    )
    return record, None


def tcm_files_from_dataset_info(
    tcm_raw_dir: str | Path,
    *,
    include_translation: bool = False,
) -> list[Path]:
    """Return TCM JSON files in dataset_info.json order when available."""

    raw_dir = Path(tcm_raw_dir)
    dataset_info_path = raw_dir / "dataset_info.json"
    files: list[Path] = []

    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as handle:
            dataset_info = json.load(handle)
        if not isinstance(dataset_info, dict):
            raise ValueError(f"dataset_info.json must contain an object: {raw_dir}")

        for dataset_name, metadata in dataset_info.items():
            if not isinstance(metadata, dict):
                continue
            file_name = metadata.get("file_name")
            if not isinstance(file_name, str):
                continue
            if not include_translation and TCM_TRANSLATION_MARKER in dataset_name:
                continue
            files.append(raw_dir / file_name)
        return files

    for path in sorted(raw_dir.glob("*.json")):
        if path.name == "dataset_info.json":
            continue
        if not include_translation and TCM_TRANSLATION_MARKER in path.stem:
            continue
        files.append(path)
    return files


def write_record(handle: TextIO, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_datasets(
    *,
    huatuogpt_input: str | Path,
    tcm_input_dir: str | Path | None,
    tcm_input_file: str | Path | None = None,
    huatuogpt_output: str | Path,
    tcm_output: str | Path,
    merged_output: str | Path,
    include_tcm_translation: bool = False,
    report_path: str | Path | None = None,
    max_huatuogpt_records: int | None = None,
    max_tcm_records_per_file: int | None = None,
) -> NormalizationReport:
    """Normalize HuatuoGPT and TCM raw datasets into canonical JSONL files."""

    huatuogpt_input_path = Path(huatuogpt_input)
    tcm_input_dir_path = Path(tcm_input_dir) if tcm_input_dir is not None else None
    tcm_input_file_path = Path(tcm_input_file) if tcm_input_file is not None else None
    output_paths = {
        "huatuogpt": str(huatuogpt_output),
        "tcm": str(tcm_output),
        "merged": str(merged_output),
    }
    skipped_counts: Counter[str] = Counter()

    if not huatuogpt_input_path.exists():
        raise FileNotFoundError(f"HuatuoGPT raw file does not exist: {huatuogpt_input}")
    if tcm_input_file_path is not None and not tcm_input_file_path.exists():
        raise FileNotFoundError(f"TCM input file does not exist: {tcm_input_file}")
    if tcm_input_file_path is None and tcm_input_dir_path is None:
        raise ValueError("Either tcm_input_file or tcm_input_dir must be provided")
    if tcm_input_file_path is None and not tcm_input_dir_path.exists():
        raise FileNotFoundError(f"TCM raw directory does not exist: {tcm_input_dir}")

    huatuogpt_output_path = Path(huatuogpt_output)
    tcm_output_path = Path(tcm_output)
    merged_output_path = Path(merged_output)
    for path in (huatuogpt_output_path, tcm_output_path, merged_output_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    huatuogpt_input_count = 0
    huatuogpt_output_count = 0
    tcm_input_count = 0
    tcm_output_count = 0

    if tcm_input_file_path is not None:
        tcm_files = [tcm_input_file_path]
    else:
        tcm_files = tcm_files_from_dataset_info(
            tcm_input_dir_path,
            include_translation=include_tcm_translation,
        )
    missing_files = [str(path) for path in tcm_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing TCM raw files: {missing_files}")

    with (
        huatuogpt_output_path.open("w", encoding="utf-8") as huatuo_handle,
        tcm_output_path.open("w", encoding="utf-8") as tcm_handle,
        merged_output_path.open("w", encoding="utf-8") as merged_handle,
    ):
        for line_number, payload, parse_errors in iter_jsonl(huatuogpt_input_path):
            if (
                max_huatuogpt_records is not None
                and huatuogpt_input_count >= max_huatuogpt_records
            ):
                break

            huatuogpt_input_count += 1
            if parse_errors:
                skipped_counts["huatuogpt.invalid_json"] += 1
                LOGGER.warning(
                    "Skipping HuatuoGPT line %s: %s",
                    line_number,
                    parse_errors,
                )
                continue

            record, error = normalize_huatuogpt_payload(
                payload,
                record_index=huatuogpt_output_count + 1,
            )
            if error is not None:
                skipped_counts[error] += 1
                continue

            write_record(huatuo_handle, record)
            write_record(merged_handle, record)
            huatuogpt_output_count += 1

        for tcm_file in tcm_files:
            safe_stem = safe_id_part(tcm_file.stem)
            file_output_count = 0
            for file_input_count, payload in enumerate(
                load_tcm_json_records(tcm_file),
                start=1,
            ):
                if (
                    max_tcm_records_per_file is not None
                    and file_input_count > max_tcm_records_per_file
                ):
                    break

                tcm_input_count += 1
                record, error = normalize_tcm_payload(
                    payload,
                    file_stem=safe_stem,
                    record_index=file_output_count + 1,
                )
                if error is not None:
                    skipped_counts[error] += 1
                    continue

                write_record(tcm_handle, record)
                write_record(merged_handle, record)
                tcm_output_count += 1
                file_output_count += 1

    report = NormalizationReport(
        huatuogpt_input_count=huatuogpt_input_count,
        huatuogpt_output_count=huatuogpt_output_count,
        tcm_input_count=tcm_input_count,
        tcm_output_count=tcm_output_count,
        merged_output_count=huatuogpt_output_count + tcm_output_count,
        tcm_files=[str(path) for path in tcm_files],
        skipped_counts=dict(skipped_counts),
        output_paths=output_paths,
    )

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    LOGGER.info(
        "Normalization complete: huatuogpt=%s tcm=%s merged=%s",
        report.huatuogpt_output_count,
        report.tcm_output_count,
        report.merged_output_count,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw PharmaLLM datasets into canonical JSONL.",
    )
    parser.add_argument("--config", default="configs/data.yaml", help="Data config.")
    parser.add_argument("--huatuogpt-input", help="HuatuoGPT raw JSONL path.")
    parser.add_argument("--tcm-input-file", help="Sampled TCM JSON file.")
    parser.add_argument("--tcm-input-dir", help="TCM raw directory.")
    parser.add_argument("--huatuogpt-output", help="Normalized HuatuoGPT JSONL path.")
    parser.add_argument("--tcm-output", help="Normalized TCM JSONL path.")
    parser.add_argument("--merged-output", help="Merged normalized JSONL path.")
    parser.add_argument("--report", help="Optional normalization JSON report path.")
    parser.add_argument(
        "--include-tcm-translation",
        action="store_true",
        help="Include the optional classical/modern Chinese translation file.",
    )
    parser.add_argument(
        "--max-huatuogpt-records",
        type=int,
        help="Optional cap for HuatuoGPT records, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-tcm-records-per-file",
        type=int,
        help="Optional per-file TCM cap, useful for smoke tests.",
    )
    return parser.parse_args()


def config_default(config: dict[str, Any], key: str, fallback: str) -> str:
    value = config.get("paths", {}).get(key)
    return value if isinstance(value, str) else fallback


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    huatuogpt_raw_dir = config_default(
        config,
        "huatuogpt_raw_dir",
        "data/raw/huatuogpt",
    )
    default_huatuogpt_input = str(
        Path(huatuogpt_raw_dir) / "HuatuoGPT_sft_data_v1.jsonl"
    )
    configured_tcm_input_file = config_default(config, "tcm_sampled", "")
    default_tcm_input_file = (
        configured_tcm_input_file
        if configured_tcm_input_file and Path(configured_tcm_input_file).exists()
        else None
    )

    report = normalize_datasets(
        huatuogpt_input=args.huatuogpt_input or default_huatuogpt_input,
        tcm_input_file=args.tcm_input_file or default_tcm_input_file,
        tcm_input_dir=args.tcm_input_dir
        or config_default(config, "tcm_raw_dir", "data/raw/tcm"),
        huatuogpt_output=args.huatuogpt_output
        or config_default(
            config,
            "huatuogpt_normalized",
            "data/interim/huatuogpt_normalized.jsonl",
        ),
        tcm_output=args.tcm_output
        or config_default(
            config,
            "tcm_normalized",
            "data/interim/tcm_normalized.jsonl",
        ),
        merged_output=args.merged_output
        or config_default(
            config,
            "merged_normalized",
            "data/interim/merged_normalized.jsonl",
        ),
        include_tcm_translation=args.include_tcm_translation,
        report_path=args.report,
        max_huatuogpt_records=args.max_huatuogpt_records,
        max_tcm_records_per_file=args.max_tcm_records_per_file,
    )
    LOGGER.info("Wrote normalization report: %s", report.to_dict())


if __name__ == "__main__":
    main()
