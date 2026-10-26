import json

from pharma_llm.data.schema import build_training_record, validate_training_record
from pharma_llm.data.validate import validate_jsonl


def test_training_record_schema_accepts_canonical_record():
    record = build_training_record(
        record_id="huatuo_000001",
        source="huatuogpt",
        category="general_medical",
        user="What is hypertension?",
        assistant="Hypertension is persistently elevated blood pressure.",
    )

    assert validate_training_record(record) == []


def test_training_record_schema_requires_assistant_message():
    record = {
        "id": "bad_001",
        "source": "huatuogpt",
        "category": "general_medical",
        "messages": [{"role": "user", "content": "Question"}],
    }

    errors = validate_training_record(record)

    assert "record must contain at least one assistant message" in errors


def test_validate_jsonl_reports_invalid_rows(tmp_path):
    input_path = tmp_path / "input.jsonl"
    valid_output = tmp_path / "valid.jsonl"
    invalid_output = tmp_path / "invalid.jsonl"
    valid_record = build_training_record(
        record_id="tcm_000001",
        source="tcm",
        category="traditional_chinese_medicine",
        user="Question",
        assistant="Answer",
    )
    invalid_record = {"id": "bad"}
    input_path.write_text(
        "\n".join(
            [
                json.dumps(valid_record, ensure_ascii=False),
                json.dumps(invalid_record, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_jsonl(
        input_path,
        output_path=valid_output,
        invalid_path=invalid_output,
    )

    assert report.valid_count == 1
    assert report.invalid_count == 1
    assert valid_output.exists()
    assert invalid_output.exists()


def test_validate_jsonl_rejects_non_object_json(tmp_path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("null\n", encoding="utf-8")

    report = validate_jsonl(input_path)

    assert report.valid_count == 0
    assert report.invalid_count == 1
    assert report.error_counts["record must be a JSON object"] == 1
