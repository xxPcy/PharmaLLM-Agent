from pharma_llm.data.dedup import exact_dedup_key, exact_deduplicate
from pharma_llm.data.schema import build_training_record


def test_exact_dedup_key_normalizes_whitespace():
    first = build_training_record(
        record_id="r1",
        source="huatuogpt",
        category="general_medical",
        user="What   is hypertension?",
        assistant="High blood pressure.",
    )
    second = build_training_record(
        record_id="r2",
        source="huatuogpt",
        category="general_medical",
        user="What is hypertension?",
        assistant="High blood pressure.",
    )

    assert exact_dedup_key(first) == exact_dedup_key(second)


def test_exact_deduplicate_preserves_first_record():
    records = [
        build_training_record(
            record_id="r1",
            source="huatuogpt",
            category="general_medical",
            user="Question",
            assistant="Answer",
        ),
        build_training_record(
            record_id="r2",
            source="huatuogpt",
            category="general_medical",
            user="Question",
            assistant="Answer",
        ),
        build_training_record(
            record_id="r3",
            source="tcm",
            category="traditional_chinese_medicine",
            user="Different question",
            assistant="Different answer",
        ),
    ]

    result = exact_deduplicate(records)

    assert [record["id"] for record in result.records] == ["r1", "r3"]
    assert result.report.duplicate_count == 1
    assert result.report.duplicate_ids == ["r2"]

