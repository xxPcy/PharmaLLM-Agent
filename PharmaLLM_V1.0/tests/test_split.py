from pharma_llm.data.schema import build_training_record
from pharma_llm.data.split import split_train_validation


def make_record(record_id, source):
    return build_training_record(
        record_id=record_id,
        source=source,
        category="general_medical" if source == "huatuogpt" else "traditional_chinese_medicine",
        user=f"Question {record_id}",
        assistant=f"Answer {record_id}",
    )


def test_split_train_validation_samples_configured_sources_without_overlap():
    records = [
        *[make_record(f"huatuogpt_{index:06d}", "huatuogpt") for index in range(10)],
        *[make_record(f"tcm_{index:06d}", "tcm") for index in range(5)],
    ]

    train, validation, report = split_train_validation(
        records,
        huatuogpt_train_samples=6,
        tcm_train_samples=3,
        validation_samples=2,
        seed=42,
    )

    train_ids = {record["id"] for record in train}
    validation_ids = {record["id"] for record in validation}
    assert len(train) == 9
    assert len(validation) == 2
    assert train_ids.isdisjoint(validation_ids)
    assert report.train_source_counts == {"huatuogpt": 6, "tcm": 3}


def test_split_train_validation_is_deterministic_for_same_seed():
    records = [
        *[make_record(f"huatuogpt_{index:06d}", "huatuogpt") for index in range(20)],
        *[make_record(f"tcm_{index:06d}", "tcm") for index in range(10)],
    ]

    first_train, first_validation, _ = split_train_validation(
        records,
        huatuogpt_train_samples=5,
        tcm_train_samples=5,
        validation_samples=3,
        seed=7,
    )
    second_train, second_validation, _ = split_train_validation(
        records,
        huatuogpt_train_samples=5,
        tcm_train_samples=5,
        validation_samples=3,
        seed=7,
    )

    assert [record["id"] for record in first_train] == [
        record["id"] for record in second_train
    ]
    assert [record["id"] for record in first_validation] == [
        record["id"] for record in second_validation
    ]
