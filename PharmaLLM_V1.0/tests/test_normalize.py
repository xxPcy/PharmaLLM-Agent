import json

from pharma_llm.data.normalize import (
    normalize_datasets,
    normalize_huatuogpt_payload,
    normalize_tcm_payload,
    tcm_files_from_dataset_info,
)


def test_normalize_huatuogpt_payload_strips_question_answer_prefixes():
    record, error = normalize_huatuogpt_payload(
        {"data": ["问：高血压是什么？", "答：高血压是血压持续升高。"]},
        record_index=1,
    )

    assert error is None
    assert record["id"] == "huatuogpt_000001"
    assert record["source"] == "huatuogpt"
    assert record["category"] == "general_medical"
    assert record["messages"][0]["content"] == "高血压是什么？"
    assert record["messages"][1]["content"] == "高血压是血压持续升高。"


def test_normalize_tcm_payload_combines_instruction_and_input():
    record, error = normalize_tcm_payload(
        {
            "instruction": "基于输入给出诊断。",
            "input": "患者咳嗽三日。",
            "output": "咳嗽",
        },
        file_stem="SFT_nlpDiseaseDiagnosed_61486",
        record_index=7,
    )

    assert error is None
    assert record["id"] == "tcm_SFT_nlpDiseaseDiagnosed_61486_000007"
    assert record["category"] == "traditional_chinese_medicine"
    assert record["messages"][0]["content"] == "基于输入给出诊断。\n\n患者咳嗽三日。"
    assert record["messages"][1]["content"] == "咳嗽"


def test_tcm_files_from_dataset_info_excludes_translation_by_default(tmp_path):
    tcm_dir = tmp_path / "tcm"
    tcm_dir.mkdir()
    (tcm_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "SFT_medicalKnowledge_source1_548404": {
                    "file_name": "SFT_medicalKnowledge_source1_548404.json"
                },
                "_SFT_traditionalTrans_1959542": {
                    "file_name": "_SFT_traditionalTrans_1959542.json"
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    files = tcm_files_from_dataset_info(tcm_dir)

    assert [path.name for path in files] == ["SFT_medicalKnowledge_source1_548404.json"]


def test_normalize_datasets_writes_split_and_merged_outputs(tmp_path):
    huatuo_file = tmp_path / "huatuo.jsonl"
    tcm_dir = tmp_path / "tcm"
    tcm_dir.mkdir()
    tcm_file = tcm_dir / "SFT_medicalKnowledge_source1_548404.json"
    huatuo_output = tmp_path / "interim" / "huatuo.jsonl"
    tcm_output = tmp_path / "interim" / "tcm.jsonl"
    merged_output = tmp_path / "interim" / "merged.jsonl"

    huatuo_file.write_text(
        json.dumps(
            {"data": ["问：什么是感冒？", "答：感冒是常见的上呼吸道感染。"]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (tcm_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "SFT_medicalKnowledge_source1_548404": {
                    "file_name": tcm_file.name
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tcm_file.write_text(
        json.dumps(
            [
                {
                    "instruction": "",
                    "input": "为什么远行奔走时脚上会起泡？",
                    "output": "可能与鞋袜摩擦和挤压有关。",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = normalize_datasets(
        huatuogpt_input=huatuo_file,
        tcm_input_dir=tcm_dir,
        huatuogpt_output=huatuo_output,
        tcm_output=tcm_output,
        merged_output=merged_output,
    )

    merged_records = [
        json.loads(line) for line in merged_output.read_text(encoding="utf-8").splitlines()
    ]
    assert report.huatuogpt_output_count == 1
    assert report.tcm_output_count == 1
    assert report.merged_output_count == 2
    assert len(merged_records) == 2
    assert merged_records[0]["source"] == "huatuogpt"
    assert merged_records[1]["source"] == "tcm"


def test_normalize_datasets_accepts_sampled_tcm_file(tmp_path):
    huatuo_file = tmp_path / "huatuo.jsonl"
    tcm_file = tmp_path / "tcm_20k.json"
    huatuo_output = tmp_path / "interim" / "huatuo.jsonl"
    tcm_output = tmp_path / "interim" / "tcm.jsonl"
    merged_output = tmp_path / "interim" / "merged.jsonl"

    huatuo_file.write_text(
        json.dumps(
            {"data": ["问：什么是中暑？", "答：中暑与高温环境下体温调节异常有关。"]},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tcm_file.write_text(
        json.dumps(
            [
                {
                    "instruction": "基于输入给出证型诊断。",
                    "input": "患者咳嗽痰黄。",
                    "output": "痰热壅肺证",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = normalize_datasets(
        huatuogpt_input=huatuo_file,
        tcm_input_file=tcm_file,
        tcm_input_dir=None,
        huatuogpt_output=huatuo_output,
        tcm_output=tcm_output,
        merged_output=merged_output,
    )

    tcm_records = [
        json.loads(line) for line in tcm_output.read_text(encoding="utf-8").splitlines()
    ]
    assert report.tcm_files == [str(tcm_file)]
    assert report.tcm_input_count == 1
    assert report.tcm_output_count == 1
    assert tcm_records[0]["id"] == "tcm_tcm_20k_000001"
    assert tcm_records[0]["messages"][0]["content"] == (
        "基于输入给出证型诊断。\n\n患者咳嗽痰黄。"
    )
