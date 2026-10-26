import json

from pharma_llm.evaluation.runner import (
    extract_choice_answer,
    load_cmb_exam,
    load_medqa_zh,
    score_examples,
)


def test_extract_choice_answer_handles_common_formats():
    choices = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}

    assert extract_choice_answer("答案：B", choices) == "B"
    assert extract_choice_answer("B。", choices) == "B"
    assert extract_choice_answer("正确答案为 B、D", choices) == "BD"


def test_load_medqa_zh_jsonl(tmp_path):
    path = tmp_path / "medqa_zh_4op.jsonl"
    row = {
        "problem_id": 7,
        "problem_text": "题干",
        "choices": ["甲", "乙", "丙", "丁"],
        "answer": ["b"],
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    examples = load_medqa_zh(tmp_path)

    assert len(examples) == 1
    assert examples[0].example_id == "7"
    assert examples[0].choices == {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}
    assert examples[0].answer == "B"


def test_load_cmb_exam_val_split(tmp_path):
    data_dir = tmp_path / "cmb_exam"
    split_dir = data_dir / "CMB" / "CMB-Exam" / "CMB-val"
    split_dir.mkdir(parents=True)
    rows = [
        {
            "id": 1,
            "question": "题干",
            "answer": "BC",
            "question_type": "多项选择题",
            "option": {"A": "甲", "B": "乙", "C": "丙"},
        }
    ]
    (split_dir / "CMB-val-merge.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    examples = load_cmb_exam(data_dir, split="val")

    assert len(examples) == 1
    assert examples[0].answer == "BC"
    assert examples[0].choices["C"] == "丙"


def test_score_examples_counts_accuracy(tmp_path):
    path = tmp_path / "medqa_zh_4op.jsonl"
    rows = [
        {"problem_id": 1, "problem_text": "题干1", "choices": ["甲", "乙"], "answer": ["a"]},
        {"problem_id": 2, "problem_text": "题干2", "choices": ["甲", "乙"], "answer": ["b"]},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    examples = load_medqa_zh(tmp_path)

    result = score_examples(examples, ["A", "A"])

    assert result["scored_samples"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
