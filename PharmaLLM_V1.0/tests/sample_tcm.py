import json
import random
import hashlib
import argparse
from pathlib import Path
from collections import Counter


SEED = 42
DEFAULT_OUTPUT_PATH = "data/interim/tcm_20k.json"


CONFIG = [
    {
        "path": "data/raw/tcm/SFT_medicalKnowledge_source2_99334.json",
        "name": "medical_knowledge",
        "target": 10000,
    },
    {
        "path": "data/raw/tcm/SFT_structPrescription_92896.json",
        "name": "prescription",
        "target": 5000,
    },
    {
        "path": "data/raw/tcm/SFT_nlpSyndromeDiagnosed_48665.json",
        "name": "syndrome",
        "target": 5000,
    },
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text):
    if text is None:
        return ""

    return " ".join(str(text).strip().split())


def user_text(item):
    instruction = normalize(item.get("instruction"))
    input_text = normalize(item.get("input"))

    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"

    return instruction or input_text


def is_valid(item):
    if not isinstance(item, dict):
        return False

    user = user_text(item)
    output = normalize(item.get("output"))

    if len(user) < 5:
        return False

    # Diagnosis files intentionally have short label-style outputs, for example
    # "湿热下注证", so this threshold must stay low.
    if len(output) < 2:
        return False

    total_len = (
        len(user)
        + len(output)
    )

    if total_len > 6000:
        return False

    return True


def sample_hash(item):
    text = (
        user_text(item)
        + "\n"
        + normalize(item.get("output"))
    )

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()


def deduplicate(data):
    seen = set()
    result = []

    for item in data:
        h = sample_hash(item)

        if h in seen:
            continue

        seen.add(h)
        result.append(item)

    return result


def get_length_bin(item):
    total_len = len(
        user_text(item)
        + normalize(item.get("output"))
    )

    if total_len < 300:
        return "short"

    elif total_len < 800:
        return "medium"

    elif total_len < 1500:
        return "long"

    else:
        return "very_long"


def stratified_sample(data, target, rng):
    if target > len(data):
        raise ValueError(
            f"target {target} is larger than available data {len(data)}"
        )

    buckets = {
        "short": [],
        "medium": [],
        "long": [],
        "very_long": [],
    }

    for item in data:
        buckets[get_length_bin(item)].append(item)

    total = len(data)
    sampled = []

    for bucket_name, bucket in buckets.items():

        if not bucket:
            continue

        ratio = len(bucket) / total
        n = round(target * ratio)

        n = min(n, len(bucket))

        sampled.extend(
            rng.sample(bucket, n)
        )

    # 修正 rounding 导致的数量偏差
    if len(sampled) < target:

        selected_ids = {
            sample_hash(x)
            for x in sampled
        }

        remaining = [
            x for x in data
            if sample_hash(x) not in selected_ids
        ]

        sampled.extend(
            rng.sample(
                remaining,
                target - len(sampled)
            )
        )

    elif len(sampled) > target:

        sampled = rng.sample(
            sampled,
            target
        )

    rng.shuffle(sampled)

    return sampled


def process_source(config, rng):

    path = config["path"]
    source_name = config["name"]
    target = config["target"]

    print(f"\nProcessing: {source_name}")

    data = load_json(path)

    print("Raw:", len(data))

    valid = [
        x for x in data
        if is_valid(x)
    ]

    print("After validation:", len(valid))

    unique = deduplicate(valid)

    print("After dedup:", len(unique))

    if len(unique) < target:
        raise ValueError(
            f"{source_name}: "
            f"only {len(unique)} valid samples, "
            f"but target is {target}"
        )

    sampled = stratified_sample(
        unique,
        target,
        rng
    )

    result = []

    for item in sampled:

        result.append({
            "instruction": normalize(
                item.get("instruction")
            ),
            "input": normalize(
                item.get("input")
            ),
            "output": normalize(
                item.get("output")
            ),
            "source": source_name,
            "source_file": Path(path).name,
        })

    print("Sampled:", len(result))

    length_distribution = Counter(
        get_length_bin(x)
        for x in result
    )

    print(
        "Length distribution:",
        dict(length_distribution)
    )

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample 20k TCM SFT records from selected raw source files."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    all_samples = []

    for config in CONFIG:
        samples = process_source(config, rng)
        all_samples.extend(samples)

    assert len(all_samples) == 20000

    rng.shuffle(all_samples)

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_samples,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n====================")
    print("Final samples:", len(all_samples))

    print(
        "Sources:",
        Counter(
            x["source"]
            for x in all_samples
        )
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
