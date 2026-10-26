#!/usr/bin/env python3
"""Check a public snapshot or Git index without displaying matched secret contents."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
PRIVATE_DIRS = {".venv", "venv", "var", "backups", "checkpoints", "logs", "node_modules"}
PRIVATE_SUFFIXES = (
    ".pth", ".pt", ".ckpt", ".safetensors", ".bin", ".onnx", ".nii", ".nii.gz", ".dcm",
    ".sqlite", ".sqlite3", ".db", ".dump", ".sql", ".log", ".zip", ".tar", ".tar.gz",
    ".pem", ".key", ".p12", ".pfx", ".pyc",
)
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "api_token": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}"),
    "personal_home_path": re.compile(rb"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
}


def private_values(path):
    """Optional local comparison; never copy, serialize or print the environment file."""
    if path is None:
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#") and any(word in key for word in ("SECRET", "PASSWORD", "API_KEY", "TOKEN")):
            value = value.strip().strip("\"'")
            if len(value) >= 12 and value != "CHANGE_ME":
                values.append(value.encode())
    return values


def path_issues(name):
    path = Path(name)
    issues = []
    if path.is_absolute() or ".." in path.parts:
        issues.append("unsafe_path")
    if any(part in PRIVATE_DIRS or part.endswith(".egg-info") for part in path.parts):
        issues.append("private_directory")
    if path.name.startswith(".env") and path.name != ".env.example":
        issues.append("private_environment")
    if path.name in {"id_rsa", "id_ed25519", ".netrc", ".pypirc"}:
        issues.append("private_credential_file")
    if name.lower().endswith(PRIVATE_SUFFIXES):
        issues.append("private_or_generated_file")
    if name.startswith(("PharmaLLM_V1.0/data/", "PharmaLLM_V1.0/outputs/", "PharmaLLM_V1.0/reports/")):
        issues.append("private_training_output")
    if path.suffix == ".jsonl" and name != "PharmaLLM_V1.1/agent_app/tests/evaluation/cases.jsonl":
        issues.append("unreviewed_dataset")
    return issues


def content_issues(data, secrets):
    issues = [label for label, pattern in PATTERNS.items() if pattern.search(data)]
    if any(secret in data for secret in secrets):
        issues.append("matches_local_secret")
    return issues


def entries(root, staged):
    if staged:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--stage", "-z"],
            check=True, capture_output=True,
        )
        for row in result.stdout.split(b"\0"):
            if not row:
                continue
            meta, raw_name = row.split(b"\t", 1)
            mode, oid, stage = meta.split()
            name = raw_name.decode("utf-8")
            if mode not in {b"100644", b"100755"} or stage != b"0":
                yield name, None, "non_regular_git_entry"
                continue
            size = int(subprocess.check_output(["git", "-C", str(root), "cat-file", "-s", oid.decode()]))
            if size > MAX_FILE_BYTES:
                yield name, None, "file_too_large"
                continue
            data = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", oid.decode()])
            yield name, data, None
        return

    def walk(folder):
        for path in sorted(folder.iterdir()):
            name = path.relative_to(root).as_posix()
            if path.name in SKIP_DIRS:
                continue
            if path.is_symlink():
                yield name, None, "symlink_not_allowed"
            elif path.is_dir():
                if path.name in PRIVATE_DIRS:
                    yield name, None, "private_directory"
                else:
                    yield from walk(path)
            elif path.is_file():
                if path.stat().st_size > MAX_FILE_BYTES:
                    yield name, None, "file_too_large"
                else:
                    yield name, path.read_bytes(), None
    yield from walk(root)


def check(root, staged=False, secrets=()):
    findings, count, total = [], 0, 0
    for name, data, error in entries(root, staged):
        count += 1
        labels = path_issues(name)
        if error:
            labels.append(error)
        if data is not None:
            total += len(data)
            labels.extend(content_issues(data, secrets))
        if labels:
            findings.append({"path": name, "reasons": sorted(set(labels))})
    if total > MAX_TOTAL_BYTES:
        findings.append({"path": ".", "reasons": ["snapshot_too_large"]})
    return {"ok": not findings, "files": count, "bytes": total, "findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--staged", action="store_true", help="Read actual index blobs, not working copies")
    parser.add_argument("--private-env", type=Path, help="Optional local secret comparison; never printed")
    args = parser.parse_args()
    try:
        result = check(args.root.resolve(), args.staged, private_values(args.private_env))
    except (OSError, ValueError, subprocess.SubprocessError):
        print("无法完成检查：确认目录、配置文件或 Git 索引有效。", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
