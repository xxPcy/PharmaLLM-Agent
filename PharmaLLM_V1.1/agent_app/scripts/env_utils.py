"""Small literal .env reader for launchers, usable inside a separate vLLM environment.

No shell evaluation or variable interpolation is performed. Command-line environment wins.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def values(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip().replace("_", "").isalnum():
            raise ValueError("只支持 KEY=VALUE 格式的配置")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key.strip()] = value
    return result


def load(profile=None):
    original = set(os.environ)
    path = ROOT / ".env"
    if not path.is_file() and not {"DATABASE_URL", "JWT_SECRET"}.issubset(os.environ):
        raise SystemExit("请先运行 python scripts/init_env.py")
    for key, value in (values(path) if path.is_file() else {}).items():
        os.environ.setdefault(key, value)
    if profile:
        path = ROOT / "configs" / "hardware" / profile
        for key, value in values(path).items():
            if key not in original:
                os.environ[key] = value
    os.chdir(ROOT)
