#!/usr/bin/env python3
"""Create local configuration only; does not install or start anything."""

import argparse
import os
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["openai", "fake"], default="openai")
args = parser.parse_args()
target = root / ".env"
if target.exists():
    raise SystemExit(".env 已存在，未覆盖。请手动修改 MODEL_MODE 等配置。")
text = (root / ".env.example").read_text(encoding="utf-8")
password = secrets.token_hex(20)
text = text.replace("POSTGRES_PASSWORD=CHANGE_ME", "POSTGRES_PASSWORD=" + password)
text = text.replace(":CHANGE_ME@", ":" + password + "@")
text = text.replace("JWT_SECRET=CHANGE_ME", "JWT_SECRET=" + secrets.token_hex(32))
text = text.replace("SEGMENTATION_API_KEY=CHANGE_ME", "SEGMENTATION_API_KEY=" + secrets.token_hex(32))
text = text.replace("MODEL_MODE=openai", "MODEL_MODE=" + args.mode)
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    handle.write(text)
print(f"已创建 {target}（权限 0600）；模式：{args.mode}。没有启动服务。")
