#!/usr/bin/env python3
import os
import sys
from env_utils import load

load()
os.execv(
    sys.executable,
    [
        sys.executable,
        "-m",
        "uvicorn",
        "pharma_agent.main:create_app",
        "--factory",
        "--host",
        os.environ.get("APP_HOST", "127.0.0.1"),
        "--port",
        os.environ.get("APP_PORT", "8080"),
        "--no-access-log",
        "--workers",
        "1",
    ],
)
