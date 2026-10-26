import hashlib
from pathlib import Path

from pharma_agent.errors import AppError


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def path(self, key: str):
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-." for c in key):
            raise AppError("INVALID_ASSET", "非法资产标识", 400)
        path = (self.root / key).resolve()
        if path.parent != self.root:
            raise AppError("INVALID_ASSET", "非法资产路径", 400)
        return path

    def write(self, key: str, content: bytes):
        path = self.path(key)
        # Only server-generated keys reach this method; no user filename or URL is accepted.
        with path.open("xb") as handle:
            path.chmod(0o600)
            handle.write(content)
        return "sha256:" + hashlib.sha256(content).hexdigest()

    def remove(self, key):
        self.path(key).unlink(missing_ok=True)


def detect_upload(content: bytes, declared: str):
    if declared == "image/png" and content.startswith(b"\x89PNG\r\n\x1a\n"):
        # Stored as an attachment; never decoded/rendered on the server.
        return "image/png"
    if declared in {"text/plain", "application/json"}:
        try:
            content.decode("utf-8")
            if declared == "application/json":
                import json

                json.loads(content)
            return declared
        except (UnicodeError, ValueError):
            pass
    raise AppError("UNSUPPORTED_FILE", "仅支持 UTF-8 TXT/JSON 和 PNG；不接受压缩包或任意服务器路径", 415)
