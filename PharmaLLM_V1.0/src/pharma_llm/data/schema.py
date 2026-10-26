from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LEGAL_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class TrainingRecord:
    id: str
    source: str
    category: str
    messages: list[ChatMessage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "category": self.category,
            "messages": [message.to_dict() for message in self.messages],
        }


def build_training_record(
    *,
    record_id: str,
    source: str,
    category: str,
    user: str,
    assistant: str,
) -> dict[str, Any]:
    """Build the canonical two-turn SFT record."""

    return TrainingRecord(
        id=record_id,
        source=source,
        category=category,
        messages=[
            ChatMessage(role="user", content=user),
            ChatMessage(role="assistant", content=assistant),
        ],
    ).to_dict()


def validate_training_record(
    record: Any,
    *,
    legal_roles: set[str] | None = None,
    require_user_message: bool = True,
    require_assistant_message: bool = True,
) -> list[str]:
    """Return schema validation errors for a canonical training record."""

    roles = legal_roles or LEGAL_ROLES
    errors: list[str] = []

    if not isinstance(record, dict):
        return ["record must be a JSON object"]

    for field in ("id", "source", "category"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
        return errors

    has_user = False
    has_assistant = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"messages[{index}] must be an object")
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in roles:
            errors.append(f"messages[{index}].role must be one of {sorted(roles)}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"messages[{index}].content must be a non-empty string")

        has_user = has_user or role == "user"
        has_assistant = has_assistant or role == "assistant"

    if require_user_message and not has_user:
        errors.append("record must contain at least one user message")
    if require_assistant_message and not has_assistant:
        errors.append("record must contain at least one assistant message")

    return errors


def normalize_training_record(record: dict[str, Any]) -> dict[str, Any]:
    """Trim scalar fields and message content without changing semantics."""

    return {
        "id": str(record["id"]).strip(),
        "source": str(record["source"]).strip(),
        "category": str(record["category"]).strip(),
        "messages": [
            {
                "role": str(message["role"]).strip(),
                "content": str(message["content"]).strip(),
            }
            for message in record["messages"]
        ],
    }

