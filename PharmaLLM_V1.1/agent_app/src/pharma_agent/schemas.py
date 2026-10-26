from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TEAM_CODE_PATTERN = r"^[a-z0-9][a-z0-9-]{2,31}$"
TEAM_CODE_HELP = "团队登录代号需为 3–32 位英文字母、数字或连字符，可用纯数字，首位不能是连字符"
PASSWORD_MIN_LENGTH = 6
PASSWORD_MAX_LENGTH = 12
LEGACY_LOGIN_PASSWORD_MAX_LENGTH = 256


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LoginInput(StrictModel):
    tenant_id: str = Field(min_length=1, max_length=64)
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    # Authenticate existing longer passwords without allowing them for new accounts.
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=LEGACY_LOGIN_PASSWORD_MAX_LENGTH)


class RegisterInput(StrictModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH)
    tenant_name: str = Field(default="演示团队", min_length=1, max_length=120)
    tenant_code: str | None = Field(None, pattern=TEAM_CODE_PATTERN)

    @field_validator("tenant_code", mode="before")
    @classmethod
    def lower_code(cls, value):
        return value.strip().lower() if isinstance(value, str) else value


class ConversationInput(StrictModel):
    title: str = Field(default="新会话", min_length=1, max_length=120)


class RunInput(StrictModel):
    message: str = Field(min_length=1, max_length=4000)
    stream: bool = True
    model_alias: str | None = Field(None, pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    segmentation_model_id: str | None = Field(None, pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    asset_ids: list[str] = Field(default_factory=list, max_length=8)
    execution_mode: Literal["ask_for_approval", "full_access"] | None = None

    @field_validator("asset_ids")
    @classmethod
    def valid_assets(cls, value):
        import re

        if any(not re.fullmatch(r"asset_[a-f0-9]{32}", item) for item in value):
            raise ValueError("非法附件标识")
        return value


class ExecutionModeInput(StrictModel):
    execution_mode: Literal["ask_for_approval", "full_access"]


class TeamInput(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    login_code: str = Field(pattern=TEAM_CODE_PATTERN)

    @field_validator("login_code", mode="before")
    @classmethod
    def lower_code(cls, value):
        return value.strip().lower() if isinstance(value, str) else value


class DepartmentInput(StrictModel):
    name: str = Field(min_length=1, max_length=80)


class MemberInput(StrictModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH)
    display_name: str = Field(min_length=1, max_length=80)
    role: Literal["doctor", "reviewer"] = "doctor"
    department_id: str = Field(min_length=1, max_length=64)


class MemberUpdate(StrictModel):
    display_name: str = Field(min_length=1, max_length=80)
    role: Literal["owner", "doctor", "reviewer"]
    department_id: str | None = Field(None, max_length=64)
    status: Literal["active", "disabled"] = "active"


class ReviewDecision(StrictModel):
    decision: Literal["approved", "changes_requested"]
    comment: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def need_reason(self):
        if self.decision == "changes_requested" and not self.comment:
            raise ValueError("退回需填写复核意见")
        return self


class MemoryInput(StrictModel):
    kind: Literal["language_preference", "answer_format", "citation_style", "ui_preference"]
    value: str
    consent: Literal[True]
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def validate_value(self):
        allowed = {
            "language_preference": {"zh", "en"},
            "answer_format": {"short", "detailed", "bullets"},
            "citation_style": {"numbered", "source"},
            "ui_preference": {"compact", "comfortable"},
        }
        if self.value not in allowed[self.kind]:
            raise ValueError("偏好值不在允许的枚举范围中")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("expires_at 必须包含时区")
        return self


class DraftInput(StrictModel):
    title: str = Field(min_length=1, max_length=120)


class ConfirmInput(StrictModel):
    confirmed: Literal[True]


class DocumentInput(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    source: str = Field(min_length=1, max_length=500)
    source_version: str = Field(min_length=1, max_length=100)
    license: str = Field(min_length=1, max_length=200)
    published_at: str = Field(min_length=1, max_length=40)
    content: str = Field(min_length=1, max_length=200000)


class FeedbackInput(StrictModel):
    run_id: str = Field(max_length=64)
    rating: Literal[-1, 1]
    category: Literal["helpful", "incorrect", "citation", "tool", "other"] = "other"


class RouteDecision(StrictModel):
    action: Literal["respond", "retrieve", "tool", "clarify"]
    tool_name: (
        Literal[
            "get_my_job",
            "search_medical_reference",
            "get_user_preference",
            "create_task_draft",
            "segment_medical_volume",
        ]
        | None
    ) = None
    arguments: dict = Field(default_factory=dict)
    reason_code: Literal[
        "GENERAL", "REFERENCE", "JOB_STATUS", "PREFERENCE", "DRAFT", "SEGMENTATION", "UNCLEAR"
    ] = "GENERAL"
    continue_: bool = Field(default=False, alias="continue")

    @model_validator(mode="after")
    def validate_action(self):
        if self.action in {"respond", "clarify"} and (self.tool_name or self.arguments):
            raise ValueError("respond/clarify 不允许携带工具参数")
        if self.action == "tool" and not self.tool_name:
            raise ValueError("tool 必须声明白名单工具")
        if self.action == "retrieve" and self.tool_name not in {None, "search_medical_reference"}:
            raise ValueError("retrieve 只能调用检索工具")
        return self


class JobQuery(StrictModel):
    job_id: str | None = Field(None, min_length=1, max_length=64, pattern=r"^job_[a-f0-9]{32}$")


class SearchInput(StrictModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(3, ge=1, le=5)


class EmptyInput(StrictModel):
    pass


class Citation(StrictModel):
    document_id: str
    chunk_id: str
    title: str
    source: str
    source_version: str
    license: str
    published_at: str
    index_version: str
    excerpt: str
    score: float


class ReferencesResult(StrictModel):
    references: list[Citation]
    retrieval_mode: str


class JobSummary(StrictModel):
    id: str = Field(pattern=r"^job_[a-f0-9]{32}$")
    status: Literal["draft", "queued", "running", "succeeded", "failed", "cancelled"]
    tool_name: str
    tool_version: str
    result: dict
    error_code: str | None
    created_at: str
    input: dict = Field(default_factory=dict)
    elapsed_seconds: int = 0


class JobsResult(StrictModel):
    jobs: list[JobSummary]


class PreferencesResult(StrictModel):
    preferences: dict[str, str]


class DraftResult(StrictModel):
    job_id: str
    status: Literal["draft", "queued", "running", "succeeded", "failed", "cancelled"] = "draft"
    requires_confirmation: bool = True


class SegmentationInput(StrictModel):
    file_id: str = Field(pattern=r"^asset_[a-f0-9]{32}$")
    task_type: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=80)


class SegmentationResult(DraftResult):
    model_id: str
    result_asset_id: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    error_code: str | None = None


class DemoChatInput(StrictModel):
    message: str = Field(min_length=1, max_length=1000)

    @field_validator("message")
    @classmethod
    def reject_empty(cls, value):
        if not value.strip():
            raise ValueError("消息不能为空")
        return value
