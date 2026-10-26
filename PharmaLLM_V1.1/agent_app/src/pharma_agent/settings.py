from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8080
    database_url: str = "postgresql+psycopg://pharma:CHANGE_ME@127.0.0.1:5433/pharma_agent"
    jwt_secret: SecretStr = SecretStr("CHANGE_ME")
    jwt_issuer: str = "pharma-agent"
    jwt_audience: str = "pharma-agent-ui"
    access_token_minutes: int = Field(120, ge=5, le=1440)
    allow_registration: bool = True
    model_mode: Literal["openai", "fake"] = "openai"
    pharma_vllm_base_url: str = "http://127.0.0.1:8000/v1"
    pharma_vllm_model_name: str = "pharmallm-v1"
    pharma_vllm_api_key: SecretStr = SecretStr("")
    model_revision: str = "unknown"
    model_timeout_seconds: float = Field(120, gt=0)
    router_model_alias: str = "pharma_llm_sft"
    router_base_url: str = "http://127.0.0.1:8000/v1"
    router_model_name: str = "pharmallm-v1"
    router_api_key: SecretStr = SecretStr("")
    model_config_path: Path = Field(Path("configs/models.yaml"), validation_alias="MODEL_CONFIG")
    tool_config: Path = Path("configs/tools.yaml")
    graph_config: Path = Path("configs/graph.yaml")
    max_active_runs: int = Field(8, ge=1, le=128)
    max_pending_runs_per_user: int = Field(4, ge=1, le=64)
    run_timeout_seconds: int = Field(240, ge=10)
    run_lease_seconds: int = Field(60, ge=30)
    history_message_limit: int = Field(20, ge=2, le=100)
    prompt_byte_budget: int = Field(12000, ge=2000, le=100000)
    answer_max_tokens: int = Field(768, ge=64, le=8192)
    enable_queue: bool = True
    redis_url: SecretStr = SecretStr("redis://127.0.0.1:6380/0")
    rate_limit_per_minute: int = Field(30, ge=0)
    rate_limit_fail_closed: bool = True
    storage_root: Path = Path("var/assets")
    max_upload_bytes: int = Field(64 * 1024 * 1024, ge=1024, le=1024**3)
    imaging_max_expanded_bytes: int = Field(512 * 1024 * 1024, ge=1024)
    imaging_max_voxels: int = Field(64_000_000, ge=1000)
    imaging_max_series_files: int = Field(1000, ge=2, le=1000)
    asset_retention_days: int = Field(30, ge=1, le=3650)
    nnunet_model_folder: Path = Path("../model/femoralhead-t1")
    nnunet_python: Path = Path("/path/to/nnunet-env/bin/python")
    segmentation_service_root: Path = Path("var/segmentation-service")
    segmentation_base_url: str = ""
    segmentation_api_key: SecretStr = SecretStr("")
    segmentation_models_config: Path = Path("configs/segmentation.yaml")
    segmentation_timeout_seconds: int = Field(1800, ge=30, le=7200)
    segmentation_load_mode: Literal["resident", "per_task"] = "resident"
    segmentation_startup_timeout_seconds: int = Field(300, ge=10, le=1800)
    embedding_provider: Literal["hash", "openai_compatible"] = "hash"
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_model_name: str = "your-embedding-model"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_dimensions: int = Field(256, ge=32, le=8192)
    embedding_revision: str = "demo-v1"
    enable_faults: bool = False
    otel_service_name: str = "pharma-agent-api"
    otel_exporter_otlp_endpoint: str = ""

    @model_validator(mode="after")
    def validate_security(self):
        secret = self.jwt_secret.get_secret_value()
        if secret == "CHANGE_ME" or len(secret.encode()) < 32:
            raise ValueError("JWT_SECRET 必须至少 32 字节，请先运行 scripts/init_env.py")
        if not self.database_url.startswith("postgresql+psycopg://"):
            raise ValueError("DATABASE_URL 需使用 postgresql+psycopg://")
        if self.app_env == "production" and (self.enable_faults or self.model_mode == "fake"):
            raise ValueError("生产模式禁止 FakeModel 和故障注入")
        return self

    @property
    def checkpoint_url(self) -> str:
        return self.database_url.replace("postgresql+psycopg://", "postgresql://", 1)

    def config_environment(self) -> dict[str, str]:
        return {
            "PHARMA_VLLM_BASE_URL": self.pharma_vllm_base_url,
            "PHARMA_VLLM_MODEL_NAME": self.pharma_vllm_model_name,
            "PHARMA_VLLM_API_KEY": self.pharma_vllm_api_key.get_secret_value(),
            "MODEL_REVISION": self.model_revision,
            "ROUTER_MODEL_ALIAS": self.router_model_alias,
            "ROUTER_BASE_URL": self.router_base_url,
            "ROUTER_MODEL_NAME": self.router_model_name,
            "ROUTER_API_KEY": self.router_api_key.get_secret_value(),
        }
