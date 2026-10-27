"""Environment settings, deliberately separate from serializable benchmark evidence."""

from typing import Literal

from pydantic import AnyHttpUrl, Field, PostgresDsn, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from inference_lab.config.schemas import Endpoint, Text


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", hide_input_in_errors=True, validate_default=True
    )

    app_env: Literal["local", "test", "kaggle", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    host: Text = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    vllm_base_url: Endpoint = AnyHttpUrl("http://127.0.0.1:8001")
    vllm_model: Text = "Qwen/Qwen2.5-3B-Instruct"
    vllm_timeout_seconds: float = Field(default=180.0, gt=0, allow_inf_nan=False)
    vllm_stream_idle_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    database_url: SecretStr | None = Field(default=None, exclude=True, repr=False)
    prometheus_enabled: bool = False
    store_prompt_content: bool = False

    @field_validator("database_url")
    @classmethod
    def database_scheme(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None:
            try:
                PostgresDsn(value.get_secret_value())
            except ValidationError:
                raise ValueError("DATABASE_URL must be a valid PostgreSQL DSN") from None
        return value
