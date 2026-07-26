"""Shared configuration for the Phase 0 exercises.

Values are loaded from the ``.env`` file and can be imported from any module in this package.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Provider credentials and default models used by the exercises."""

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="OPENAI_EMBEDDING_MODEL",
    )
    voyage_api_key: SecretStr | None = Field(default=None, validation_alias="VOYAGE_API_KEY")
    voyage_embedding_model: str = Field(
        default="voyage-3",
        validation_alias="VOYAGE_EMBEDDING_MODEL",
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "GOOGLE_MODEL"),
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        validation_alias=AliasChoices(
            "GEMINI_EMBEDDING_MODEL", "GOOGLE_EMBEDDING_MODEL"
        ),
    )

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def require_voyage_api_key(self) -> str:
            """Return the Voyage API key or raise a clear setup error."""
    
            if self.voyage_api_key is None:
                raise RuntimeError(
                    "Set VOYAGE_API_KEY in the repository .env file."
                )
            return self.voyage_api_key.get_secret_value()

    def require_gemini_api_key(self) -> str:
        """Return the Gemini API key or raise a clear setup error."""

        if self.gemini_api_key is None:
            raise RuntimeError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the repository .env file."
            )
        return self.gemini_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()
