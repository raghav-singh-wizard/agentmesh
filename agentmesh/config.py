"""Centralised configuration loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for AgentMesh.

    Values are read from environment variables (prefix is optional per field).
    Defaults favour a local-dev setup (localhost Redis, ./data directories).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    planner_model: str = Field(default="claude-sonnet-4-5", alias="AGENTMESH_PLANNER_MODEL")
    critic_model: str = Field(default="claude-sonnet-4-5", alias="AGENTMESH_CRITIC_MODEL")
    max_tokens: int = Field(default=2048, alias="AGENTMESH_MAX_TOKENS")
    temperature: float = Field(default=0.2, alias="AGENTMESH_TEMPERATURE")

    # --- Memory ---
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    procedural_store_path: str = Field(
        default="./data/procedural.json", alias="PROCEDURAL_STORE_PATH"
    )

    # --- Runtime ---
    max_steps: int = Field(default=12, alias="AGENTMESH_MAX_STEPS")
    critic_enabled: bool = Field(default=True, alias="AGENTMESH_CRITIC_ENABLED")
    log_level: str = Field(default="INFO", alias="AGENTMESH_LOG_LEVEL")

    # --- API ---
    host: str = Field(default="0.0.0.0", alias="AGENTMESH_HOST")
    port: int = Field(default=8000, alias="AGENTMESH_PORT")

    def ensure_dirs(self) -> None:
        """Create persistence directories if they don't exist."""
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.procedural_store_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor."""
    s = Settings()
    s.ensure_dirs()
    return s
