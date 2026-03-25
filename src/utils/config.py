from __future__ import annotations

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Fireworks AI — LLM generation
    # ------------------------------------------------------------------
    fireworks_api_key: str = ""
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"

    # ------------------------------------------------------------------
    # Weaviate
    # ------------------------------------------------------------------
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str | None = None
    weaviate_collection: str = "MedicalChunk"

    # ------------------------------------------------------------------
    # Google AI Studio — multimodal embeddings
    # Get a key at https://aistudio.google.com/apikey
    # ------------------------------------------------------------------
    gemini_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-2-preview"
    # Flexible output dimension: 128–3072. Recommended: 768, 1536, 3072.
    # NOTE: changing this after data has been ingested requires dropping and
    # recreating the Weaviate collection (vector dimensions must be uniform).
    embedding_dimension: int = 3072

    # ------------------------------------------------------------------
    # Prompt versioning
    # ------------------------------------------------------------------
    prompt_version: str = "v1"

    # ------------------------------------------------------------------
    # Retrieval tuning
    # ------------------------------------------------------------------
    retrieval_top_k: int = 5
    hybrid_alpha: float = 0.5
    rerank_top_m: int = 20
    rerank_top_k: int = 5
    confidence_threshold: float = 0.5  # Threshold for raising low_confidence flag

    # ------------------------------------------------------------------
    # Fireworks AI — Reranking
    # ------------------------------------------------------------------
    fireworks_rerank_model: str = "accounts/fireworks/models/qwen3-reranker-8b"


    # ------------------------------------------------------------------
    # MinIO / S3 Storage
    # ------------------------------------------------------------------
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "medical-docs"
    minio_secure: bool = False

    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    app_env: str = "development"
    testing: bool = False

    @model_validator(mode="after")
    def _validate_required_for_production(self) -> Settings:
        """Fail fast with clear messages when critical vars are missing."""
        if self.testing:
            return self  # skip validation in test mode

        missing: list[str] = []
        if not self.fireworks_api_key:
            missing.append("FIREWORKS_API_KEY")
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the values."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
