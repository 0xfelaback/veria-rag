from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    MINIO_BUCKET_NAME: str = Field(default=...)
    MINIO_SECRET_KEY: str = Field(default=...)
    MINIO_ACCESS_KEY: str = Field(default=...)
    MINIO_ENDPOINT: str = Field(default=...)
    ELASTIC_URL: str = Field(default=...)
    ELASTIC_USERNAME: str = Field(default=...)
    ELASTIC_PASSWORD: str = Field(default=...)
    # ELASTIC_TIMEOUT: int = Field(default=...)
    ELASTIC_IGNORE_SSL_ERRORS: str = Field(default=...)
    ES_INDEX_NAME: str = Field(default=...)
    ES_CLUSTER_NAME: str = Field(default=...)
    OLLAMA_BASE_URL: str = Field(default=...)
    OLLAMA_LLM_MODEL: str = Field(default=...)
    OLLAMA_EMBEDDING_MODEL: str = Field(default=...)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
