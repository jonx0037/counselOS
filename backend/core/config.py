from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""

    # Embeddings
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-2-preview"  # verified: ai.google.dev/gemini-api/docs/embeddings
    google_api_key: str = ""
    chroma_collection_name: str = "counsel_kb"

    # Server
    backend_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
