from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    neon_database_url: str | None = Field(default=None, alias="NEON_DATABASE_URL")
    ollama_host: str | None = Field(default=None, alias="OLLAMA_HOST")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    developer_bypass_code: str = Field(
        default="Cukur301r",
        alias="DEVELOPER_BYPASS_CODE",
    )


settings = Settings()
