import logging
import os
#from datetime import timedelta
from functools import lru_cache
from typing import Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""
    temperature: float = 0.0
    max_tokens: Optional[int] = 1024 #None
    max_retries: int = 3

class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")

class AnthropicSettings(LLMSettings):
    """Anthropic-specific settings extending LLMSettings."""
    api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    default_model: str = Field(default="claude-3-5-sonnet-20241022")


class OllamaSettings(LLMSettings):
    """Ollama-specific settings extending LLMSettings."""
    base_url: str = Field(default="http://0.0.0.0:11434/v1")
    api_key: str = Field(default="ollama")
    default_model: str = Field(default="llama3.2")




class Settings(BaseModel):
    """Main settings class combining all sub-settings."""
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings