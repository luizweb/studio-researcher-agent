from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaEmbeddings

from config.settings import get_settings

class LLMProvider:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.model = self._initialize_model()

    def _initialize_model(self) -> BaseLanguageModel:
        model_initializers = {
            "openai": lambda s: ChatOpenAI(
                api_key=s.api_key,
                model=s.default_model,
                temperature=s.temperature,
                max_tokens=s.max_tokens
            ),
            "anthropic": lambda s: ChatAnthropic(
                api_key=s.api_key,
                model=s.default_model,
                temperature=s.temperature,
                max_tokens=s.max_tokens
            ),
            "ollama": lambda s: ChatOpenAI(
                base_url=s.base_url,
                api_key=s.api_key,
                model=s.default_model,
                temperature=s.temperature,
                max_tokens=s.max_tokens
            ),
            "ollama_embedding": lambda s: OllamaEmbeddings(
                model=s.embedding_model,
            )
        }
        
        initializer = model_initializers.get(self.provider)
        if initializer:
            return initializer(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def __getattr__(self, name):
        """Delegate attribute access to the model instance."""
        return getattr(self.model, name)



if __name__ == "__main__":
    llm = LLMProvider("openai")  # "openai""anthropic" or "ollama"
    response = llm.invoke("What is the capital of Brazil?")
    print(response)