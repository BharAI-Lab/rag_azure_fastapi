import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load `.env` only for local development
load_dotenv()

REQUIRED_VARS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_API_KEY",
    "AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT_NAME",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_INDEX_NAME",
]

def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}\n"
            f"Make sure it's set in your .env (local) or Azure Container App env/secret (prod)."
        )
    return value

@dataclass(frozen=True)
class Settings:
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_KEY: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str
    AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME: str

    # Azure AI Foundry
    AZURE_OPENAI_CHAT_DEPLOYMENT: str
    AZURE_OPENAI_CHAT_DEPLOYMENT_API_KEY: str
    AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT_NAME: str
    
    # Azure AI Search
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_API_KEY: str
    AZURE_SEARCH_INDEX_NAME: str

    # RAG knobs (with defaults)
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "800"))

    @staticmethod
    def load() -> "Settings":
        # Validate all required variables are present
        for var in REQUIRED_VARS:
            _require(var)

        return Settings(
            AZURE_OPENAI_ENDPOINT=_require("AZURE_OPENAI_ENDPOINT"),
            AZURE_OPENAI_KEY=_require("AZURE_OPENAI_KEY"),
            AZURE_OPENAI_API_VERSION=_require("AZURE_OPENAI_API_VERSION"),
            AZURE_OPENAI_CHAT_DEPLOYMENT=_require("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            AZURE_OPENAI_CHAT_DEPLOYMENT_API_KEY=_require("AZURE_OPENAI_CHAT_DEPLOYMENT_API_KEY"),
            AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT_NAME=_require("AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT_NAME"),
            AZURE_OPENAI_EMBEDDING_DEPLOYMENT=_require("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME=_require("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME"),
            AZURE_SEARCH_ENDPOINT=_require("AZURE_SEARCH_ENDPOINT"),
            AZURE_SEARCH_API_KEY=_require("AZURE_SEARCH_API_KEY"),
            AZURE_SEARCH_INDEX_NAME=_require("AZURE_SEARCH_INDEX_NAME"),
            TOP_K=int(os.getenv("TOP_K", "4")),
            TEMPERATURE=float(os.getenv("TEMPERATURE", "0.2")),
            MAX_TOKENS=int(os.getenv("MAX_TOKENS", "800")),
        )

settings = Settings.load()
