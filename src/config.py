"""Centralized Configuration for Contract Risk Analyzer."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    TEST_CONTRACTS_DIR: Path = DATA_DIR / "test_contracts"
    MODELS_DIR: Path = BASE_DIR / "models"
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma_db"
    
    # Model Artifacts
    CLASSIFIER_PATH: Path = MODELS_DIR / "classifier.joblib"
    LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.joblib"

    # Embedding & Reranker
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Server / App
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # LLM Settings (Optional API Key for generation)
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
