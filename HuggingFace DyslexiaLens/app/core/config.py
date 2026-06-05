import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Security
    api_key: str = "KunciRahasia123!"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Model paths
    dyslexia_model_path: str = "models/dyslexialens_model.keras"
    ocr_model_path: str = "models/ocr_model_v4.keras"

    # Inference
    decision_threshold: float = 0.52

    # App meta
    app_title: str = "DyslexiaLens AI API"
    app_version: str = "1.0.0"
    api_v1_prefix: str = "/api/v1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
