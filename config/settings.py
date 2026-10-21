"""
Configuration and settings module for Clinical Copilot.
Handles environment variables, storage paths, model configs, and logging setup.
"""
from pathlib import Path
import os
from pydantic import BaseModel, Field, ConfigDict

# Base Directory of Project
BASE_DIR = Path(__file__).resolve().parent.parent

class AppSettings(BaseModel):
    app_name: str = "Clinical Copilot"
    app_version: str = "1.0.0-rc"
    environment: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")

    # Data paths
    data_dir: Path = BASE_DIR / "data"
    datasets_dir: Path = BASE_DIR / "data" / "datasets"
    guidelines_dir: Path = BASE_DIR / "data" / "guidelines"
    sample_images_dir: Path = BASE_DIR / "data" / "sample_images"
    vector_store_dir: Path = BASE_DIR / "data" / "vector_store"
    models_dir: Path = BASE_DIR / "data" / "models"

    # Device configuration
    device: str = os.getenv("DEVICE", "auto")

    # Embedding and LLM configuration
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # Medical Imaging configuration
    imaging_model_name: str = "densenet121-cxr"
    supported_image_types: list[str] = ["png", "jpg", "jpeg", "webp"]
    max_image_size_mb: int = 15

    # Risk model configuration
    risk_dataset_name: str = "heart_failure_prediction_kaggle"
    risk_outcome: str = "Cardiovascular / Heart Disease Risk"

    model_config = ConfigDict(arbitrary_types_allowed=True)

# Ensure all directories exist
settings = AppSettings()
for folder in [
    settings.data_dir,
    settings.datasets_dir,
    settings.guidelines_dir,
    settings.sample_images_dir,
    settings.vector_store_dir,
    settings.models_dir,
]:
    folder.mkdir(parents=True, exist_ok=True)
