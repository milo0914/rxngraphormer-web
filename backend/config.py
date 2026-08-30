"""Backend configuration via environment variables."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_url: str = "http://localhost:8000"
    
    # Model paths
    forward_model_dir: str = "models/seq-v2-USPTO_STEREO-20250509_070206_ft"
    retro_model_dir: str = "models/USPTO_50k"
    forward_vocab_path: str = "dataset/USPTO_STEREO/vocab_smiles.txt"
    retro_vocab_path: str = "dataset/USPTO_50k/vocab_smiles.txt"
    
    # Inference
    device: str = "cpu"
    beam_size: int = 20
    n_best: int = 10
    top_k_default: int = 5
    
    # Startup
    download_models_on_start: bool = True
    github_releases_repo: str = "milo0914/rxngraphormer-web"
    github_token: Optional[str] = None
    
    # CORS
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "https://milo0914.github.io",
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()