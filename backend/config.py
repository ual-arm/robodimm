"""
Configuration module for Robodimm.
Loads environment variables and provides configuration settings.
"""
import os
from pathlib import Path
from typing import List

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Try to load .env.local first, then .env
    env_file = Path(__file__).parent.parent / ".env.local"
    if not env_file.exists():
        env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # python-dotenv not installed, use os.environ directly
    pass


class Config:
    """Application configuration loaded from environment variables."""
    
    # Environment type: "local" or "server"
    ENV: str = os.getenv("ENV", "local")
    
    # API base URL for frontend
    API_BASE_URL: str = os.getenv("API_BASE_URL", "")
    
    # Backend host for uvicorn
    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    
    # Backend port
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    
    # CORS origins (comma-separated string -> list)
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    
    @classmethod
    def is_local(cls) -> bool:
        """Check if running in local environment."""
        return cls.ENV == "local"
    
    @classmethod
    def is_server(cls) -> bool:
        """Check if running in server environment."""
        return cls.ENV == "server"
    
    @classmethod
    def get_api_base_url(cls) -> str:
        """Get the API base URL for frontend requests."""
        return cls.API_BASE_URL
    
    @classmethod
    def display_config(cls) -> None:
        """Display current configuration for debugging."""
        print(f"[CONFIG] Environment: {cls.ENV}")
        print(f"[CONFIG] API Base URL: '{cls.API_BASE_URL}'")
        print(f"[CONFIG] Backend Host: {cls.BACKEND_HOST}")
        print(f"[CONFIG] Backend Port: {cls.BACKEND_PORT}")
        print(f"[CONFIG] CORS Origins: {cls.CORS_ORIGINS}")


# Display configuration on import
Config.display_config()
