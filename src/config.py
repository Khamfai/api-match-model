import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration"""

    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"
    MODEL_PATH = os.environ.get("MODEL_PATH") or "models/en_th_matching_model"
    DB_PATH = os.environ.get("DB_PATH") or "./data/name_matching.db"
    DEBUG = False
    TESTING = False

    # Additional configuration options
    USE_REMOTE_MODEL = os.environ.get("USE_REMOTE_MODEL", "false").lower() == "true"
    REMOTE_MODEL_API_URL = os.environ.get("REMOTE_MODEL_API_URL", "")
    MAX_CONTENT_LENGTH = int(
        os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)
    )  # 16MB default


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY")


class TestingConfig(Config):
    """Testing configuration"""

    TESTING = True
    DEBUG = True


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
