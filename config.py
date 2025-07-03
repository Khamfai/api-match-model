import os


class Config:
    """Base configuration"""

    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"
    MODEL_PATH = os.environ.get("MODEL_PATH") or "../trained_matching_model"
    DB_PATH = os.environ.get("DB_PATH") or "../data"
    DEBUG = False
    TESTING = False


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
