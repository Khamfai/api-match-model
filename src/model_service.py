import os
import logging
import numpy as np
import pickle
import re
import unicodedata
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self):
        self.model = None
        self.threshold = 0.7
        self.thai_pattern = re.compile(r"[\u0E00-\u0E7F]+")
        self.english_pattern = re.compile(r"[a-zA-Z]+")

    def load_model(self, model_path: str) -> bool:
        """Load the trained model using SentenceTransformer directly"""
        try:
            if os.path.exists(model_path):
                # Load the SentenceTransformer model directly
                self.model = SentenceTransformer(model_path)

                # Load threshold and other parameters if config exists
                try:
                    with open(f"{model_path}/config.pkl", "rb") as f:
                        config = pickle.load(f)
                        self.threshold = config.get("threshold", 0.7)
                except FileNotFoundError:
                    logger.warning("Config file not found, using default threshold")
                    self.threshold = 0.7

                logger.info("Model loaded successfully")
                return True
            else:
                logger.error(f"Model not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def get_threshold(self) -> float:
        """Get current model threshold"""
        if self.model is None:
            raise ValueError("Model not loaded")
        return self.threshold

    def set_threshold(self, new_threshold: float) -> float:
        """Set new model threshold and return old threshold"""
        if self.model is None:
            raise ValueError("Model not loaded")
        old_threshold = self.threshold
        self.threshold = float(new_threshold)
        return old_threshold

    def normalize_english_name(self, name: str) -> str:
        """Normalize English names - implemented locally"""
        if not name:
            return ""

        # Convert to lowercase and remove extra whitespace
        name = re.sub(r"\s+", " ", name.strip().lower())

        # Remove special characters except spaces and hyphens
        name = re.sub(r"[^a-zA-Z\s\-]", "", name)

        return name

    def normalize_thai_name(self, name: str) -> str:
        """Normalize Thai names - implemented locally"""
        if not name:
            return ""

        # Remove extra whitespace
        name = re.sub(r"\s+", " ", name.strip())

        # Normalize Unicode characters
        name = unicodedata.normalize("NFC", name)

        return name

    def predict_similarity(self, english_name: str, thai_name: str) -> float:
        """Predict similarity between names using SentenceTransformer"""
        if self.model is None:
            raise ValueError("Model not loaded")

        try:
            # Encode both names to get embeddings
            embeddings = self.model.encode([english_name, thai_name])

            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )

            return float(similarity)
        except Exception as e:
            logger.error(f"Error predicting similarity: {str(e)}")
            return 0.0

    def is_match(self, english_name: str, thai_name: str) -> bool:
        """Determine if names match based on similarity threshold"""
        if self.model is None:
            raise ValueError("Model not loaded")

        similarity = self.predict_similarity(english_name, thai_name)
        return similarity >= self.threshold


class RemoteModelService:
    """Service for handling remote model API calls when model is deployed separately"""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.threshold = 0.7
        self.thai_pattern = re.compile(r"[\u0E00-\u0E7F]+")
        self.english_pattern = re.compile(r"[a-zA-Z]+")

    def is_model_loaded(self) -> bool:
        """Always return True for remote service"""
        return True

    def get_threshold(self) -> float:
        """Get current threshold"""
        return self.threshold

    def set_threshold(self, new_threshold: float) -> float:
        """Set new threshold"""
        old_threshold = self.threshold
        self.threshold = float(new_threshold)
        return old_threshold

    def normalize_english_name(self, name: str) -> str:
        """Normalize English names - implemented locally"""
        if not name:
            return ""

        # Convert to lowercase and remove extra whitespace
        name = re.sub(r"\s+", " ", name.strip().lower())

        # Remove special characters except spaces and hyphens
        name = re.sub(r"[^a-zA-Z\s\-]", "", name)

        return name

    def normalize_thai_name(self, name: str) -> str:
        """Normalize Thai names - implemented locally"""
        if not name:
            return ""

        # Remove extra whitespace
        name = re.sub(r"\s+", " ", name.strip())

        # Normalize Unicode characters
        name = unicodedata.normalize("NFC", name)

        return name

    def predict_similarity(self, english_name: str, thai_name: str) -> float:
        """Predict similarity via remote API call"""
        try:
            import requests

            response = requests.post(
                f"{self.api_url}/predict",
                json={"english_name": english_name, "thai_name": thai_name},
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return float(result.get("similarity", 0.0))
            else:
                logger.error(f"Remote API error: {response.status_code}")
                return 0.0

        except Exception as e:
            logger.error(f"Error calling remote model API: {str(e)}")
            return 0.0

    def is_match(self, english_name: str, thai_name: str) -> bool:
        """Determine if names match based on similarity threshold"""
        similarity = self.predict_similarity(english_name, thai_name)
        return similarity >= self.threshold
