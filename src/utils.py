import time
import uuid
import re
from typing import Tuple


def validate_input_string(input_str, max_length=100) -> Tuple[bool, str]:
    """Validate and sanitize input strings to prevent malicious content"""
    if not input_str or not isinstance(input_str, str):
        return False, "Invalid input type"

    # Check length
    if len(input_str) > max_length:
        return False, f"Input too long (max {max_length} characters)"

    # Check for suspicious patterns (URLs, scripts, etc.)
    suspicious_patterns = [
        r"https?://",  # URLs
        r"ftp://",  # FTP URLs
        r"<script",  # Script tags
        r"javascript:",  # JavaScript URLs
        r"data:",  # Data URLs
        r"vbscript:",  # VBScript URLs
    ]

    input_lower = input_str.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, input_lower):
            return False, "Input contains suspicious content"

    return True, "Valid"


def calculate_confidence(similarity: float, threshold: float) -> str:
    """Calculate confidence level based on similarity score and threshold"""
    confidence_score = abs(similarity - threshold)
    if confidence_score > 0.3:
        return "High"
    elif confidence_score > 0.15:
        return "Medium"
    else:
        return "Low"


def process_name_pair(english_name: str, thai_name: str, model_service) -> dict:
    """Process a single name pair and return prediction results"""
    start_time = time.time()

    # Validate input strings
    is_valid_en, msg_en = validate_input_string(english_name)
    if not is_valid_en:
        raise ValueError(f"Invalid english_name: {msg_en}")

    is_valid_th, msg_th = validate_input_string(thai_name)
    if not is_valid_th:
        raise ValueError(f"Invalid thai_name: {msg_th}")

    # Preprocess names
    english_clean = model_service.normalize_english_name(english_name)
    thai_clean = model_service.normalize_thai_name(thai_name)

    # Predict
    similarity = model_service.predict_similarity(english_clean, thai_clean)
    is_match = model_service.is_match(english_clean, thai_clean)
    confidence = calculate_confidence(similarity, model_service.get_threshold())

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    return {
        "english_name_normalized": english_clean,
        "thai_name_normalized": thai_clean,
        "similarity_score": float(similarity),
        "is_match": bool(is_match),
        "confidence": confidence,
        "processing_time_ms": processing_time,
    }


def generate_batch_id() -> str:
    """Generate a unique batch ID"""
    return str(uuid.uuid4())
