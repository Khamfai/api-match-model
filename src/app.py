from datetime import datetime, time
import os
import tempfile
import atexit
import gc

from src.config import Config
from src.utils import generate_batch_id, process_name_pair, validate_input_string
from src.model_service import ModelService

# Fix matplotlib and multiprocessing issues
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import logging
from flask_swagger_ui import get_swaggerui_blueprint
import re
from urllib.parse import quote, unquote
import pandas as pd

# Import our database module
from src.db import MatchingDB

# Add this after the existing imports
import signal
import sys
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
config = Config()
model_service = None
db = None
model_path = config.MODEL_PATH
db_path = config.DB_PATH


def validate_input_string(input_str, max_length=100):
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


def load_model():
    """Load the trained model using ModelService"""
    global model_service, db
    try:
        # Initialize ModelService
        model_service = ModelService()

        # Load the model
        if model_service.load_model(model_path):
            # Initialize database
            db = MatchingDB(db_path)
            logger.info("Model and database loaded successfully")
            return True
        else:
            logger.error(f"Model not found at {model_path}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False


# Load model immediately when the module is imported
try:
    if load_model():
        logger.info("✓ Model loaded successfully on startup")
    else:
        logger.warning("⚠ Warning: Model not loaded. Some endpoints may not work.")
except Exception as e:
    logger.error(f"Failed to load model on startup: {str(e)}")


# Static file serving for Swagger
@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files securely"""
    # Validate filename to prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid file path"}), 400

    # Only allow specific file types
    allowed_extensions = {".yaml", ".yml", ".json", ".css", ".js"}
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in allowed_extensions:
        return jsonify({"error": "File type not allowed"}), 400

    try:
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


@app.route("/", methods=["GET"])
def index_check():
    """Index check endpoint"""
    return jsonify(
        {
            "status": "200",
            "message": "Welcome to model macthing",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model_service is not None
            and model_service.is_model_loaded(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict_match():
    """Predict if English and Thai names match"""
    try:
        # Check if model is loaded
        if not model_service or not model_service.is_model_loaded():
            return (
                jsonify(
                    {
                        "error": "Model not loaded. Please ensure the model is trained and available."
                    }
                ),
                500,
            )

        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        english_name = data.get("english_name", "").strip()
        thai_name = data.get("thai_name", "").strip()

        if not english_name or not thai_name:
            return (
                jsonify({"error": "Both english_name and thai_name are required"}),
                400,
            )

        # Process the name pair using utility function
        try:
            result = process_name_pair(english_name, thai_name, model_service)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Save to database
        try:
            if db:
                db.save_prediction(
                    english_name=english_name,
                    thai_name=thai_name,
                    english_name_normalized=result["english_name_normalized"],
                    thai_name_normalized=result["thai_name_normalized"],
                    similarity_score=result["similarity_score"],
                    is_match=result["is_match"],
                    confidence=result["confidence"],
                    threshold_used=model_service.get_threshold(),
                    request_ip=request.remote_addr,
                    processing_time_ms=result["processing_time_ms"],
                )
        except Exception as db_error:
            logger.error(f"Database save error: {str(db_error)}")
            # Continue without failing the request

        return jsonify(
            {
                "english_name": english_name,
                "thai_name": thai_name,
                "english_name_normalized": result["english_name_normalized"],
                "thai_name_normalized": result["thai_name_normalized"],
                "similarity_score": round(result["similarity_score"], 4),
                "is_match": result["is_match"],
                "confidence": result["confidence"],
                "threshold": model_service.get_threshold(),
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": round(result["processing_time_ms"], 2),
            }
        )

    except Exception as e:
        logger.error(f"Error in predict_match: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Predict multiple name pairs at once"""
    start_time = datetime.now()
    batch_id = generate_batch_id()

    try:
        if not model_service or not model_service.is_model_loaded():
            return (
                jsonify(
                    {
                        "error": "Model not loaded. Please ensure the model is trained and available."
                    }
                ),
                500,
            )

        data = request.get_json()

        if not data or "name_pairs" not in data:
            return jsonify({"error": "name_pairs array is required"}), 400

        name_pairs = data["name_pairs"]

        if not isinstance(name_pairs, list):
            return jsonify({"error": "name_pairs must be an array"}), 400

        # Limit batch size to prevent abuse
        if len(name_pairs) > 100:
            return jsonify({"error": "Batch size limited to 100 pairs"}), 400

        results = []
        successful_predictions = 0
        failed_predictions = 0

        for i, pair in enumerate(name_pairs):
            try:
                english_name = pair.get("english_name", "").strip()
                thai_name = pair.get("thai_name", "").strip()

                if not english_name or not thai_name:
                    results.append(
                        {
                            "index": i,
                            "error": "Both english_name and thai_name are required",
                        }
                    )
                    failed_predictions += 1
                    continue

                # Process the name pair using utility function
                try:
                    result = process_name_pair(english_name, thai_name, model_service)
                except ValueError as e:
                    results.append({"index": i, "error": str(e)})
                    failed_predictions += 1
                    continue

                # Save individual prediction to database
                try:
                    if db:
                        db.save_prediction(
                            english_name=english_name,
                            thai_name=thai_name,
                            english_name_normalized=result["english_name_normalized"],
                            thai_name_normalized=result["thai_name_normalized"],
                            similarity_score=result["similarity_score"],
                            is_match=result["is_match"],
                            confidence=result["confidence"],
                            threshold_used=model_service.get_threshold(),
                            request_ip=request.remote_addr,
                        )
                except Exception as db_error:
                    logger.error(f"Database save error for pair {i}: {str(db_error)}")

                results.append(
                    {
                        "index": i,
                        "english_name": english_name,
                        "thai_name": thai_name,
                        "similarity_score": round(result["similarity_score"], 4),
                        "is_match": result["is_match"],
                        "confidence": result["confidence"],
                    }
                )
                successful_predictions += 1

            except Exception as e:
                results.append(
                    {"index": i, "error": f"Error processing pair: {str(e)}"}
                )
                failed_predictions += 1

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Save batch metadata to database
        try:
            if db:
                db.save_batch_prediction(
                    batch_id=batch_id,
                    total_pairs=len(name_pairs),
                    successful_predictions=successful_predictions,
                    failed_predictions=failed_predictions,
                    request_ip=request.remote_addr,
                    processing_time_ms=processing_time,
                )
        except Exception as db_error:
            logger.error(f"Database batch save error: {str(db_error)}")

        return jsonify(
            {
                "batch_id": batch_id,
                "results": results,
                "total_processed": len(results),
                "successful_predictions": successful_predictions,
                "failed_predictions": failed_predictions,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": round(processing_time, 2),
            }
        )

    except Exception as e:
        logger.error(f"Error in batch_predict: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# Add new database-related endpoints
@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    """Get stored predictions with optional filtering"""
    try:
        if not db:
            return jsonify({"error": "Database not available"}), 500

        # Get query parameters
        limit = min(int(request.args.get("limit", 100)), 1000)  # Max 1000
        offset = int(request.args.get("offset", 0))
        is_match = request.args.get("is_match")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        # Convert is_match to boolean if provided
        if is_match is not None:
            is_match = is_match.lower() in ["true", "1", "yes"]

        predictions = db.get_predictions(
            limit=limit,
            offset=offset,
            is_match=is_match,
            start_date=start_date,
            end_date=end_date,
        )

        return jsonify(
            {
                "predictions": predictions,
                "count": len(predictions),
                "limit": limit,
                "offset": offset,
            }
        )

    except Exception as e:
        logger.error(f"Error retrieving predictions: {str(e)}")
        return jsonify({"error": f"Error retrieving predictions: {str(e)}"}), 500


@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    """Get database statistics"""
    try:
        if not db:
            return jsonify({"error": "Database not available"}), 500

        stats = db.get_statistics()
        return jsonify(stats)

    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({"error": f"Error getting statistics: {str(e)}"}), 500


@app.route("/api/export", methods=["POST"])
def export_predictions():
    """Export predictions to CSV"""
    try:
        if not db:
            return jsonify({"error": "Database not available"}), 500

        data = request.get_json() or {}
        limit = data.get("limit")

        # Create export filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_export_{timestamp}.csv"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        success = db.export_predictions_csv(filepath, limit)

        if success:
            return jsonify(
                {
                    "message": "Export completed successfully",
                    "filename": filename,
                    "filepath": filepath,
                }
            )
        else:
            return jsonify({"error": "Export failed"}), 500

    except Exception as e:
        logger.error(f"Error exporting predictions: {str(e)}")
        return jsonify({"error": f"Error exporting predictions: {str(e)}"}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    """Get model information"""
    try:
        if not model_service or not model_service.is_model_loaded():
            return jsonify({"error": "Model not loaded"}), 500

        return jsonify(
            {
                "model_loaded": True,
                "threshold": model_service.get_threshold(),
                "model_path": model_path,
                "model_exists": os.path.exists(model_path),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error getting model info: {str(e)}"}), 500


@app.route("/model/threshold", methods=["POST"])
def update_threshold():
    """Update model threshold"""
    try:
        if not model_service or not model_service.is_model_loaded():
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        new_threshold = data.get("threshold")

        if new_threshold is None:
            return jsonify({"error": "threshold value is required"}), 400

        if not isinstance(new_threshold, (int, float)) or not (0 <= new_threshold <= 1):
            return jsonify({"error": "threshold must be a number between 0 and 1"}), 400

        old_threshold = model_service.set_threshold(new_threshold)

        return jsonify(
            {
                "message": "Threshold updated successfully",
                "old_threshold": old_threshold,
                "new_threshold": model_service.get_threshold(),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error updating threshold: {str(e)}"}), 500


# ... existing code ...
# Enhanced error handlers
@app.errorhandler(404)
def not_found(error):
    """Enhanced 404 error handler with helpful information"""
    # Get the requested path
    requested_path = request.path
    method = request.method

    # List of available endpoints
    available_endpoints = {
        "GET": ["/health", "/model/info", "/api/docs", "/static/swagger.yaml"],
        "POST": ["/predict", "/batch_predict", "/model/threshold"],
    }

    # Check if it's a method not allowed case
    all_paths = []
    for methods, paths in available_endpoints.items():
        all_paths.extend(paths)

    response_data = {
        "error": "Endpoint not found",
        "message": f'The requested endpoint "{method} {requested_path}" does not exist',
        "requested_path": requested_path,
        "requested_method": method,
        "timestamp": datetime.now().isoformat(),
        "available_endpoints": available_endpoints,
    }

    # Suggest similar endpoints if path is close to existing ones
    suggestions = []
    for path in all_paths:
        if (
            requested_path.lower() in path.lower()
            or path.lower() in requested_path.lower()
        ):
            suggestions.append(path)

    if suggestions:
        response_data["suggestions"] = suggestions

    # Add helpful hints for common mistakes
    hints = []
    if requested_path.endswith("/"):
        hints.append(f"Try without trailing slash: {requested_path.rstrip('/')}")
    elif not requested_path.endswith("/") and requested_path + "/" in all_paths:
        hints.append(f"Try with trailing slash: {requested_path}/")

    if requested_path.startswith("/api/") and requested_path != "/api/docs":
        hints.append("API documentation is available at /api/docs")

    if hints:
        response_data["hints"] = hints

    logger.warning(f"404 Error: {method} {requested_path} - {request.remote_addr}")

    return jsonify(response_data), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle method not allowed errors"""
    requested_path = request.path
    method = request.method

    # Get allowed methods for this path
    allowed_methods = []
    for rule in app.url_map.iter_rules():
        if rule.rule == requested_path:
            allowed_methods.extend(rule.methods)

    # Remove OPTIONS and HEAD methods from display
    allowed_methods = [m for m in allowed_methods if m not in ["OPTIONS", "HEAD"]]

    response_data = {
        "error": "Method not allowed",
        "message": f'Method "{method}" is not allowed for endpoint "{requested_path}"',
        "requested_path": requested_path,
        "requested_method": method,
        "allowed_methods": allowed_methods,
        "timestamp": datetime.now().isoformat(),
    }

    logger.warning(f"405 Error: {method} {requested_path} - {request.remote_addr}")

    return jsonify(response_data), 405


@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    response_data = {
        "error": "Bad request",
        "message": "The request could not be understood by the server",
        "timestamp": datetime.now().isoformat(),
    }

    # Add specific error message if available
    if hasattr(error, "description"):
        response_data["details"] = error.description

    logger.warning(
        f"400 Error: {request.method} {request.path} - {request.remote_addr}"
    )

    return jsonify(response_data), 400


@app.errorhandler(500)
def internal_error(error):
    """Enhanced 500 error handler"""
    response_data = {
        "error": "Internal server error",
        "message": "An unexpected error occurred on the server",
        "timestamp": datetime.now().isoformat(),
    }

    # In debug mode, include more details
    if app.debug and hasattr(error, "description"):
        response_data["details"] = error.description

    logger.error(
        f"500 Error: {request.method} {request.path} - {request.remote_addr} - {str(error)}"
    )

    return jsonify(response_data), 500


# Add request size limit
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max


# Add graceful shutdown handling
def signal_handler(sig, frame):
    logger.info("Gracefully shutting down...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Add request entity too large handler
@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def request_entity_too_large(error):
    return (
        jsonify(
            {
                "error": "Request entity too large",
                "message": "The uploaded data is too large. Maximum size is 16MB.",
                "timestamp": datetime.now().isoformat(),
            }
        ),
        413,
    )


# Add request timeout handler
@app.errorhandler(408)
def request_timeout(error):
    return (
        jsonify(
            {
                "error": "Request timeout",
                "message": "The request took too long to process.",
                "timestamp": datetime.now().isoformat(),
            }
        ),
        408,
    )


# Improve the existing error handlers
@app.before_request
def before_request():
    # Log incoming requests
    logger.info(
        f"Incoming request: {request.method} {request.path} from {request.remote_addr}"
    )

    # Add request validation
    if (
        request.content_length
        and request.content_length > app.config["MAX_CONTENT_LENGTH"]
    ):
        return (
            jsonify(
                {
                    "error": "Request entity too large",
                    "message": "Request size exceeds maximum allowed size",
                }
            ),
            413,
        )


@app.after_request
def after_request(response):
    # Log response status
    logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
    return response


# Add a catch-all route for better 404 handling
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    """Catch-all route to handle undefined endpoints with helpful response"""
    # Check if this is a request for the root path
    if path == "":
        return jsonify(
            {
                "message": "Welcome to English-Thai Name Matching API",
                "version": "1.0.0",
                "documentation": "/api/docs",
                "health_check": "/health",
                "endpoints": {
                    "predict": "POST /predict",
                    "batch_predict": "POST /batch_predict",
                    "model_info": "GET /model/info",
                    "update_threshold": "POST /model/threshold",
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

    # For all other undefined paths, trigger 404 handler
    from flask import abort

    abort(404)


# Swagger UI setup - Fixed URL handling
SWAGGER_URL = "/api/docs"
API_URL = "/static/swagger.yaml"

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        "app_name": "English-Thai Name Matching API",
        "validatorUrl": None,  # Disable validator to prevent external URL calls
    },
)

app.register_blueprint(swaggerui_blueprint)


def cleanup_resources():
    """Clean up resources on shutdown"""
    global model_service, db
    try:
        if model_service is not None:
            del model_service
        if db is not None:
            db.close()
            del db
        gc.collect()
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


# Register cleanup function
atexit.register(cleanup_resources)

if __name__ == "__main__":
    # Model is already loaded above, just run the app
    app.run(host="0.0.0.0", port=3000, debug=True)
