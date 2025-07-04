# Use Python 3.9 slim image for better performance
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    git \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MPLCONFIGDIR=/tmp/matplotlib

# Create matplotlib config directory
RUN mkdir -p /tmp/matplotlib

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gdown

 #URL Download and Directory inside the container where the model will be saved
ENV MODEL_NAME="en_th_matching_model"
ARG MODEL_DRIVE_ID="1vsv-suGgyCTfEnIZhtg-10poeT72xKd2"
# ENV MODEL_DOWNLOAD_URL=""
ARG MODEL_SAVE_DIR="/app/models"

# Create the directory for the model
RUN mkdir -p ${MODEL_SAVE_DIR}

# Download the model during the Docker build process using CURL
# RUN echo "Downloading model from ${MODEL_DOWNLOAD_URL}..." && \
#     curl -L "${MODEL_DOWNLOAD_URL}" -o "${MODEL_SAVE_DIR}/${MODEL_NAME}.zip" && \
#     echo "Model downloaded."

# Download the model using gdown
RUN echo "Downloading model with gdown (ID: ${MODEL_DRIVE_ID})..." && \
    gdown --id "${MODEL_DRIVE_ID}" --output "${MODEL_SAVE_DIR}/${MODEL_NAME}.zip" --no-cookies && \
    echo "Model downloaded."

# Unzip the model (assuming it's a .zip file, adjust if it's .tar.gz or a directory)
RUN unzip "${MODEL_SAVE_DIR}/${MODEL_NAME}.zip" -d "${MODEL_SAVE_DIR}/${MODEL_NAME}" && \
    rm "${MODEL_SAVE_DIR}/${MODEL_NAME}.zip" && \
    rm -rf /var/lib/apt/lists/* # Clean up apt cache

    # Update your Flask app's model path to point to the downloaded model
ENV MODEL_PATH=${MODEL_SAVE_DIR}/${MODEL_NAME}

# Copy all files
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs
RUN mkdir -p /app/data

# Set proper permissions
# RUN chmod +x api/start_server.sh

# Expose port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:4000/health || exit 1

# Start the application
CMD ["gunicorn", "--config", "src/gunicorn.conf.py", "wsgi:app"]