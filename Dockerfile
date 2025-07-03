# Use Python 3.9 slim image for better performance
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
COPY . .
RUN pip install --no-cache-dir -r api/requirements.txt


# Copy the trained model (if it exists)
COPY ./trained_matching_model ./trained_matching_model

# Create necessary directories
RUN mkdir -p /app/logs
RUN mkdir -p /app/data

# Set proper permissions
RUN chmod +x api/start_server.sh

# Expose port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:4000/health || exit 1

# Start the application
CMD ["gunicorn", "--config", "api/gunicorn.conf.py", "wsgi:app"]