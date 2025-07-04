#!/bin/bash

# Kill any existing processes on port 3000
echo "Stopping any existing processes on port 3000..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Wait a moment
sleep 2

# Start the server with simple configuration (like the working command)
echo "Starting Gunicorn server..."
cd "$(dirname "$0")"
gunicorn --bind 0.0.0.0:4000 wsgi:app

# Run multi process
# gunicorn --bind 0.0.0.0:3000 --workers=4 --threads=1 wsgi:app