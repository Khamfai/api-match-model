#!/bin/bash

# Kill any existing processes on port 4000
echo "Stopping any existing processes on port 4000..."
lsof -ti:4000 | xargs kill -9 2>/dev/null || true

# Wait a moment
sleep 2

# Start the server with proper configuration
echo "Starting Gunicorn server..."
cd "$(dirname "$0")"
gunicorn --config gunicorn.conf.py wsgi:application