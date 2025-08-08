#!/bin/bash

# RAG-Anything Inference UI Startup Script
# This script ensures proper environment setup and runs the Gradio UI

echo "=========================================="
echo "Starting RAG-Anything Inference UI"
echo "=========================================="

# Change to the script directory
cd "$(dirname "$0")"

# Set SSL environment variables
export SSL_CERT_FILE=./.venv/lib/python3.12/site-packages/certifi/cacert.pem
export REQUESTS_CA_BUNDLE=./.venv/lib/python3.12/site-packages/certifi/cacert.pem

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e ..
else
    echo "✓ Virtual environment found"
    source .venv/bin/activate
fi

# Check required environment variables
echo ""
echo "Checking environment variables..."
if [ -f ".env" ]; then
    echo "✓ .env file found"
    set -a
    source .env
    set +a
else
    echo "❌ .env file not found! Please create one."
    exit 1
fi

# Kill any existing Gradio UI process
echo ""
echo "Checking for existing processes..."
if pgrep -f "run_inference.py" > /dev/null; then
    echo "Found existing Inference UI process. Killing it..."
    pkill -f "run_inference.py"
    sleep 2
fi

# Start the Gradio UI
echo ""
echo "Starting Inference UI..."
echo "=========================================="
echo "Access the UI at: http://localhost:7861"
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Run with real-time output
python3 run_inference.py
