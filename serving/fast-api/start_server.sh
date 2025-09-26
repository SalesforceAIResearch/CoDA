#!/bin/bash
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# API Server Startup Script

# Change to this script's directory so relative paths are correct
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting API Server..."
echo "=========================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8 or later."
    exit 1
fi

# Check if requirements are installed
python -c "import fastapi, torch, transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing requirements..."
    pip install -r ../requirements.txt
fi

# Set environment variables if not set
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-"8000"}
export MODEL_NAME=${MODEL_NAME:-"Salesforce/CoDA-v0-Instruct"}
# Pass HF token through environment if provided
export HF_TOKEN=${HF_TOKEN}
export DEVICE=${DEVICE:-"cuda"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Model: $MODEL_NAME"
echo "  Device: $DEVICE"
echo "  Log Level: $LOG_LEVEL"
echo ""

# Start the server
echo "Starting server..."
python main.py

