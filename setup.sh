#!/bin/bash

echo "Starting MediaPipe Project Environment Setup..."

# 1. Check for correct Python version range (3.8 to 3.11)
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if [ "$PYTHON_VERSION" != "3.8" ] && [ "$PYTHON_VERSION" != "3.9" ] && [ "$PYTHON_VERSION" != "3.10" ] && [ "$PYTHON_VERSION" != "3.11" ]; then
    echo "Error: MediaPipe 0.10.14 requires Python 3.8, 3.9, 3.10, or 3.11."
    echo "Current version is Python $PYTHON_VERSION."
    echo "Please install Python 3.11 (e.g., 'brew install python@3.11') and try again."
    exit 1
fi

# 2. Create the virtual environment if it doesn't exist
if [ ! -d "mp_env" ]; then
    echo "📦 Creating virtual environment 'mp_env' using Python $PYTHON_VERSION..."
    python3 -m venv mp_env
else
    echo "Virtual environment 'mp_env' already exists."
fi

# 3. Activate environment and install locked dependencies
echo "Activating environment and installing requirements..."
source mp_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "===================================================="
echo "   Setup complete! To start your environment, run:"
echo "   source mp_env/bin/activate"
echo "   python main.py"
echo "===================================================="
