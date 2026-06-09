#!/bin/bash
# Robodimm PRO local backend startup script for Linux/macOS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Setting up Robodimm PRO Local Backend ==="

# 1. Detect conda or mamba
CONDA_EXE=""
if command -v mamba &> /dev/null; then
    CONDA_EXE="mamba"
elif command -v conda &> /dev/null; then
    CONDA_EXE="conda"
else
    echo "❌ Error: Neither conda nor mamba detected."
    echo "Please install Miniforge or Anaconda first to run the backend."
    echo "Visit: https://github.com/conda-forge/miniforge#miniforge3"
    exit 1
fi

echo "Using package manager: $CONDA_EXE"

# 2. Create environment if it doesn't exist
ENV_NAME="robodimm-pro-backend"
if ! $CONDA_EXE env list | grep -q "$ENV_NAME"; then
    echo "Creating Conda environment '$ENV_NAME' from environment.yml..."
    $CONDA_EXE env create -f "$PROJECT_ROOT/environment.yml"
else
    echo "Conda environment '$ENV_NAME' already exists."
fi

# 3. Launch backend using uvicorn inside environment
echo "Starting FastAPI backend server on 127.0.0.1:8001..."
$CONDA_EXE run -n "$ENV_NAME" python "$PROJECT_ROOT/backend/main.py"
