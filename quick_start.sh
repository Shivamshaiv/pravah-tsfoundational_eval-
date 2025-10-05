#!/bin/bash
# Quick Start Script for Running Model Evaluation with GPU
# Run this after cloning the repository on your local machine

set -e  # Exit on error

echo "=============================================="
echo "Model Evaluation Quick Start"
echo "=============================================="
echo ""

# Step 1: Check if we're in the right directory
if [ ! -f "evaluate_gifteval_models.py" ]; then
    echo "ERROR: Please run this script from the project root directory"
    echo "cd pravah-tsfoundational_eval-"
    exit 1
fi

# Step 2: Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Step 3: Activate virtual environment
echo ""
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
echo "✓ Virtual environment activated"

# Step 4: Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✓ Pip upgraded"

# Step 5: Install PyTorch with CUDA (adjust version as needed)
echo ""
echo "Checking PyTorch installation..."
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo "✓ PyTorch installed"
else
    echo "✓ PyTorch already installed"
fi

# Step 6: Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt -q
echo "✓ Dependencies installed"

# Step 7: Verify GPU
echo ""
echo "=============================================="
echo "GPU Configuration Check"
echo "=============================================="
python models/gpu_utils.py

# Step 8: Ask user if they want to proceed
echo ""
read -p "GPU configuration shown above. Proceed with evaluation? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled. You can run manually with:"
    echo "  python evaluate_gifteval_models.py"
    exit 0
fi

# Step 9: Run evaluation
echo ""
echo "=============================================="
echo "Starting Model Evaluation"
echo "=============================================="
echo ""
echo "This will train and evaluate all models."
echo "Progress will be saved to: evaluation_$(date +%Y%m%d_%H%M%S).log"
echo ""

LOG_FILE="evaluation_$(date +%Y%m%d_%H%M%S).log"
python -u evaluate_gifteval_models.py 2>&1 | tee "$LOG_FILE"

# Step 10: Show results
echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - results/evaluation_results.csv"
echo "  - results/evaluation_results.json"
echo "  - $LOG_FILE"
echo ""

if [ -f "results/evaluation_results.csv" ]; then
    echo "Quick Summary:"
    cat results/evaluation_results.csv
fi

echo ""
echo "Done! 🎉"
