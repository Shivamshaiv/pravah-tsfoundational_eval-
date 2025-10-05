# Complete Instructions to Run Model Evaluation on Your Local Machine with GPU

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] NVIDIA GPU with CUDA support
- [ ] NVIDIA drivers installed
- [ ] CUDA toolkit installed (11.8+ or 12.x)
- [ ] Python 3.8 or higher
- [ ] Git installed

---

## Step 1: Verify GPU Setup

First, verify your GPU is working:

```bash
# Check NVIDIA driver
nvidia-smi

# Should show your GPU (e.g., RTX 3090, A100, etc.)
```

---

## Step 2: Clone Repository and Setup Environment

```bash
# Clone the repository
git clone https://github.com/Shivamshaiv/pravah-tsfoundational_eval-.git
cd pravah-tsfoundational_eval-

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust CUDA version as needed)
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt

# Install additional foundation model packages (optional but recommended)
pip install chronos-forecasting  # For Chronos
pip install uni2ts              # For Moirai
pip install timesfm             # For TimesFM
```

---

## Step 3: Verify PyTorch GPU Detection

```bash
# Test GPU availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Expected output should show:
# PyTorch version: 2.x.x+cuXXX
# CUDA available: True
# CUDA version: 12.1 (or your version)
# GPU: NVIDIA GeForce RTX 3090 (or your GPU)
```

---

## Step 4: Check GPU Configuration in Your Code

```bash
# Run GPU utilities to verify everything is configured
python models/gpu_utils.py

# Should display:
# ============================================================
# GPU/DEVICE CONFIGURATION
# ============================================================
# PyTorch Version: x.x.x
# Device: cuda:0
# Device Name: Your GPU Name
# CUDA Available: True
# CUDA Version: x.x
# GPU Memory Total: XX.XX GB
# ============================================================
```

---

## Step 5: Verify Your Data Files

```bash
# Check if data files exist
ls -lh data/

# Should see:
# merged_timeseries.csv (your main electricity demand data)
# sample_demand_timeseries.csv (backup sample data)

# Verify data format (should have datetime and demand columns)
python check_data.py
```

If your data is in a different format, ensure it has:
- A datetime column (named: `datetime`, `ds`, `timestamp`, or `time`)
- A demand/value column (named: `demand`, `y`, `value`, or `electricity_demand`)

---

## Step 6: Run Complete Model Evaluation

Now run the full evaluation pipeline:

```bash
# Run all models with GPU acceleration
python evaluate_gifteval_models.py
```

This will:
1. Display GPU configuration at startup
2. Load and prepare your electricity demand data
3. Create train/validation/test splits (2016-2022 train, 2023 validation, 2024 test)
4. Train and evaluate each model:
   - Baseline models (fast): Naive, SeasonalNaive, MovingAverage
   - Deep learning models (GPU-accelerated): TiDE, PatchTST, TFT, NBEATS, DLinear, NHiTS
   - Foundation models (GPU-accelerated, zero-shot): TimesFM, Moirai, Chronos
5. Calculate metrics: MAPE, MAE, RMSE, MASE, CRPS
6. Save results to `results/` folder

---

## Step 7: Monitor GPU Usage During Training

While the evaluation is running, monitor GPU usage in another terminal:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or just check once
nvidia-smi
```

You should see:
- GPU utilization increasing during model training
- Memory usage increasing (will vary by model and batch size)
- Temperature rising (normal during training)

---

## Step 8: Understand the Output

### Console Output

You'll see output like:

```
================================================================================
GIFTEval Model Evaluation Framework
================================================================================

============================================================
GPU/DEVICE CONFIGURATION
============================================================
PyTorch Version: 2.x.x+cuXXX
Device: cuda:0
Device Name: NVIDIA GeForce RTX 3090
CUDA Available: True
...
============================================================

Loading demand data...
Loaded demand data: XXXXX records from 2016-01-01 to 2024-12-31

Data splits prepared:
Train: XXXXX records (2016-01-01 to 2022-12-31)
Validation: XXXXX records (2023-01-01 to 2023-12-31)
Test: XXXXX records (2024-01-01 to 2024-12-31)
Features: XX features

Evaluating model: Naive
----------------------------------------
...

Evaluating model: TiDE
----------------------------------------
Creating TiDE model
Device: cuda:0 (NVIDIA GeForce RTX 3090)
Batch size: 128
Training TiDE model...
...

==================================================
TIDE MODEL EVALUATION
==================================================
TiDE Results:
  MAPE: XX.XXXX%
  MAE: XXX.XXXX
  RMSE: XXX.XXXX
  MASE: X.XXXX
...
```

### Results Files

After completion, check the results folder:

```bash
ls -lh results/

# You should see:
# evaluation_results.csv  - Results in CSV format
# evaluation_results.json - Results in JSON format
```

View results:

```bash
# View CSV results
cat results/evaluation_results.csv

# Or view JSON results (prettier)
cat results/evaluation_results.json
```

---

## Step 9: Customize Evaluation (Optional)

### Run Specific Models Only

Edit `evaluate_gifteval_models.py` and modify the `models_to_evaluate` list:

```python
# In the main() function, around line 662:
models_to_evaluate = [
    # Comment out models you don't want to run
    "Naive",
    "TiDE",
    "PatchTST",
    # "Chronos",  # Skip this one
]
```

### Adjust Model Hyperparameters

Edit individual model files to adjust hyperparameters:

**For TiDE** (`models/tide_model.py`):
```python
def _get_default_config(self) -> Dict:
    return {
        'n_epochs': 100,  # Increase from 50
        'batch_size': 128,  # Adjust based on GPU memory
        'learning_rate': 1e-3,  # Tune learning rate
        # ... other parameters
    }
```

**For Darts models** (`models/darts_models.py`):
```python
def _get_default_config(self, model_name: str) -> Dict:
    base_config = {
        'n_epochs': 100,  # Increase epochs
        'batch_size': 64,  # Adjust batch size
        # ... other parameters
    }
```

---

## Step 10: Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```python
# In model config files, reduce batch_size
'batch_size': 32,  # or 16, or 8
```

**Solution 2**: Clear GPU cache
```python
import torch
torch.cuda.empty_cache()
```

### Issue: CUDA Not Available

**Check:**
```bash
# Verify NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Foundation Models Not Found

**Install missing packages:**
```bash
pip install chronos-forecasting uni2ts timesfm
```

If still failing, comment out foundation models in `evaluate_gifteval_models.py`:
```python
models_to_evaluate = [
    "Naive",
    "TiDE",
    "PatchTST",
    # "TimesFM",  # Skip if not installed
    # "Moirai",   # Skip if not installed
    # "Chronos",  # Skip if not installed
]
```

### Issue: Module Import Errors

**Fix Python path:**
```bash
# Ensure you're in the project root
cd pravah-tsfoundational_eval-

# Run from project root
python evaluate_gifteval_models.py
```

### Issue: Slow Training on GPU

**Check:**
```bash
# Verify GPU is being used
nvidia-smi

# Check if model is actually on GPU
python -c "from models.gpu_utils import print_gpu_info; print_gpu_info()"
```

---

## Step 11: Advanced Usage

### Run Individual Model Tests

**Test TiDE model only:**
```python
python -c "
from models.tide_model import TiDEModelWrapper
from models.gpu_utils import print_gpu_info
import pandas as pd

print_gpu_info()

# Load your data
train_data = pd.read_csv('data/merged_timeseries.csv')
# ... prepare data ...

model = TiDEModelWrapper()
# model.train(train_data, val_data, train_features, val_features)
"
```

### Monitor Training Progress

Add verbose logging in `evaluate_gifteval_models.py`:
```python
# In the model.fit() calls, set verbose=True
model.fit(
    series=train_y,
    future_covariates=train_features_scaled,
    val_series=val_y,
    val_future_covariates=val_features_scaled,
    verbose=True  # Enable detailed logging
)
```

---

## Expected Results Format

After successful completion, `results/evaluation_results.csv` will look like:

```csv
model,mape,mae,rmse,mase,crps
Naive,15.2345,234.5678,345.6789,1.2345,234.5678
SeasonalNaive,12.3456,198.7654,298.7654,1.0987,198.7654
MovingAverage,13.4567,210.8765,310.8765,1.1234,210.8765
TiDE,8.5432,156.4321,230.4321,0.8765,156.4321
PatchTST,9.2345,167.5432,245.6543,0.9123,167.5432
TFT,9.8765,178.6543,256.7654,0.9876,178.6543
NBEATS,10.1234,189.7654,267.8765,1.0234,189.7654
DLinear,8.9876,165.4321,240.5432,0.9012,165.4321
NHiTS,9.5432,172.3456,250.4567,0.9456,172.3456
TimesFM,11.2345,190.5678,280.6789,1.0567,190.5678
Moirai,10.8765,185.4321,275.5432,1.0123,185.4321
Chronos,11.5678,195.6789,285.7890,1.0789,195.6789
```

---

## Performance Expectations

### With GPU (e.g., RTX 3090, A100):
- **Baseline models**: < 1 minute each
- **Deep learning models**: 10-30 minutes each (depends on epochs)
- **Foundation models**: 5-15 minutes each (zero-shot, no training)
- **Total runtime**: 1-3 hours for all models

### GPU Memory Usage:
- **TiDE/PatchTST**: 2-4 GB
- **TFT**: 4-6 GB
- **NBEATS/NHiTS**: 2-4 GB
- **Foundation models**: 4-8 GB

### Batch Size Guidelines:
- **24GB GPU (RTX 3090)**: batch_size = 64-128
- **40GB GPU (A100)**: batch_size = 128-256
- **16GB GPU (RTX 4060)**: batch_size = 32-64
- **8GB GPU**: batch_size = 16-32

---

## Quick Start Summary

For copy-paste execution:

```bash
# 1. Clone and setup
git clone https://github.com/Shivamshaiv/pravah-tsfoundational_eval-.git
cd pravah-tsfoundational_eval-
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Verify GPU
python models/gpu_utils.py

# 4. Run evaluation
python evaluate_gifteval_models.py

# 5. View results
cat results/evaluation_results.csv
```

---

## Need Help?

If you encounter issues:
1. Check GPU is detected: `python models/gpu_utils.py`
2. Verify data is loaded: `python check_data.py`
3. Run with fewer models first to test
4. Check GPU memory usage: `nvidia-smi`
5. Review error messages carefully

For detailed debugging, run with Python's verbose mode:
```bash
python -u evaluate_gifteval_models.py 2>&1 | tee evaluation_log.txt
```

This saves all output to `evaluation_log.txt` for review.

---

## Success Indicators

You'll know everything worked when you see:
- ✅ GPU detected and used (shows in startup logs)
- ✅ All models complete without errors
- ✅ Results files generated in `results/` folder
- ✅ CSV/JSON files contain metrics for all models
- ✅ MAPE, MAE, RMSE, MASE, CRPS values are reasonable (not inf or nan)

Good luck with your evaluation!
