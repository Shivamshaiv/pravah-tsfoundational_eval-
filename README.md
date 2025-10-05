# Time Series Foundation Models Evaluation

Evaluation framework for electricity demand forecasting using state-of-the-art time series models with GPU support.

## Features

- GPU-accelerated training and inference with automatic CPU fallback
- Multiple model architectures:
  - Foundation models: TimesFM, Moirai, Chronos
  - Deep learning models: TiDE, PatchTST, TFT, NBEATS, DLinear, NHiTS
  - Baseline models: Naive, Seasonal Naive, Moving Average
- Comprehensive evaluation metrics (MAPE, MASE, CRPS)
- Automatic batch size optimization based on device
- Data pipeline for electricity demand forecasting

## GPU Support

The framework automatically detects and uses GPU when available. Key features:

- Automatic device detection (CUDA GPU or CPU)
- Optimized batch sizes based on available GPU memory
- Clear logging of device usage
- Seamless fallback to CPU if GPU unavailable

### Check GPU Status

```python
from models.gpu_utils import print_gpu_info
print_gpu_info()
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA toolkit (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd tsfoundational_eval
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### GPU Setup (Optional)

For GPU acceleration, install NVIDIA drivers and CUDA toolkit:

**Ubuntu/Debian:**
```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA drivers
sudo apt-get install -y cuda-drivers

# Reboot
sudo reboot
```

**Verify GPU:**
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Run Evaluation

```bash
python evaluate_gifteval_models.py
```

### Check GPU Configuration

```bash
python models/gpu_utils.py
```

### Train Specific Models

```python
from models.tide_model import TiDEModelWrapper
from models.darts_models import DartsModelWrapper
from models.foundation_models import FoundationModelWrapper

# TiDE model
tide = TiDEModelWrapper()
tide.train(train_data, val_data, train_features, val_features)

# PatchTST model
patchtst = DartsModelWrapper('PatchTST')
patchtst.train(train_data, val_data, train_features, val_features)

# Chronos foundation model
chronos = FoundationModelWrapper('Chronos')
chronos.load_model()
predictions = chronos.predict(history, horizon=192)
```

## Project Structure

```
tsfoundational_eval/
├── models/
│   ├── gpu_utils.py          # GPU detection and configuration
│   ├── foundation_models.py   # Foundation model wrappers
│   ├── darts_models.py        # Deep learning models
│   ├── tide_model.py          # TiDE model implementation
│   └── baseline_models.py     # Baseline models
├── data/
│   └── merged_timeseries.csv  # Electricity demand data
├── evaluate_gifteval_models.py  # Main evaluation script
├── requirements.txt
└── README.md
```

## Model Details

### Foundation Models

- **TimesFM** (Google): 200M parameter foundation model
- **Moirai** (Salesforce): Universal time series forecasting
- **Chronos** (Amazon): T5-based forecasting model

### Deep Learning Models

- **TiDE**: Time-series Dense Encoder
- **PatchTST**: Patch-based Transformer
- **TFT**: Temporal Fusion Transformer
- **NBEATS**: Neural Basis Expansion Analysis
- **DLinear**: Decomposition Linear
- **NHiTS**: Neural Hierarchical Interpolation

## Performance

The framework automatically optimizes performance based on available hardware:

- **With GPU (A100 40GB)**: Batch size 64-128, fast training
- **CPU only**: Reduced batch size (8-16), slower but functional

## Data Format

Expected CSV format for time series data:

```csv
datetime,demand
2016-01-01 00:00:00,1234.5
2016-01-01 00:15:00,1245.8
...
```

## Results

Evaluation results are saved to:
- `results/evaluation_results.csv`
- `results/evaluation_results.json`

## Contributing

Contributions welcome! Please ensure:
- Code follows existing patterns
- GPU utilities are used for device management
- Tests pass with both CPU and GPU

## License

MIT License

## Citation

If you use this code, please cite the relevant model papers and the GIFTEval benchmark.
