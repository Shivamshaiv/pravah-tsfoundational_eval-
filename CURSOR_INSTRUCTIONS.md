# Instructions to Paste into Cursor on Your Local Machine

Copy and paste this into Cursor after cloning the repository to your local machine with GPU.

---

## INSTRUCTION FOR CURSOR AI:

I have cloned this time series forecasting evaluation project to my local machine which has an NVIDIA GPU. I need to:

1. **Setup the environment with GPU support**
2. **Install all dependencies including PyTorch with CUDA**
3. **Verify GPU is detected and working**
4. **Run the complete model evaluation pipeline**
5. **Generate final results in CSV and JSON format**

### My System:
- OS: [Windows/Linux/Mac]
- GPU: [Your GPU model, e.g., RTX 3090, A100]
- Python version: [Your Python version]

### What I need you to do:

1. **First, verify my GPU setup:**
   - Check if NVIDIA drivers are installed
   - Verify CUDA is available
   - Test PyTorch GPU detection

2. **Setup the environment:**
   - Create/activate virtual environment in `.venv/`
   - Install PyTorch with the correct CUDA version for my system
   - Install all dependencies from `requirements.txt`
   - Install optional foundation model packages (chronos-forecasting, uni2ts, timesfm)

3. **Verify everything is working:**
   - Run `models/gpu_utils.py` to check GPU configuration
   - Confirm GPU memory, CUDA version, and PyTorch can see the GPU
   - Check that data files exist in `data/` folder

4. **Run the complete evaluation:**
   - Execute `evaluate_gifteval_models.py` with GPU acceleration
   - This should train and evaluate ALL models:
     * Baseline: Naive, SeasonalNaive, MovingAverage
     * Deep Learning: TiDE, PatchTST, TFT, NBEATS, DLinear, NHiTS
     * Foundation: TimesFM, Moirai, Chronos
   - Monitor progress and GPU usage
   - Handle any errors that come up

5. **Generate and show me the final results:**
   - Results should be saved in `results/evaluation_results.csv` and `results/evaluation_results.json`
   - Show me a summary table with all models and their metrics (MAPE, MAE, RMSE, MASE, CRPS)
   - Identify which model performed best

### Important Notes:
- **Use GPU throughout**: All deep learning models should use CUDA, not CPU
- **Optimize batch sizes**: Adjust based on my GPU memory (show me current batch sizes)
- **Handle memory issues**: If GPU runs out of memory, reduce batch sizes automatically
- **Save progress**: If training takes long, ensure checkpoints are saved
- **Show detailed logs**: I want to see GPU utilization, training progress, and validation metrics

### Expected Output:
At the end, I should have:
- ✅ Confirmation GPU was used for training
- ✅ `results/evaluation_results.csv` with all model metrics
- ✅ `results/evaluation_results.json` with detailed results
- ✅ Summary showing which model performed best on each metric
- ✅ Training time for each model
- ✅ GPU memory usage statistics

### If something fails:
- Try to fix it automatically (e.g., install missing packages, adjust batch sizes)
- If a specific model fails, skip it and continue with others
- At the end, tell me what failed and why

### Data Information:
- Data is in `data/merged_timeseries.csv`
- Format: 15-minute interval electricity demand data
- Time period: 2016-2024
- Splits: Train (2016-2022), Validation (2023), Test (2024)
- Features: Calendar features (hour, day, month, holidays) + weather data

Start by checking GPU availability and then proceed step by step. Show me the output of each major step.

---

## Alternative: Run Quick Start Script

If you prefer automated setup, I can also just run:

**On Linux/Mac:**
```bash
chmod +x quick_start.sh
./quick_start.sh
```

**On Windows:**
```cmd
quick_start.bat
```

This will automatically:
- Setup virtual environment
- Install all dependencies
- Verify GPU
- Run complete evaluation
- Show results

Which approach would you like me to take?

---

## Manual Commands (if needed):

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt
pip install chronos-forecasting uni2ts timesfm

# 4. Verify GPU
python models/gpu_utils.py

# 5. Run evaluation
python evaluate_gifteval_models.py

# 6. View results
cat results/evaluation_results.csv
```

Please help me get this running with full GPU acceleration and generate the final evaluation results!
