@echo off
REM Quick Start Script for Running Model Evaluation with GPU (Windows)
REM Run this after cloning the repository on your local machine

echo ==============================================
echo Model Evaluation Quick Start (Windows)
echo ==============================================
echo.

REM Step 1: Check if we're in the right directory
if not exist "evaluate_gifteval_models.py" (
    echo ERROR: Please run this script from the project root directory
    echo cd pravah-tsfoundational_eval-
    exit /b 1
)

REM Step 2: Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    echo Virtual environment created
) else (
    echo Virtual environment exists
)

REM Step 3: Activate virtual environment
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated

REM Step 4: Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip -q
echo Pip upgraded

REM Step 5: Install PyTorch with CUDA
echo.
echo Checking PyTorch installation...
python -c "import torch" 2>nul
if errorlevel 1 (
    echo Installing PyTorch with CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo PyTorch installed
) else (
    echo PyTorch already installed
)

REM Step 6: Install requirements
echo.
echo Installing dependencies...
pip install -r requirements.txt -q
echo Dependencies installed

REM Step 7: Verify GPU
echo.
echo ==============================================
echo GPU Configuration Check
echo ==============================================
python models/gpu_utils.py

REM Step 8: Ask user if they want to proceed
echo.
set /p proceed="GPU configuration shown above. Proceed with evaluation? (y/n): "
if /i not "%proceed%"=="y" (
    echo Evaluation cancelled. You can run manually with:
    echo   python evaluate_gifteval_models.py
    exit /b 0
)

REM Step 9: Run evaluation
echo.
echo ==============================================
echo Starting Model Evaluation
echo ==============================================
echo.
echo This will train and evaluate all models.
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set LOG_FILE=evaluation_%mydate%_%mytime%.log
echo Progress will be saved to: %LOG_FILE%
echo.

python -u evaluate_gifteval_models.py 2>&1 | tee %LOG_FILE%

REM Step 10: Show results
echo.
echo ==============================================
echo Evaluation Complete!
echo ==============================================
echo.
echo Results saved to:
echo   - results\evaluation_results.csv
echo   - results\evaluation_results.json
echo   - %LOG_FILE%
echo.

if exist "results\evaluation_results.csv" (
    echo Quick Summary:
    type results\evaluation_results.csv
)

echo.
echo Done!
pause
