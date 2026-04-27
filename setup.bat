@echo off
echo ================================================
echo   Virtual Try-On - Windows Setup
echo ================================================

:: ── Step 1: Python version check ─────────────────
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10 or 3.11 from python.org
    pause
    exit /b 1
)

:: ── Step 2: Create virtual environment ───────────
echo.
echo [1/5] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: ── Step 3: Upgrade pip ───────────────────────────
echo.
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip

:: ── Step 4: Install PyTorch ───────────────────────
echo.
echo [3/5] Installing PyTorch...
echo.
echo Choose your setup:
echo   1) NVIDIA GPU - CUDA 11.8
echo   2) NVIDIA GPU - CUDA 12.1
echo   3) CPU only (slow)
echo.
set /p choice="Enter choice (1/2/3): "

if "%choice%"=="1" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%choice%"=="2" (
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    pip install torch torchvision torchaudio
)

:: ── Step 5: Install all dependencies ─────────────
echo.
echo [4/5] Installing project dependencies...
pip install -r requirements.txt

:: ── Step 6: Done ──────────────────────────────────
echo.
echo [5/5] Setup complete!
echo.
echo ================================================
echo To run the Web UI:   python app.py
echo To run the API:      python api.py
echo ================================================
pause
