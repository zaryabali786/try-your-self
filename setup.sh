#!/bin/bash
set -e

echo "================================================"
echo "  Virtual Try-On - Linux/Mac Setup"
echo "================================================"

# ── Step 1: Python check ──────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Install Python 3.10 or 3.11"
    exit 1
fi
python3 --version

# ── Step 2: Virtual environment ───────────────────
echo ""
echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ── Step 3: Upgrade pip ───────────────────────────
echo ""
echo "[2/5] Upgrading pip..."
pip install --upgrade pip

# ── Step 4: PyTorch ───────────────────────────────
echo ""
echo "[3/5] Installing PyTorch..."
echo ""
echo "Choose your setup:"
echo "  1) NVIDIA GPU - CUDA 11.8"
echo "  2) NVIDIA GPU - CUDA 12.1"
echo "  3) CPU only (slow)"
echo ""
read -p "Enter choice (1/2/3): " choice

if [ "$choice" = "1" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "$choice" = "2" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch torchvision torchaudio
fi

# ── Step 5: All dependencies ──────────────────────
echo ""
echo "[4/5] Installing project dependencies..."
pip install -r requirements.txt

# ── Step 6: Done ──────────────────────────────────
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "================================================"
echo "To run Web UI:  source venv/bin/activate && python app.py"
echo "To run API:     source venv/bin/activate && python api.py"
echo "================================================"
