#!/bin/bash
# Piper TTS Fine-Tuning Local Setup for macOS (Apple Silicon)
# Requires: Homebrew

set -e

echo "=================================================="
echo "Piper TTS Local Setup for Mac (M4)"
echo "=================================================="

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew not found. Install from https://brew.sh"
    exit 1
fi

# Install Python 3.10 if not present
if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10..."
    brew install python@3.10
fi

echo "Python 3.10 location: $(which python3.10)"
python3.10 --version

# Set up working directory
PIPER_DIR="$HOME/piper-training"
mkdir -p "$PIPER_DIR"
cd "$PIPER_DIR"

# Clone Piper if not already cloned
if [ ! -d "piper" ]; then
    echo "Cloning Piper repository..."
    git clone https://github.com/rhasspy/piper.git
else
    echo "Piper already cloned"
fi

# Create virtual environment
echo "Creating Python 3.10 virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the "Golden Trio" dependencies
echo ""
echo "=================================================="
echo "Installing Golden Trio dependencies..."
echo "=================================================="
pip install torch==2.1.0 pytorch-lightning==1.9.0 torchmetrics==0.11.4
pip install onnx==1.16.1 onnxruntime==1.17.1

# Install other Piper dependencies
echo "Installing Piper dependencies..."
pip install cython librosa numpy==1.26

# NOTE: piper-phonemize is NOT installed because it doesn't build on macOS ARM64
# Preprocessing must be done on Colab/Linux, then copy preprocessed data here for training
echo ""
echo "NOTE: piper-phonemize skipped (not available for macOS ARM64)"
echo "You must preprocess your dataset on Colab/Linux first, then copy it here."

# Build monotonic_align
echo ""
echo "Building monotonic_align..."
cd piper/src/python
bash build_monotonic_align.sh

# Apply ONNX export patches
echo ""
echo "Applying ONNX export patches..."

# Patch 1: Comment out math assertion in transforms.py
sed -i '' 's/assert (discriminant >= 0).all(), discriminant/# assert (discriminant >= 0).all(), discriminant/' piper_train/vits/transforms.py

# Patch 2: Add .detach() to mask guard in modules.py
sed -i '' 's/h = self.pre(x0) \* x_mask/h = self.pre(x0) * x_mask.detach()/' piper_train/vits/modules.py

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "To activate the environment:"
echo "  cd $PIPER_DIR"
echo "  source venv/bin/activate"
echo ""
echo "To train a model:"
echo "  cd piper/src/python"
echo "  python -m piper_train \\"
echo "    --dataset-dir /path/to/preprocessed/data \\"
echo "    --accelerator mps \\"
echo "    --devices 1 \\"
echo "    --batch-size 8 \\"
echo "    --quality medium \\"
echo "    --max_epochs 3000 \\"
echo "    --resume_from_checkpoint /path/to/pretrained.ckpt \\"
echo "    --precision 32"
echo ""
