#!/bin/bash
# Piper TTS Fine-Tuning Script for macOS (Apple Silicon)
#
# Usage: ./train_local.sh
#
# Before running:
# 1. Run setup_local.sh first
# 2. Preprocess your dataset (see below)
# 3. Update the paths in this script

set -e

# ================================================
# CONFIGURATION - Update these paths!
# ================================================

# Path to your preprocessed dataset directory
DATASET_DIR="$HOME/piper-training/my_voice"

# Path to pretrained checkpoint (for fine-tuning)
PRETRAINED_CKPT="$HOME/piper-training/en_US-kushal-medium.ckpt"

# Training settings
BATCH_SIZE=8          # Reduce to 4 if you get memory errors
QUALITY="medium"      # x-low, medium, or high
MAX_EPOCHS=3000
CHECKPOINT_EPOCHS=5
LOG_EVERY_N_STEPS=100

# ================================================
# Setup
# ================================================

PIPER_DIR="$HOME/piper-training"
cd "$PIPER_DIR"
source venv/bin/activate
cd piper/src/python

# Set environment variable for PyTorch
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

# Allow PyTorch to use MPS
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "=================================================="
echo "Piper TTS Fine-Tuning (Apple Silicon)"
echo "=================================================="
echo "Dataset: $DATASET_DIR"
echo "Pretrained: $PRETRAINED_CKPT"
echo "Quality: $QUALITY"
echo "Batch size: $BATCH_SIZE"
echo "Max epochs: $MAX_EPOCHS"
echo "=================================================="
echo ""

# Verify paths exist
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    echo ""
    echo "You need to preprocess your dataset first:"
    echo "  python -m piper_train.preprocess \\"
    echo "    --language en-us \\"
    echo "    --input-dir /path/to/raw/dataset \\"
    echo "    --output-dir $DATASET_DIR \\"
    echo "    --dataset-name my_voice \\"
    echo "    --dataset-format ljspeech \\"
    echo "    --sample-rate 22050 \\"
    echo "    --single-speaker"
    exit 1
fi

if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "Error: Pretrained checkpoint not found: $PRETRAINED_CKPT"
    exit 1
fi

# Run training
python -m piper_train \
    --dataset-dir "$DATASET_DIR" \
    --accelerator mps \
    --devices 1 \
    --batch-size $BATCH_SIZE \
    --validation-split 0 \
    --num-test-examples 0 \
    --quality $QUALITY \
    --checkpoint-epochs $CHECKPOINT_EPOCHS \
    --num_ckpt 1 \
    --log_every_n_steps $LOG_EVERY_N_STEPS \
    --max_epochs $MAX_EPOCHS \
    --resume_from_checkpoint "$PRETRAINED_CKPT" \
    --precision 32

echo ""
echo "=================================================="
echo "Training complete!"
echo "=================================================="
