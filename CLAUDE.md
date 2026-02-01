# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Google Colab notebooks for training Piper TTS (text-to-speech) models. Piper is an open-source neural TTS system.

## Notebooks

### piper_finetune_notebook.ipynb (Recommended)

A streamlined 10-cell notebook for fine-tuning Piper models:

1. **Environment Setup** - Mount Drive, clone Piper, install "Golden Trio" dependencies
2. **Extract Dataset** - Unzip audio files from Google Drive
3. **Upload Transcript & Preprocess** - Configure language/sample rate, run preprocessing
4. **Training Settings** - Batch size, quality, epochs
5. **Download Pretrained Model** - Select and download base model for fine-tuning
6. **Run Fine-Tuning** - Execute training with `--resume_from_checkpoint`
7. **Export to ONNX** - Convert checkpoint to ONNX format
8. **Download Model** - Download ONNX + config locally
9. **Usage Instructions** (Markdown) - How to use the exported model

### STEVE_piper_multilingual_training_notebook.ipynb (Legacy)

The original 40-cell notebook with full training options including training from scratch. Kept for reference.

## Critical Dependencies ("Golden Trio")

These specific versions are required for ONNX export to work:

```
torch==2.1.0
pytorch-lightning==1.9.0
torchmetrics==0.11.4
onnx==1.16.1
onnxruntime==1.17.1
```

## Required Patches for ONNX Export

1. **transforms.py**: Comment out math assertion
   ```bash
   sed -i 's/assert (discriminant >= 0).all(), discriminant/# assert ...' transforms.py
   ```

2. **modules.py**: Add `.detach()` to mask guard
   ```bash
   sed -i 's/h = self.pre(x0) \* x_mask/h = self.pre(x0) * x_mask.detach()/' modules.py
   ```

## Data Format

- **Audio**: WAV files, 16000 or 22050Hz, 16-bit mono, numbered sequentially (1.wav, 2.wav, ...)
- **Transcript**: Pipe-delimited format: `wavs/<filename>.wav|<transcription text>`

## Key Commands

**Fine-tuning:**
```bash
python -m piper_train \
    --dataset-dir <output_dir> \
    --resume_from_checkpoint /content/pretrained.ckpt \
    --quality medium \
    --max_epochs 3000
```

**ONNX Export:**
```bash
python3 -m piper_train.export_onnx <checkpoint_path> <output.onnx>
```
