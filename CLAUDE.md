# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Google Colab notebook for training Piper TTS (text-to-speech) models. Piper is an open-source neural TTS system. The notebook guides users through the complete workflow of training a custom voice model using the Piper framework.

## Notebook Structure

The notebook (`STEVE_piper_multilingual_training_notebook.ipynb`) is designed to run in Google Colab and follows this workflow:

1. **Environment Setup** (Cells 1-18): Python downgrade to 3.10, GPU checks, Google Drive mounting, Piper repository cloning, dependency installation
2. **Training Pipeline** (Cells 29-38):
   - Extract audio dataset (WAV files: 16000 or 22050Hz, 16-bit, mono)
   - Upload transcript file (format: `wavs/1.wav|Transcript text here.`)
   - Preprocess dataset (optional Whisper-based auto-transcription)
   - Configure training settings
   - Monitor with TensorBoard
   - Run training
3. **Export** (Cell 27): Convert trained checkpoint to ONNX format for inference

## Key Dependencies

- `piper-phonemize` / `piper-phonemize-cross`
- `pytorch-lightning==1.9.0`
- `torch==2.1.0`
- `onnxruntime`, `onnx`, `onnxscript`
- `faster-whisper` (for auto-transcription)

## Data Format

- Audio: WAV files, 16000 or 22050Hz, 16-bit mono, numbered sequentially (1.wav, 2.wav, ...)
- Transcript: Pipe-delimited format: `wavs/<filename>.wav|<transcription text>`
