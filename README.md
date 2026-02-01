# corrosive-stones

A Google Colab notebook for training custom Piper TTS (text-to-speech) voice models.

## Overview

This repository contains a Jupyter notebook (`STEVE_piper_multilingual_training_notebook.ipynb`) that guides you through training a custom neural TTS voice using the [Piper](https://github.com/rhasspy/piper) framework.

## Features

- Complete training pipeline from dataset preparation to ONNX export
- Optional Whisper-based automatic transcription
- TensorBoard integration for monitoring training progress
- Google Drive integration for persistent storage

## Requirements

- Google Colab (with GPU runtime recommended)
- Audio dataset: WAV files (16000 or 22050Hz, 16-bit, mono)
- Transcript file in pipe-delimited format: `wavs/1.wav|Transcript text here.`

## Usage

1. Open the notebook in Google Colab
2. Mount your Google Drive
3. Upload your audio dataset and transcript
4. Run the preprocessing cells
5. Configure training settings
6. Train and export your model to ONNX format
