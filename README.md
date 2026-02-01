# Piper TTS Training Notebooks

Google Colab notebooks for training custom [Piper](https://github.com/rhasspy/piper) TTS (text-to-speech) voice models.

## Notebooks

### [piper_finetune_notebook.ipynb](piper_finetune_notebook.ipynb) (Recommended)

A streamlined notebook for fine-tuning existing Piper models on your own voice data. 10 cells covering:

1. Environment setup (Drive mount, dependencies, patches)
2. Dataset extraction
3. Transcript upload & preprocessing
4. Training configuration
5. Pretrained model download
6. Fine-tuning execution
7. ONNX export
8. Local download

**Best for:** Creating a custom voice with a small dataset (5+ minutes of audio).

### [STEVE_piper_multilingual_training_notebook.ipynb](STEVE_piper_multilingual_training_notebook.ipynb) (Legacy)

The original comprehensive notebook with all training options including:
- Training from scratch
- Multi-speaker models
- Whisper auto-transcription
- TensorBoard monitoring

**Best for:** Advanced users who need full control or want to train from scratch with large datasets.

## Quick Start

1. Open [piper_finetune_notebook.ipynb](https://colab.research.google.com/github/PyEatingContest/corrosive-stones/blob/master/piper_finetune_notebook.ipynb) in Google Colab
2. Enable GPU runtime (Runtime > Change runtime type > T4 GPU)
3. Prepare your dataset:
   - Audio: WAV files (22050Hz, 16-bit, mono)
   - Transcript: `wavs/1.wav|Text spoken in audio 1.`
4. Upload dataset ZIP to Google Drive
5. Run cells sequentially
6. Download your `.onnx` model when complete

## Requirements

- Google Colab with GPU runtime
- Audio dataset: WAV files (16000 or 22050Hz, 16-bit, mono)
- Transcript file in pipe-delimited format

## Data Format

**Audio files:**
- Format: WAV
- Sample rate: 16000 or 22050 Hz
- Bit depth: 16-bit
- Channels: Mono
- Naming: Sequential numbers (1.wav, 2.wav, ...)

**Transcript file (metadata.csv):**
```
wavs/1.wav|This is the text spoken in the first audio file.
wavs/2.wav|This is what is said in the second recording.
```

For multi-speaker datasets:
```
wavs/1.wav|speaker1|Text spoken by speaker one.
wavs/2.wav|speaker2|Text spoken by speaker two.
```

## Using Your Trained Model

After export, use your model with Piper:

```bash
# Install piper-tts
pip install piper-tts

# Generate speech
echo "Hello world!" | piper --model my_voice.onnx --output_file output.wav
```

## Resources

- [Piper GitHub](https://github.com/rhasspy/piper)
- [Piper Training Documentation](https://github.com/rhasspy/piper/blob/master/TRAINING.md)
- [Piper Voice Samples](https://rhasspy.github.io/piper-samples/)
