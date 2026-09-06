"""
Central configuration for the music mood classification pipeline.
Edit MOOD_CLASSES and paths here to match your dataset.
"""

import os

# ---- Dataset ----
# Expected folder layout under DATA_RAW_DIR:
#   data/raw/<mood_label>/*.wav  (or .mp3)
# e.g. data/raw/happy/song1.wav, data/raw/sad/song2.wav ...
MOOD_CLASSES = ["happy", "sad", "energetic", "calm"]

DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ---- Audio ----
SAMPLE_RATE = 22050          # Hz
CLIP_DURATION = 30           # seconds analyzed per track (center segment)

# ---- Mel-spectrogram feature extraction ----
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
# Resulting spectrogram time-frames for a CLIP_DURATION-second clip:
# frames = int(SAMPLE_RATE * CLIP_DURATION / HOP_LENGTH) + 1
IMG_HEIGHT = N_MELS
IMG_WIDTH = int(SAMPLE_RATE * CLIP_DURATION / HOP_LENGTH) + 1

# ---- Training ----
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

MODEL_PATH = os.path.join(MODELS_DIR, "mood_cnn.keras")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_classes.npy")
HISTORY_PATH = os.path.join(OUTPUTS_DIR, "training_history.json")
