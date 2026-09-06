"""
Run mood prediction on a single audio file using the trained model.

Usage:
    python predict.py path/to/song.mp3
"""

import sys
import numpy as np
from tensorflow import keras

import config
from features import audio_file_to_feature, pad_or_crop_width


def predict_mood(audio_path, model=None, classes=None):
    if model is None:
        model = keras.models.load_model(config.MODEL_PATH)
    if classes is None:
        classes = np.load(config.LABEL_ENCODER_PATH, allow_pickle=True)

    feat = audio_file_to_feature(audio_path)
    feat = pad_or_crop_width(feat)
    feat = feat[np.newaxis, ..., np.newaxis]  # (1, n_mels, time, 1)

    probs = model.predict(feat, verbose=0)[0]
    ranked = sorted(zip(classes, probs), key=lambda p: -p[1])

    return ranked  # list of (mood_label, probability), most likely first


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file>")
        sys.exit(1)

    results = predict_mood(sys.argv[1])
    print(f"\nPredictions for {sys.argv[1]}:")
    for label, prob in results:
        print(f"  {label:12s} {prob*100:5.1f}%")
    print(f"\n=> Predicted mood: {results[0][0]}")
