"""
Audio -> log-mel-spectrogram feature extraction.

A mel-spectrogram is used (instead of raw waveform) because it turns audio
into a 2D time-frequency image, which is exactly the input CNNs are built
for. Log scaling compresses the dynamic range, mimicking human loudness
perception and making the CNN's job easier.
"""

import numpy as np
import librosa

import config


def load_audio_segment(path, sr=config.SAMPLE_RATE, duration=config.CLIP_DURATION):
    """Load an audio file and take a fixed-length center segment.

    Using the center of the track avoids silent intros/outros and gives a
    consistent, representative window for mood.
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)

    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    else:
        # pad short clips with zeros (silence) so all inputs are equal length
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")

    return y, sr


def extract_melspectrogram(y, sr=config.SAMPLE_RATE):
    """Compute a log-scaled mel-spectrogram, normalized to zero mean / unit std."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # per-clip standardization
    mean, std = log_mel.mean(), log_mel.std()
    log_mel = (log_mel - mean) / (std + 1e-8)

    return log_mel.astype(np.float32)  # shape: (N_MELS, time_frames)


def audio_file_to_feature(path):
    """Full pipeline: file path -> normalized (n_mels, time) feature array."""
    y, sr = load_audio_segment(path)
    return extract_melspectrogram(y, sr)


def pad_or_crop_width(feature, target_width=config.IMG_WIDTH):
    """Force a feature map to a fixed width so all samples batch together."""
    h, w = feature.shape
    if w == target_width:
        return feature
    if w > target_width:
        start = (w - target_width) // 2
        return feature[:, start:start + target_width]
    pad_total = target_width - w
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(feature, ((0, 0), (pad_left, pad_right)), mode="constant")
