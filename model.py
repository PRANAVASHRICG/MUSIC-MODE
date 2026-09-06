"""
CNN architecture for music mood classification from mel-spectrogram images.

Design notes:
- 4 conv blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout) progressively
  shrink the time-frequency map while learning increasingly abstract features
  (edges/onsets -> timbre patterns -> mood-level structure).
- GlobalAveragePooling instead of Flatten+Dense keeps parameter count low and
  reduces overfitting, which matters a lot for small music datasets.
- Class weights / label smoothing hooks are left in train.py since mood
  datasets are often imbalanced.
"""

from tensorflow import keras
from tensorflow.keras import layers

import config


def build_model(num_classes, input_shape=None):
    if input_shape is None:
        input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, 1)

    inputs = keras.Input(shape=input_shape, name="mel_spectrogram")

    x = inputs

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="mood")(x)

    model = keras.Model(inputs, outputs, name="music_mood_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_model(num_classes=len(config.MOOD_CLASSES))
    m.summary()
