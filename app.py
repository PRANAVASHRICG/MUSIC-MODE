import os
import tempfile

import numpy as np
import streamlit as st
from tensorflow import keras

import config
from features import audio_file_to_feature, pad_or_crop_width


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Music Mood Classification",
    page_icon="🎵",
    layout="centered"
)

# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Load trained CNN model
# -----------------------------
@st.cache_resource
def load_model():
    return keras.models.load_model(config.MODEL_PATH)


model = load_model()


# -----------------------------
# Prediction function
# -----------------------------
def predict_mood(audio_path):

    # Extract mel-spectrogram
    feature = audio_file_to_feature(audio_path)

    # Make width identical to training input
    feature = pad_or_crop_width(feature)

    # Add batch and channel dimensions
    feature = feature[np.newaxis, ..., np.newaxis]

    # CNN prediction
    probabilities = model.predict(feature, verbose=0)[0]

    # Get predicted class
    predicted_index = int(np.argmax(probabilities))
    predicted_mood = config.MOOD_CLASSES[predicted_index]

    return predicted_mood, probabilities


# -----------------------------
# UI
# -----------------------------
st.markdown(
    '<div class="main-title">🎵 Music Mood Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a music file and let the CNN classify its mood'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "🎧 Upload your music file",
    type=["wav", "mp3", "flac", "ogg", "m4a"]
)


if uploaded_file is not None:

    st.audio(uploaded_file)

    st.success(f"File uploaded: {uploaded_file.name}")

    if st.button("🔮 Predict Mood", use_container_width=True):

        with st.spinner("Analyzing your music... 🎵"):

            # Save uploaded audio temporarily
            suffix = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            try:

                predicted_mood, probabilities = predict_mood(temp_path)

            finally:

                if os.path.exists(temp_path):
                    os.remove(temp_path)


        # -----------------------------
        # Display prediction
        # -----------------------------

        mood_icons = {
            "happy": "😊",
            "sad": "😢",
            "energetic": "⚡",
            "calm": "😌"
        }

        icon = mood_icons.get(predicted_mood, "🎵")

        st.markdown(
            f"""
            <div class="prediction-box">
                <h2>Predicted Mood</h2>
                <h1>{icon} {predicted_mood.upper()}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Mood Probabilities")

        # Display probabilities
        for mood, probability in zip(
            config.MOOD_CLASSES,
            probabilities
        ):

            percentage = float(probability) * 100

            st.write(
                f"**{mood.capitalize()}** — {percentage:.1f}%"
            )

            st.progress(float(probability))