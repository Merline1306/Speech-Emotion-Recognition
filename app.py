import streamlit as st
import tempfile

from predict import predict

st.set_page_config(
    page_title="Speech Emotion Recognition",
    layout="wide"
)

st.title("Speech Emotion Recognition")

uploaded = st.file_uploader(
    "Upload WAV file",
    type=["wav"]
)

if uploaded:

    st.audio(uploaded)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(uploaded.read())

        audio_path = tmp.name

    # Show loading message
    with st.spinner("Analyzing emotion... Please wait..."):
        emotion, confidence, probs = predict(audio_path)

    st.success(f"Predicted Emotion: {emotion}")

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )
