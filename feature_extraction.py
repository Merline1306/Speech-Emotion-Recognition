import librosa
import numpy as np

from preprocessing import preprocess_audio

TARGET_SR = 16000


def extract_features(audio_path):

    audio = preprocess_audio(audio_path)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = mel_db[..., np.newaxis]

    return mel_db

