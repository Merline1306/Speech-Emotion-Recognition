import librosa
import numpy as np

TARGET_SR = 16000
MAX_LENGTH = 48000


def preprocess_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR
    )

    audio, _ = librosa.effects.trim(audio)

    audio = librosa.util.normalize(audio)

    if len(audio) < MAX_LENGTH:
        audio = np.pad(
            audio,
            (0, MAX_LENGTH-len(audio))
        )

    audio = audio[:MAX_LENGTH]

    return audio

