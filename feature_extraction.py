import librosa
import numpy as np

from preprocessing import preprocess_audio

TARGET_SR = 16000

def extract_features(audio_path):

    audio = preprocess_audio(audio_path)

    # Use a maximum of 3 seconds
    max_samples = TARGET_SR * 3
    audio = audio[:max_samples]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Fixed size
    if mel_db.shape[1] < 94:
        pad = 94 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)))
    else:
        mel_db = mel_db[:, :94]

    mel_db = mel_db[..., np.newaxis]

    return mel_db
