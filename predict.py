import os
import json
import numpy as np

from tensorflow.keras.models import load_model
from feature_extraction import extract_features

print("Current Directory:", os.getcwd())
print("Files in Current Directory:", os.listdir())

model = load_model("emotion_model.keras")

with open("emotion_labels.json", "r", encoding="utf-8") as f:
    content = f.read()

print("JSON Content:", repr(content))

labels = json.loads(content)-[]


def predict(audio_path):

    feature = extract_features(audio_path)

    feature = np.expand_dims(feature, axis=0)

    prediction = model.predict(feature, verbose=0)

    index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    emotion = labels[str(index)]

    return emotion, confidence, prediction[0]

