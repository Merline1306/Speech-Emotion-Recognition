import time
import json
import numpy as np
from tensorflow.keras.models import load_model
from feature_extraction import extract_features

# Load trained model
model = load_model("emotion_model.keras")

# Load emotion labels
with open("emotion_labels.json", "r") as f:
    labels = json.load(f)

# Prediction function
import time

def predict(audio_path):

    print("=" * 50)
    print("Prediction Started")

    start = time.time()

    feature = extract_features(audio_path)
    print("Feature shape:", feature.shape)
    print("Feature extraction:", time.time() - start)

    start = time.time()

    feature = np.expand_dims(feature, axis=0)

    print("Input shape:", feature.shape)

    prediction = model.predict(feature, verbose=0)

    print("Prediction completed")
    print("Prediction time:", time.time() - start)

    index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    emotion = labels[str(index)]

    print("Predicted Emotion:", emotion)
    print("=" * 50)

    return emotion, confidence, prediction[0]
