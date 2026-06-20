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
def predict(audio_path):

    total_start = time.time()

    # Feature extraction
    start = time.time()
    feature = extract_features(audio_path)
    print(f"Feature Extraction Time: {time.time() - start:.2f} seconds")

    # Prediction
    start = time.time()
    feature = np.expand_dims(feature, axis=0)
    prediction = model.predict(feature, verbose=0)
    print(f"Prediction Time: {time.time() - start:.2f} seconds")

    index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    emotion = labels[str(index)]

    print(f"Total Time: {time.time() - total_start:.2f} seconds")

    return emotion, confidence, prediction[0]
