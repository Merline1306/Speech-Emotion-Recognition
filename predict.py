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

    print("STEP 1: Start")

    t = time.time()

    feature = extract_features(audio_path)

    print("STEP 2: Feature extracted")
    print(feature.shape)
    print("Time:", time.time()-t)

    t = time.time()

    feature = np.expand_dims(feature, axis=0)

    print("STEP 3: Before model.predict")

    prediction = model.predict(feature, verbose=0)

    print("STEP 4: Prediction completed")
    print("Time:", time.time()-t)

    index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    emotion = labels[str(index)]

    print("STEP 5: Finished")

    return emotion, confidence, prediction[0]
