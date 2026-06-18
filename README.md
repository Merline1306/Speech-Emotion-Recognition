# AI-Based Speech Emotion Recognition using Deep Learning

## Overview

Speech Emotion Recognition (SER) is a Deep Learning application that automatically identifies the emotional state of a speaker from speech signals. This project utilizes the RAVDESS Emotional Speech Audio Dataset to classify speech into eight different emotions using a Convolutional Neural Network (CNN).

The system performs complete audio processing, including preprocessing, feature extraction using Mel Spectrograms, model training, evaluation, and real-time emotion prediction through an interactive Streamlit web application.

---

## Features

- Speech Emotion Recognition from audio files
- Audio preprocessing and normalization
- Mel Spectrogram feature extraction
- CNN-based emotion classification
- Real-time prediction using Streamlit
- Interactive waveform visualization
- Mel Spectrogram visualization
- Emotion confidence score
- Probability distribution for all emotions
- Model performance evaluation
- Clean and responsive user interface

---

## Emotion Classes

The model classifies speech into the following eight emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fear
- Disgust
- Surprised

---

## Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

Dataset Features:

- 24 Professional Actors
- Male and Female Speakers
- 1440 Audio Files
- High-quality WAV format
- Eight emotion categories

Dataset Link:

https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

---

## Project Workflow

```
Speech Audio
      │
      ▼
Audio Preprocessing
      │
      ▼
Feature Extraction
(Mel Spectrogram)
      │
      ▼
CNN Model
      │
      ▼
Emotion Prediction
      │
      ▼
Streamlit Web Application
```

---

## Project Structure

```
Speech-Emotion-Recognition/
│
├── app.py
├── preprocessing.py
├── feature_extraction.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── emotion_model.keras
│   └── emotion_labels.json
│
├── notebooks/
│   └── ser.ipynb
│
├── assets/
│   ├── architecture.png
│   ├── home_page.png
│   ├── prediction_page.png
│   ├── waveform.png
│   └── mel_spectrogram.png
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── classification_report.txt
│
└── sample_audio/
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- Librosa
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Plotly

---

## Installation

### Clone the repository

```bash
git clone https://github.com/Merline1306/Speech-Emotion-Recognition.git
```

### Navigate to the project folder

```bash
cd Speech-Emotion-Recognition
```

### Install the required packages

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## Model Architecture

The model is built using a Convolutional Neural Network consisting of:

- Conv2D Layers
- Batch Normalization
- Max Pooling
- Global Average Pooling
- Dense Layers
- Dropout
- Softmax Output Layer

---

## Audio Preprocessing

The preprocessing pipeline includes:

- Audio Loading
- Resampling to 16 kHz
- Silence Removal
- Audio Normalization
- Padding and Trimming
- Fixed-length Audio Processing

---

## Feature Extraction

Mel Spectrograms are generated using Librosa and used as input to the CNN model.

Feature Extraction Steps:

- Audio Loading
- Mel Spectrogram Generation
- Power to Decibel Conversion
- Feature Normalization

---

## Model Training

Training Configuration:

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Batch Size: 32
- Early Stopping
- Learning Rate Reduction
- Model Checkpoint

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## Application Interface

The Streamlit application provides:

- Upload WAV Audio
- Audio Playback
- Emotion Prediction
- Confidence Score
- Waveform Visualization
- Mel Spectrogram Visualization
- Prediction Probability Chart

---

## Results

The trained CNN model successfully classifies speech into eight emotional categories using Mel Spectrogram features extracted from speech audio.

Model evaluation includes:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Confusion Matrix
- Classification Report

---


## Future Enhancements

- Gender Recognition
- Real-Time Microphone Input
- Transformer-based Speech Emotion Recognition
- Attention Mechanism
- Mobile Application
- Multi-language Emotion Recognition
- Cloud Deployment
- Explainable AI (XAI)
- Speaker Identification
- Emotion Timeline Visualization

---

## Learning Outcomes

This project demonstrates practical knowledge in:

- Deep Learning
- Audio Signal Processing
- Speech Emotion Recognition
- Feature Engineering
- CNN Architecture
- Model Evaluation
- Data Visualization
- Streamlit Deployment
- Python Application Development

---



## License

This project is developed for educational and research purposes.

The RAVDESS dataset is the property of its respective creators and is used in accordance with its license.
