
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load trained BiLSTM model
model = tf.keras.models.load_model(
    "models/bilstm_sentiment_model.keras"
)

# Load BiLSTM tokenizer
tokenizer = joblib.load(
    "models/bilstm_tokenizer.pkl"
)

# Load test dataset
test_data = pd.read_csv("dataset/test.csv")

# Settings
max_length = 200

# Convert reviews into sequences
X_test = tokenizer.texts_to_sequences(
    test_data["text"].astype(str)
)

# Apply padding
X_test_padded = pad_sequences(
    X_test,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

# Actual labels
y_true = test_data["label"].astype(int).tolist()

# Make predictions
prediction_scores = model.predict(
    X_test_padded,
    batch_size=64,
    verbose=1
)

# Convert scores into classes
y_pred = (prediction_scores.flatten() >= 0.5).astype(int)

# Display accuracy
print("\nBiLSTM Test Accuracy:")
print(accuracy_score(y_true, y_pred) * 100, "%")

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Negative", "Positive"]
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
