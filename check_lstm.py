
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load model and tokenizer
model = tf.keras.models.load_model(
    "models/lstm_sentiment_model.keras"
)

tokenizer = joblib.load("models/lstm_tokenizer.pkl")

# Load actual test dataset
test_data = pd.read_csv("dataset/test.csv")

max_length = 200

# Test one negative and one positive review
for label in [0, 1]:

    sample = test_data[test_data["label"] == label].sample(
        n=1,
        random_state=42
    )

    review = sample["text"].iloc[0]

    sequence = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded, verbose=0)[0][0]

    print("\n-----------------------------")
    print("Actual sentiment:", "POSITIVE" if label == 1 else "NEGATIVE")
    print("Review:", review[:300])
    print("Prediction score:", float(prediction))
    print("Predicted sentiment:", "POSITIVE" if prediction >= 0.5 else "NEGATIVE")
# STEP: Evaluate the complete test dataset

# Convert reviews into sequences
X_test = tokenizer.texts_to_sequences(
    test_data["text"].astype(str)
)

# Apply padding
X_test_padded = pad_sequences(
    X_test,
    maxlen=200,
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

# Convert prediction scores into classes
y_pred = (prediction_scores.flatten() >= 0.5).astype(int)

# Display accuracy
print("\nTest Accuracy:")
print(accuracy_score(y_true, y_pred) * 100, "%")

# Display classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Negative", "Positive"]
))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
 

  