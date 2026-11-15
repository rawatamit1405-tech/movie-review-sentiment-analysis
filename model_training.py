import os
import joblib

import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("STEP 1: Loading dataset...")

# Load training and testing data
train_data = pd.read_csv("dataset/train.csv")
test_data = pd.read_csv("dataset/test.csv")


# Clean movie reviews
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("STEP 2: Cleaning reviews...")

X_train = train_data["text"].apply(clean_text)
X_test = test_data["text"].apply(clean_text)

y_train = train_data["label"]
y_test = test_data["label"]


print("STEP 3: Converting text into numbers...")

# TF-IDF
vectorizer = TfidfVectorizer(max_features=20000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("STEP 4: Training Logistic Regression model...")

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train_tfidf, y_train)


print("STEP 5: Making predictions...")

predictions = model.predict(X_test_tfidf)


print("\nSTEP 6: Evaluating model...")

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy * 100, "%")

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"]
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")

# Create a folder to store the trained model
os.makedirs("models", exist_ok=True)

# Save the trained model
joblib.dump(model, "models/sentiment_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
