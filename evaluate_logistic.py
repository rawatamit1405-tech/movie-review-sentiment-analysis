
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading Logistic Regression model...")

# Load saved model and vectorizer
model = joblib.load("models/sentiment_model.pkl")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load test dataset
test_data = pd.read_csv("dataset/test.csv")

X_test = test_data["text"].astype(str)
y_true = test_data["label"].astype(int).tolist()

print("Converting reviews into TF-IDF features...")

X_test_tfidf = vectorizer.transform(X_test)

print("Predicting sentiment...")

y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_true, y_pred)

print("\nLogistic Regression Test Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Negative", "Positive"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
