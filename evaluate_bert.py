
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load test dataset
test_data = pd.read_csv("dataset/test.csv")

# Load saved BERT model and tokenizer
model_path = "models/bert_sentiment_model"

print("Loading BERT model...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

print("BERT model loaded successfully!")

# Prepare reviews and actual labels
reviews = test_data["text"].astype(str).tolist()
y_true = test_data["label"].astype(int).tolist()

predictions = []

# Process reviews in small batches
batch_size = 16

print("\nEvaluating all test reviews...")

for start in range(0, len(reviews), batch_size):

    batch_reviews = reviews[start:start + batch_size]

    inputs = tokenizer(
        batch_reviews,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    batch_predictions = torch.argmax(
        outputs.logits,
        dim=1
    )

    predictions.extend(batch_predictions.cpu().tolist())

    if (start // batch_size + 1) % 100 == 0:
        print(f"Processed {min(start + batch_size, len(reviews))} reviews")

# Calculate performance
accuracy = accuracy_score(y_true, predictions)

print("\nBERT Test Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        predictions,
        target_names=["Negative", "Positive"]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, predictions))

print("\nBERT evaluation completed!")
