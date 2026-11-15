
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

model_path = "models/bert_sentiment_model"

print("Loading trained BERT model...")

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

print("BERT model loaded successfully!")

reviews = [
    "This movie was absolutely fantastic. I loved every moment.",
    "This movie was terrible and completely boring."
]

inputs = tokenizer(
    reviews,
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

predictions = torch.argmax(outputs.logits, dim=1)

for review, prediction in zip(reviews, predictions):
    sentiment = "POSITIVE" if prediction.item() == 1 else "NEGATIVE"

    print("\nReview:", review)
    print("Predicted Sentiment:", sentiment)
