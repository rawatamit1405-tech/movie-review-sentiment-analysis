
import os
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
train_data = pd.read_csv("dataset/train.csv")

print("Dataset loaded!")

# 2. Load BERT tokenizer
model_name = "bert-base-uncased"
max_length = 256

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Tokenize training reviews
train_encodings = tokenizer(
    train_data["text"].astype(str).tolist(),
    truncation=True,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt"
)

train_labels = torch.tensor(
    train_data["label"].values,
    dtype=torch.long
)

print("Training data tokenized!")

# 4. Create dataset
dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_labels
)

# 5. Split training and validation data
indices = list(range(len(dataset)))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=train_data["label"]
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# 6. Create DataLoaders
batch_size = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size
)

print("Training and validation loaders created!")

# 7. Load BERT classification model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 8. Select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

print("Model loaded successfully!")
print("Training device:", device)

# 9. Set optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

# 10. Training settings
epochs = 2
best_accuracy = 0.0

os.makedirs("models", exist_ok=True)

# 11. Start training
for epoch in range(epochs):

    model.train()
    total_loss = 0

    print(f"\nStarting Epoch {epoch + 1}/{epochs}")

    for batch_number, batch in enumerate(train_loader):

        input_ids, attention_mask, labels = [
            item.to(device) for item in batch
        ]

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        if (batch_number + 1) % 100 == 0:
            print(
                f"Batch {batch_number + 1}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    # 12. Validation
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in val_loader:

            input_ids, attention_mask, labels = [
                item.to(device) for item in batch
            ]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    val_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print(f"\nEpoch {epoch + 1} completed!")
    print(f"Average Training Loss: {average_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # 13. Save the best model
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        model.save_pretrained("models/bert_sentiment_model")
        tokenizer.save_pretrained("models/bert_sentiment_model")

        print("Best BERT model saved!")

print("\nBERT training completed!")
print(f"Best Validation Accuracy: {best_accuracy * 100:.2f}%")


