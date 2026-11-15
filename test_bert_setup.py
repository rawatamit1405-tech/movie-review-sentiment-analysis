
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading BERT tokenizer...")

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Tokenizer loaded successfully!")

print("Loading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

print("BERT model loaded successfully!")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

print("Selected device:", device)
print("BERT setup completed successfully!")
