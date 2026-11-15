
import pandas as pd
import re

# Load the dataset
data = pd.read_csv("dataset/train.csv")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Clean the reviews
data["cleaned_review"] = data["text"].apply(clean_text)

# Tokenization function
def tokenize_text(text):
    tokens = text.split()
    return tokens

# Apply tokenization to the first review
sample_review = data["cleaned_review"].iloc[0]

tokens = tokenize_text(sample_review)

print("Original Review:")
print(data["text"].iloc[0])

print("\nCleaned Review:")
print(sample_review)

print("\nTokens:")
print(tokens)

print("\nTotal number of tokens:", len(tokens))
