
import pandas as pd
import os
from datasets import load_dataset

print("STEP 1: Starting dataset download...", flush=True)

print("STEP 2: Connecting to Hugging Face...", flush=True)

# Correct dataset repository name
dataset = load_dataset("stanfordnlp/imdb")

print("STEP 3: Dataset downloaded successfully!", flush=True)

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

print("STEP 4: Converting dataset into DataFrames...", flush=True)

train_data = pd.DataFrame(dataset["train"])
test_data = pd.DataFrame(dataset["test"])

print("STEP 5: Saving CSV files...", flush=True)

train_data.to_csv("dataset/train.csv", index=False)
test_data.to_csv("dataset/test.csv", index=False)

print("\nSUCCESS! Dataset saved.")

print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

print("\nFirst review:")
print(train_data["text"].iloc[0])

print("\nFirst review label:")
print(train_data["label"].iloc[0])

import os

print("Current working directory:", os.getcwd())
print("Dataset folder location:", os.path.abspath("dataset"))

