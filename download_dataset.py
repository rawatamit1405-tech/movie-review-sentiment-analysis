
from datasets import load_dataset
import pandas as pd

print("Downloading IMDB dataset...")

# Download the IMDB dataset
dataset = load_dataset("imdb")

# Convert training and testing data into DataFrames
train_data = pd.DataFrame(dataset["train"])
test_data = pd.DataFrame(dataset["test"])

# Save datasets as CSV files
train_data.to_csv("imdb_train.csv", index=False)
test_data.to_csv("imdb_test.csv", index=False)

print("Dataset downloaded successfully!")

print("Training dataset:", train_data.shape)
print("Testing dataset:", test_data.shape)

print("Dataset files saved successfully!")

