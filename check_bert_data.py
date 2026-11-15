
import pandas as pd

train_data = pd.read_csv("dataset/train.csv")
test_data = pd.read_csv("dataset/test.csv")

print("Training dataset shape:", train_data.shape)
print("Testing dataset shape:", test_data.shape)

print("\nTraining dataset columns:")
print(train_data.columns)

print("\nSample reviews:")
print(train_data[["text", "label"]].head())

print("\nLabel distribution:")
print(train_data["label"].value_counts())
