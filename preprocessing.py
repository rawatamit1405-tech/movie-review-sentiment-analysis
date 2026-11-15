
import pandas as pd
import re

# Load training dataset
data = pd.read_csv("dataset/train.csv")

# Function to clean movie reviews
def clean_text(text):

    # Convert text into lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning to reviews
data["cleaned_review"] = data["text"].apply(clean_text)

# Display original and cleaned reviews
print("\nOriginal Review:")
print(data["text"].iloc[0])

print("\nCleaned Review:")
print(data["cleaned_review"].iloc[0])

print("\nPreprocessing completed successfully!")
