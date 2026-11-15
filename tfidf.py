
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer

print("STEP 1: Loading dataset...")

# Load training and testing data
train_data = pd.read_csv("dataset/train.csv")
test_data = pd.read_csv("dataset/test.csv")


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("STEP 2: Cleaning movie reviews...")

# Clean training and testing reviews
X_train = train_data["text"].apply(clean_text)
X_test = test_data["text"].apply(clean_text)

# Extract sentiment labels
y_train = train_data["label"]
y_test = test_data["label"]


print("STEP 3: Applying TF-IDF...")

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=20000)

# Learn vocabulary and convert training reviews
X_train_tfidf = vectorizer.fit_transform(X_train)

# Convert testing reviews using the same vocabulary
X_test_tfidf = vectorizer.transform(X_test)


print("\nSTEP 4: TF-IDF completed successfully!")

print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)

print("\nTotal vocabulary words:",
      len(vectorizer.get_feature_names_out()))

# Display some words learned by TF-IDF
print("\nFirst 20 vocabulary words:")
print(vectorizer.get_feature_names_out()[:20])

# Display the numerical representation of first review
print("\nFirst review numerical representation:")
print(X_train_tfidf[0])

print("\nAll TF-IDF operations completed!")
