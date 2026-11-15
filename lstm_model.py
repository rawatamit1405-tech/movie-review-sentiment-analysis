
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import text, sequence

print("STEP 1: Loading IMDB dataset...")

# Load datasets
train_data = pd.read_csv("dataset/train.csv")
test_data = pd.read_csv("dataset/test.csv")

# Shuffle training dataset
train_data = train_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nTraining dataset shuffled successfully!")
print(train_data["label"].value_counts())


print("\n========== DATASET CHECK ==========")

print("\nTraining dataset label counts:")
print(train_data["label"].value_counts())

print("\nTesting dataset label counts:")
print(test_data["label"].value_counts())

print("\n========== NEGATIVE REVIEWS ==========")

negative_reviews = train_data[train_data["label"] == 0]

for review in negative_reviews["text"].head(3):
    print("\n", review)

print("\n========== POSITIVE REVIEWS ==========")

positive_reviews = train_data[train_data["label"] == 1]

for review in positive_reviews["text"].head(3):
    print("\n", review)

print("\n========== CHECK COMPLETED ==========")


print("Training data:", train_data.shape)
print("Testing data:", test_data.shape)


print("\nSTEP 2: Preparing reviews...")

X_train = train_data["text"].astype(str)
X_test = test_data["text"].astype(str)

y_train = train_data["label"].values
y_test = test_data["label"].values



print("\nSTEP 3: Tokenizing reviews...")

# Maximum number of words
vocab_size = 10000

# Create tokenizer
tokenizer = text.Tokenizer(
    num_words=vocab_size,
    oov_token="<OOV>"
)

# Learn words from training reviews
tokenizer.fit_on_texts(X_train)

# Convert reviews into numbers
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

print("Tokenization completed!")
print("Sample numerical sequence:")
print(X_train_sequences[0][:20])


print("\nSTEP 4: Applying padding...")


print("\nSTEP 4: Applying padding...")

# Maximum review length
max_length = 200

X_train_padded = sequence.pad_sequences(
    X_train_sequences,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_test_padded = sequence.pad_sequences(
    X_test_sequences,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

print("Padding completed!")
print("Training shape:", X_train_padded.shape)
print("Testing shape:", X_test_padded.shape)

print("\nSTEP 5: Building LSTM model...")

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=64,
        input_length=max_length
    ),

    tf.keras.layers.LSTM(64),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation="sigmoid")

])


print("\nSTEP 6: Compiling model...")

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


print("\nSTEP 7: Training LSTM model...")

history = model.fit(
    X_train_padded,
    y_train,
    epochs=7,
    batch_size=64,
    validation_split=0.2
)


print("\nSTEP 8: Evaluating model...")

loss, accuracy = model.evaluate(
    X_test_padded,
    y_test
)

print("\nLSTM Test Accuracy:", accuracy * 100, "%")


print("\nLSTM TRAINING COMPLETED!")

print("\nSTEP 9: Saving LSTM model...")

model.save("models/lstm_sentiment_model.keras")

print("LSTM model saved successfully!")

import os
import joblib

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Save trained LSTM model
model.save("models/lstm_sentiment_model.keras")

# Save tokenizer
joblib.dump(tokenizer, "models/lstm_tokenizer.pkl")

print("LSTM model and tokenizer saved successfully!")
