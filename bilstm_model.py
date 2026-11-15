import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
# Load training and testing datasets

train_data = pd.read_csv("dataset/train.csv")
test_data = pd.read_csv("dataset/test.csv")

print("Training dataset:", train_data.shape)
print("Testing dataset:", test_data.shape)

# Shuffle training data

train_data = train_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Separate reviews and labels

X_train = train_data["text"].astype(str)
y_train = train_data["label"].values

X_test = test_data["text"].astype(str)
y_test = test_data["label"].values

# Tokenization

vocab_size = 10000
max_length = 200

tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(
    X_train_sequences,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

X_test_padded = pad_sequences(
    X_test_sequences,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

print("Tokenization completed!")

# Build BiLSTM model

model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_length
    ),

    Bidirectional(
        LSTM(64)
    ),

    Dense(32, activation="relu"),

    Dropout(0.5),

    Dense(1, activation="sigmoid")
])

# Compile model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nBiLSTM Model Summary:")
model.summary()

# Train BiLSTM model

# Train improved BiLSTM model

epochs = 10
batch_size = 64

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

# Save the best model
model_checkpoint = ModelCheckpoint(
    "models/bilstm_best.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
# Save improved BiLSTM model

model.save("models/bilstm_sentiment_model.keras")

joblib.dump(
    tokenizer,
    "models/bilstm_tokenizer.pkl"
)

print("\nImproved BiLSTM model saved successfully!")

# Train model
history = model.fit(
    X_train_padded,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Save BiLSTM model

model.save("models/bilstm_sentiment_model.keras")

joblib.dump(
    tokenizer,
    "models/bilstm_tokenizer.pkl"
)

print("\nBiLSTM model and tokenizer saved successfully!")