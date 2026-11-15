
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# STEP 1: Load trained LSTM model
print("Loading LSTM model...")

model = tf.keras.models.load_model(
    "models/lstm_sentiment_model.keras"
)

# STEP 2: Load tokenizer
tokenizer = joblib.load(
    "models/lstm_tokenizer.pkl"
)

print("Model and tokenizer loaded successfully!")

# STEP 3: Maximum review length
max_length = 200

# STEP 4: Take review from user
review = input("\nEnter a movie review: ")

# STEP 5: Convert review into sequences
review_sequence = tokenizer.texts_to_sequences([review])

# STEP 6: Apply padding
review_padded = pad_sequences(
    review_sequence,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

# STEP 7: Make prediction
prediction = model.predict(review_padded)

# STEP 8: Display prediction score
score = float(prediction[0][0])

print("\nRaw prediction score:", score)

# STEP 9: Display sentiment
if score >= 0.5:
    print("Sentiment: POSITIVE")
    print("Predicted class: 1")
else:
    print("Sentiment: NEGATIVE")
    print("Predicted class: 0")

