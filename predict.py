
import re
import joblib

print("Loading trained model...")

# Load saved model and TF-IDF vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("Model loaded successfully!")


# Clean the user's review
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("\n===== MOVIE REVIEW SENTIMENT ANALYSIS =====")

# Take review from the user
review = input("\nEnter your movie review: ")

if not review.strip():
    print("Please enter a valid review.")

else:
    # Clean the review
    cleaned_review = clean_text(review)

    # Convert review into numerical form
    review_vector = vectorizer.transform([cleaned_review])

    # Predict sentiment
    prediction = model.predict(review_vector)

    # Display result
    if prediction[0] == 1:
        print("\nSentiment: POSITIVE 😊")
    else:
        print("\nSentiment: NEGATIVE 😞")

    # Display prediction confidence
    probabilities = model.predict_proba(review_vector)[0]
    confidence = max(probabilities) * 100

    print("Prediction confidence:", round(confidence, 2), "%")
