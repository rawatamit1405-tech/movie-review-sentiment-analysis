
import streamlit as st
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="MovieSentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS DESIGN ----------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0b1120, #111827);
    color: #f8fafc;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

section[data-testid="stSidebar"] p {
    color: #cbd5e1;
}

/* Headings */
h1, h2, h3 {
    color: #f8fafc !important;
}

/* Paragraphs */
p {
    color: #cbd5e1;
}

/* Text area */
.stTextArea textarea {
    background-color: #1e293b !important;
    color: white !important;
    border: 1px solid #475569 !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.5rem;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
}

/* Custom cards */
.custom-card {
    background: #1e293b;
    padding: 22px;
    border: 1px solid #334155;
    border-radius: 16px;
    margin-bottom: 15px;
}

.hero-card {
    background: linear-gradient(120deg, #312e81, #1e3a8a);
    padding: 35px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.positive-card {
    background: #064e3b;
    border: 1px solid #10b981;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.negative-card {
    background: #7f1d1d;
    border: 1px solid #ef4444;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.small-label {
    color: #94a3b8;
    font-size: 14px;
}

.metric-number {
    color: #a5b4fc;
    font-size: 30px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD BERT MODEL ----------------

@st.cache_resource
def load_bert_model():

    model_path = "models/bert_sentiment_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_bert_model()


# ---------------- SIDEBAR ----------------

st.sidebar.markdown("# 🎬 MovieSentiment")
st.sidebar.caption("AI Powered Review Analyzer")

st.sidebar.divider()

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Home",
        "Sentiment Prediction",
        "Model Performance",
        "About Project"
    ]
)


page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Sentiment Prediction",
        "Model Performance",
        "About Project"
    ]
)


# ---------------- HOME PAGE ----------------

if page == "Home":

    st.markdown("""
    <div class="hero-card">
        <h1>🎬 Movie Review Intelligence</h1>
        <h3>Understand the emotion behind every review.</h3>
        <p>
        An AI-powered sentiment analysis system that
        classifies movie reviews as positive or negative
        using Natural Language Processing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="small-label">Total Dataset</div>
            <div class="metric-number">50,000</div>
            <p>Movie Reviews</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="small-label">Best Model</div>
            <div class="metric-number">BERT</div>
            <p>Transformer Architecture</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="custom-card">
            <div class="small-label">Test Accuracy</div>
            <div class="metric-number">91.19%</div>
            <p>25,000 Test Reviews</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("⚙️ How It Works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("""
        <div class="custom-card">
            <h3>01. Input</h3>
            <p>Enter your movie review into the application.</p>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div class="custom-card">
            <h3>02. Analysis</h3>
            <p>BERT processes the text and understands its context.</p>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div class="custom-card">
            <h3>03. Prediction</h3>
            <p>Get the predicted sentiment and confidence score.</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- PREDICTION PAGE ----------------

elif page == "Sentiment Prediction":

    st.title("💬 Sentiment Analyzer")

    st.write(
        "Enter a movie review and let our AI model "
        "analyze its sentiment."
    )

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)

    review = st.text_area(
        "Your Movie Review",
        placeholder="Example: The movie was amazing, the acting was brilliant...",
        height=180
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("✨ Analyze Sentiment", type="primary"):

        if not review.strip():

            st.warning("Please enter a movie review first.")

        else:

            with st.spinner("AI is analyzing your review..."):

                inputs = tokenizer(
                    review,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )

                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                with torch.no_grad():
                    outputs = model(**inputs)

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=1
                )

                prediction = torch.argmax(
                    probabilities,
                    dim=1
                ).item()
                prediction = int(prediction)

                confidence = (
                    probabilities[0][prediction].item() * 100
                )

            st.divider()

            st.subheader("📌 Analysis Result")

            if prediction == 1:

                st.markdown("""
                <div class="positive-card">
                    <h1>😊 POSITIVE</h1>
                    <h3>Your review expresses a positive sentiment.</h3>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="negative-card">
                    <h1>😞 NEGATIVE</h1>
                    <h3>Your review expresses a negative sentiment.</h3>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(confidence / 100)

            st.caption(
                "Confidence represents the model's predicted "
                "probability for the selected class, not a guarantee."
            )


# ---------------- PERFORMANCE PAGE ----------------


elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Comparison of four Machine Learning and "
        "Deep Learning models evaluated on the complete "
        "IMDB test dataset."
    )

    # Model accuracy results
    performance = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "LSTM",
            "BiLSTM",
            "BERT"
        ],
        "Accuracy": [
            88.29,
            79.61,
            84.512,
            91.19
        ]
    })

    # Sort models from highest to lowest accuracy
    performance = performance.sort_values(
        by="Accuracy",
        ascending=False
    ).reset_index(drop=True)

    st.divider()

    # Best model information
    st.subheader("🏆 Best Performing Model")

    st.success(
        "BERT achieved the highest test accuracy of 91.19%."
    )

    # Accuracy chart
    st.subheader("📈 Model Accuracy Comparison")

    st.bar_chart(
        performance,
        x="Model",
        y="Accuracy",
        width="stretch"
    )

    st.caption(
        "Accuracy is measured on the complete test dataset "
        "containing 25,000 movie reviews."
    )

    st.divider()

    # Accuracy table
    st.subheader("📋 Accuracy Details")

    st.dataframe(
        performance,
        hide_index=True,
        width="stretch"
    )

    # Individual model results
    st.subheader("🔍 Individual Model Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Logistic Regression",
            "88.29%"
        )

        st.metric(
            "LSTM",
            "79.61%"
        )

    with col2:
        st.metric(
            "BiLSTM",
            "84.512%"
        )

        st.metric(
            "BERT",
            "91.19%"
        )




# ---------------- ABOUT PAGE ----------------

elif page == "About Project":

    st.title("📚 About This Project")

    st.markdown("""
    <div class="custom-card">
        <h3>Movie Review Sentiment Analysis</h3>
        <p>
        This project uses Machine Learning, Deep Learning
        and Natural Language Processing to identify the
        sentiment of movie reviews.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🛠️ Technologies Used")

    technologies = [
        "Python",
        "Pandas and NumPy",
        "Scikit-learn",
        "TensorFlow and Keras",
        "PyTorch",
        "Hugging Face Transformers",
        "BERT",
        "Streamlit"
    ]

    for item in technologies:
        st.write("✓", item)

    st.subheader("📂 Dataset Information")

    st.write("""
    - Dataset: IMDB Movie Reviews
    - Total reviews: 50,000
    - Training reviews: 25,000
    - Testing reviews: 25,000
    - Categories: Positive and Negative
    """)

    st.subheader("🏆 Best Performing Model")

    st.success("BERT — 91.19% Test Accuracy")


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<center><p>🎬 MovieSentiment AI | "
    "Machine Learning & Deep Learning Project</p></center>",
    unsafe_allow_html=True
)

