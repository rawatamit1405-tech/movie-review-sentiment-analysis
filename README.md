# Movie Review Sentiment Analysis 🎬

A Machine Learning and Deep Learning project that analyzes movie reviews and predicts whether the sentiment expressed is **Positive** or **Negative**.

The project compares four different models: Logistic Regression, LSTM, BiLSTM, and BERT. It also provides an interactive web application built using Streamlit.

## 🚀 Project Overview

The main objective of this project is to understand how Natural Language Processing (NLP) can be used to classify movie reviews automatically.

The system accepts a movie review as input, processes the text, and predicts its sentiment using a trained BERT model.

## ✨ Features

- Predicts positive and negative movie reviews.
- Uses four different Machine Learning and Deep Learning models.
- Compares model performance using accuracy.
- Displays positive and negative prediction probabilities.
- Provides an interactive Streamlit web interface.
- Uses the IMDB movie review dataset containing 50,000 reviews.

## 🧠 Models Used

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 88.29% |
| LSTM | 79.61% |
| BiLSTM | 84.512% |
| BERT | 91.19% |

**Best Performing Model:** BERT achieved the highest test accuracy of 91.19%.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow and Keras
- PyTorch
- Hugging Face Transformers
- BERT
- Streamlit
- Natural Language Processing (NLP)

## 📂 Project Structure

```text
movie_sentiment_data/
│
├── app.py
├── bert_model.py
├── bilstm_model.py
├── lstm_model.py
├── model_training.py
├── preprocessing.py
├── tokenization.py
├── tfidf.py
├── predict.py
├── predict_lstm.py
├── evaluate_bert.py
├── evaluate_logistic.py
├── check_bert.py
├── check_bilstm.py
├── check_lstm.py
├── download_dataset.py
├── requirements.txt
├── README.md
│
├── dataset/
├── datasets/
└── models/
```

*Note: Dataset and trained model files are excluded from this GitHub repository because of their size. They must be downloaded or generated before running the application.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rawatamit1405-tech/movie-review-sentiment-analysis.git
```

### 2. Open the project folder

```bash
cd movie-review-sentiment-analysis
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS:

```bash
source .venv/bin/activate
```

### 4. Install the required libraries

```bash
pip install -r requirements.txt
```

### 5. Download the dataset and prepare the trained model

Use the provided dataset download and model training scripts. The trained BERT model must be saved in the `models/bert_sentiment_model/` directory before launching the application.

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 📊 Dataset Information

- Dataset: IMDB Movie Reviews
- Total Reviews: 50,000
- Training Reviews: 25,000
- Testing Reviews: 25,000
- Sentiment Classes: Positive and Negative

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Natural Language Processing.
- Text preprocessing and tokenization.
- Machine Learning model development.
- Deep Learning using LSTM and BiLSTM.
- Transformer-based sentiment classification using BERT.
- Model evaluation and performance comparison.
- Building interactive applications using Streamlit.
- Managing a project using Git and GitHub.

## 👨‍💻 Author

**A B.Tech Engineering Student**

GitHub: [rawatamit1405-tech](https://github.com/rawatamit1405-tech)

---

*This project was developed for learning and academic purposes.*