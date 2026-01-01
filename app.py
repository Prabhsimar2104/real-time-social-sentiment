import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))

# -----------------------------
# Text cleaning (same as training)
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🟦 Real-Time Social Media Sentiment Analyzer")
st.write("Type a tweet/post and get instant sentiment analysis")

user_input = st.text_area("Enter tweet text:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == "Positive":
            st.success(f"Sentiment: {prediction} 😊")
        elif prediction == "Negative":
            st.error(f"Sentiment: {prediction} 😠")
        else:
            st.info(f"Sentiment: {prediction} 😐")