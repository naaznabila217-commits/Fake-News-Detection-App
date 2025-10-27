# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# --------------------------
# Helper functions
# --------------------------
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

def predict_news(text, model, vectorizer):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(vectorized))
    else:
        confidence = None
    return prediction, confidence

# --------------------------
# Load model
# --------------------------
model, vectorizer = load_model()

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Sidebar
st.sidebar.title("Options")
news_type = st.sidebar.selectbox("Choose News Type:", ["Global News", "Local News"])
country = st.sidebar.text_input("Country code (for local news, e.g., 'in')", value="in")

st.title("📰 Fake News Detection System")
st.markdown("Enter any news text below to check if it's **Fake** or **Real**!")

# Text input
user_input = st.text_area("Enter News Here:")

# --------------------------
# Live News Fetch
# --------------------------
API_KEY = "02aeba635b80ceb5afeff0b27d518518"  # Replace with your NewsAPI key
if st.sidebar.button("Fetch Latest News"):
    if news_type == "Global News":
        url = f"https://newsapi.org/v2/top-headlines?language=en&apiKey={API_KEY}"
    else:
        url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={API_KEY}"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    
    st.subheader("Latest News:")
    for art in articles[:5]:
        st.markdown(f"**[{art['title']}]({art['url']})**")
        if art['urlToImage']:
            st.image(art['urlToImage'], width=400)
        st.write(art['description'])
        st.write("---")

# --------------------------
# Predict Button
# --------------------------
if st.button("Check News"):
    if user_input.strip() != "":
        pred, conf = predict_news(user_input, model, vectorizer)
        st.write(f"**Prediction:** {'Fake' if pred=='FAKE' else 'Real'}")
        if conf:
            st.write(f"**Confidence Score:** {conf*100:.2f}%")
        # Modification detection (simple check)
        if len(user_input.split()) > 50:
            st.info("⚠️ Long text, may have been edited or aggregated.")
        # Save history
        history = pd.DataFrame({"News": [user_input], "Prediction": [pred], "Time": [datetime.now()]})
        history.to_csv("prediction_history.csv", mode="a", header=False, index=False)
        st.success("✅ Prediction saved to history")
    else:
        st.warning("Please enter some news text first.")

# --------------------------
# Prediction History & Charts
# --------------------------
if st.sidebar.checkbox("Show Prediction History"):
    try:
        hist = pd.read_csv("prediction_history.csv", names=["News", "Prediction", "Time"])
        st.subheader("Prediction History")
        st.dataframe(hist.tail(10))
        chart_data = hist['Prediction'].value_counts()
        st.bar_chart(chart_data)
    except:
        st.info("No history found yet.")

