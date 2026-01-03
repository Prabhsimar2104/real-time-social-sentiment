# Real-Time Social Media Sentiment Analyzer

## Project Overview
This project implements a Real-Time Social Media Sentiment Analyzer that classifies social media-style text (tweets/posts) into Positive, Negative, or Neutral sentiment using Machine Learning. The system demonstrates real-time data analytics, preprocessing, model training, evaluation, and deployment.

---

## Hosted Application (Public Link)
The application is deployed online using Streamlit Community Cloud.

Public URL:
https://real-time-social-sentiment.streamlit.app/

This link allows instant evaluation of the project without any local setup.

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- NLTK

---

## Project Structure
- app.py – Streamlit application for real-time sentiment prediction  
- preprocessing.py – Data cleaning and preprocessing  
- train_model.py – Model training script  
- evaluation.py – Model evaluation and performance analysis  
- eda_twitter_sentiment.py – Exploratory Data Analysis  
- model/ – Saved trained model and TF-IDF vectorizer  
- clean_twitter_train.csv – Cleaned training dataset  
- clean_twitter_val.csv – Cleaned validation dataset  

---

## How to Run the Project Locally

### Step 1: Install Python
Ensure Python 3.9 or above is installed.

Check version:
python --version

yaml
Copy code

---

### Step 2: (Optional) Create and Activate Virtual Environment

Create virtual environment:
python -m venv venv

Windows:
venv\Scripts\activate

macOS / Linux:
source venv/bin/activate

---

### Step 3: Install Required Dependencies

pip install -r requirements.txt

---

### Step 4: Run the Streamlit Application

streamlit run app.py

If the streamlit command is not recognized:
python -m streamlit run app.py

The application will open in a browser at:
http://localhost:8501

---

## Model Performance
The Logistic Regression model achieved approximately 85% validation accuracy. Training and validation performance indicate good generalization with minimal overfitting.

---

## AI Exploration Experiment Summary
The deployed model was tested with unexpected inputs such as sarcasm, mixed sentiments, slang, and noisy text. The model performed well on standard inputs but showed limitations with sarcasm and highly informal language, highlighting real-world challenges in sentiment analysis.

---

## Notes for Evaluator
- The hosted version is recommended for quick evaluation using the public URL.
- Local execution steps are provided for reproducibility.
- The system performs real-time sentiment analysis with low latency.

---

## Student Details
Name: Prabhsimar Singh  
Roll No.: 102483078
Branch: COE
