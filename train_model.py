import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load cleaned data
# -----------------------------
train_df = pd.read_csv("clean_twitter_train.csv")
val_df   = pd.read_csv("clean_twitter_val.csv")

# 🔥 FIX: Remove empty / NaN text rows
train_df = train_df.dropna(subset=["Clean_Text"])
val_df   = val_df.dropna(subset=["Clean_Text"])

train_df = train_df[train_df["Clean_Text"].str.strip() != ""]
val_df   = val_df[val_df["Clean_Text"].str.strip() != ""]

X_train = train_df["Clean_Text"]
y_train = train_df["Sentiment"]

X_val = val_df["Clean_Text"]
y_val = val_df["Sentiment"]

# -----------------------------
# 2. TF-IDF Feature Extraction
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)

# -----------------------------
# 3. Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# 4. Evaluate Model
# -----------------------------
y_pred = model.predict(X_val_tfidf)

accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))

# -----------------------------
# 5. Save model & vectorizer
# -----------------------------
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")