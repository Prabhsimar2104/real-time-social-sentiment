import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load cleaned data
# -----------------------------
df = pd.read_csv("clean_twitter_train.csv")

df = df.dropna(subset=["Clean_Text"])
df = df[df["Clean_Text"].str.strip() != ""]

X = df["Clean_Text"]
y = df["Sentiment"]

# -----------------------------
# 2. Train-validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)

# -----------------------------
# 4. Train model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# 5. Accuracy calculation
# -----------------------------
train_acc = accuracy_score(y_train, model.predict(X_train_tfidf))
val_acc   = accuracy_score(y_val, model.predict(X_val_tfidf))

print("Training Accuracy:", train_acc)
print("Validation Accuracy:", val_acc)

# -----------------------------
# 6. Plot comparison
# -----------------------------
plt.bar(["Training Accuracy", "Validation Accuracy"], [train_acc, val_acc])
plt.title("Training vs Validation Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()