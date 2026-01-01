import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (run once)
nltk.download('stopwords')

# -----------------------------
# 1. Load dataset
# -----------------------------
train_df = pd.read_csv("C:\\Users\\HP\\Desktop\\real_time_social_sentiment\\data\\twitter_training.csv", header=None)
val_df   = pd.read_csv("C:\\Users\\HP\\Desktop\\real_time_social_sentiment\\data\\twitter_validation.csv", header=None)

train_df.columns = ["ID", "Entity", "Sentiment", "Text"]
val_df.columns   = ["ID", "Entity", "Sentiment", "Text"]

# -----------------------------
# 2. Remove 'Irrelevant' tweets
# -----------------------------
train_df = train_df[train_df["Sentiment"] != "Irrelevant"]
val_df   = val_df[val_df["Sentiment"] != "Irrelevant"]

# -----------------------------
# 3. Text cleaning function
# -----------------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()                     # lowercase
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)             # remove mentions
    text = re.sub(r"#\w+", "", text)             # remove hashtags
    text = re.sub(r"[^a-z\s]", "", text)         # remove punctuation & numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)

# -----------------------------
# 4. Apply preprocessing
# -----------------------------
train_df["Clean_Text"] = train_df["Text"].apply(clean_text)
val_df["Clean_Text"]   = val_df["Text"].apply(clean_text)

# -----------------------------
# 5. Show before vs after
# -----------------------------
print("\nBEFORE CLEANING:")
print(train_df["Text"].iloc[0])

print("\nAFTER CLEANING:")
print(train_df["Clean_Text"].iloc[0])

# -----------------------------
# 6. Save cleaned data
# -----------------------------
train_df.to_csv("clean_twitter_train.csv", index=False)
val_df.to_csv("clean_twitter_val.csv", index=False)

print("\nPreprocessing complete. Clean files saved.")