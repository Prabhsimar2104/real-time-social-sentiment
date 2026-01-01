import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load the datasets
# -----------------------------
train_df = pd.read_csv("C:\\Users\\HP\\Desktop\\real_time_social_sentiment\\data\\twitter_training.csv", header=None)
val_df   = pd.read_csv("C:\\Users\\HP\\Desktop\\real_time_social_sentiment\\data\\twitter_validation.csv", header=None)

# Assign column names
train_df.columns = ["ID", "Entity", "Sentiment", "Text"]
val_df.columns   = ["ID", "Entity", "Sentiment", "Text"]

print("Training Data Shape:", train_df.shape)
print("Validation Data Shape:", val_df.shape)

# -----------------------------
# 2. View basic information
# -----------------------------
print("\nDataset Info:")
print(train_df.info())

print("\nFirst 5 rows:")
print(train_df.head())

# -----------------------------
# 3. Check sentiment labels
# -----------------------------
print("\nUnique Sentiments:")
print(train_df["Sentiment"].unique())

# -----------------------------
# 4. Sentiment Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x="Sentiment")
plt.title("Sentiment Distribution in Training Data")
plt.show()

# -----------------------------
# 5. Show sample tweets
# -----------------------------
print("\nSample Tweets:\n")
for i in range(5):
    print(f"Tweet {i+1}:")
    print(train_df["Text"].iloc[i])
    print("-" * 50)

# -----------------------------
# 6. Text Length Analysis
# -----------------------------
train_df["Text_Length"] = train_df["Text"].astype(str).apply(len)

plt.figure(figsize=(6,4))
plt.hist(train_df["Text_Length"], bins=50)
plt.title("Tweet Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.show()

print("\nText Length Statistics:")
print(train_df["Text_Length"].describe())

# -----------------------------
# 7. Identify Noise Examples
# -----------------------------
print("\nTweets with URLs / Mentions / Hashtags:\n")

noise_samples = train_df[
    train_df["Text"].str.contains("http|@|#", na=False)
].head(5)

for i, text in enumerate(noise_samples["Text"], start=1):
    print(f"Noise Tweet {i}:")
    print(text)
    print("-" * 50)
