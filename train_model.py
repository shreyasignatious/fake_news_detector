import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle, os, zipfile

# --- Sample mini datasets ---
fake_news = [
    "Aliens built the pyramids.",
    "NASA confirms the sun rises in the west.",
    "Scientists discover unicorn DNA.",
    "Coca-Cola cures all diseases.",
    "World ends tomorrow, officials confirm.",
]
true_news = [
    "NASA announces new mission to Mars.",
    "UN releases annual climate change report.",
    "Apple launches latest iPhone model.",
    "Scientists discover new species in Amazon rainforest.",
    "Government introduces digital literacy program.",
]

pd.DataFrame({"text": fake_news}).to_csv("Fake.csv", index=False)
pd.DataFrame({"text": true_news}).to_csv("True.csv", index=False)

# --- Load and label ---
df_fake = pd.read_csv("Fake.csv"); df_fake["label"] = 0
df_true = pd.read_csv("True.csv"); df_true["label"] = 1
df = pd.concat([df_fake, df_true]).sample(frac=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=300)
model.fit(X_train_tfidf, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test_tfidf)))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# --- Zip all outputs ---
with zipfile.ZipFile("fake_news_detector.zip", "w") as zipf:
    for f in ["Fake.csv", "True.csv", "model.pkl", "vectorizer.pkl"]:
        zipf.write(f)
print("\n✅ fake_news_detector.zip created!")
