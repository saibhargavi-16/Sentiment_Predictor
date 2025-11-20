# train.py
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords

# Uncomment if running for the first time
# nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@', ' ', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(tokens)

def main():
    os.makedirs('models', exist_ok=True)

    csv_file = 'sample_tweets.csv'
    df = pd.read_csv(csv_file)

    df['clean_text'] = df['text'].fillna('').astype(str).apply(clean_text)

    X = df['clean_text']
    y = df['label'].copy()

    # Convert labels to 0/1
    if y.dtype == object:
        y = y.map(lambda v: 1 if str(v).lower() in ['positive', 'pos', '1', 'yes'] else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    print("Test Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    joblib.dump(model, 'models/lr_model.joblib')

    print("Model + vectorizer saved inside /models folder.")

if __name__ == '__main__':
    main()
