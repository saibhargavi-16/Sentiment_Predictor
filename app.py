import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("🔥 Sentiment Analysis Dashboard")
st.write("Analyze text as Positive, Neutral, or Negative!")

# ---------------------------------------------------
# Load 3-class sentiment model (Positive / Neutral / Negative)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

label_mapping = {
    0: ("Negative", "😞"),
    1: ("Neutral", "😐"),
    2: ("Positive", "😀")
}

# ---------------------------------------------------
# Function to predict sentiment
# ---------------------------------------------------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(scores).item()
    label_text, emoji = label_mapping[label_id]
    confidence = scores[0][label_id].item()
    return label_text, emoji, confidence

# ---------------------------------------------------
# Single text UI
# ---------------------------------------------------
st.subheader("🔍 Analyze Single Text")
user_text = st.text_area("Enter text:")

if st.button("Analyze"):
    try:
        label, emoji, confidence = predict_sentiment(user_text)
        st.write(f"### **Prediction:** {emoji} {label}")
        st.write(f"### **Confidence:** {confidence:.3f}")

    except Exception as e:
        st.error(f"Error analyzing text: {e}")

# ---------------------------------------------------
# CSV upload
# ---------------------------------------------------
st.subheader("📂 Upload CSV")
uploaded_file = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column")
    else:
        st.info("Analyzing... please wait ⏳")

        df["prediction"] = df["text"].apply(lambda x: predict_sentiment(str(x))[0])
        df["emoji"] = df["text"].apply(lambda x: predict_sentiment(str(x))[1])
        df["confidence"] = df["text"].apply(lambda x: predict_sentiment(str(x))[2])

        st.success("Done!")
        st.dataframe(df[["text", "prediction", "emoji", "confidence"]])

        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv_download, "sentiment_results.csv", "text/csv")
