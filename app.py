import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("🔥 Sentiment Analysis Dashboard")
st.write("Analyze the text as positive and negative!")

# Load BERT model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# -------------------------
# Single text input
# -------------------------
st.subheader("🔍 Analyze Single Text")

user_text = st.text_area("Enter text:")

if st.button("Analyze"):
    try:
        result = classifier(user_text)[0]
        label = result["label"]
        score = result["score"]

        st.write(f"### **Prediction:** {label}")
        st.write(f"### **Confidence:** {score:.3f}")

    except Exception as e:
        st.error("Error analyzing text.")

# -------------------------
# CSV Upload
# -------------------------
st.subheader("📂 Upload CSV")

uploaded_file = st.file_uploader("Upload CSV containing a 'text' column", type=["csv"])

if uploaded_file:
    import pandas as pd

    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a column named 'text'")
    else:
        st.info("Analyzing... please wait ⏳")

        df["result"] = df["text"].apply(lambda x: classifier(str(x))[0])
        df["label"] = df["result"].apply(lambda x: x["label"])
        df["confidence"] = df["result"].apply(lambda x: x["score"])

        st.success("Done!")
        st.dataframe(df[["text", "label", "confidence"]])

        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results", csv_download, "sentiment_results.csv", "text/csv")

