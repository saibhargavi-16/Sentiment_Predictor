import streamlit as st
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Advanced Text + Emotion Analyzer", layout="wide")
st.title("🔥 Sentiment Text Analyzer (Sentiment • Emotion • Type • Idiom)")
st.write("Sentiment + Emotion + Sentence Type + Idiom Detection with emojis")

# -----------------------------
# Load sentiment (3-class) model
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

sent_tokenizer, sent_model = load_sentiment_model()

sent_label_mapping = {
    0: ("Negative", "😞"),
    1: ("Neutral", "😐"),
    2: ("Positive", "😀")
}

# -----------------------------
# Load emotion model
# -----------------------------
@st.cache_resource
def load_emotion_model():
    # small / good emotion model
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

emo_tokenizer, emo_model = load_emotion_model()

# Emotion -> emoji mapping (tweakable)
emotion_emoji = {
    "anger": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😀",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😮"
}

# ---------------------------------------------
# Idiom dictionary (you can extend this)
# ---------------------------------------------
idioms = {
    "a piece of cake": "Something very easy",
    "break the ice": "To make people comfortable",
    "hit the sack": "To go to sleep",
    "under the weather": "Feeling sick",
    "once in a blue moon": "Very rare",
    "bite the bullet": "Accept something difficult",
}

# ---------------------------------------------
# Helper functions
# ---------------------------------------------
def detect_sentence_type(text):
    text = text.strip()
    if text.endswith("?"):
        return "Question", "❓"
    elif text.endswith("!"):
        return "Exclamation", "😮"
    elif re.match(r"^(go|stop|listen|do|make|get|take|try|open|close)\b", text.lower()):
        return "Command", "⚡"
    else:
        return "Statement", "📘"

def detect_idiom(text):
    text_lower = text.lower()
    for phrase, meaning in idioms.items():
        if phrase in text_lower:
            return phrase, meaning, "✨"
    return "None", "No idiom detected", "—"

@torch.no_grad()
def predict_sentiment(text: str):
    # returns (label_text, emoji, confidence)
    inputs = sent_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = sent_model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    label_id = int(torch.argmax(scores, dim=1).item())
    label_text, emoji = sent_label_mapping[label_id]
    confidence = float(scores[0][label_id].item())
    return label_text, emoji, confidence

@torch.no_grad()
def predict_emotion(text: str):
    # returns (emotion_label, emoji, confidence)
    inputs = emo_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = emo_model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).squeeze()
    top_idx = int(torch.argmax(scores).item())
    label = emo_model.config.id2label[top_idx] if hasattr(emo_model.config, "id2label") else None
    # Some models store labels like 'LABEL_0' - map via label mapping if needed.
    # For j-hartmann/emotion-english-distilroberta-base the labels are strings like 'anger', 'joy', etc.
    emotion_label = label.lower() if isinstance(label, str) else str(label)
    emoji = emotion_emoji.get(emotion_label, "🙂")
    confidence = float(scores[top_idx].item())
    return emotion_label, emoji, confidence

# ---------------------------------------------
# UI - Single Text
# ---------------------------------------------
st.subheader("🔍 Analyze Single Text")
user_text = st.text_area("Enter text:")

if st.button("Analyze"):
    if not user_text.strip():
        st.error("Please enter some text.")
    else:
        try:
            # Sentiment
            sentiment, senti_emoji, senti_conf = predict_sentiment(user_text)

            # Emotion
            emotion, emo_emoji, emo_conf = predict_emotion(user_text)

            # Sentence type + idiom
            stype, type_emoji = detect_sentence_type(user_text)
            idiom_phrase, idiom_meaning, idiom_emoji = detect_idiom(user_text)

            st.write("### 🎯 Results")
            st.markdown(f"**Sentiment:** {senti_emoji} **{sentiment}** ({senti_conf:.3f})")
            st.markdown(f"**Emotion:** {emo_emoji} **{emotion.capitalize()}** ({emo_conf:.3f})")
            st.markdown(f"**Sentence Type:** {type_emoji} **{stype}**")
            st.markdown(f"**Idiom:** {idiom_emoji} **{idiom_phrase}**")
            st.markdown(f"**Meaning:** {idiom_meaning}")

        except Exception as e:
            st.error(f"Error analyzing text: {e}")

# ---------------------------------------------
# UI - CSV Upload
# ---------------------------------------------
st.subheader("📂 Upload CSV (with 'text' column)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = None

    if df is not None:
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            st.info("Analyzing... please wait ⏳")

            texts = df["text"].astype(str).tolist()

            # Predict sentiment, emotion, type, idiom for each text
            sentiments, sentiment_emojis, sentiment_conf = [], [], []
            emotions, emotion_emojis, emotion_conf = [], [], []
            sentence_types, type_emojis = [], []
            idiom_phrases, idiom_meanings, idiom_emojis = [], [], []

            for t in texts:
                s_label, s_emoji, s_conf = predict_sentiment(t)
                e_label, e_emoji, e_conf = predict_emotion(t)
                stype, st_emoji = detect_sentence_type(t)
                idiom_phrase, idiom_meaning, idiom_emoji = detect_idiom(t)

                sentiments.append(s_label)
                sentiment_emojis.append(s_emoji)
                sentiment_conf.append(s_conf)

                emotions.append(e_label)
                emotion_emojis.append(e_emoji)
                emotion_conf.append(e_conf)

                sentence_types.append(stype)
                type_emojis.append(st_emoji)

                idiom_phrases.append(idiom_phrase)
                idiom_meanings.append(idiom_meaning)
                idiom_emojis.append(idiom_emoji)

            # Attach to dataframe
            df["sentiment"] = sentiments
            df["sentiment_emoji"] = sentiment_emojis
            df["sentiment_confidence"] = sentiment_conf

            df["emotion"] = emotions
            df["emotion_emoji"] = emotion_emojis
            df["emotion_confidence"] = emotion_conf

            df["sentence_type"] = sentence_types
            df["type_emoji"] = type_emojis

            df["idiom"] = idiom_phrases
            df["idiom_meaning"] = idiom_meanings
            df["idiom_emoji"] = idiom_emojis

            st.success("Done!")
            st.dataframe(df)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv_data, "enhanced_emotion_results.csv", "text/csv")

