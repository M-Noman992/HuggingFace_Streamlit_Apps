import streamlit as st
from PIL import Image
from transformers import pipeline
import pyttsx3

emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
text_to_speech_engine = pyttsx3.init()

st.title("Emotion Analysis, Sentiment Analysis, and Text-to-Speech with AI")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    description = "Emotion Analysis, Sentiment Analysis, and Text-to-Speech."

    with st.spinner("Analyzing the emotion..."):
        emotion_result = emotion_model(description)
    emotion_label = emotion_result[0]['label']
    st.write(f"Detected Emotion: **{emotion_label}**")

    with st.spinner("Analyzing sentiment..."):
        sentiment_result = sentiment_model(description)
    sentiment_label_id = sentiment_result[0]['label']
    sentiment_score = sentiment_result[0]['score']

    sentiment_labels = {
        'LABEL_0': 'NEGATIVE',
        'LABEL_1': 'NEUTRAL',
        'LABEL_2': 'POSITIVE'
    }
    sentiment_label = sentiment_labels.get(sentiment_label_id, 'UNKNOWN')
    st.write(f"Sentiment Analysis Result: **{sentiment_label}** (Score: {sentiment_score:.2f})")

    with st.spinner("Converting text to speech..."):
        text_to_speech_engine.say(description)
        text_to_speech_engine.runAndWait()
    st.write("**Text-to-Speech completed.**")

