# Hugging Face AI Streamlit Apps

This repository contains three interactive web applications built with Streamlit, showcasing the capabilities of Hugging Face Transformer models for Natural Language Processing (NLP), Computer Vision, and Text-to-Speech tasks.

## 🛠 Setup & Run Commands

Open your terminal or command prompt. Run the installation command first, and then execute the specific application you want to run directly from the same terminal:

```bash
# 1. Install all required dependencies
pip install streamlit transformers torch torchvision torchaudio Pillow pyttsx3

# 2. Run the Multilingual Text Generation & Sentiment App
streamlit run Multilingual_Sentiment_Gen.py

# 3. Run the Emotion Analysis & Text-to-Speech App
streamlit run Sentiment_Voice_Reader.py

# 4. Run the Visual Question Answering (VQA) & Summarization App
streamlit run VQA_to_Summary.py
