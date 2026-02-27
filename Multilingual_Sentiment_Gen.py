import streamlit as st
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
text_gen_model = GPT2LMHeadModel.from_pretrained("distilgpt2")  
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
sentiment_analyzer = pipeline("sentiment-analysis")
translator = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru")
st.title("Text generation, sentiment analysis, and translation with AI")
user_input = st.text_input("Enter text:")
if user_input.strip():
    prompt = f"{user_input}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with st.spinner("Generating text..."):
        output = text_gen_model.generate(
            input_ids,
            max_length=50,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write("Generated Text:")
    st.write(generated_text)

    with st.spinner("Analyzing sentiment..."):
        sentiment_result = sentiment_analyzer(user_input)
    sentiment = sentiment_result[0]['label']
    st.write("Sentiment Analysis:")
    st.write(f"Sentiment: {sentiment}")

    with st.spinner("Translating text..."):
        translation_result = translator(generated_text[:256]) 
    translated_text = translation_result[0]['translation_text']
    st.write("Translation (English to Russian):")
    st.write(translated_text)
