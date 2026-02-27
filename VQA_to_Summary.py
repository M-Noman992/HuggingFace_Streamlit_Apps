import streamlit as st
from PIL import Image
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
vqa_pipe = pipeline("visual-question-answering", framework="pt")
model_name = "gpt2"
text_gen_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model="t5-small", framework="pt")
st.title("VQA, text generation and text summarization with AI")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    question = "What is in the picture?"
    with st.spinner("Analyzing the image..."):
        result = vqa_pipe(image, question, top_k=1)
    vqa_answer = result[0]['answer']
    st.write("Answer of the asked question:", vqa_answer)

prompt = (
    f"The image analysis suggests: '{vqa_answer}'. "
    f"Based on this, describe the image in detail, including the main objects, colors, "
    f"and any notable features present in the picture."
)
input_ids = tokenizer.encode(prompt, return_tensors="pt")
with st.spinner("Generating detailed description..."):
    output = text_gen_model.generate(
        input_ids,
        max_length=100, 
        top_p=0.90,      
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  
    )
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
st.write("Generated Text:")
st.write(generated_text)

with st.spinner("Summarizing the description..."):
    summary = summarizer(generated_text, max_length=20, min_length=10, do_sample=False)
    summary_text = summary[0]['summary_text']
    st.write("Summarized Text:")
    st.write(summary_text)
