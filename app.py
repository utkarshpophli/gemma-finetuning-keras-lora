import streamlit as st
from src.model import load_gemma_model, enable_lora
from src.utils import generate_text, format_prompt
from config import Config

config = Config()

@st.cache_resource
def load_model():
    model = load_gemma_model(config.MODEL_PRESET)
    model = enable_lora(model, config.LORA_RANK)
    model.load_weights(config.MODEL_SAVE_PATH)
    return model

st.title("Gemma Fine-tuned with LoRA")

model = load_model()

instruction = st.text_area("Enter your instruction:", height=100)
max_length = st.slider("Max response length:", min_value=50, max_value=500, value=256)

if st.button("Generate"):
    if instruction:
        with st.spinner("Generating response..."):
            prompt = format_prompt(instruction)
            response = generate_text(model, prompt, max_length=max_length)
            st.write("Generated Response:")
            st.write(response)
    else:
        st.warning("Please enter an instruction.")