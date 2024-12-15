from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import streamlit as st
import torch
import os


# Load or save multilingual model
def load_multilingual_model():
    """
    Loads the multilingual model for translation. If the model is not already
    downloaded, it downloads the model and tokenizer from the Hugging Face
    model hub, saves them locally, and then loads them. If the model is already
    downloaded, it loads the model and tokenizer from the local directory.
    Returns:
        tuple: A tuple containing the tokenizer and the model.
    """
    model_name = "facebook/m2m100_418M"
    model_dir = os.path.join("models", model_name.replace("/", "_"))

    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        st.sidebar.write("Downloading model for the first time...")

        # Save tokenizer and model locally
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    else:
        st.sidebar.write("Loading model from saved directory...")
        tokenizer = M2M100Tokenizer.from_pretrained(model_dir)
        model = M2M100ForConditionalGeneration.from_pretrained(model_dir)

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model


# Translate text
def translate_multilingual(text, tokenizer, model, source_lang, target_lang):
    """
    Translates text from a source language to a target language using a multilingual model.

    Args:
        text (str): The text to be translated.
        tokenizer (PreTrainedTokenizer): The tokenizer to preprocess the text.
        model (PreTrainedModel): The multilingual model to perform the translation.
        source_lang (str): The language code of the source text.
        target_lang (str): The language code of the target text.

    Returns:
        str: The translated text.
    """
    tokenizer.src_lang = source_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    outputs = model.generate(
        **inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang)
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Streamlit UI
st.title("Real-Time Language Translation")
st.sidebar.header("Settings")


# Load model
st.sidebar.write("Loading multilingual translation model...")
with st.spinner("Loading model..."):
    tokenizer, model = load_multilingual_model()
st.sidebar.success("Model loaded successfully!")

# Language selection
supported_languages = [
    "en",
    "fr",
    "de",
    "es",
    "zh",
    "ar",
    "hi",
    "ja",
    "ru",
    "ko",
    "it",
    "pt",
]
language_mapping = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Chinese": "zh",
    "Arabic": "ar",
    "Hindi": "hi",
    "Japanese": "ja",
    "Russian": "ru",
    "Korean": "ko",
    "Italian": "it",
    "Portuguese": "pt",
}
source_language = st.sidebar.selectbox("Source Language", language_mapping.keys())
target_language = st.sidebar.selectbox("Target Language", language_mapping.keys())

if source_language == target_language:
    st.sidebar.warning("Source and target languages should be different.")

source_lang_code = language_mapping[source_language]
target_lang_code = language_mapping[target_language]

# Input and translation
st.subheader(f"Translation: {source_language} to {target_language}")
input_text = st.text_area("Enter text to translate:", "")

if st.button("Translate") and input_text:
    with st.spinner("Translating..."):
        translated_text = translate_multilingual(
            input_text, tokenizer, model, source_lang_code, target_lang_code
        )
    st.text_area("Translated text:", translated_text, height=150)
