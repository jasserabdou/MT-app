from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import streamlit as st
import torch


# Load multilingual model
def load_multilingual_model():
    """
    Loads the multilingual model for translation from the Hugging Face model hub.
    Returns:
        tuple: A tuple containing the tokenizer and the model.
    """
    model_name = "facebook/m2m100_418M"
    st.sidebar.write("Downloading and loading model...")

    # Load tokenizer and model from Hugging Face model hub
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    model.to("cpu")
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
        "cpu"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang)
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
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
