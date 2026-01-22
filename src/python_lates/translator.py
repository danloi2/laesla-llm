from transformers import MarianMTModel, MarianTokenizer
import os
from . import utils


def load_model(model_path=None):
    """
    Load the model and tokenizer from the specified directory.
    """
    if model_path is None:
        model_path = utils.get_model_path()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at: {model_path}")

    print(f"Loading model from {model_path}...")
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    print("Model loaded successfully.")
    return model, tokenizer


def translate(text, model, tokenizer):
    """
    Translate the given text from Latin to Spanish.
    """
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Generate translation
    translated = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

    # Decode the translation
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    return result
