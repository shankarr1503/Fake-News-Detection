"""Text preparation shared by training and serving.

Both the training scripts and the API import from here so the model always
sees text prepared the same way it was trained on.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Punctuation (including Unicode quotes/dashes), digits and underscores.
# Replaced with a space so "well-known" -> "well known", not "wellknown".
_NON_LETTERS = re.compile(r"[\W\d_]+")


def combine_title_text(title, text):
    """Join title and body the same way for training and inference."""
    title = title if isinstance(title, str) else ""
    text = text if isinstance(text, str) else ""
    return f"{title} {text}".strip()


def clean_text(text):
    """Lowercase, strip non-letters and drop stopwords (input for the TF-IDF model)."""
    if not isinstance(text, str):
        return ""
    text = _NON_LETTERS.sub(" ", text.lower())
    return " ".join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)
