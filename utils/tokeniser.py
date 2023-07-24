"""Utility file for tokenising text data."""
import spacy
from typing import Optional


class Tokeniser:
    """Tokeniser class for tokenising text data using spacy."""

    def __init__(self, spacy_model_name: Optional[str] = 'en_core_web_sm'):
        """Initialise the tokeniser with a spacy model."""
        self.nlp = spacy.load(spacy_model_name)

    def tokenise(self, text: str) -> list:
        """Token-ise a string of text."""
        return [token.text for token in self.nlp.tokenizer(text)]

    def detokenise(self, tokens: list[str]) -> str:
        """De-tokenise a list of tokens."""
        return ''.join(tokens)

    def __call__(self, text: str) -> list:
        """Token-ise a string of text."""
        return self.tokenise(text)

    def __repr__(self) -> str:
        """Return the string representation of the tokeniser."""
        return f"Tokeniser(spacy_model_name='{self.nlp.meta['name']}')"

    def __str__(self) -> str:
        """Return the string representation of the tokeniser."""
        return f"Tokeniser(spacy_model_name='{self.nlp.meta['name']}')"
