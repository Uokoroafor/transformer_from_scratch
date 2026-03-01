import pytest

from utils.tokeniser import BPETokeniser

_SAMPLE_TEXT = (
    "the quick brown fox jumps over the lazy dog and the dog barked loudly at the fox"
)


@pytest.fixture
def trained_tokeniser():
    tok = BPETokeniser(_SAMPLE_TEXT, only_lower_case=True)
    tok.train(20)
    return tok
