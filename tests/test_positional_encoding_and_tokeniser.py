import torch

from embeddings.positional_encoding import PositionalEncoding
from utils.tokeniser import BPETokeniser


def test_positional_encoding_preserves_shape_for_short_sequences():
    layer = PositionalEncoding(d_model=8, max_len=16)
    x = torch.zeros(2, 5, 8)

    output = layer(x)

    assert output.shape == (2, 5, 8)
    assert not torch.allclose(output[:, 0, :], output[:, 1, :])


def test_tokeniser_round_trip_preserves_known_text():
    tokeniser = BPETokeniser("hello world", only_lower_case=True)
    tokeniser.train(0)

    encoded = tokeniser.encode("hello world")
    decoded = tokeniser.decode_words(encoded)

    assert decoded == "hello world"


# ---------------------------------------------------------------------------
# Positional encoding – value correctness
# ---------------------------------------------------------------------------


def test_positional_encoding_position_zero_has_sin_zero_cos_one():
    """PE at position 0: even dims are sin(0)=0, odd dims are cos(0)=1."""
    layer = PositionalEncoding(d_model=8, max_len=16)
    x = torch.zeros(1, 1, 8)

    output = layer(x)

    assert torch.allclose(output[0, 0, 0::2], torch.zeros(4), atol=1e-6)
    assert torch.allclose(output[0, 0, 1::2], torch.ones(4), atol=1e-6)


def test_positional_encoding_all_positions_are_distinct():
    """Every position must produce a unique encoding vector."""
    d_model, max_len = 16, 20
    layer = PositionalEncoding(d_model=d_model, max_len=max_len)
    x = torch.zeros(1, max_len, d_model)

    output = layer(x)
    positions = output[0]  # (max_len, d_model)

    for i in range(max_len):
        for j in range(i + 1, max_len):
            assert not torch.allclose(positions[i], positions[j]), (
                f"Positions {i} and {j} have identical encodings"
            )


# ---------------------------------------------------------------------------
# BPE – merge behaviour, unk, special token indices
# ---------------------------------------------------------------------------


def test_tokeniser_bpe_merges_produce_subword_tokens():
    """After training, at least one token longer than one char must exist."""
    # "abababab" guarantees ('a','b') is the most frequent pair on the first merge.
    # We stop at 2 iterations: iter 3 would reduce the text to a single token
    # with no remaining pairs, triggering the known empty-counts edge case.
    tok = BPETokeniser("abababab", only_lower_case=False)
    tok.train(2)

    subword_tokens = [
        t for t in tok.lookup_table if len(t) > 1 and t not in tok.special_tokens
    ]

    assert len(subword_tokens) > 0
    assert "ab" in tok.lookup_table


def test_tokeniser_bpe_round_trip_with_trained_vocab(trained_tokeniser):
    """Encode then decode must reproduce the original string after real merges."""
    for phrase in ["the fox", "the dog", "brown fox"]:
        encoded = trained_tokeniser.encode(phrase)
        decoded = trained_tokeniser.decode_words(encoded)
        assert decoded == phrase, f"Round-trip failed for '{phrase}': got '{decoded}'"


def test_tokeniser_unk_for_unseen_character(trained_tokeniser):
    """A character absent from training data must map to the unk token."""
    encoded = trained_tokeniser.encode("@")
    unk_idx = trained_tokeniser.lookup_table[trained_tokeniser.unk]
    assert encoded == [unk_idx]


def test_tokeniser_special_token_indices_are_fixed():
    """pad=0, sos=1, eos=2, unk=3 must always hold regardless of vocab content."""
    tok = BPETokeniser("hello world", only_lower_case=True)
    tok.train(0)

    assert tok.lookup_table[tok.pad] == 0
    assert tok.lookup_table[tok.sos] == 1
    assert tok.lookup_table[tok.eos] == 2
    assert tok.lookup_table[tok.unk] == 3
