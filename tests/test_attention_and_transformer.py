import torch
import torch.nn as nn

from layers.attention import Attention
from layers.multi_head_attention import MultiHeadAttention
from models.encoder import Encoder
from models.transformer import Transformer


def test_multi_head_attention_returns_expected_shapes():
    layer = MultiHeadAttention(d_model=8, num_heads=2)
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 4, 8)
    value = torch.randn(2, 4, 8)

    output, weights = layer(query, key, value)

    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 2, 4, 4)


def test_transformer_masks_respect_padding_and_causality():
    model = Transformer(
        src_pad=0,
        trg_pad=0,
        trg_sos=1,
        vocab_size_enc=16,
        vocab_size_dec=16,
        d_model=8,
        d_ff=16,
        max_seq_len=6,
        num_layers=1,
        num_heads=2,
        dropout_prob=0.0,
        device="cpu",
    )

    src = torch.tensor([[1, 2, 0, 0]])
    trg = torch.tensor([[1, 3, 4, 0]])

    src_mask = model.get_src_mask(src)
    trg_mask = model.get_trg_mask(trg)

    expected_src_mask = torch.tensor([[[True, True, False, False]]])
    expected_trg_mask = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, False],
            ]
        ]
    )

    assert torch.equal(src_mask, expected_src_mask)
    assert torch.equal(trg_mask, expected_trg_mask)


def test_transformer_forward_returns_vocab_logits():
    model = Transformer(
        src_pad=0,
        trg_pad=0,
        trg_sos=1,
        vocab_size_enc=32,
        vocab_size_dec=24,
        d_model=8,
        d_ff=16,
        max_seq_len=6,
        num_layers=1,
        num_heads=2,
        dropout_prob=0.0,
        device="cpu",
    )
    src = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]])
    trg = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 0]])

    output = model(src, trg)

    assert output.shape == (2, 4, 24)


# ---------------------------------------------------------------------------
# Attention – weight correctness and masking behaviour
# ---------------------------------------------------------------------------


def test_attention_weights_sum_to_one():
    """Softmax over keys must produce a valid probability distribution."""
    torch.manual_seed(0)
    layer = Attention()
    q = torch.randn(2, 2, 5, 4)
    k = torch.randn(2, 2, 5, 4)
    v = torch.randn(2, 2, 5, 4)

    _, weights = layer(q, k, v)

    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_attention_masked_positions_get_near_zero_weight():
    """Key positions where mask==0 must receive ~0 attention after softmax."""
    torch.manual_seed(0)
    layer = Attention()
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)

    # Keep key positions 0 and 1; mask out 2 and 3.
    mask = torch.tensor([[[[1, 1, 0, 0]]]])  # (1, 1, 1, 4) – broadcasts over queries
    _, weights = layer(q, k, v, mask=mask)

    assert weights[..., 2].abs().max().item() < 1e-6
    assert weights[..., 3].abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# Encoder – output shape
# ---------------------------------------------------------------------------


def test_encoder_output_preserves_batch_and_sequence_shape():
    enc = Encoder(
        vocab_size_enc=32,
        d_model=8,
        max_seq_len=10,
        num_layers=2,
        num_heads=2,
        d_ff=16,
        dropout_prob=0.0,
    )
    src = torch.randint(1, 32, (3, 7))

    output = enc(src)

    assert output.shape == (3, 7, 8)


# ---------------------------------------------------------------------------
# Transformer – training step
# ---------------------------------------------------------------------------


def test_transformer_training_step_produces_gradients():
    """A forward + backward pass must populate gradients for every parameter."""
    model = Transformer(
        src_pad=0,
        trg_pad=0,
        trg_sos=1,
        vocab_size_enc=16,
        vocab_size_dec=16,
        d_model=8,
        d_ff=16,
        max_seq_len=6,
        num_layers=1,
        num_heads=2,
        dropout_prob=0.0,
        device="cpu",
    )
    src = torch.tensor([[1, 2, 3, 0, 0, 0]])
    trg = torch.tensor([[1, 2, 3, 4, 0]])

    # Teacher-forcing: decoder sees trg[:-1], loss is against trg[1:]
    output = model(src, trg[:, :-1])
    loss = nn.CrossEntropyLoss(ignore_index=0)(
        output.reshape(-1, 16), trg[:, 1:].reshape(-1)
    )
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
