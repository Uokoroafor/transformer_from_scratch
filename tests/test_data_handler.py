from pathlib import Path

import torch

from utils.data_utils import DataHandler
from utils.tokeniser import BPETokeniser


def build_tokeniser(text: str) -> BPETokeniser:
    tokeniser = BPETokeniser(text, only_lower_case=True)
    tokeniser.train(0)
    return tokeniser


def test_data_handler_collate_adds_special_tokens_and_padding(tmp_path: Path):
    src_path = tmp_path / "src.txt"
    trg_path = tmp_path / "trg.txt"
    src_path.write_text("hello world\nsmall test\n", encoding="utf-8")
    trg_path.write_text("bonjour monde\npetit test\n", encoding="utf-8")

    src_tokeniser = build_tokeniser(src_path.read_text(encoding="utf-8"))
    trg_tokeniser = build_tokeniser(trg_path.read_text(encoding="utf-8"))

    handler = DataHandler(
        src_file_path=str(src_path),
        trg_file_path=str(trg_path),
        src_tokeniser=src_tokeniser,
        trg_tokeniser=trg_tokeniser,
        src_max_seq_len=6,
        trg_max_seq_len=6,
        batch_size=2,
    )

    src_batch, trg_batch = next(iter(handler.get_data_loader()))

    assert src_batch.shape == (2, 6)
    assert trg_batch.shape == (2, 7)
    assert torch.all(src_batch[:, -1] == handler.src_eos_idx)
    assert torch.all(trg_batch[:, 0] == handler.trg_sos_idx)

    eos_positions = (trg_batch == handler.trg_eos_idx).sum(dim=1)
    assert torch.all(eos_positions >= 1)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_data_handler_truncates_long_source_sequences(tmp_path: Path):
    # Source has many more tokens than max_seq_len after encoding.
    long_src = " ".join(list("abcdefghijklmnop")) + "\n"
    short_trg = "x y\n"

    src_path = tmp_path / "src.txt"
    trg_path = tmp_path / "trg.txt"
    src_path.write_text(long_src, encoding="utf-8")
    trg_path.write_text(short_trg, encoding="utf-8")

    src_tok = build_tokeniser(long_src)
    trg_tok = build_tokeniser(short_trg)

    max_len = 4
    handler = DataHandler(
        src_file_path=str(src_path),
        trg_file_path=str(trg_path),
        src_tokeniser=src_tok,
        trg_tokeniser=trg_tok,
        src_max_seq_len=max_len,
        trg_max_seq_len=10,
        batch_size=1,
    )

    src_batch, _ = next(iter(handler.get_data_loader()))

    assert src_batch.shape == (1, max_len)


# ---------------------------------------------------------------------------
# prep_string
# ---------------------------------------------------------------------------


def test_data_handler_prep_string_pads_to_max_len(tmp_path: Path):
    src_path = tmp_path / "src.txt"
    trg_path = tmp_path / "trg.txt"
    src_path.write_text("hello world\n", encoding="utf-8")
    trg_path.write_text("bonjour\n", encoding="utf-8")

    src_tok = build_tokeniser("hello world")
    trg_tok = build_tokeniser("bonjour")

    max_len = 10
    handler = DataHandler(
        src_file_path=str(src_path),
        trg_file_path=str(trg_path),
        src_tokeniser=src_tok,
        trg_tokeniser=trg_tok,
        src_max_seq_len=max_len,
        trg_max_seq_len=max_len,
        batch_size=1,
    )

    # "he" encodes to 2 tokens; with eos + padding the result must be max_len long.
    result = handler.prep_string("he")

    assert result.shape == torch.Size([max_len])
    assert result[-1].item() == handler.src_pad_idx


# ---------------------------------------------------------------------------
# output_string
# ---------------------------------------------------------------------------


def test_data_handler_output_string_returns_a_string(tmp_path: Path):
    src_path = tmp_path / "src.txt"
    trg_path = tmp_path / "trg.txt"
    src_path.write_text("hello world\n", encoding="utf-8")
    trg_path.write_text("bonjour monde\n", encoding="utf-8")

    src_tok = build_tokeniser("hello world")
    trg_tok = build_tokeniser("bonjour monde")

    handler = DataHandler(
        src_file_path=str(src_path),
        trg_file_path=str(trg_path),
        src_tokeniser=src_tok,
        trg_tokeniser=trg_tok,
        src_max_seq_len=10,
        trg_max_seq_len=10,
        batch_size=1,
    )

    _, trg_batch = next(iter(handler.get_data_loader()))
    result = handler.output_string(trg_batch[0])

    assert isinstance(result, str)
    assert len(result) > 0
