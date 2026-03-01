from __future__ import annotations

import argparse
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "europarl_fr_en"


@dataclass
class TranslateConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    checkpoint: Path | None = None
    text: str = ""
    max_len: int = 100
    d_model: int = 512
    d_ff: int = 2048
    num_layers: int = 6
    num_heads: int = 8
    dropout_prob: float = 0.0
    tokeniser_epochs: int = 50


def parse_args() -> TranslateConfig:
    parser = argparse.ArgumentParser(
        description="Translate EN to FR with a trained transformer checkpoint."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout-prob", type=float, default=0.0)
    parser.add_argument("--tokeniser-epochs", type=int, default=50)

    args = parser.parse_args()
    return TranslateConfig(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        text=args.text,
        max_len=args.max_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_prob=args.dropout_prob,
        tokeniser_epochs=args.tokeniser_epochs,
    )


def load_tokeniser(path: Path):
    with path.open("rb") as tokeniser_file:
        return pkl.load(tokeniser_file)


def load_tokenisers(data_dir: Path, tokeniser_epochs: int):
    src_tokeniser_path = data_dir / f"english_tokeniser_{tokeniser_epochs}_epochs.pkl"
    trg_tokeniser_path = data_dir / f"french_tokeniser_{tokeniser_epochs}_epochs.pkl"

    if not src_tokeniser_path.exists() or not trg_tokeniser_path.exists():
        raise FileNotFoundError(
            "Tokenisers not found. Run examples/train_fr_en.py first or set "
            "--tokeniser-epochs to match the saved tokenisers."
        )

    return load_tokeniser(src_tokeniser_path), load_tokeniser(trg_tokeniser_path)


def build_model(
    config: TranslateConfig, src_tokeniser, trg_tokeniser, device
) -> "Transformer":
    import torch
    from models.transformer import Transformer

    model = Transformer(
        src_pad=src_tokeniser.encode(src_tokeniser.pad)[0],
        trg_pad=trg_tokeniser.encode(trg_tokeniser.pad)[0],
        trg_sos=trg_tokeniser.encode(trg_tokeniser.sos)[0],
        vocab_size_enc=len(src_tokeniser),
        vocab_size_dec=len(trg_tokeniser),
        d_model=config.d_model,
        d_ff=config.d_ff,
        max_seq_len=config.max_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout_prob=config.dropout_prob,
        device=device,
    )

    state = torch.load(config.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def encode_source(text: str, src_tokeniser, max_len: int, device) -> "torch.Tensor":
    import torch

    if src_tokeniser.only_lower_case:
        text = text.lower()
    tokens = src_tokeniser.encode(text)
    eos_idx = src_tokeniser.encode(src_tokeniser.eos)[0]
    pad_idx = src_tokeniser.encode(src_tokeniser.pad)[0]

    tokens = tokens[: max_len - 1] + [eos_idx]
    tokens += [pad_idx] * (max_len - len(tokens))

    return torch.tensor(tokens, device=device).unsqueeze(0)


def greedy_decode(
    model: "Transformer",
    src: "torch.Tensor",
    trg_sos: int,
    trg_eos: int,
    max_len: int,
) -> "torch.Tensor":
    import torch

    device = src.device
    trg_tokens = torch.tensor([[trg_sos]], device=device)

    for _ in range(max_len - 1):
        output = model(src, trg_tokens)
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        trg_tokens = torch.cat([trg_tokens, next_token], dim=1)
        if next_token.item() == trg_eos:
            break

    return trg_tokens.squeeze(0)


def decode_target(tokens: "torch.Tensor", trg_tokeniser) -> str:
    return "".join(trg_tokeniser.decode(tokens.tolist(), ignore_special_tokens=True))


def main() -> None:
    config = parse_args()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_tokeniser, trg_tokeniser = load_tokenisers(
        config.data_dir, config.tokeniser_epochs
    )
    model = build_model(config, src_tokeniser, trg_tokeniser, device)
    src = encode_source(config.text, src_tokeniser, config.max_len, device)

    trg_tokens = greedy_decode(
        model=model,
        src=src,
        trg_sos=trg_tokeniser.encode(trg_tokeniser.sos)[0],
        trg_eos=trg_tokeniser.encode(trg_tokeniser.eos)[0],
        max_len=config.max_len,
    )
    translation = decode_target(trg_tokens, trg_tokeniser)

    print(translation)


if __name__ == "__main__":
    main()
