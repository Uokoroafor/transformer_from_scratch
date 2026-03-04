from __future__ import annotations

import argparse
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

from typing import TYPE_CHECKING

from examples.data_prep import train_tokenisers

if TYPE_CHECKING:
    from models.transformer import Transformer
    from utils.data_utils import DataHandler
    from utils.train_utils import Trainer


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "europarl_fr_en"

TRAIN_PRESETS = {
    "local": {
        "num_epochs": 5,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "max_seq_len": 64,
        "d_model": 256,
        "d_ff": 1024,
        "num_layers": 4,
        "num_heads": 4,
        "dropout_prob": 0.1,
        "tokeniser_epochs": 20,
        "eval_every": 1,
        "early_stopping_patience": 5,
    },
    "benchmark": {
        "num_epochs": 10,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "max_seq_len": 100,
        "d_model": 512,
        "d_ff": 2048,
        "num_layers": 6,
        "num_heads": 8,
        "dropout_prob": 0.1,
        "tokeniser_epochs": 50,
        "eval_every": 1,
        "early_stopping_patience": 10,
    },
}


def resolve_override(value, default):
    return value if value is not None else default


@dataclass
class TrainConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_DATA_DIR
    num_epochs: int = TRAIN_PRESETS["local"]["num_epochs"]
    batch_size: int = TRAIN_PRESETS["local"]["batch_size"]
    learning_rate: float = 1e-4
    max_seq_len: int = TRAIN_PRESETS["local"]["max_seq_len"]
    d_model: int = TRAIN_PRESETS["local"]["d_model"]
    d_ff: int = TRAIN_PRESETS["local"]["d_ff"]
    num_layers: int = TRAIN_PRESETS["local"]["num_layers"]
    num_heads: int = TRAIN_PRESETS["local"]["num_heads"]
    dropout_prob: float = 0.1
    tokeniser_epochs: int = TRAIN_PRESETS["local"]["tokeniser_epochs"]
    eval_every: int = 1
    early_stopping_patience: int = TRAIN_PRESETS["local"]["early_stopping_patience"]
    plot_losses: bool = True
    verbose: bool = True
    preset: str = "local"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train the EN-FR transformer example on the Europarl dataset."
    )
    parser.add_argument(
        "--preset",
        choices=tuple(TRAIN_PRESETS.keys()),
        default="local",
        help="Training preset. `local` is the fast default; `benchmark` restores the larger baseline settings.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dropout-prob", type=float, default=None)
    parser.add_argument("--tokeniser-epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument(
        "--no-plot-losses",
        action="store_true",
        help="Disable loss plotting during training.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce training log output.",
    )

    args = parser.parse_args()
    preset = TRAIN_PRESETS[args.preset].copy()
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=resolve_override(args.num_epochs, preset["num_epochs"]),
        batch_size=resolve_override(args.batch_size, preset["batch_size"]),
        learning_rate=resolve_override(args.learning_rate, preset["learning_rate"]),
        max_seq_len=resolve_override(args.max_seq_len, preset["max_seq_len"]),
        d_model=resolve_override(args.d_model, preset["d_model"]),
        d_ff=resolve_override(args.d_ff, preset["d_ff"]),
        num_layers=resolve_override(args.num_layers, preset["num_layers"]),
        num_heads=resolve_override(args.num_heads, preset["num_heads"]),
        dropout_prob=resolve_override(args.dropout_prob, preset["dropout_prob"]),
        tokeniser_epochs=resolve_override(
            args.tokeniser_epochs, preset["tokeniser_epochs"]
        ),
        eval_every=resolve_override(args.eval_every, preset["eval_every"]),
        early_stopping_patience=resolve_override(
            args.early_stopping_patience, preset["early_stopping_patience"]
        ),
        plot_losses=not args.no_plot_losses,
        verbose=not args.quiet,
        preset=args.preset,
    )


def load_tokeniser(path: Path):
    with path.open("rb") as tokeniser_file:
        return pkl.load(tokeniser_file)


def ensure_tokenisers(data_dir: Path, tokeniser_epochs: int):
    src_tokeniser_path = data_dir / f"english_tokeniser_{tokeniser_epochs}_epochs.pkl"
    trg_tokeniser_path = data_dir / f"french_tokeniser_{tokeniser_epochs}_epochs.pkl"

    if not (src_tokeniser_path.exists() and trg_tokeniser_path.exists()):
        train_tokenisers(
            num_epochs=tokeniser_epochs,
            folder_path=data_dir,
        )

    return load_tokeniser(src_tokeniser_path), load_tokeniser(trg_tokeniser_path)


def build_file_paths(data_dir: Path, language: str) -> dict[str, Path]:
    return {
        "train": data_dir / f"{language}_train.txt",
        "val": data_dir / f"{language}_val.txt",
        "test": data_dir / f"{language}_test.txt",
    }


def build_data_handler(
    src_file_path: Path,
    trg_file_path: Path,
    src_tokeniser,
    trg_tokeniser,
    config: TrainConfig,
) -> "DataHandler":
    from utils.data_utils import DataHandler

    return DataHandler(
        src_file_path=str(src_file_path),
        trg_file_path=str(trg_file_path),
        src_tokeniser=src_tokeniser,
        trg_tokeniser=trg_tokeniser,
        src_max_seq_len=config.max_seq_len,
        trg_max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
    )


def build_model(
    config: TrainConfig, src_tokeniser, trg_tokeniser, device
) -> "Transformer":
    from models.transformer import Transformer

    return Transformer(
        src_pad=src_tokeniser.encode(src_tokeniser.pad)[0],
        trg_pad=trg_tokeniser.encode(trg_tokeniser.pad)[0],
        trg_sos=trg_tokeniser.encode(trg_tokeniser.sos)[0],
        vocab_size_enc=len(src_tokeniser),
        vocab_size_dec=len(trg_tokeniser),
        d_model=config.d_model,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout_prob=config.dropout_prob,
        device=device,
    )


def run_training(config: TrainConfig) -> tuple["Trainer", float]:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from utils.train_utils import Trainer

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    src_paths = build_file_paths(config.data_dir, "english")
    trg_paths = build_file_paths(config.data_dir, "french")
    src_tokeniser, trg_tokeniser = ensure_tokenisers(
        config.data_dir,
        config.tokeniser_epochs,
    )

    train_data = build_data_handler(
        src_paths["train"], trg_paths["train"], src_tokeniser, trg_tokeniser, config
    )
    val_data = build_data_handler(
        src_paths["val"], trg_paths["val"], src_tokeniser, trg_tokeniser, config
    )
    test_data = build_data_handler(
        src_paths["test"], trg_paths["test"], src_tokeniser, trg_tokeniser, config
    )

    train_loader = train_data.get_data_loader()
    val_loader = val_data.get_data_loader()
    test_loader = test_data.get_data_loader()

    model = build_model(config, src_tokeniser, trg_tokeniser, device)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=train_data.trg_pad_idx)
    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate)

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        val_data=val_loader,
        loss_fn=loss_fn,
        optimiser=optimiser,
        device=device,
        path=str(config.output_dir),
        verbose=config.verbose,
    )
    trainer.logger.log_info(f"Using device: {device}")

    trainer.train(
        epochs=config.num_epochs,
        save_model=True,
        plotting=config.plot_losses,
        verbose=config.verbose,
        eval_every=config.eval_every,
        early_stopping=True,
        early_stopping_patience=config.early_stopping_patience,
    )
    test_loss = trainer.evaluate(test_loader)
    trainer.logger.log_info(f"Test loss: {test_loss:.4f}")

    return trainer, test_loss


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
