"""Download and split the Europarl FR-EN corpus into train/val/test text files.

Downloads the fr-en language pair from statmt.org, extracts the aligned English
and French text files, removes markup lines, and writes six files:

    data/europarl_fr_en/english_train.txt
    data/europarl_fr_en/english_val.txt
    data/europarl_fr_en/english_test.txt
    data/europarl_fr_en/french_train.txt
    data/europarl_fr_en/french_val.txt
    data/europarl_fr_en/french_test.txt

Usage:
    uv run python examples/download_data.py
    uv run python examples/download_data.py --preset benchmark
    uv run python examples/download_data.py --max-train 200000
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

EUROPARL_URL = "https://www.statmt.org/europarl/v7/fr-en.tgz"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "europarl_fr_en"

DATA_PRESETS = {
    "local": {
        "val_size": 2_000,
        "test_size": 2_000,
        "max_train": 50_000,
    },
    "benchmark": {
        "val_size": 5_000,
        "test_size": 5_000,
        "max_train": None,
    },
}

# Names of the files inside the archive
ARCHIVE_EN = "europarl-v7.fr-en.en"
ARCHIVE_FR = "europarl-v7.fr-en.fr"


def _report_progress(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        print(f"\r  {pct:.1f}% ({downloaded // 1_048_576} MB / {total_size // 1_048_576} MB)", end="", flush=True)


def download_archive(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "fr-en.tgz"
    if archive_path.exists():
        print(f"Archive already present at {archive_path}, skipping download.")
        return archive_path
    print(f"Downloading Europarl FR-EN corpus from {EUROPARL_URL} ...")
    urllib.request.urlretrieve(EUROPARL_URL, archive_path, reporthook=_report_progress)
    print()  # newline after progress
    return archive_path


def extract_lines(archive_path: Path, filename: str) -> list[str]:
    print(f"Extracting {filename} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        member = tar.getmember(filename)
        f = tar.extractfile(member)
        lines = [line.decode("utf-8") for line in f]
    return lines


def strip_markup(lines: list[str]) -> list[str]:
    """Remove XML-style markup lines (lines starting with '<')."""
    return [line for line in lines if not line.startswith("<")]


def write_split(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  Wrote {len(lines):,} lines to {path}")


def split_and_write(
    en_lines: list[str],
    fr_lines: list[str],
    data_dir: Path,
    val_size: int,
    test_size: int,
    max_train: int | None = None,
) -> None:
    assert len(en_lines) == len(fr_lines), (
        f"Line count mismatch: {len(en_lines)} English vs {len(fr_lines)} French"
    )

    total = len(en_lines)
    test_start = total - test_size
    val_start = test_start - val_size
    train_end = min(val_start, max_train) if max_train is not None else val_start

    splits = {
        "train": (0, train_end),
        "val": (val_start, test_start),
        "test": (test_start, total),
    }

    print(f"\nSplitting {total:,} sentence pairs ...")
    for split, (start, end) in splits.items():
        write_split(en_lines[start:end], data_dir / f"english_{split}.txt")
        write_split(fr_lines[start:end], data_dir / f"french_{split}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the Europarl FR-EN corpus.")
    parser.add_argument(
        "--preset",
        choices=tuple(DATA_PRESETS.keys()),
        default="local",
        help="Dataset preset. `local` is the fast default; `benchmark` keeps the larger split sizes.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Number of sentence pairs for the validation set.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Number of sentence pairs for the test set.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Cap on training sentence pairs. Defaults to the selected preset value.",
    )
    args = parser.parse_args()
    preset = DATA_PRESETS[args.preset]
    val_size = args.val_size if args.val_size is not None else preset["val_size"]
    test_size = args.test_size if args.test_size is not None else preset["test_size"]
    max_train = args.max_train if args.max_train is not None else preset["max_train"]

    archive = download_archive(args.data_dir)

    en_lines = strip_markup(extract_lines(archive, ARCHIVE_EN))
    fr_lines = strip_markup(extract_lines(archive, ARCHIVE_FR))

    split_and_write(
        en_lines,
        fr_lines,
        args.data_dir,
        val_size,
        test_size,
        max_train,
    )

    print("\nDone. You can now run: make train")


if __name__ == "__main__":
    main()
