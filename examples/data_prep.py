# This reads in the data and creates tokenisers
from pathlib import Path
from typing import Optional

from utils.tokeniser import BPETokeniser


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "europarl_fr_en"


def train_tokenisers(
    num_epochs: Optional[int] = 0, folder_path: Path = DEFAULT_DATA_DIR
) -> None:
    """Train the tokenisers on the training data. Saves the tokenisers to disk as pickle files.

    Args:
        num_epochs: The number of epochs to train the tokenisers for. If 0, the tokenisers are only character level.
        folder_path: Directory containing the train splits.
    """
    # We tokenise over the training data only

    folder_path = Path(folder_path)
    paths = [folder_path / "english_train.txt", folder_path / "french_train.txt"]
    saved_paths = [
        folder_path / f"english_tokeniser_{num_epochs}_epochs.pkl",
        folder_path / f"french_tokeniser_{num_epochs}_epochs.pkl",
    ]

    for path, saved_path in zip(paths, saved_paths):
        with path.open("r", encoding="utf-8") as f:
            data = f.readlines()
        data = "".join(data)
        tokeniser = BPETokeniser(data, only_lower_case=True)
        tokeniser.train(num_epochs)
        tokeniser.save(str(saved_path))


if __name__ == "__main__":
    train_tokenisers(100)
