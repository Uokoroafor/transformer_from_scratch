# This reads in the data and creates tokenisers
from typing import Optional
from utils.tokeniser import BPETokeniser


def train_tokenisers(num_epochs: Optional[int] = 0) -> None:
    """Train the tokenisers on the training data. Saves the tokenisers to disk as pickle files.

    Args:
        num_epochs: The number of epochs to train the tokenisers for. If 0, the tokenisers are only character level.
    """
    folder_path = "../data/europarl_fr_en/"
    # We tokenise over the training data only

    paths = [folder_path + "english_train.txt", folder_path + "french_train.txt"]
    saved_paths = [
        f"{folder_path}english_tokeniser_{num_epochs}_epochs.pkl",
        f"{folder_path}french_tokeniser_{num_epochs}_epochs.pkl",
    ]

    for path, saved_path in zip(paths, saved_paths):
        with open(path, "r", encoding="utf-8") as f:
            data = f.readlines()
        data = "".join(data)
        tokeniser = BPETokeniser(data, only_lower_case=True)
        tokeniser.train(num_epochs)
        tokeniser.save(saved_path)


if __name__ == "__main__":
    train_tokenisers(100)
