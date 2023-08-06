import os
import time
from typing import List, Optional, Dict, Any


def create_training_folder(path: Optional[str] = None) -> str:
    """Create a training folder to save the model checkpoints and training logs. Folder name is today's date and time.
    Args:
        path (Optional[str], optional): The path to save the training folder to. Defaults to None.
    Returns:
        str: The path to the training folder.
    """

    # Get today's date and time in the YY-MM-DD-HH-MM format
    date_time = time.strftime("%y%m%d-%H%M")
    path_ = path if path is not None else "."
    # If the training folder doesn't exist, create it
    if not os.path.exists(path_ + "/training"):
        os.makedirs(path_ + "/training")

    # Create the path to the training folder
    path = f"{path_}/training/{date_time}"

    # Create the training folder, if it already exists, add a number to the end
    try:
        os.makedirs(path)
    except FileExistsError:
        i = 1
        while True:
            try:
                os.makedirs(f"{path}_{i}")
                path = f"{path}_{i}"
                break
            except FileExistsError:
                i += 1

    # Also make a folder for the saved models and training logs
    os.makedirs(f"{path}/saved_models")
    os.makedirs(f"{path}/training_logs")

    return path


def save_losses(train_losses: List[float], val_losses: List[float], path: str) -> None:
    """Save the training and validation losses to a txt file.
    Args:
        train_losses (List[float]): Training losses.
        val_losses (List[float]): Validation losses.
        path (str): The path to save the losses to.
    """
    # Save the training and validation losses to a csv file
    with open(f"{path}/training_logs/losses.csv", "w") as f:
        f.write("train_loss,val_loss\n")
        for train_loss, val_loss in zip(train_losses, val_losses):
            f.write(f"{train_loss},{val_loss}\n")


def save_config(config: dict, path: Optional[str] = None) -> None:
    """Save the config to a txt file.
    Args:
        config (dict): The config.
        path (str): The path to save the config to.
    """
    # Save the config to a txt file
    if path is None:
        save_path = "config.txt"
    else:
        save_path = path
    with open(f"{save_path}", "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the config from a txt file.
    Args:
        config_path (str): The path to load the config from.
    Returns:
        dict: The config.
    """
    # Load the config from a txt file
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            key, value = line.split(": ")
            config[key] = value.strip()
            # Convert to int or float if possible
            try:
                config[key] = int(config[key])
            except ValueError:
                try:
                    config[key] = float(config[key])
                except ValueError:
                    pass
    return config
