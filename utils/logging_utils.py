import logging
import logging.handlers
from typing import List, Optional
import matplotlib.pyplot as plt
import torch


class Logger:
    def __init__(
        self,
        log_file_path: str,
        name: Optional[str] = None,
        log_level: int = logging.INFO,
        verbose: Optional[bool] = False,
    ):
        """Initialize the logger object

        Args:
            log_file_path (str): Path to the log file
            name (str, optional): Name of the logger. Defaults to None.
            log_level (int, optional): Log level. Defaults to logging.INFO.
            verbose (bool, optional): Whether to print the logs to stdout. Defaults to False.
        """
        self.log_file_path = log_file_path
        if name is None:
            name = __name__
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.messages = []
        self.verbose = verbose

        self._setup_file_handler()

    def _setup_file_handler(self):
        """Set up the file handler for logging"""
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file_path, maxBytes=1024 * 1024, backupCount=5
            )
            # maxBytes = 1024 * 1024 = 1 MB and backupCount = 5 means that at most 5 files will be created
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up log file handler: {str(e)}")

    def log_info(self, message: str):
        """Log an info message"""
        self.logger.info(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def log_warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def log_error(self, message: str):
        """Log an error message"""
        self.logger.error(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def log_critical(self, message: str):
        """Log a critical message"""
        self.logger.critical(message)
        self.messages.append(message)
        self.print_last_message() if self.verbose else None

    def print_last_message(self):
        """Print the last message in the log file"""
        if self.messages:
            print(self.messages[-1])
        else:
            print("No messages logged.")


def plot_losses(
    train_losses: List[float],
    val_losses: List[float],
    model_name: Optional[str] = None,
    num_epochs: Optional[int] = None,
    saved_path: Optional[str] = None,
) -> None:
    """Plot the training and validation losses
    Args:

        train_losses (List[float]): Training losses
        val_losses (List[float]): Validation losses
        model_name (Optional[str], optional): Name of the model. Defaults to None.
        num_epochs (Optional[int], optional): Number of epochs. Defaults to None.
        saved_path: (Optional[str], optional): Path to save the plot. Defaults to None.
    """
    if num_epochs is not None:
        steps = num_epochs
        x = torch.arange(0, num_epochs + 1, num_epochs // (len(train_losses) - 1))
        # Make the x-axis start at 1
        x[0] = 1
    else:
        steps = len(train_losses)
        x = torch.arange(1, len(train_losses) + 1)
    plt.plot(x, train_losses, label="train")
    plt.plot(x, val_losses, label="val", linestyle="dashed")
    if model_name is not None:
        plt.title(f"Losses for the {model_name} model over {steps} iterations")
    else:
        plt.title(f"Losses over {len(train_losses)} steps")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    if saved_path is not None:
        plt.savefig(saved_path)
    plt.show()
    plt.close()
