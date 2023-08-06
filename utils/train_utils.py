from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.file_utils import create_training_folder
from utils.logging_utils import Logger, plot_losses


class Trainer:

    def __init__(self, model: nn.Module, train_data: DataLoader, val_data: DataLoader,
                 loss_fn: nn.Module, optimiser: optim.Optimizer, device: torch.device, path: Optional[str] = None,
                 verbose: bool = True):
        """ Initialise the trainer object

        Args:
            model: The model to train
            train_data: The training data
            val_data: The validation data
            loss_fn: The loss function to use
            optimiser: The optimiser to use
            device: The device to use
            path: The path to save the model to
            verbose: Whether to print the training logs to stdout
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.device = device
        self.path = create_training_folder(path)

        self.logger = Logger(self.path + "/training_logs/training_log.txt", name="training_log", verbose=verbose)

    def model_loop(self, method: str = 'train', test_data: Optional[DataLoader] = None) -> float:
        """ Train the model for one epoch

        Args:
            method: The method to use, either 'train', 'val' or 'test'
            test_data: The test data to use if method is 'test' otherwise None

        Returns:
            The average loss for the epoch or evaluation
        """

        if method == 'train':
            self.model.train()
            data_loader = self.train_data
        elif method == 'val':
            self.model.eval()
            data_loader = self.val_data
        elif method == 'test':
            self.model.eval()
            data_loader = test_data
        else:
            raise ValueError(f'Invalid method: {method}. Only "train", "val" and "test" are valid methods.')

        epoch_loss = 0
        batch_idx = 0

        for batch_idx, (src_input, trg_input) in enumerate(data_loader):
            src_input = src_input.to(self.device)
            trg_input = trg_input.to(self.device)

            # Forward pass
            output = self.model(src_input, trg_input[:, :-1])

            # Calculate the loss
            loss = self.loss_fn(output.reshape(-1, output.shape[-1]), trg_input[:, 1:].reshape(-1))

            if method == 'train':
                # Backward pass
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

            epoch_loss += loss.item()

        return epoch_loss / (batch_idx + 1)

    def train(self, epochs: int, save_model: bool = True, save_model_path: Optional[str] = None,
              plotting: bool = True, verbose: bool = True, eval_every: int = 1, early_stopping: bool = True,
              early_stopping_patience: int = 10) -> Tuple[nn.Module, List[float], List[float]]:
        """ Train the model

        Args:
            epochs: The number of epochs to train for
            save_model: Whether to save the model
            save_model_path: The path to save the model to
            plotting: Whether to plot the training and validation loss
            verbose: Whether to print the training and validation loss
            eval_every: The number of epochs to wait before evaluating the model
            early_stopping: Whether to use early stopping
            early_stopping_patience: The number of epochs to wait before stopping

        Returns:
            The training and validation loss
        """
        self.logger.log_info(f'Training for {epochs} epochs')
        if save_model and save_model_path is None:
            save_model_path = f"{self.path}/saved_models/{type(self.model).__name__}_best_model.pt"

        train_loss = []
        val_loss = []

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            train_epoch_loss = self.model_loop(method='train')

            if epoch % eval_every == 0:
                val_epoch_loss = self.model_loop(method='val')

                train_loss.append(train_epoch_loss)
                val_loss.append(val_epoch_loss)

                if verbose:
                    self.logger.log_info(
                        f'Epoch {epoch + 1}/{epochs}: Train loss: {train_epoch_loss:.4f}, Val loss: {val_epoch_loss:.4f}')

                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    epochs_without_improvement = 0

                    if save_model:
                        self.logger.log_info(f'Saving model to {save_model_path}')
                        torch.save(self.model.state_dict(), save_model_path)
                else:
                    epochs_without_improvement += 1

                    if early_stopping and epochs_without_improvement == early_stopping_patience:
                        self.logger.log_info(f'Early stopping after {epoch + 1} epochs')
                        break

        if plotting:
            saved_path = f"{self.path}/training_logs/{type(self.model).__name__}_losses.png"
            plot_losses(train_loss, val_loss, model_name=self.model.__class__.__name__, saved_path=saved_path)

        return self.model, train_loss, val_loss

    def evaluate(self, data: DataLoader) -> float:
        """ Evaluate the model

        Args:
            data: The data to evaluate the model on

        Returns:
            The average loss
        """
        return self.model_loop(method='test', test_data=data)
