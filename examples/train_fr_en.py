# This file contains a training loop for the Transformer model on the EuroParl dataset -
# see data/euro_parl_fr_en/Europarl_Parallel_Corpus.html for more information.

import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import DataHandler
from utils.tokeniser import BPETokeniser as Tokeniser
from models.transformer import Transformer
import os
import pickle as pkl
from typing import Tuple, Optional

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the training hyperparameters
training_params = {
    "num_epochs": 10,
    "batch_size": 32,
    "lr": 0.0001
}

# Define the data hyperparameters
data_params = {
    "max_seq_len": 100,
    "max_vocab_size": 10000,
    "batch_size": training_params["batch_size"],
    "device": device,
}

# Define the file paths
src_file_paths = {
    'train': '../data/euro_parl_fr_en/english_train.txt',
    'val': '../data/euro_parl_fr_en/english_val.txt',
    'test': '../data/euro_parl_fr_en/english_test.txt',
}

trg_file_paths = {
    'train': '../data/euro_parl_fr_en/french_train.txt',
    'val': '../data/euro_parl_fr_en/french_val.txt',
    'test': '../data/euro_parl_fr_en/french_test.txt',
}

src_tokeniser_pth = '../data/euro_parl_fr_en/english_tokeniser_50_epochs.pkl'
trg_tokeniser_pth = '../data/euro_parl_fr_en/french_tokeniser_50_epochs.pkl'

# Load the tokenisers if they exist else create them
if not (os.path.exists(src_tokeniser_pth) and os.path.exists(trg_tokeniser_pth)):
    # Run the data_prep.py script to create the tokenisers
    os.system('python data_prep.py')

with open(src_tokeniser_pth, 'rb') as src_tokeniser_file:
    src_tokeniser = pkl.load(src_tokeniser_file)
with open(trg_tokeniser_pth, 'rb') as trg_tokeniser_file:
    trg_tokeniser = pkl.load(trg_tokeniser_file)

# Define the data handler
train_data = DataHandler(
    src_file_path=src_file_paths['train'],
    trg_file_path=trg_file_paths['train'],
    src_tokeniser=src_tokeniser,
    trg_tokeniser=trg_tokeniser,
    src_max_seq_len=data_params['max_seq_len'],
    trg_max_seq_len=data_params['max_seq_len'],
    batch_size=data_params['batch_size'],
)

val_data = DataHandler(
    src_file_path=src_file_paths['val'],
    trg_file_path=trg_file_paths['val'],
    src_tokeniser=src_tokeniser,
    trg_tokeniser=trg_tokeniser,
    src_max_seq_len=data_params['max_seq_len'],
    trg_max_seq_len=data_params['max_seq_len'],
    batch_size=data_params['batch_size'],
)

test_data = DataHandler(
    src_file_path=src_file_paths['test'],
    trg_file_path=trg_file_paths['test'],
    src_tokeniser=src_tokeniser,
    trg_tokeniser=trg_tokeniser,
    src_max_seq_len=data_params['max_seq_len'],
    trg_max_seq_len=data_params['max_seq_len'],
    batch_size=data_params['batch_size'],
)

"""
, src_pad: int, trg_pad: int, trg_sos: int, vocab_size_enc: int, vocab_size_dec: int, d_model: int,
                 d_ff: int, max_seq_len: int, num_layers: Optional[int] = 6, num_heads: Optional[int] = 8,
                 dropout_prob: Optional[float] = 0.1, device: Optional[str] = 'cpu'
"""
# Define the model hyperparameters
transformer_params = {
    "src_pad": src_tokeniser.encode('<pad>')[0],
    "trg_pad": trg_tokeniser.encode('<pad>')[0],
    "trg_sos": trg_tokeniser.encode('<sos>')[0],
    "vocab_size_enc": len(src_tokeniser),
    "vocab_size_dec": len(trg_tokeniser),
    "d_model": 512,
    "d_ff": 2048,  # Four times the size of d_model
    "max_seq_len": data_params['max_seq_len'],
    "num_layers": 6,
    "num_heads": 8,
    "dropout_prob": 0.1,
    "device": device}

# Make the data loader
train_loader = train_data.get_data_loader()
val_loader = val_data.get_data_loader()
test_loader = test_data.get_data_loader()

# Define the model
model = Transformer(**transformer_params)
model.to(device)

# test a random batch tensor through the model
src, trg = next(iter(train_loader))
src = src.to(device)
trg = trg.to(device)

output = model(src, trg[:, :-1])

# Define the loss function
src_pad = src_tokeniser.encode('<pad>')[0]
loss_fn = nn.CrossEntropyLoss(ignore_index=src_pad)

# Define the optimiser
optimiser = optim.Adam(model.parameters(), lr=training_params['lr'])


# Define the training loop
def train(model: nn.Module, data_loader: torch.utils.data.DataLoader, optimiser: torch.optim.Optimizer,
          loss_fn: nn.Module, device: torch.device, epoch: Optional[int], log_interval: int = 100):
    """ Train the model for one epoch

    Args:
        model: The model to train
        data_loader: The data loader to use
        optimiser: The optimiser to use
        loss_fn: The loss function to use
        device: The device to use
        epoch: The epoch number
        log_interval: The interval at which to log the loss
    """

    model.train()
    epoch_loss = 0
    for batch_idx, (src_input, trg_input) in enumerate(data_loader):
        src_input = src_input.to(device)
        trg_input = trg_input.to(device)

        # Forward pass
        output = model(src_input, trg_input[:, :-1])

        # Calculate the loss
        loss = loss_fn(output.reshape(-1, output.shape[-1]), trg_input[:, 1:].reshape(-1))

        # Backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_loss += loss.item()

        # Print the loss
        if batch_idx % log_interval == 0:
            epoch_str = f'Epoch: {epoch}, ' if epoch is not None else ''
            print(f'{epoch_str}Batch: {batch_idx}, Loss: {loss.item()}')
        pass

    return epoch_loss / len(data_loader)


_ = train(model, train_loader, optimiser, loss_fn, device, 1)
# Define the validation loop


pass
