# This file contains a training loop for the Transformer model on the EuroParl dataset -
# see data/euro_parl_fr_en/Europarl_Parallel_Corpus.html for more information.

import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import DataHandler
from utils.tokeniser import BPETokeniser as Tokeniser  # This is the object to be un-pickled
from utils.train_utils import Trainer
from models.transformer import Transformer
import os
import pickle as pkl

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the training hyperparameters
training_params = {
    "num_epochs": 10,
    "batch_size": 32,
    "lr": 0.0001,
    "path": "../data/euro_parl_fr_en",
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
src_pad = train_data.src_pad_idx
loss_fn = nn.CrossEntropyLoss(ignore_index=src_pad)

# Define the optimiser
optimiser = optim.Adam(model.parameters(), lr=training_params['lr'])

## Define the training loop
# def model_loop(model: nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: nn.Module, device: torch.device,
#                optimiser: Optional[torch.optim.Optimizer] = None, epoch: Optional[int] = None,
#                log_interval: Optional[int] = None, method: str = 'train') -> float:
#     """ Train the model for one epoch
#
#     Args:
#         model: The model to train
#         data_loader: The data loader to use
#         loss_fn: The loss function to use
#         device: The device to use
#         optimiser: The optimiser to use
#         epoch: The epoch number. Only useful for logging
#         log_interval: The interval at which to log the loss
#         method: The method to use, either 'train' or 'val'
#
#     Returns:
#         The average loss for the epoch or evaluation
#     """
#
#     if method == 'train':
#         model.train()
#     elif method == 'val':
#         model.eval()
#     else:
#         raise ValueError(f'Invalid method: {method}. Only "train" and "val" are valid methods.')
#     epoch_loss = 0
#     batch_idx = 0
#
#     for batch_idx, (src_input, trg_input) in enumerate(data_loader):
#         src_input = src_input.to(device)
#         trg_input = trg_input.to(device)
#
#         # Forward pass
#         output = model(src_input, trg_input[:, :-1])
#
#         # Calculate the loss
#         loss = loss_fn(output.reshape(-1, output.shape[-1]), trg_input[:, 1:].reshape(-1))
#
#         if method == 'train':
#             # Backward pass
#             optimiser.zero_grad()
#             loss.backward()
#             optimiser.step()
#
#         epoch_loss += loss.item()
#
#         # Print the loss every log_interval batches
#         if log_interval is not None:
#             if batch_idx % log_interval == 0:
#                 epoch_str = f'Epoch: {epoch}, ' if epoch is not None else ''
#                 print(f'{epoch_str}Batch: {batch_idx}, Loss: {loss.item(): 4f}')
#
#     return epoch_loss / (batch_idx + 1)
#
#
# train_loss = model_loop(model, train_loader, loss_fn, device, optimiser, epoch=1)
# val_loss = model_loop(model, val_loader, loss_fn, device, method='val')
#
# print(f'Train loss: {train_loss: 4f}')
# print(f'Val loss: {val_loss: 4f}')
# # Define the validation loop

# Create the trainer
trainer = Trainer(model=model, train_data=train_loader, val_data=val_loader, loss_fn=loss_fn, optimiser=optimiser,
                  device=device, path=training_params['path'])

# Train the model
model, _, _ = trainer.train(epochs=training_params['num_epochs'], save_model=True, plotting=True,
              verbose=True, eval_every=1, early_stopping=False, early_stopping_patience=10)

# Test a phrase
phrase = 'The way around an obstacle is through it.'
phrase_tokens = train_data.prep_string(phrase)
phrase_tokens = torch.tensor(phrase_tokens).unsqueeze(0).to(device)

# Get the model predictions
preds = model(phrase_tokens)

# Convert the predictions to tokens
preds = torch.argmax(preds, dim=-1).squeeze(0)

# Convert the tokens to a string
pred_str = train_data.output_string(preds)
print(pred_str)

