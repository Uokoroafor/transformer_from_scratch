# This file contains a training loop for the Transformer model on the EuroParl dataset -
# see data/europarl_fr_en/Europarl_Parallel_Corpus.html for more information.

import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import DataHandler
from utils.tokeniser import (
    BPETokeniser as Tokeniser,
)  # This is the object to be un-pickled
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
    "path": "../data/europarl_fr_en",
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
    "train": "../data/europarl_fr_en/english_train.txt",
    "val": "../data/europarl_fr_en/english_val.txt",
    "test": "../data/europarl_fr_en/english_test.txt",
}

trg_file_paths = {
    "train": "../data/europarl_fr_en/french_train.txt",
    "val": "../data/europarl_fr_en/french_val.txt",
    "test": "../data/europarl_fr_en/french_test.txt",
}

src_tokeniser_pth = "../data/europarl_fr_en/english_tokeniser_50_epochs.pkl"
trg_tokeniser_pth = "../data/europarl_fr_en/french_tokeniser_50_epochs.pkl"

# Load the tokenisers if they exist else create them
if not (os.path.exists(src_tokeniser_pth) and os.path.exists(trg_tokeniser_pth)):
    # Run the data_prep.py script to create the tokenisers
    os.system("python data_prep.py")

with open(src_tokeniser_pth, "rb") as src_tokeniser_file:
    src_tokeniser = pkl.load(src_tokeniser_file)
with open(trg_tokeniser_pth, "rb") as trg_tokeniser_file:
    trg_tokeniser = pkl.load(trg_tokeniser_file)

# Define the data handler
train_data = DataHandler(
    src_file_path=src_file_paths["train"],
    trg_file_path=trg_file_paths["train"],
    src_tokeniser=src_tokeniser,
    trg_tokeniser=trg_tokeniser,
    src_max_seq_len=data_params["max_seq_len"],
    trg_max_seq_len=data_params["max_seq_len"],
    batch_size=data_params["batch_size"],
)

val_data = DataHandler(
    src_file_path=src_file_paths["val"],
    trg_file_path=trg_file_paths["val"],
    src_tokeniser=src_tokeniser,
    trg_tokeniser=trg_tokeniser,
    src_max_seq_len=data_params["max_seq_len"],
    trg_max_seq_len=data_params["max_seq_len"],
    batch_size=data_params["batch_size"],
)

test_data = DataHandler(
    src_file_path=src_file_paths["test"],
    trg_file_path=trg_file_paths["test"],
    src_tokeniser=src_tokeniser,
    trg_tokeniser=trg_tokeniser,
    src_max_seq_len=data_params["max_seq_len"],
    trg_max_seq_len=data_params["max_seq_len"],
    batch_size=data_params["batch_size"],
)

# Define the model hyperparameters
transformer_params = {
    "src_pad": src_tokeniser.encode("<pad>")[0],
    "trg_pad": trg_tokeniser.encode("<pad>")[0],
    "trg_sos": trg_tokeniser.encode("<sos>")[0],
    "vocab_size_enc": len(src_tokeniser),
    "vocab_size_dec": len(trg_tokeniser),
    "d_model": 512,
    "d_ff": 2048,  # Four times the size of d_model
    "max_seq_len": data_params["max_seq_len"],
    "num_layers": 6,
    "num_heads": 8,
    "dropout_prob": 0.1,
    "device": device,
}

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
optimiser = optim.Adam(model.parameters(), lr=training_params["lr"])

# Create the trainer
trainer = Trainer(
    model=model,
    train_data=train_loader,
    val_data=val_loader,
    loss_fn=loss_fn,
    optimiser=optimiser,
    device=device,
    path=training_params["path"],
)

# Train the model
model, _, _ = trainer.train(
    epochs=training_params["num_epochs"],
    save_model=True,
    plotting=True,
    verbose=True,
    eval_every=1,
    early_stopping=True,
    early_stopping_patience=10,
)

# Test a phrase
phrase = "The way around an obstacle is through it."
phrase_tokens = train_data.prep_string(phrase)
phrase_tokens = torch.tensor(phrase_tokens).unsqueeze(0).to(device)

# Get the model predictions
preds = model(phrase_tokens)

# Convert the predictions to tokens
preds = torch.argmax(preds, dim=-1).squeeze(0)

# Convert the tokens to a string
pred_str = train_data.output_string(preds)
print(pred_str)
