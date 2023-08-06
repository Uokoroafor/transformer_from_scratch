import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        """
        Class for sinusoidal positional encoding. This is added to the input embeddings at the beginning of the encoder/decoder.
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Un-squeezing position to get a column vector (1D tensor to 2D tensor)

        # define div_term in 2 steps to avoid overflow
        div_term = torch.arange(0, d_model, step=2).float()
        div_term = torch.exp(-torch.log(torch.tensor(10000.0)) * div_term / d_model)

        # Using logs instead of raising to a power
        pe[:, 0::2] = torch.sin(position * div_term)
        # 0::2 to get only even indices

        pe[:, 1::2] = torch.cos(position * div_term)
        # 1::2 to get only odd indices

        pe = pe.unsqueeze(0)  # Want it to be of shape (1, max_len, d_model)

        self.register_buffer('pe', pe) # Registering as a buffer so that it is not considered a model parameter

    def forward(self, x):
        # x is of shape (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()

        return x + self.pe[:seq_len, :]
