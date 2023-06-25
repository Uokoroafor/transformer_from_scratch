from typing import Optional
import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, d_out: Optional[int], dropout: Optional[float] = 0.1):
        """Constructor class for the Positionwise Feed Forward layer

        Args:
            d_model (int): Dimension of the model
            d_ff (int): Dimension of the feed forward layer
            d_out (int): Dimension of the output
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(FeedForward, self).__init__()

        if d_out is None:
            d_out = d_model

        # Create the feed forward layers
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()  # For Non-linearity
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Positionwise Feed Forward layer

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x
