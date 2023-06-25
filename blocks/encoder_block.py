from typing import Optional
import torch
from torch import nn
from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.feed_forward import FeedForward


class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout_prob: Optional[float] = 0.1):
        """Constructor class for the Encoder Block of the Transformer
        Args:
            d_model (int): Dimension of the model
            d_ff (int): Hidden Dimension of the feed forward layer
            num_heads (int): Number of heads
            dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(EncoderBlock,self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm_1 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, d_out=d_model, dropout=dropout_prob) # Specifying arguments here to avoid ambiguity
        self.layer_norm_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Encoder Block
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor): Mask tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Attention
        _x = self.attention(x, x, x, mask)

        # Add and Norm
        _x = self.dropout(_x)
        x = self.layer_norm_1(x + _x)

        # Feed forward
        _x = self.feed_forward(x)

        # Add and Norm
        _x = self.dropout(_x)
        x = self.layer_norm_2(x + _x)

        return x


