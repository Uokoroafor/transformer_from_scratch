from typing import Optional, Tuple
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self):
        """Constructor class for computing scaled dot-product attention weights. Only performs the forward pass and
        does not store any weights
        """
        super(Attention, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the attention layer. Computes the attention weights and the attention output

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, d_v)
            mask (Optional[torch.Tensor], optional): Mask tensor of shape (batch_size, seq_len, seq_len). Defaults is None

        Returns:
            torch.Tensor: Attention output of shape (batch_size, num_heads, seq_len, d_v)
            torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Get the shape of the query
        batch_size, num_heads, seq_len, d_k = query.shape

        # Compute the attention weights
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k + 1e-12)
        # Adding a small constant for numerical stability

        # Apply the mask if it exists
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply the softmax to get the attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Compute the attention output
        attention_output = torch.matmul(attention_weights, value)

        return attention_output, attention_weights
