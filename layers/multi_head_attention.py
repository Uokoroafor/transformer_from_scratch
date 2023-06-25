import torch
from torch import nn
from attention import Attention
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """Constructor class for the Multi-Head Attention layer

        Args:
            d_model (int): Dimension of the model
            num_heads (int): Number of heads
        """
        super(MultiHeadAttention, self).__init__()

        # Check if the d_model is divisible by the number of heads
        assert d_model % num_heads == 0, 'd_model must be divisible by the number of heads'

        # Set the d_model and num_heads
        self.d_model = d_model
        self.num_heads = num_heads

        # Set the depth of each head
        self.depth = d_model // num_heads

        # Create the query, key, value and linear layers
        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

        # Create the attention layer
        self.attention = Attention()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the MultiHeadAttention layer

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model)
            mask (Optional[torch.Tensor], optional): Mask tensor of shape (batch_size, seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: MultiHeadAttention output of shape (batch_size, seq_len, d_model)
            torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Get the batch size
        batch_size = query.shape[0]

        # Pass the query, key and value through their respective linear layers
        query = self.query_layer(query)
        key = self.key_layer(key)
        value = self.value_layer(value)

        # Reshape the query, key and value
        # The reshaping is necessary to divide the `d_model` dimensions into `num_heads` number of heads,
        # each having `depth` dimensions.
        query = query.reshape(batch_size, -1, self.num_heads, self.depth)
        key = key.reshape(batch_size, -1, self.num_heads, self.depth)
        value = value.reshape(batch_size, -1, self.num_heads, self.depth)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute the attention output and the attention weights
        attention_output, attention_weights = self.attention(query, key, value, mask)

        # Reshape the attention output
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        # Have seen people use contiguous().view() instead of reshape() here. Will use reshape() for now

        # Pass the attention output through the linear layer
        mha_output = self.linear_layer(attention_output)

        return mha_output, attention_weights
