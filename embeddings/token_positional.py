from torch import nn
from embeddings.positional_encoding import PositionalEncoding
from embeddings.token_embeddings import TokenEmbeddings
from typing import Optional


class TransformerEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: Optional[float] = 0.0,
    ):
        """Class for Transformer Embeddings. Combines the token embeddings and the positional encoding and applies
        dropout if needed
        Args:
            vocab_size: Vocabulary size
            d_model: Dimension of the model
            max_seq_len: Maximum sequence length
            dropout: Dropout probability (default=0.o so no dropout is applied)
        """
        super(TransformerEmbeddings, self).__init__()
        self.token_embeddings = TokenEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """Forward pass of the Transformer Embeddings layer
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Get the embedding of the input
        x = self.token_embeddings(x)

        # Add the positional encoding
        x = self.positional_encoding(x)

        return self.dropout(x)
