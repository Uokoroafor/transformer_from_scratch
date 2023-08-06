from typing import Optional
import torch
from torch import nn
from blocks.encoder_block import EncoderBlock
from embeddings.token_positional import TransformerEmbeddings


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size_enc: int,
        d_model: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout_prob: float,
    ):
        """Constructor class for the encoder of the transformer

        Args:
            vocab_size_enc (int): Size of the vocabulary of the encoder
            d_model (int): Dimension of the model
            max_seq_len (int): Maximum sequence length
            num_layers (int): Number of encoder layers
            num_heads (int): Number of heads in the multi-head attention
            d_ff (int): Hidden dimension of the Feed Forward Layer
            dropout_prob (float): Dropout probability
        """
        super(Encoder, self).__init__()

        self.embedding_dim = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        self.embedding = TransformerEmbeddings(
            vocab_size=vocab_size_enc,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout_prob,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the encoder

        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_len)
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Apply token and positional embeddings
        src = self.embedding(src)  # Will now have size (batch_size, seq_len, d_model)

        # Apply encoder blocks
        for encoder_block in self.encoder_blocks:
            src = encoder_block(src, src_mask)

        return src
