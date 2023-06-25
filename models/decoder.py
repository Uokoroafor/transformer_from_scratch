from typing import Optional
import torch
from torch import nn
from blocks.decoder_block import DecoderBlock
from embeddings.token_positional import TransformerEmbeddings


class Decoder(nn.Module):
    def __init__(self, vocab_size_dec: int, d_model: int, max_seq_len: int, num_layers: int, num_heads: int,
                 d_ff: int, dropout_prob: float):
        """Constructor class for the decoder of the transformer

        Args:
            vocab_size_dec (int): Size of the vocabulary of the decoder
            d_model (int): Dimension of the model
            max_seq_len (int): Maximum sequence length
            num_layers (int): Number of decoder layers
            num_heads (int): Number of heads in the multi-head attention
            d_ff (int): Hidden dimension of the Feed Forward Layer
            dropout_prob (float): Dropout probability
        """
        super(Decoder, self).__init__()

        self.embedding_dim = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        self.embedding = TransformerEmbeddings(vocab_size=vocab_size_dec, d_model=d_model,
                                               max_seq_len=max_seq_len, dropout=dropout_prob)

        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                                                          dropout_prob=dropout_prob) for _ in range(num_layers)])

        self.linear = nn.Linear(d_model, vocab_size_dec)

    def forward(self, trg: torch.Tensor, src: torch.Tensor, trg_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the decoder

        Args:
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len)
            src (torch.Tensor): Source tensor of shape (batch_size, seq_len) - output of the encoder
            trg_mask (torch.Tensor): Target mask tensor of shape (batch_size, seq_len, seq_len)
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Apply token and positional embeddings
        trg = self.embedding(trg)

        # Apply decoder blocks
        for decoder_block in self.decoder_blocks:
            trg = decoder_block(trg, src, trg_mask, src_mask)

        # Apply linear layer
        output = self.linear(trg)
        return output
