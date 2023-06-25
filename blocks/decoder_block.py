from typing import Optional
import torch
from torch import nn
from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.feed_forward import FeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout_prob: Optional[float] = 0.1):
        """Constructor class for the decoder block of the transformer

        Args:
            d_model (int): Dimension of the Model
            d_ff (int): Hidden dimension of the Feed Forward Layer
            num_heads (int): Number of heads in the multi-head attention
            dropout_prob (float): Dropout probability
        """
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm1 = LayerNorm(d_model)

        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm2 = LayerNorm(d_model)

        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, d_out=d_model,
                                        dropout=dropout_prob)  # Specifying arguments here to avoid ambiguity
        self.layer_norm3 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, trg: torch.Tensor, enc_src: Optional[torch.Tensor] = None,
                trg_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the decoder block using the pre-Norm Architecture from the original paper

        Args:
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len, embedding_dim)
            enc_src (torch.Tensor): Source tensor of shape (batch_size, seq_len, embedding_dim)
            trg_mask (torch.Tensor): Target mask tensor of shape (batch_size, seq_len, seq_len)
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Multi-head Self Attention
        trg2, _ = self.self_attention(trg, trg, trg, trg_mask)

        # Add and Norm
        trg = trg + self.dropout(trg2)
        trg = self.layer_norm1(trg)

        if enc_src is not None:
            # Cross attention
            trg2, _ = self.encoder_attention(trg, enc_src, enc_src, src_mask)

            # Add and Norm
            trg = trg + self.dropout(trg2)
            trg = self.layer_norm2(trg)

        # Feed forward
        trg2 = self.feed_forward(trg)

        # Add and Norm
        trg = trg + self.dropout(trg2)
        trg = self.layer_norm3(trg)

        return trg
