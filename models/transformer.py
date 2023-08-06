from typing import Optional
import torch
from torch import nn
from models.decoder import Decoder
from models.encoder import Encoder


class Transformer(nn.Module):

    def __init__(self, src_pad: int, trg_pad: int, trg_sos: int, vocab_size_enc: int, vocab_size_dec: int, d_model: int,
                 d_ff: int, max_seq_len: int, num_layers: Optional[int] = 6, num_heads: Optional[int] = 8,
                 dropout_prob: Optional[float] = 0.1, device: Optional[str] = 'cpu'):
        """ Constructor class for the transformer. It consists of both the encoder and the decoder.
        Args:

            src_pad (int): Source padding index
            trg_pad (int): Target padding index
            trg_sos (int): Target start of sentence token
            vocab_size_enc (int): Size of the vocabulary of the encoder
            vocab_size_dec (int): Size of the vocabulary of the decoder
            d_model (int): Dimension of the model
            d_ff (int): Hidden dimension of the Feed Forward Layer
            max_seq_len (int): Maximum sequence length
            num_layers (int): Number of decoder layers
            num_heads (int): Number of heads in the multi-head attention
            dropout_prob (float): Dropout probability
            device (str): Device - 'cpu' or 'cuda'
        """
        super(Transformer, self).__init__()
        self.src_pad = src_pad
        self.trg_pad = trg_pad
        self.trg_sos = trg_sos
        self.encoder = Encoder(vocab_size_enc, d_model, max_seq_len, num_layers, num_heads, d_ff, dropout_prob)
        self.decoder = Decoder(vocab_size_dec, d_model, max_seq_len, num_layers, num_heads, d_ff, dropout_prob)
        self.device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer
        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_len)
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        src_mask = self.get_src_mask(src)
        trg_mask = self.get_trg_mask(trg)
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(trg, encoder_output, trg_mask, src_mask)
        return decoder_output

    def get_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create source mask
        Args:
            src (torch.Tensor): Source tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Source mask tensor of shape (batch_size, seq_len, seq_len)
        """
        # Want to ignore the padding tokens and want shape to be (batch_size, seq_len, seq_len)
        src_mask = (src != self.src_pad).unsqueeze(-2)  # (batch_size, 1, seq_len)

        # repeat the mask so that the shape is (batch_size, seq_len, seq_len)
        src_mask = src_mask & src_mask.transpose(-2, -1)

        return src_mask

    def get_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """Create target mask
        Args:
            trg (torch.Tensor): Target tensor of shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Target mask tensor of shape (batch_size, seq_len, seq_len)
        """
        # What to ignore the padding tokens
        trg_pad_mask = (trg != self.trg_pad).unsqueeze(-2)  # (batch_size, 1, seq_len)
        trg_len = trg.shape[1]
        # What to ignore the future tokens (i.e. tokens that are not yet predicted)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # Final mask ignores both padding and future tokens
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
