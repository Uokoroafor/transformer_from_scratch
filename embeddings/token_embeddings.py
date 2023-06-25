from torch import nn


class TokenEmbeddings(nn.Embedding):

    def __init__(self, vocab_size: int, d_model: int):
        """ Class for token embeddings. These are the embeddings of the input/output tokens before positional encoding.
        Args:
            vocab_size: Vocabulary size
            d_model: Dimension of the model
        """
        super().__init__(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Will just use the default initialisation and forward method for nn.Embedding
