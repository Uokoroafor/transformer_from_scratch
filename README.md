# transformer_from_scratch

This is a PyTorch implementation of the Transformer model in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
I did this to help me better understand work I had already done on the Tutorial by Andrej Karpathy for nanoGPT and has certainly been helped by other open source repositories.

It enacts the components of the transformer architecture in the pre-Norm style, which is the style used in the original paper.

The key components are:
- *Positional Encoding*: This is a sine and cosine function that is added to the input embeddings to give the model a sense of position in the sequence.

- *Scaled Dot Product Attention*: This is the attention mechanism used in the Transformer. It is a dot product between the query and key vectors, scaled by the square root of the dimension of the key vectors. The output is a weighted sum of the value vectors.

- *Multi-Head Attention*: This is a concatenation of multiple attention heads. Each head is a scaled dot product attention mechanism. The output of each head is concatenated and then projected to the output dimension.

- *Feed Forward Network*: This is a two layer fully connected network with a ReLU activation function in between the layers.

- *Residual Connections*: These are connections that allow the gradients to flow through the network. They are added to the output of each sub-layer and then normalised by layer normalisation.

- *Layer Normalisation*: This is a normalisation of the output of each sub-layer. It is a normalisation across the feature dimension.

- *Masking*: This is a masking of the attention weights to prevent the model from attending to future tokens in the sequence.

## Installation
```
git clone https://github.com/Uokoroafor/transformer_from_scratch
cd transformer_from_scratch
pip install -r requirements.txt
```
## Usage
I have not yet included a training loop. This is purely for my edification. I will add a training loop in the future.

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)


## License
MIT


