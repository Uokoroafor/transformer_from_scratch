# transformer_from_scratch

This is a PyTorch implementation of the Transformer model in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
I did this to help me better understand work I had already done on the Tutorial by Andrej Karpathy for nanoGPT and has certainly been helped by other open source repositories.

It enacts the components of the transformer architecture in the post-Norm style, which is the style used in the original paper.

The key components are:
- *Positional Encoding*: This is a sine or cosine function that is added to the input embeddings to give the model a sense of position in the sequence.
<br><br>
- *Scaled Dot Product Attention*: This is the attention mechanism used in the Transformer. It is a dot product between the query and key vectors, scaled by the square root of the dimension of the key vectors. The output is a weighted sum of the value vectors.
<br><br>
- *Multi-Head Attention*: This is a concatenation of multiple attention heads. Each head is a scaled dot product attention mechanism. The output of each head is concatenated and then projected to the output dimension.
<br><br>
- *Feed Forward Network*: This is a two layer fully connected network with a ReLU activation function in between the layers.
<br><br>
- *Residual Connections*: These are connections that allow the gradients to flow through the network. They are added to the output of each sub-layer and then normalised by layer normalisation.
<br><br>
- *Layer Normalisation*: This is a normalisation of the output of each sub-layer. It is a normalisation across the feature dimension.
<br><br>
- *Masking*: This is a masking of the attention weights to prevent the model from attending to future tokens in the sequence.

## Installation
```
git clone https://github.com/Uokoroafor/transformer_from_scratch
cd transformer_from_scratch
pip install -r requirements.txt
```

## Project Structure
```bash
├── README.md
├── data
│   ├── __init__.py
│   └── europarl_fr_en
├── examples
│   ├── __init__.py
│   └── train_fr_en.py
├── models
│   ├── __init__.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── multi_head_attention.py
│   ├── positional_encoding.py
│   ├── residual_block.py
│   └── transformer.py
├── embeddings
│   ├── __init__.py
│   ├── multi_head_attention.py
│   ├── positional_encoding.py
├── requirements.txt
└── utils
    ├── __init__.py
    ├── file_utils.py
    ├── train_utils.py
    ├── data_utils.py
    ├── logging_utils.py
    └── tokeniser.py
```

## Usage
I have now included a number of utility files in the utils folder to help with handling the data and training the model. 
The main file to train on the europarl dataset is train_fr_en.py in the examples folder.

This file can be run with the following command:
```
python examples/train_fr_en.py
```
Note that it is training a model to translate from English to French but it is fairly easy to change this to any other language pair.
## Results
TBC - the run will take a while to complete so this will be updated when there is capacity to run it.

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


