# File for prepping data for transformer model for machine translation.
from typing import Optional, List, Tuple, Iterator

import torch
from torch.utils.data import IterableDataset, DataLoader

from utils.tokeniser import BPETokeniser


class DataHandler:
    """Data loader class for loading data for machine translation."""

    def __init__(self, src_file_path: str, trg_file_path: str, src_tokeniser: BPETokeniser, trg_tokeniser: BPETokeniser,
                 src_max_seq_len: Optional[int] = None, trg_max_seq_len: Optional[int] = None, batch_size: int = 32):
        """Initialise the data loader with the data and tokenisers.

        Args:
            src_file_path: The source language sentences.
            trg_file_path: The target language sentences.
            src_tokeniser: The tokeniser for the source language.
            trg_tokeniser: The tokeniser for the target language.
            src_max_seq_len: The maximum sequence length for the source language. Longer sequences will be truncated.
            trg_max_seq_len: The maximum sequence length for the target language. Longer sequences will be truncated.
            batch_size: The batch size.
        """
        self.src = src_file_path
        self.trg = trg_file_path
        self.src_tokeniser = src_tokeniser
        self.trg_tokeniser = trg_tokeniser
        self.batch_size = batch_size

        self.src_pad_idx = self.src_tokeniser.encode(self.src_tokeniser.pad)[0]
        self.trg_pad_idx = self.trg_tokeniser.encode(self.trg_tokeniser.pad)[0]

        self.src_sos_idx = self.src_tokeniser.encode(self.src_tokeniser.sos)[0]
        self.trg_sos_idx = self.trg_tokeniser.encode(self.trg_tokeniser.sos)[0]

        self.src_eos_idx = self.src_tokeniser.encode(self.src_tokeniser.eos)[0]
        self.trg_eos_idx = self.trg_tokeniser.encode(self.trg_tokeniser.eos)[0]

        self.src_unk_idx = self.src_tokeniser.encode(self.src_tokeniser.unk)[0]
        self.trg_unk_idx = self.trg_tokeniser.encode(self.trg_tokeniser.unk)[0]

        self.src_max_seq_len = src_max_seq_len
        self.trg_max_seq_len = trg_max_seq_len

        self.dataset = TranslationIterableDataset(self.src, self.trg, self.src_tokeniser, self.trg_tokeniser)

        self.dataloader = self.get_data_loader()

    def _collate_fn(self, batch: List[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate the batch.

        Args:
            batch: The batch of data.

        Returns:
            The source and target tensors.
        """

        src_batch = []
        trg_batch = []

        for src_line, trg_line in batch:
            # Add eos to end of sentence for source, pad and convert to tensor
            src_line = src_line[:self.src_max_seq_len - 1] + [self.src_eos_idx]
            src_line += [self.src_pad_idx] * (self.src_max_seq_len - len(src_line))
            src_batch.append(torch.tensor(src_line))

            # Add sos to start of sentence for target, pad and convert to tensor
            trg_line = [self.trg_sos_idx] + trg_line[:self.trg_max_seq_len - 2] + [self.trg_eos_idx]
            trg_line += [self.trg_pad_idx] * (self.trg_max_seq_len - len(trg_line))
            trg_batch.append(torch.tensor(trg_line))

        return torch.stack(src_batch), torch.stack(trg_batch)

    def get_data_loader(self) -> DataLoader:
        """Get the data loader.

        Returns:
            The data loader.
        """
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          collate_fn=self._collate_fn)


class TranslationIterableDataset(IterableDataset):

    def __init__(self, src_file_path: str, trg_file_path: str, src_tokenizer: BPETokeniser,
                 trg_tokenizer: BPETokeniser):
        """Initialise the dataset.

        Args:
            src_file_path: The path to the source file.
            trg_file_path: The path to the target file.
            src_tokenizer: The tokenizer for the source language.
            trg_tokenizer: The tokenizer for the target language.
        """
        super().__init__()
        self.src_file_path = src_file_path
        self.trg_file_path = trg_file_path
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

    def __iter__(self) -> Iterator[Tuple[List[int], List[int]]]:
        """Iterate over the dataset.

        Yields:
            The source and target sentences.
        """
        try:
            with open(self.src_file_path, 'r', encoding='utf-8') as src_file, open(self.trg_file_path, 'r', encoding='utf-8') as trg_file:
                for src_line, trg_line in zip(src_file, trg_file):
                    yield self.process_lines(src_line, trg_line)
        except FileNotFoundError:
            print(f"File not found at the specified path.")
            yield None
        except Exception as e:
            print(f"An error occurred: {e}")
            yield None

    def process_lines(self, src_line: str, trg_line: str) -> Tuple[List[int], List[int]]:
        """Process the source and target lines.

        Args:
            src_line: The source line.
            trg_line: The target line.

        Returns:
            The processed (tokenized) source and target lines.
        """
        # make lower case if required by the tokenizers
        if self.src_tokenizer.only_lower_case:
            src_line = src_line.lower()
        if self.trg_tokenizer.only_lower_case:
            trg_line = trg_line.lower()

        src_line = self.src_tokenizer.encode(src_line.strip())
        trg_line = self.trg_tokenizer.encode(trg_line.strip())

        return src_line, trg_line


if __name__ == '__main__':
    src_file_path = '../data/euro_parl_fr_en/english_test.txt'
    trg_file_path = '../data/euro_parl_fr_en/french_test.txt'

    # load data from text files
    with open(src_file_path, 'r', encoding='utf-8') as src_file, open(trg_file_path, 'r', encoding='utf-8') as trg_file:
        src_lines = src_file.readlines()
        trg_lines = trg_file.readlines()

    # amalgamate the data keeping in the new line characters
    src_lines = ''.join(src_lines)
    trg_lines = ''.join(trg_lines)

    # create tokenizers
    src_tokenizer = BPETokeniser(src_lines, only_lower_case=False)
    trg_tokenizer = BPETokeniser(trg_lines, only_lower_case=False)

    # train tokenizers
    src_tokenizer.train(10)
    trg_tokenizer.train(10)

    # create the datahandler
    data_handler = DataHandler(src_file_path, trg_file_path, src_tokenizer, trg_tokenizer, batch_size=32,
                               src_max_seq_len=100, trg_max_seq_len=100)

    # get the data loader
    data_loader = data_handler.get_data_loader()

    data_iter = iter(data_loader)
    for _ in range(5):
        src, trg = next(data_iter)
        print('English: ', ''.join(src_tokenizer.decode(src[0, :].tolist(), True)), '\nFrench: ',
              ''.join(trg_tokenizer.decode(trg[0, :].tolist(), True)), '\n')
