# File containing a Byte Pair Encoding Class
import pickle as pkl
import re
from collections import Counter
from typing import List, Tuple, Optional, Dict, Set


class BPETokeniser:
    """Byte Pair Encoding Class for tokenising text data and creating a vocabulary."""

    def __init__(
        self,
        data: str,
        vocab_size: Optional[int] = 1_000,
        sos: Optional[str] = "<sos>",
        eos: Optional[str] = "<eos>",
        pad: Optional[str] = "<pad>",
        unk: Optional[str] = "<unk>",
        only_lower_case: Optional[bool] = False,
    ):
        """Byte Pair Encoding Class
        Args:
            data(str): string to be encoded
            vocab_size(int): Maximum size of the vocabulary
            sos(str): start of sentence symbol
            eos(str): end of sentence symbol
            pad(str): padding symbol
            unk(str): unknown symbol
            only_lower_case(bool): whether to convert all characters to lower case
        """
        self.vocab_size = vocab_size
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.unknown = unk
        self.special_tokens = {self.sos, self.eos, self.pad, self.unknown}
        self.lookup_table = None
        self.reverse_lookup_table = None

        if only_lower_case:
            data = data.lower()

        self.data = self._get_words(data)
        self.tokens = set(data)
        self.vocab = self.create_initial_vocab()

        assert vocab_size > len(
            self.vocab
        ), "Vocab size must be greater than or equal to the size of the initial vocab"

        if self.vocab_size < len(set(data)):
            # raise a warning
            print('Warning: vocab_size is less than the number of unique characters in the data setting vocab_size to '
                  'the number of unique characters in the data')
            self.vocab_size = len(set(data)) + 4 # add 4 for the special tokens

    @staticmethod
    def _get_words(data: str) -> List[List[str]]:
        """Get the words from the data
        Args:
            data(str): string to be encoded
        Returns:
            words(List[List[str]]): list of words where each word is a list of characters
        """
        words = re.findall(r"\S+|\s+", data)

        words = [list(word) for word in words]

        return words

    def create_initial_vocab(self) -> Set[str]:
        """Create the initial vocabulary
        Returns:
            vocab(set): set of characters in the data"""
        vocab = self.tokens
        return vocab | self.special_tokens

    def get_counts(self) -> Counter:
        """Looks at the pairs of tokens and counts their frequency
        Returns:
            counts(Counter): Counter object containing the counts of pairs of characters
        """
        counts = Counter()
        for word in self.data:
            for i in range(len(word) - 1):
                counts[(word[i], word[i + 1])] += 1
        return counts

    @staticmethod
    def get_most_frequent_pair(counts: Counter) -> Tuple[str, str]:
        """Get the most frequent pair of characters
        Args:
            counts(Counter): Counter object containing the counts of pairs of characters
        Returns:
            most_frequent_pair(Tuple[str, str]): most frequent pair of characters
        """
        most_frequent_pair = counts.most_common(1)[0][0]
        return most_frequent_pair

    def merge_most_frequent_pair(self, most_frequent_pair: Tuple[str, str]) -> None:
        """Merge the most frequent pair of characters in the vocabulary and create a new entry in the vocabulary
        Args:
            most_frequent_pair(Tuple[str, str]): most frequent pair of characters
        """
        self.vocab.add("".join(most_frequent_pair))
        # Also update the data
        # if char[0], char[1] in data, replace with char[0] + char[1]
        for word in self.data:
            # Want to check the sublists without using a for loop since we are modifying the list
            i = 0
            while i < len(word) - 1:
                if (
                    word[i] == most_frequent_pair[0]
                    and word[i + 1] == most_frequent_pair[1]
                ):
                    word[i] = "".join(most_frequent_pair)
                    del word[i + 1]
                else:
                    i += 1

    def train(
        self, num_iters: Optional[int] = 1000, verbose: Optional[bool] = False
    ) -> None:
        """Train the BPE model
        Args:
            num_iters(int): number of iterations to train the model
            verbose(bool): whether to print out the number of tokens in the vocab
        """
        for _ in range(num_iters):
            counts = self.get_counts()
            most_frequent_pair = self.get_most_frequent_pair(counts)
            self.merge_most_frequent_pair(most_frequent_pair)
            if len(self.vocab) >= self.vocab_size:
                break

        # Once the model is trained, create a lookup table
        self.create_lookup_table()

        # Create a reverse lookup table
        self.create_reverse_lookup_table()

        # Data file is likely to be large, so once the model is trained, we want to delete the data from the object
        self.data = None

        if verbose:
            print("Training complete")
            self.report_size()

    @staticmethod
    def fill_gaps(my_dict: Dict[str, int]) -> Dict[str, int]:
        """Makes sure the lookup table goes from 0 to vocab_size"""
        keys = list(my_dict.keys())

        for i, key in enumerate(keys):
            my_dict[key] = i

        return my_dict

    def create_lookup_table(self) -> None:
        """Create a lookup table for the vocabulary"""
        # pad, sos, eos are the first three entries in the vocab
        lookup_table = dict()
        lookup_table[self.pad] = 0
        lookup_table[self.sos] = 1
        lookup_table[self.eos] = 2
        lookup_table[self.unknown] = 3

        for i, token in enumerate(self.vocab):
            lookup_table[token] = i + len(lookup_table)

        lookup_table = self.fill_gaps(lookup_table)

        self.lookup_table = lookup_table

    def create_reverse_lookup_table(self) -> None:
        """Reverses the lookup table so that we can decode the data"""
        self.reverse_lookup_table = {v: k for k, v in self.lookup_table.items()}


    def encode(self, data: str) -> List[int]:
        """Encode the data
        Args:
            data(str): string to be encoded
        Returns:
            enc_data(List[int]): list of integers representing the encoded data
        """
        enc_data = []
        for word in self._get_words(data):
            i = 0
            while i < len(word):
                found_token = False
                for j in range(len(word), i, -1):
                    token = "".join(word[i:j])
                    if token in self.vocab:
                        enc_data.append(
                            self.lookup_table.get(
                                token, self.lookup_table[self.unknown]
                            )
                        )
                        i = j
                        found_token = True
                        break
                if not found_token:
                    enc_data.append(
                        self.lookup_table.get(word[i], self.lookup_table[self.unknown])
                    )
                    i += 1
        return enc_data

    def decode(
        self, enc_data: List[int], ignore_special_tokens: Optional[bool] = False
    ) -> List[str]:
        """Decode the encoded data
        Args:
            enc_data(List[int]): list of integers representing the encoded data
            ignore_special_tokens(bool): whether to ignore the special tokens
        Returns:
            dec_data(str): List of decoded tokens
        """
        dec_data = []
        for token in enc_data:
            try:
                # perform a reverse lookup
                word = self.reverse_lookup_table[token]
                if ignore_special_tokens:
                    if word not in [self.pad, self.sos, self.eos]:
                        dec_data.append(word)
                else:
                    dec_data.append(word)

            except KeyError:
                # If the token is not in the vocab, append the unknown token and print a warning
                dec_data.append(self.unknown)
                print("Warning: {} not in vocab".format(token))
        return dec_data

    def decode_words(
        self, enc_data: List[int], ignore_special_tokens: Optional[bool] = True
    ) -> str:
        """Decode the encoded data
        Args:
            enc_data(List[int]): list of integers representing the encoded data
            ignore_special_tokens(bool): whether to ignore the special tokens
        Returns:
            dec_data(str): joined string of decoded tokens
        """
        dec_data = self.decode(enc_data, ignore_special_tokens)
        return "".join(dec_data)

    def save(self, path: str) -> None:
        """Save the BPE model
        Args:
            path(str): path to save the model
        """
        with open(path, "wb") as f:
            pkl.dump(self, f)

    def report_size(self) -> None:
        """Report the size of the lookup table"""
        print("Number of tokens in vocab: {}".format(len(self.lookup_table)))

    def __repr__(self) -> str:
        return "BPE(vocab_size={})".format(self.vocab_size)

    def __str__(self) -> str:
        return "BPE(vocab_size={})".format(self.vocab_size)

    def __len__(self) -> int:
        return self.vocab_size


if __name__ == "__main__":
    # Example Usage
    data = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. Sed sit amet ipsum mauris. \n"
    data += "Maecenas congue ligula ac quam viverra nec consectetur ante hendrerit. Donec et mollis dolor. \n"
    data += "Praesent et diam eget libero egestas mattis sit amet vitae augue. Nam tincidunt congue enim, \n"
    data += "ut porta lorem lacinia consectetur. Donec ut libero sed arcu vehicula ultricies a non tortor. \n"
    data += "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean ut gravida lorem."

    # Create the BPE object
    bpe = BPETokeniser(data)

    # Train the model
    bpe.train(num_iters=10)

    print("Look Up Table: ", bpe.lookup_table)
    print("Reverse Look Up Table: ", bpe.reverse_lookup_table)

    # Encode the data
    encoded_data = bpe.encode("colour coordination is the best. Some might use 'color' instead of 'colour' but I am "
                              "not one of them.")

    # Decode the data
    decoded_data = bpe.decode(encoded_data)
    decoded_words = bpe.decode_words(encoded_data)

    # Print the encoded data
    print("encoded_data", encoded_data)

    # Print the decoded data
    print("decoded_data", decoded_data)
    print("decoded_words", decoded_words)
