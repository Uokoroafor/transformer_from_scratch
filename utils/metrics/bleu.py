from nltk.translate.bleu_score import sentence_bleu
from typing import Union, Optional, List
from spacy.tokenizer import Tokenizer


def bleu_score(
    predictions: Union[List[str], str],
    targets: Union[List[str], str],
    n_gram: Optional[int] = 1,
    tokenizer: Optional[Tokenizer] = None,
) -> float:
    """Calculate the BLEU score between two lists of strings.
    Args:
        predictions (Union[List[str],str]): The predicted sentences.
        targets (Union[List[str],str]): The target sentences.
        n_gram (Optional[int], optional): The n-gram to use. Defaults to 1.
        tokenizer (Tokenizer, optional): The tokenizer to use. Defaults to None.

    Returns:
        float: The BLEU score.
    """

    # Tokenize strings if they are not lists
    if isinstance(predictions, str):
        predictions = (
            predictions.split()
            if tokenizer is None
            else [str(token) for token in tokenizer(predictions)]
        )
    if isinstance(targets, str):
        targets = (
            [targets.split()]
            if tokenizer is None
            else [[str(token) for token in tokenizer(targets)]]
        )

    # Modify this line to compute weights correctly
    weights = [1.0 if i == n_gram - 1 else 0.0 for i in range(n_gram)]

    # Notice the change in order for targets and predictions
    return sentence_bleu(targets, predictions, weights=weights)


if __name__ == "__main__":
    prediction1 = "This is a test"
    target1 = "This is a test"
    print(bleu_score(prediction1, target1))

    prediction2 = "This is another test"
    target2 = "This is a test"
    print(bleu_score(prediction2, target2))

    prediction3 = "This is a test"
    target3 = "This is another test"
    print(bleu_score(prediction3, target3))
