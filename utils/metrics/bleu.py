# TODO: Complete bleu score implementation
from nltk.translate.bleu_score import sentence_bleu
from typing import List, Union, Optional


def bleu_score(predictions: Union[List[str], str], targets: Union[List[str], str], n_gram: Optional[int] = 4) -> float:
    """Calculate the BLEU score between two lists of strings.
    Args:
        predictions (Union[List[str],str]): The predicted sentences.
        targets (Union[List[str],str]): The target sentences.
        n_gram (Optional[int], optional): The n-gram to use. Defaults to 4.
    Returns:
        float: The BLEU score.
    """

    if isinstance(predictions, str):
        predictions = list(predictions)
    if isinstance(targets, str):
        targets = list(targets)

    return sentence_bleu(predictions, targets, weights=(1 / n_gram,) * n_gram)
