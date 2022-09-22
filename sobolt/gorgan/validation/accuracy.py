from typing import List, Tuple


def calc_accuracy(
    predictions: List[int], truths: List[int], label: int
) -> Tuple[float, float, float, float]:
    """Calculate classification metrics.

    This function calculates precision, recall, specificity and recall for the given
    label based on the predictions and truths given.

    Parameters
    ----------
    predictions: List[int]
        The predictor's list of classes represented by an integer.
    truths: List[int]
        The true classes of the samples. Samples are paired with predictions by index.
        This list must therefore have the same length as the predictions list.
    label: int
        The class of which we want to calculate metrics.

    Returns
    -------
    Tuple[float, float, float, float]
        A 4-tuple of f1, precision, recall, specifity.
    """
    # TODO Rewrite as matrix operations on torch tensors
    assert len(predictions) == len(truths)
    items = list(zip(predictions, truths))

    true_pos = sum(1 for (p, t) in items if p == label and t == label)
    false_pos = sum(1 for (p, t) in items if p == label and t != label)

    true_neg = sum(1 for (p, t) in items if p != label and t != label)
    false_neg = sum(1 for (p, t) in items if p != label and t == label)

    eps = 1e-10

    precision = true_pos / (true_pos + false_pos + eps)
    recall = true_pos / (true_pos + false_neg + eps)
    specificity = true_neg / (true_neg + false_pos + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return f1, precision, recall, specificity
