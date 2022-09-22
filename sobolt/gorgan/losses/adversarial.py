from typing import Dict, Union, List

import torch
import torch.nn as nn

from sobolt.gorgan.validation import calc_accuracy


class AdversarialDiscriminatorLoss(nn.Module):
    __predictions: Dict[str, List[int]] = {}
    __truths: Dict[str, List[int]] = {}

    class_map = {1: "Real", 0: "Generated"}

    def __init__(self):
        """Initialize the adversarial loss."""
        super().__init__()

    def zero_accuracy(self):
        """To start tracking new accuracy metrics, first zero out previously tracked
        results by calling this method.
        """
        self.__predictions = {}
        self.__truths = {}

    def calc_accuracy(self, bins: Dict[str, List[int]]) -> Dict[str, float]:
        """Calculate the accuracy for the given bins, returning for each the f1,
        precision, recall and specificity metrics.

        These metrics are calculated based on batch metrics collected using
        `track_batch_accuracy`. Call this method with the results of several batches
        to get a reliable result.

        Parameters
        ----------
        bins: Dict[str, int]
            A mapping of bin names to bin class labels. For example:
            {"real_b": 1, "fake_b": 0, "cycled_b": 0}. These names are used to choose
            for which bins to calculate the metrics and to define the name in the result
            dictionary.

        Results
        -------
        Dict[str, float]:
            A dictionary with the metrics for each bin accessible under keys containing
            the bin name.
        """
        assert all(k in self.__predictions for k in bins)
        assert all(k in self.__truths for k in bins)

        accuracy = {}
        for bin in bins:
            for bin_truth in bins[bin]:
                predictions = self.__predictions[bin]
                item_truth = self.__truths[bin]

                f1, prec, rec, spec = calc_accuracy(predictions, item_truth, bin_truth)

                tag = self.class_map[bin_truth]
                accuracy[f"Confusion {bin} / {tag}: F1"] = f1
                accuracy[f"Confusion {bin} / {tag}: Precision"] = prec
                accuracy[f"Confusion {bin} / {tag}: Recall"] = rec
                accuracy[f"Confusion {bin} / {tag}: Specificity"] = spec

        return accuracy

    def track_batch_accuracy(
        self, discriminated: torch.Tensor, truths: Union[int, List[int]], bin: str
    ):
        """Track batch classification results.

        Use this method to add classification results for a discriminator. It is a
        separate method to allow calculating and tracking results for more data than can
        be computed in a single batch.

        Call `calc_accuracy` after several of these calls to get the metrics.

        Parameters
        ----------
        discriminated: torch.Tensor
            The direct outputs of a discriminator trained with this adversarial loss
            function.
        truths: Union[int, List[int]]
            The real class this output belongs to. Can either be an class integer which
            counts for all discriminated items, or a list of integers which must be the
            same length as the amount of items that are discriminated.
        bin: str
            The name of the bin to track with this call. Useful when different bins share
            the same class integer (such as fake (0) and cycled (0) data).
        """
        # Normalize the raw discriminator's output to be classification (0, 1, fake, real)
        classif = self.classify(discriminated)

        # Ensure the true classification is a list of equal length as the contents of
        # the predictor's classification
        if isinstance(truths, int):
            truths = [truths for _ in range(classif.shape[0])]
        assert isinstance(truths, list)
        assert len(truths) == classif.shape[0]

        # Get or create the lists in which we will put our data and extend it with the
        # new classifications
        classif_bin = self.__predictions.get(bin, [])
        classif_truths = self.__truths.get(bin, [])

        classif_bin.extend(classif.tolist())
        classif_truths.extend(truths)

        self.__predictions[bin] = classif_bin
        self.__truths[bin] = classif_truths

    def classify(self, discriminated: torch.Tensor) -> torch.Tensor:
        """Convert discriminator's outputs to classes.

        For LSE and Minimax, the inputs are thresholded with 0.5, values below it becoming
        class 0 and above it class 1. Other losses, such as Wasserstein, want to override
        this method.

        Returns
        -------
        torch.Tensor
            1D tensor converted to classes 0 and 1.
        """
        return (discriminated > 0.5).float().flatten()


class AdversarialGeneratorLoss(nn.Module):
    def __init__(self):
        """Initialize the adversarial loss."""
        super().__init__()
