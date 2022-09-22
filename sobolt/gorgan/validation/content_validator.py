from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssim import Ssim
from .psnr import Psnr
from .cosine_similarity import CosineSimilarity


class ContentValidator(nn.Module):
    """A validation module to calculate metrics that compares the contents of two tensors.

    These metrics therefore only make sense for two images intended to contain the same
    or similar pixel content. Some metrics make assumptions about (e.g. cosine receiving
    an image, SSIM receiving data normalized to be between 0 and 1), so enable/disable
    as applicable.
    """

    __val_functions: nn.ModuleDict
    __tracked_metrics: Dict[Tuple[Optional[str], str], List[float]] = {}

    def __init__(
        self,
        num_channels: int,
        psnr: bool = True,
        ssim: bool = True,
        cosine: bool = True,
        cosine_model: str = "resnet18",
    ):
        """Initialize the content validator.

        This validator combines validation functions that compare the content of the one
        tensor with another. In other words, the contents are 'paired'.
        """
        super().__init__()
        self.__val_functions = nn.ModuleDict()

        if ssim:
            self.__val_functions["SSIM"] = Ssim(num_channels).eval()
        if psnr:
            self.__val_functions["PSNR"] = Psnr().eval()
        if cosine:
            self.__val_functions["Cosine similarity"] = CosineSimilarity().eval()

    def forward(
        self, generated: torch.Tensor, target: torch.Tensor, tag: Optional[str] = None
    ):
        """Calculates validation metrics regarding the contents of one tensor to another
        functioning as the label.

        The validation metrics are stored internally to allow calculation over multiple
        batches. Retrieve them using `calc_metrics()`.

        The shapes of the tensors are further subject to the constraints of validation
        metrics used. For example, enabling cosine similarity means that data must have
        3 or more channels. The CS will then only be calculated for the first three.

        In case the generated data is larger than the target data, the target data will be
        automatically bilinearly upsampled to have the same size, thus serving as the
        bilinear benchmark.

        Parameters
        ----------
        generated: torch.Tensor
            The tensor under scrutinity. Accepted shape is [B, ...] where B is the batch
            dimension.
        target: torch.Tensor
            The 'label' of the other tensor, the one serving as the reference of
            correctness. Should have the same dimensions as the generated tensor.
        tag: Optional[str]
            Include a tag in the returned metrics name. This helps distinguishing the same
            metric (e.g. SSIM) calculated for different comparisons (e.g. Domain a: SSIM
            versus Domain b: SSIM).
        """
        assert generated.shape[:2] == target.shape[:2]
        assert len(generated.shape) == len(target.shape)

        # Check generated and target shape for the bilinear benchmark
        if len(generated.shape) == 4:
            # We are dealing with cubic data, e.g. channels of 2D data. Check if the
            # generated data is larger than the target data and upsample the latter if so.
            # Written like this to prevent applying this operation on other tasks
            g_h, g_w = generated.shape[2], generated.shape[3]
            t_h, t_w = target.shape[2], target.shape[3]
            if g_h > t_h and g_w > t_w:
                # Wrong shape, upsample to the right shape
                target = self.up(target, (g_h, g_w))

        # Calculate metrics
        for name, val_function in self.__val_functions.items():
            # Validation function should return a metric per batch item, so a float tensor
            # of shape [B]
            metric = val_function(generated, target)
            metric = metric.flatten().cpu().tolist()
            key = (tag, name)
            self.__tracked_metrics[key] = self.__tracked_metrics.get(key, [])
            self.__tracked_metrics[key].extend(metric)

    def calc_metrics(self, tag: Optional[str] = None) -> Dict[str, float]:
        """Reduce the tracked validation metrics to a single value per metric, suitable
        for logging or plotting.

        Returns
        -------
        Dict[str, float]
            A dictionary of metric names mapping to metric values.
        """
        # Update the metric name to include a tag in the generated metric names
        reduced: Dict[str, float] = {}
        for (tag, name), val_results in self.__tracked_metrics.items():
            metric_key = (
                f"Validation (content) / {name}"
                if tag is None
                else f"Validation {tag} (content) / {name}"
            )
            metric_val = sum(val_results) / len(val_results)
            reduced[metric_key] = metric_val
        return reduced

    def zero_metrics(self):
        """Clean the slate of tracked metrics.

        Necessary after having calculating metrics before, trained a model further, and
        wanting to calculate new metrics.
        """
        self.__tracked_metrics = {}

    def up(self, data: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(data, size, mode="bilinear", align_corners=False)
