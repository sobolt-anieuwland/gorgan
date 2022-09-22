from typing import Dict, List, Union, Any
import numpy as np
import io

import torch
from torchvision import transforms

from .compute_metrics import *
from sobolt.gorgan.validation.performance_metrics.functional import *


class MetricsCollector:
    """Collects all performance metrics computed across batched iterations into a
    single dictionary (format is "metric_name": [a list of metric per batch]).
    """

    __feature_extractor: CosineSimilarity
    __device: torch.device

    def __init__(
        self, device: torch.device = torch.device("cpu"), complete: bool = False
    ):
        """Initializes the MetricsCollector class.

        Parameters
        ----------
        device: torch.device
            The device to carry computation, cuda if available else CPU.
        """
        self.__feature_extractor = CosineSimilarity(device)
        self.__device = device
        self.__complete = complete

    def gather_metrics(
        self,
        originals: torch.Tensor,
        generated: torch.Tensor,
        batch_metrics: Dict[str, Any] = {},
        index: int = 1,
        directory: str = "",
        factor: int = 4,
        cosine: bool = False,
        canny: bool = False,
        radiometric: bool = False,
        saturation: bool = False,
        gradient: bool = False,
        fft: bool = False,
        tva: bool = False,
        performance: bool = False,
    ) -> Dict[str, Any]:
        """Calculates resolution score and similarity score based on original and generated
        data.

        Parameters
        ----------
        originals: torch.Tensor
            The original tensor of shape BxCxHxW we want to generate.
        generated: torch.Tensor
            The generated input tensor of shape BxCxHxW.
        batch_metrics: Dict
            The dictionary that contains all performance metrics.
        index: int
            The current index when iterating through a dataloader object.
        directory: int
            A directory to save images to.
        factor: int
            The upsample factor for an input tensor.
        cosine: bool
            Computes cosine similarity.
        canny: bool
            Computes canny edge density ratio.
        radiometric: bool
            Computes radiometric consistency.
        saturation: bool
            Computes saturation accuracy.
        gradient: bool
            Computes inputs gradients' orientations & magnitudes.
        fft: bool
            Computes fast fourier transform median difference.
        tva: bool
            Computes total variation.
        performance: bool = False,
            Computes summary of metrics in the form of performance metrics of
            similarity & resolution scores.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the performance metrics name and corresponding
            batch-wise outputs as a list, where each index refers to a batch.
        """
        # Dictionary specifying the metrics to compute
        batch_metrics = self.construct_dict(batch_metrics)

        # Get cosine similarity
        if cosine:
            for img_nr in range(generated.shape[0]):
                cosine_similarity = compute_cosine_similarity(
                    originals[img_nr], generated[img_nr], self.__feature_extractor
                )
                batch_metrics["cosine_similarity"].append(cosine_similarity)

        # Get Canny density ratio
        if canny:
            for img_nr in range(generated.shape[0]):
                canny_density = compute_canny_density_ratio(
                    generated[img_nr].unsqueeze(0), originals[img_nr].unsqueeze(0)
                )
                batch_metrics["canny_edge_density_increase"].append(canny_density)

        # Get radiometric consistency & max radiometric distance
        if radiometric:
            (
                radiometric_similarity,
                max_radiometric_distance,
            ) = compute_radiometric_similarity(originals, generated)
            batch_metrics["radiometric_consistency"] = radiometric_similarity
            batch_metrics["max_radiometric_distance"] = max_radiometric_distance

        # Saturation accuracy
        if saturation:
            saturation_accuracy = compute_saturation_accuracy(originals, generated)
            batch_metrics["saturation_accuracy"] = saturation_accuracy

        # Get gradient orientation & magnitude earth mover distance & magnitude increase
        if gradient:
            grad_ori_emd, grad_mag_emd, grad_mag_increase = compute_gradient_stats(
                originals, generated, self.__device
            )
            batch_metrics["grad_orientation_emd"] = grad_ori_emd
            batch_metrics["grad_magnitude_emd"] = grad_mag_emd

        # Increase in median absolute spatial frequency (FFT)
        if fft:
            median_freq_increase = compute_median_frequency_increase(originals, generated)
            batch_metrics["median_absolute_frequency_increase"] = median_freq_increase

        # Get total variation difference
        if tva:
            total_variation = compute_total_variation_difference(originals, generated)
            batch_metrics["tva_difference"] = total_variation

        # Performance metrics
        if performance:
            resolution_score = compute_resolution_score(batch_metrics)
            similarity_score = compute_similarity_score(batch_metrics)
            batch_metrics["similarity"] = similarity_score
            batch_metrics["resolution"] = resolution_score

        # Get fidelity & enhancement
        fidelity, fidelity_mask = compute_fidelity_score(
            originals, generated, index, directory
        )
        enhancement, enhancement_mask = compute_enhancement_score(
            originals, generated, index, directory
        )
        batch_metrics["fidelity"] = fidelity
        batch_metrics["fidelity_mask"] = fidelity_mask
        batch_metrics["enhancement"] = enhancement
        batch_metrics["enhancement_mask"] = enhancement_mask

        return batch_metrics

    def __call__(
        self, lores: np.ndarray, hires: np.ndarray, masks: bool = False
    ) -> List[Dict[str, Any]]:
        """Creates a dictionary of performance metrics for comparing a generated high-
        resolution image to its low-resolution counterpart.

        Parameters
        ----------
        lores: ndarray
            Numpy array containing a low-resolution image (shape =
            batchxchannelsxheightxwidth).
        hires: ndarray
            Numpy array containing a high-resolution image (shape =
            batchxchannelsxheightxwidth)
        masks: bool
            Allows to visualize the full mask (CxHxW) for fidelity and enhancement metrics
            if set to True. Default is False.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionary containing the performance metrics for evaluating a
            generated high-resolution image from a low-resolution one. The default in
            the dictionary are fidelity & enhancement scores and their corresponding
            masks (shape is the same as input numpy arrays) for visualization.
        """
        if len(lores.shape) < 4:
            raise ValueError(
                "Input array needs 4 dimensions but "
                "only {} dimensions found".format(len(lores.shape))
            )

        lores, hires = (
            torch.from_numpy(lores.astype("float32")).unsqueeze(0),
            torch.from_numpy(hires.astype("float32")).unsqueeze(0),
        )

        # Get metrics
        metrics: Dict = {}
        metrics.update(
            {
                "cosine": self.__complete,
                "canny": self.__complete,
                "radiometric": self.__complete,
                "saturation": self.__complete,
                "gradient": self.__complete,
                "fft": self.__complete,
                "tva": self.__complete,
                "performance": self.__complete,
            }
        )

        performance_metrics: List[Dict[str, Any]] = [
            self.gather_metrics(lores[:, idx, :, :, :], hires[:, idx, :, :, :], **metrics)
            for idx in range(lores.shape[1])
        ]

        performance_metrics = [
            {
                metric: value
                for metric, value in pms.items()
                if metric == "fidelity"
                or metric == "enhancement"
                or metric == "fidelity_mask"
                or metric == "enhancement_mask"
            }
            for pms in performance_metrics
        ]

        if not masks:
            performance_metrics = [
                {
                    metric: value
                    for metric, value in pms.items()
                    if metric == "fidelity" or metric == "enhancement"
                }
                for pms in performance_metrics
            ]

        return performance_metrics

    @staticmethod
    def mask_to_memory(mask: torch.Tensor):
        to_image = transforms.ToPILImage()
        mask_img = to_image(mask.cpu())

        # Save to memory
        to_memory = io.BytesIO()
        mask_img.save(to_memory, format="PNG")
        return to_memory.getvalue()

    @staticmethod
    def construct_dict(batch_metrics: Dict = {}) -> Dict[str, Any]:
        """Adds or constructs a dictionary of lists to append all metrics computed
        during each batch.

        Parameters
        ----------
        batch_metrics: Dict
            The batch dictionary to add performance metrics to.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing metric name as keys and values can be floats (
            metrics), lists or numpy ndarray (masks).
        """
        metric_names: List = [
            "cosine_similarity",
            "radiometric_consistency",
            "max_radiometric_distance",
            "saturation_accuracy",
            "grad_orientation_emd",
            "grad_magnitude_emd",
            "median_absolute_frequency_increase",
            "canny_edge_density_increase",
            "tva_difference",
            "similarity",
            "resolution",
            "fidelity",
            "enhancement",
            "fidelity_mask",
            "enhancement_mask",
        ]

        return {
            metric_name: batch_metrics[metric_name]
            if metric_name in batch_metrics
            else []
            for metric_name in metric_names
        }
