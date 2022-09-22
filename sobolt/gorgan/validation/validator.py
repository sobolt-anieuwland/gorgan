import os
from math import log10
from typing import Dict, Optional, Tuple, Any, List, Union
from skimage.exposure import match_histograms
from sklearn.metrics import f1_score, recall_score, precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from numpy.core.multiarray import ndarray
from pytorch_msssim import SSIM, MS_SSIM
from scipy import stats
from torch_fidelity import calculate_metrics
from torchvision import transforms
from torchvision.models import vgg16, resnet18

from sobolt.gorgan.data import (
    Dataset,
    ProgressiveUpsamplingDecorator,
    pad_renders,
    tensor_to_auxiliary,
    add_title,
)
from sobolt.gorgan.nn import (
    OrientationMagnitudeExtractor,
    FrequencyExtractor,
    NaiveDownsampler,
    PhysicalDownsampler,
)
from sobolt.gorgan.nn import dominant_frequency_percent, get_gradient_metrics
from sobolt.gorgan.graphs.factory import graph_factory
from sobolt.gorgan.graphs.generators import LinearGenerator
from sobolt.gorgan.data.transforms import normalize_sentinel2_l2a_cran

import sobolt

from . import DiscriminatorWrapper
from .. import Gan, gan_factory


class Validator:
    __config: Dict[str, Any]
    __device: torch.device
    __bil_benchmark: nn.Module
    __generator: nn.Module
    __discriminator: nn.Module
    __normalization: Tuple[float, float]
    __VGG16: str = "vgg16"
    __ResNet18: str = "resnet18"

    @staticmethod
    def from_config(
        config: Dict[str, Any],
        model_path: Optional[str] = None,
        opt_args: Dict[str, Any] = {},
        **kwargs,
    ):
        # Check whether a path to model weights has been given
        model_path = model_path if model_path else config["generator"].get("weights")
        config["generator"]["weights"] = model_path  # graph_factory reads this var

        # Specify for detailed output logging to terminal
        quiet = config.get("quiet", False)

        # Load generator graph
        generator = graph_factory(
            config["generator"],
            quiet,
            shape_originals=config["shape_originals"],
            shape_targets=config["shape_targets"],
            **opt_args,
        )

        # Load discriminator graph
        discriminator = graph_factory(
            config["discriminator"], quiet, shape_targets=config["shape_targets"]
        )

        use_gpu = torch.cuda.is_available() and config["use_gpu"]
        device = torch.device("cuda" if use_gpu else "cpu")
        generator.to(device)
        discriminator.to(device)

        gan = gan_factory(config)
        return Validator(config, generator, discriminator, gan, **kwargs)

    def __init__(
        self,
        config: Dict[str, Any],
        generator: nn.Module,
        discriminator: nn.Module,
        gan: Gan,
        quiet: bool = False,
        band_order: str = "RGBI",
        # Expected values: RGB, BGR, RGBI, BGRI, RGBI+, BGRI+
        pretrained_model_name: str = "resnet18",
        pretrained_path: Optional[str] = None,
    ):
        # Initialize basic class variables
        use_gpu = torch.cuda.is_available() and config["use_gpu"]
        self.__device = torch.device("cuda" if use_gpu else "cpu")
        self.__band_order = band_order

        self.__config = config
        self.__renders_rgb = config.get("validation", {}).get("renders", [])
        self.__renders_false_color = (
            config.get("validation", {}).get("args", {}).get("false_color", False)
        )
        self.__upsample_factor = config["upsample_factor"]
        self.__bil_benchmark = LinearGenerator(
            config["shape_originals"], upsample_factor=config["upsample_factor"]
        )
        self.__cycle_graph = gan.cycle.graph
        self.__generator = generator
        self.__discriminator = discriminator
        self.__z_score = config.get("z_score", False)
        self.__normalization = config.get("normalization", (-1.0, 1.0))
        self.__use_auxiliary = config.get("aux_gan", False)
        self.__use_condition = config.get("conditional_gan", False)
        self.__pretrained_model_name = pretrained_model_name
        self.__pretrained_path = pretrained_path
        self.__color_match = config.get("color_match", False)
        self.__color_transfer = config.get("color_transfer", False)
        if self.__color_transfer:
            self.__unet_color_transfer = gan.color_transfer_model.graph

        self.__quiet = quiet

        self.__feature_extractor = self.__get_layers(self.__pretrained_path)
        self.__feature_extractor.eval()
        self.__feature_extractor = self.__feature_extractor.to(self.__device)

        self.__edge_detector = OrientationMagnitudeExtractor(self.__device)
        self.__frequency_extractor = FrequencyExtractor()

        self.__n_bands = config["shape_originals"][0]
        self.__shape_target = config["shape_targets"][-1]
        self.__ssim = SSIM(data_range=1, size_average=False, channel=self.__n_bands)
        self.__ms_ssim = MS_SSIM(data_range=1, size_average=False, channel=self.__n_bands)

    def __get_layers(self, weight_path):
        default = self.__config.get("validation_metrics", {}).get(
            "weights_pretrained", None
        )
        weight_path = weight_path if weight_path else default
        with_weights = weight_path is not None
        graph = nn.Module()
        if self.__pretrained_model_name == Validator.__VGG16:
            graph = vgg16(pretrained=not with_weights)
        elif self.__pretrained_model_name == Validator.__ResNet18:
            graph = resnet18(pretrained=not with_weights)

        if with_weights:
            graph.load_state_dict(torch.load(weight_path))
        layers = []
        if self.__pretrained_model_name == Validator.__VGG16:
            layers = list(graph.features.children())[:-1]
        elif self.__pretrained_model_name == Validator.__ResNet18:
            layers = list(graph.children())[:-1]

        return nn.Sequential(*layers)

    def validate(
        self,
        data_val: DataLoader,
        set_object: Dataset,
        render: bool = True,
        psnr: bool = True,
        ndvi: bool = False,
        mad: bool = False,
        cycle: bool = False,
        cos: bool = True,
        fid: bool = False,
        msssim: bool = False,
        ssim: bool = True,
        edge: bool = False,
        fft: bool = False,
        accuracy: bool = False,
        test: bool = False,
        directory="",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute statistics per tensors.

        Parameters
        ----------
        data_val: Dataloader
            The dataloader to be evaluated during training or test
        set_object: Dataset
            The dataset object containing auxliary classes
        render: Bool
            Convert tensor to image for qualitative analysis
        psnr: Bool
            Peak signal noise ratio between original and generated
        ndvi: Bool
            Per pixel and assesses significance in super-resolution
        mad: Bool
            Median absolute deviation between original & generated
        cos: Bool
            Cosine similarity between learned feature vectors of original & generated
        fid: Bool
            Frechet Inception Distance between original & generated
        cycle: Bool
            MSE between naively-downsampled-generated & original
        mad: Bool
            Mean absolute deviation between original and generated
        msssim: Bool
            Multi-Scale Structural Similarity (SSIM) index score for original and
            generated image
        edge: Bool
            Use sobel filter to get image edge orientation and magnitudes
        fft: Bool
            Computes FFT over input to get frequency histogram
        accuracy: bool
            Computes the accuracy of the discriminator on generated and target images.
        test: Bool
            Returns a single value summary statistics over histogram metrics when using
            the in1 test command
        directory: string
            Target directory for saving image
        Returns
        -------
        Dict[str, float]
            A dictionary with format "metric_name": metric_value.
        """
        if not self.__quiet:
            print("Calculating scores")
        with_nir = False  # "I" in self.__band_order

        batch_metrics: Dict[str, Any] = {}
        histogram_metrics: Dict[str, Any] = {}

        for val_batch_nr, data in enumerate(data_val, 1):
            # Apply generator to validation data
            originals, generated, results, targets = self.__generate(data)
            aux_originals = torch.Tensor([0])

            generated_discriminated = self.__discriminate(generated)
            targets_discriminated = self.__discriminate(targets)

            if self.__use_auxiliary:
                aux_originals = data["originals_auxiliary"].to(self.__device)

            if render and self.__renders_rgb:
                os.makedirs(directory, exist_ok=True, mode=0o777)
                for img_nr in range(generated.shape[0]):
                    self.__render_tensor(
                        generated_discriminated,
                        results,
                        val_batch_nr,
                        img_nr,
                        data,
                        set_object,
                        directory,
                    )

            # Discriminator accuracy
            if accuracy:
                d_g = generated_discriminated["discriminated"]
                d_t = targets_discriminated["discriminated"]

                accuracy_generated = self.__compute_accuracy(d_g, 0)
                accuracy_targets = self.__compute_accuracy(d_t, 1)

                d_gt = self.__cycle(data["targets"].to(self.__device))
                d_gt = self.__generator(d_gt)["generated"]
                d_gt = self.__discriminator(d_gt)["discriminated"]

                confusion_generated = self.reduce(d_g)
                confusion_targets = self.reduce(d_t)
                confusion_gen_targets = self.reduce(d_gt)

                if "val_confusion_g" not in batch_metrics:
                    batch_metrics["val_confusion_g"] = []
                batch_metrics["val_confusion_g"].append(confusion_generated)

                if "val_confusion_t" not in batch_metrics:
                    batch_metrics["val_confusion_t"] = []
                batch_metrics["val_confusion_t"].append(confusion_targets)

                if "val_confusion_gt" not in batch_metrics:
                    batch_metrics["val_confusion_gt"] = []
                batch_metrics["val_confusion_gt"].append(confusion_gen_targets)

                if "val_accuracy_generated" not in batch_metrics:
                    batch_metrics["val_accuracy_generated"] = []
                batch_metrics["val_accuracy_generated"].append(accuracy_generated)

                if "val_accuracy_targets" not in batch_metrics:
                    batch_metrics["val_accuracy_targets"] = []
                batch_metrics["val_accuracy_targets"].append(accuracy_targets)

            # PSNR
            if psnr:
                for img_nr in range(generated.shape[0]):
                    psnr_score = self.__compute_psnr(generated[img_nr], originals[img_nr])
                    if "val_psnr" not in batch_metrics:
                        batch_metrics["val_psnr"] = []
                    batch_metrics["val_psnr"].append(psnr_score)

            # NDVI
            if ndvi:
                for img_nr in range(generated.shape[0]):
                    if with_nir:
                        ndvi_score_o = self.__compute_ndvi(originals[img_nr])
                        ndvi_score_g = self.__compute_ndvi(generated[img_nr])
                        ndvi_score = self.__t_test(ndvi_score_g, ndvi_score_o)
                        if "val_ndvi" not in batch_metrics:
                            batch_metrics["val_ndvi"] = []
                        batch_metrics["val_ndvi"].append(ndvi_score)

            # MAD
            if mad:  # FIXME unsure about meaningfulness of this metric
                for img_nr in range(generated.shape[0]):
                    mad_score_o = self.__compute_mad(originals[img_nr])
                    mad_score_g = self.__compute_mad(generated[img_nr])
                    mad_difference = mad_score_g - mad_score_o
                    if "val_mad" not in batch_metrics:
                        batch_metrics["val_mad"] = []
                    batch_metrics["val_mad"].append(mad_difference)

            # Cosine Similarity
            if cos:
                for img_nr in range(generated.shape[0]):
                    cosine_similarity = self.__compute_cosine_sim(
                        generated[img_nr][:3, :, :], originals[img_nr][:3, :, :]
                    )

                    if "val_cos" not in batch_metrics:
                        batch_metrics["val_cos"] = []
                    batch_metrics["val_cos"].append(cosine_similarity.item())

            # Cycle loss
            if False and cycle:
                cycle_loss = self.__compute_cycle_loss(generated, originals)
                if "val_cycle" not in batch_metrics:
                    batch_metrics["val_cycle"] = []
                batch_metrics["val_cycle"].extend(
                    [value.item() for value in torch.flatten(cycle_loss)]
                )

            # FID & KID
            if fid:
                fid_score = self.__compute_fid(generated, originals)
                if "val_fid" not in batch_metrics or "val_fid" not in batch_metrics:
                    batch_metrics["val_fid"] = []
                    # batch_metrics['batch_kid'] = []
                batch_metrics["val_fid"].append(fid_score["frechet_inception_distance"])
                # batch_metrics['batch_kid'].append(fid['kernel_inception_distance_mean'])

            # MS-SSIM
            if msssim:
                msssim_score = self.__compute_msssim(generated, originals)
                if "val_msssim" not in batch_metrics:
                    batch_metrics["val_msssim"] = []
                for score in msssim_score:
                    batch_metrics["val_msssim"].append(score.item())

            # SSIM
            if ssim:
                ssim_score = self.__compute_ssim(generated, originals)
                if "val_ssim" not in batch_metrics:
                    batch_metrics["val_ssim"] = []
                for score in ssim_score:
                    batch_metrics["val_ssim"].append(score.item())

            # Edge Detection
            if edge or test:
                gradient_magnitude, gradient_orientation = self.__edge_detector(
                    generated[:, :3, :, :]
                )
                gradient_metrics = get_gradient_metrics(
                    gradient_orientation, gradient_magnitude
                )
                if (
                    "val_magnitude_mean" not in batch_metrics
                    or "val_magnitude_std" not in batch_metrics
                    or "val_orientation" not in batch_metrics
                    or "val_orientation_mean" not in batch_metrics
                    or "val_orientation_std" not in batch_metrics
                    or "val_orientation_kurtosis" not in batch_metrics
                    or "val_orientation_skewness" not in batch_metrics
                ):
                    batch_metrics["val_magnitude_mean"] = []
                    batch_metrics["val_magnitude_std"] = []
                    batch_metrics["val_orientation_mean"] = []
                    batch_metrics["val_orientation_std"] = []
                    batch_metrics["val_orientation_kurtosis"] = []
                    batch_metrics["val_magnitude_skewness"] = []
                    histogram_metrics["val_orientation"] = []

                if not test:
                    histogram_metrics["val_orientation"].append(gradient_orientation)

                batch_metrics.update(gradient_metrics)

            # Dominant frequency
            if fft or test:
                shifted_fft = self.__frequency_extractor(generated[:, :3, :, :])
                dominant_frequencies = dominant_frequency_percent(shifted_fft)

                if (
                    "val_fft_dominant" not in batch_metrics
                    or "val_fft" not in batch_metrics
                ):
                    batch_metrics["val_fft_dominant"] = []
                    histogram_metrics["val_fft"] = []

                if not test:
                    histogram_metrics["val_fft"].append(shifted_fft[:, :, :, :, 0])

                batch_metrics["val_fft_dominant"].append(dominant_frequencies)

        if accuracy:
            conf_g = batch_metrics["val_confusion_g"]
            conf_t = batch_metrics["val_confusion_t"]
            true, pred = self.concatenate(conf_g, conf_t)
            self.__confusion_metrics(pred, true, "generated", batch_metrics)

            conf_gt = batch_metrics["val_confusion_gt"]
            true, pred = self.concatenate(conf_gt, conf_t)
            self.__confusion_metrics(pred, true, "gt", batch_metrics)

            batch_metrics.pop("val_confusion_gt")
            batch_metrics.pop("val_confusion_g")
            batch_metrics.pop("val_confusion_t")

        batch_metrics = {
            metric: np.mean(value) for metric, value in batch_metrics.items()
        }

        if not self.__quiet:
            print("Metrics logged.")
            for metric, value in batch_metrics.items():
                print(metric, ": ", value)

        # Remove unused variables for memory optimization
        del generated
        del originals
        del targets
        del generated_discriminated
        del targets_discriminated
        del data
        del results["generated"]
        return batch_metrics, histogram_metrics

    def __zscore_denormalize(
        self, sample: torch.Tensor, stats: torch.Tensor
    ) -> torch.Tensor:

        for channel in range(sample.shape[0]):
            mean = stats[channel][0].item()
            std = stats[channel][1].item()
            sample[channel] = (sample[channel] * std) + mean
            min_ = mean - (2 * std)
            max_ = mean + (2 * std)
            sample[channel] = torch.clip(sample[channel], min_, max_)
            sample[channel] = (sample[channel] - min_) / (max_ - min_)
        return sample

    def concatenate(self, generated: ndarray, target: ndarray) -> Tuple[ndarray, ndarray]:
        target_true = np.full(len(target), 1)
        generated_true = np.full(len(generated), 0)

        true = np.concatenate((generated_true, target_true))
        pred = np.concatenate((generated, target))
        return true, pred

    def reduce(self, value: torch.Tensor) -> float:
        reduced = torch.mean(value).item()
        reduced = 0.0 if reduced < 0.5 else 1.0
        return reduced

    def __confusion_metrics(
        self,
        pred_value: ndarray,
        true_value: ndarray,
        name: str = "",
        batch_metrics: Dict[str, Any] = {},
    ):
        # F1
        (
            batch_metrics["val_{}_f1_g".format(name)],
            batch_metrics["val_{}_f1_t".format(name)],
        ) = f1_score(true_value, pred_value, average=None)

        # Precision
        (
            batch_metrics["val_{}_precision_g".format(name)],
            batch_metrics["val_{}_precision_t".format(name)],
        ) = precision_score(true_value, pred_value, average=None)

        # Recall
        (
            batch_metrics["val_{}_recall_g".format(name)],
            batch_metrics["val_{}_recall_t".format(name)],
        ) = recall_score(true_value, pred_value, average=None)

    def __compute_accuracy(self, value: torch.Tensor, true_value: int) -> int:
        if self.__config["base_loss"] == "wasserstein":
            value = torch.sigmoid(value)

        pred_value = self.reduce(value)

        if true_value == pred_value:
            return 1
        else:
            return 0

    def check_size(self, original: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        if original.size() != generated.size():
            size = generated.size(3)
            original = F.interpolate(original, (size, size))
        return original

    def __compute_msssim(self, generated, original) -> torch.Tensor:
        original = self.check_size(original, generated)
        return self.__ms_ssim(original, generated)

    def __compute_ssim(self, generated, original) -> torch.Tensor:
        original = self.check_size(original, generated)
        return self.__ssim(original, generated)

    def __t_test(self, original, target) -> float:
        t_statistic, p_value = stats.ttest_ind(original, target)
        return p_value.round(2)

    def __compute_ndvi(self, image) -> float:
        image = image.detach().cpu().numpy().astype(np.float32)

        nir_idx = self.__band_order.find("I")  # -1 if no I in bands
        red_idx = self.__band_order.find("R")

        nir_band = image[nir_idx]  # Warning: assumes NIR band exists. If not, handle
        red_band = image[red_idx]  # in caller.

        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir_band - red_band) / (nir_band + red_band)
        return ndvi

    def __compute_mad(self, image) -> ndarray:
        image = image.detach().cpu().numpy()
        return np.median(np.absolute(image - np.median(image)))

    def __compute_cosine_sim(self, generated, original) -> torch.Tensor:
        normalize = lambda x: x
        if self.__pretrained_model_name == Validator.__VGG16:
            normalize = transforms.Normalize(mean=[103, 116, 123], std=[1, 1, 1])
        elif self.__pretrained_model_name == Validator.__ResNet18:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        generated_norm, original_norm = normalize(generated), normalize(original)
        original_norm = F.interpolate(original_norm.unsqueeze(0), size=(224, 224))
        generated_features, original_features = (
            self.__feature_extractor(generated_norm.unsqueeze(0)),
            self.__feature_extractor(original_norm),
        )
        cos = nn.CosineSimilarity()
        return cos(generated_features[0].view(1, -1), original_features[0].view(1, -1))

    def __compute_keypoint_score(self) -> float:
        pass  # TODO

    def __wavelet_transform(self) -> float:
        pass  # TODO

    def __compute_fid(self, generated, original) -> Dict[str, float]:
        generated_tmp_dataset = TemporaryDataset(generated.type(torch.uint8))
        original_tmp_dataset = TemporaryDataset(original.type(torch.uint8))
        FID = calculate_metrics(
            generated_tmp_dataset,
            original_tmp_dataset,
            cuda=False,
            fid=True,
            kid=False,
            verbose=False
            # save_cpu_ram=True,
        )

        if not self.__quiet:
            print("FID score:", FID)
        return FID

    def __compute_psnr(self, generated, reference) -> float:
        generated_resized = generated
        if generated.shape != reference.shape:
            generated_resized = F.interpolate(
                generated.unsqueeze(0), size=reference.shape[1:]
            )
            generated = generated_resized.squeeze()
        mse = F.mse_loss(generated, reference)
        divisor = mse.item() + 0.00000000000001
        return 10 * log10(1 / divisor)

    def __compute_cycle_loss(self, generated, original) -> torch.Tensor:
        _, _, h_o, w_o = original.shape
        _, _, h_g, w_g = generated.shape
        factor = h_g // h_o

        naive_downsampler = NaiveDownsampler(factor)
        physical_downsampler = PhysicalDownsampler(factor)
        if self.__config["cycle"]["type"] == "NaiveDownsampler":
            generated_downsampled = naive_downsampler(generated)
        elif self.__config["cycle"]["type"] == "PhysicalDownsampler":
            generated_downsampled = physical_downsampler(generated)
        else:
            msg = "Unrecognized downsampling type, currently supported options are \
                   NaiveDownsampler and PhysicalDownsampler."
            raise NotImplementedError(msg)

        return F.mse_loss(generated_downsampled, original, reduction="none")

    def __render_tensor(
        self,
        generated_discriminated: Dict[str, torch.Tensor],
        results: Dict[str, torch.Tensor],
        val_batch_nr: int,
        img_nr: int,
        data: Dict[str, torch.Tensor],
        set_object: Dataset,
        directory: str = "",
    ):
        """Renders a tensor to png.
        This function returns the original, generated and target tensors as an RGB image
        for qualitative analysis of GAN performance. If auxiliary classification is
        specified in config, the resulting image includes the true class and the
        discriminator predicted class.

        Parameters
        ----------
        generated_discriminated: Dict[str, torch.Tensor]
            Tensor of a batch of generated images discriminated by the discriminator.
        results: Dict[str, torch.Tensor]
            The generated output dictionary (contains original, generated and target).
        val_batch_nr: int
            The index of a batch.
        img_nr: int
            The index of a single tensor within a batch.
        data: Dict[str, torch.Tensor]
            A dataloader object.
        set_object: Dataset
            A dataset dictionary that can contain class labels for auxiliary.
        directory: str
            Target directory for saving images.
        """
        # Placeholder variables to save rendered tensors
        template = os.path.join(directory, "batch{}, valimg {} of {}.png")
        file_name = template.format(val_batch_nr, img_nr + 1, data["originals"].shape[0])

        # Set variables for rendering
        imgs_combined: List = []
        rendered_dict: Dict = {}

        if self.__color_transfer:
            data["targets"] = self.__unet_color_transfer(
                data["targets"].to(self.__device)
            )["generated"].detach()

        if self.__color_match:
            data = self.match_spectral(data)

        data["originals"] = self.__bil_benchmark(data["originals"])["generated"]

        # When performing haze removal, cycle is not required
        if "haze_mask" in results.keys():
            target_generated = self.__generator(data["targets"].to(self.__device))
            results["target_generated"] = target_generated["generated"]
            results["target_haze_mask"] = target_generated["haze_mask"]
            results["generated_cycled"] = results["generated"] + results["haze_mask"]
        else:
            results["target_generated"] = self.__cycle(data["targets"].to(self.__device))
            results["target_generated"] = self.__generator(results["target_generated"])[
                "generated"
            ]
            results["generated_cycled"] = self.__cycle(results["generated"])

        combined_inputs = {**data, **results, **generated_discriminated}

        # Renders a dictionary of input tensors
        combined_inputs = {k: v[img_nr] for k, v in combined_inputs.items()}

        # Denormalize z score for rendering
        if self.__z_score:  # FIXME how to make this more generic/flexible
            c_g = combined_inputs["generated"]
            c_gt = combined_inputs["gen_sv"]
            c_t = combined_inputs["targets"]
            c_o = combined_inputs["originals"]
            o_stats = combined_inputs["originals_stats"]
            t_stats = combined_inputs["targets_stats"]

            combined_inputs["generated"] = self.__zscore_denormalize(c_g, o_stats)
            combined_inputs["originals"] = self.__zscore_denormalize(c_o, o_stats)
            combined_inputs["targets"] = self.__zscore_denormalize(c_t, t_stats)
            combined_inputs["gen_sv"] = self.__zscore_denormalize(c_gt, t_stats)

        rendered_dict = set_object.render(combined_inputs, self.__renders_false_color)
        for config_dict in self.__renders_rgb:
            key = config_dict["type"]
            # check first if key is present among tensors
            assert key in rendered_dict, "Tensor not found for {}".format(key)
            tensor = rendered_dict[key]
            if key == "infrared":
                assert self.__n_bands == 4, "Can't render infrared for rgb images."
            if tensor.shape[-1] != self.__shape_target:
                tensor = F.interpolate(
                    tensor, scale_factor=self.__upsample_factor, mode="bilinear"
                )
            tensor = tensor.squeeze()
            imgs_combined += [tensor]
            del tensor

        # Pad rendered tensors if size mismatch
        imgs_combined = pad_renders(imgs_combined)

        # Convert tensor to PIL image format
        imgs_combined = [imgs.to(self.__device) for imgs in imgs_combined]
        before_after = torch.cat(imgs_combined, dim=2)
        to_image = transforms.ToPILImage()
        image = to_image(before_after.cpu())

        # Add title on top of combined image for each rendering type class
        image = add_title(image, self.__renders_rgb, self.__shape_target)

        # Add auxiliary variables if available
        if self.__use_auxiliary:
            tensor_to_auxiliary(combined_inputs, set_object, image, img_nr)

        image.save(file_name)

        # Remove unused variables for memory optimization
        del before_after
        del imgs_combined
        del image
        del rendered_dict["originals"]
        del rendered_dict["targets"]
        del rendered_dict["generated"]
        del rendered_dict
        del combined_inputs["originals"]
        del combined_inputs["generated"]
        del combined_inputs["targets"]
        del combined_inputs

    def match_spectral(
        self, batch_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Apply histogram matching between targets and inputs
        matched = match_histograms(
            batch_inputs["targets"].cpu().numpy(), batch_inputs["originals"].cpu().numpy()
        )
        batch_inputs["targets"] = torch.from_numpy(matched.astype(np.float32)).to(
            self.__device
        )
        return batch_inputs

    def __generate(self, data: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
        originals = data["originals"].to(self.__device)
        targets = data["targets"].to(self.__device)

        with torch.no_grad():
            if self.__use_condition:
                conditions_originals = data["originals_conditions"].to(self.__device)
                results = self.__generator(originals, conditions_originals)
            else:
                results = self.__generator(originals)
            generated = results["generated"]

        return originals, generated, results, targets

    def __cycle(self, generated: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cycled = self.__cycle_graph(generated)["generated"]
        return cycled

    def __discriminate(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            discriminated = self.__discriminator(data)

        return discriminated


class TemporaryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)
