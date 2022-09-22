from typing import List
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as tvt
import torchvision.models as tvm


class CosineSimilarity(nn.Module):
    """Module to calculate the cosine similarity per batch item.

    A feature extraction module provided by torchvision is used for this. Available
    are resnet18 and vgg16, although vgg16 is currently not advised to be used, because
    its normalization seem off.
    """

    __feature_extractor: nn.Module
    __normalize: nn.Module
    __cos_sim: nn.CosineSimilarity

    def __init__(self, model: str = "resnet18"):
        """Initialize cosine similarity metric module.

        Parameters
        ----------
        model: str
            Choose a model to use for feature extraction. Options are resnet18 (default)
            and vgg16.
        """
        super().__init__()

        # Select a model and create a feature extracting sequence of layers out of it,
        # including a normalization layer
        layers: List[nn.Module]
        normalize: nn.Module
        if model == "resnet18":
            graph = tvm.resnet18(pretrained=True)
            layers = list(graph.children())[:-1]

            means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            normalize = tvt.Normalize(mean=means, std=stds)
        elif model == "vgg16":
            warnings.warn("VGG16 normalization has not been vetted.", RuntimeWarning)
            graph = tvm.vgg16(pretrained=True)
            layers = list(graph.features.children())[:-1]

            # TODO Revisit before usage! Values don't seem to be correct.
            # The integers only apply to uint8 data, while the floats (calculated by
            # dividing by 255) are in BGR order and different from the resnet ones. The
            # values of the stds are also suspicious.
            # means, stds = [0.404, 0.455, 0.482], [1, 1, 1]
            means, stds = [103, 116, 123], [1, 1, 1]
            normalize = tvt.Normalize(mean=means, std=stds)
        else:
            raise ValueError(f"Unknown CosineSimilarity validation model {model}")

        # Final class objects for calculating the CS metric
        self.__normalize = normalize
        self.__feature_extractor = nn.Sequential(*layers)
        self.__cos_sim = nn.CosineSimilarity()

        for param in self.__feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the cosine similarity per batch item.

        Parameters
        ----------
        generated: torch.Tensor
            Assumed to be a tensor of image data with the shape [B, C, H, W] where
            C is at least 3. Only the first 3 channels are taken into accounts; others
            are ignored. H and W may be any size but are recommended to be square and
            over 224.
        target: torch.Tensor
            Tensor that is the label for the generated tensor. The same conditions apply
            as for generated. Shape must be the same as generated.

        Returns
        -------
        torch.Tensor
            A tensor of shape [B] with floats corresponding to the cosine similarity
            metrics per batch item.
        """
        assert generated.shape == target.shape
        assert generated.shape[1] >= 3

        cos_sims = torch.zeros((generated.shape[0],), device=generated.device)
        for batch_item in range(generated.shape[0]):
            # Apply normalization torchvision models expect
            # This is probably not correct because our data is already normalized
            # TODO Rewrite so it works without the loop and normalization per item
            features_g = self.__normalize(generated[batch_item, :3]).unsqueeze(dim=0)
            features_t = self.__normalize(target[batch_item, :3]).unsqueeze(dim=0)

            # Torchvision requires inputs to be at least 224 x 224
            if features_g.shape[2] < 224 or features_g.shape[3] < 224:
                features_g = F.interpolate(features_g, size=(224, 224), mode="bicubic")
                features_t = F.interpolate(features_t, size=(224, 224), mode="bicubic")

            # Extract features from torchvision model
            features_g = self.__feature_extractor(features_g)
            features_t = self.__feature_extractor(features_t)

            # Reshape tensors and calculate CosSim
            assert features_g.shape[0] == features_t.shape[0] == 1
            features_g = features_g[0].view(1, -1)
            features_t = features_t[0].view(1, -1)
            cos_sims[batch_item] = self.__cos_sim(features_g, features_t)
        return cos_sims
