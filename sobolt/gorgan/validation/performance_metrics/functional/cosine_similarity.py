from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.transforms import transforms


class CosineSimilarity:
    def __init__(self, device: torch.device):
        self.__feature_extractor = self.__get_layers()
        self.__feature_extractor.eval()
        self.__feature_extractor = self.__feature_extractor.to(device)

    def __get_layers(self):
        graph = resnet18(pretrained=True)
        layers = list(graph.children())[:-1]
        return nn.Sequential(*layers)

    def extract_features(
        self, generated: torch.Tensor, original: torch.Tensor
    ) -> Tuple[resnet18, resnet18]:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        generated_norm, original_norm = (
            normalize(generated[:3, :, :]),
            normalize(original[:3, :, :]),  # Cosine similarity is limited to 3 channels
        )
        original_norm = F.interpolate(original_norm.unsqueeze(0), size=(224, 224))

        return (
            self.__feature_extractor(generated_norm.unsqueeze(0)),
            self.__feature_extractor(original_norm),
        )
