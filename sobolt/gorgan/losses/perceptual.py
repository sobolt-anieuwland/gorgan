import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3


class PerceptualLoss(nn.Module):
    """Perceptual loss requires a pre-trained model (inception V3) that
    can extract features from both original image and generated image. L1 loss is then
    computed between features from original and generated tensors.
    """

    __feature_extractor: nn.Module

    def __init__(self):
        """Initializes the PerceptualLoss class.
        """
        super(PerceptualLoss, self).__init__()

        model = inception_v3(pretrained=True)

        feature_extractor = nn.Sequential(*list(model.children())[:-7]).eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False

        self.__feature_extractor = feature_extractor

    def forward(self, generated: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Get difference between the input and target tensors.

        Parameters
        ----------
        generated: torch.Tensor
            A generated input tensor.
        targets: torch.Tensor
            A reference input tensor we want to compare a generated input with.

        Returns
        -------
        torch.Tensor
            The mean absolute error (L1 loss) returns the difference between
            features extracted from the target input versus generated input.
        """
        targets = self.check_size(targets, generated)

        return F.l1_loss(
            self.__feature_extractor(generated[:, :3, :, :]),
            self.__feature_extractor(targets[:, :3, :, :]),
        )

    def check_size(self, original: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        if original.size() != generated.size():
            size = generated.size(3)
            original = F.interpolate(original, (size, size))
        return original
