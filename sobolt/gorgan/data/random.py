import numpy as np
from typing import Dict, Any, Tuple, List

import torch
import torch.nn.functional as F


class RandomDataset:
    """ Class to read the Celeb A dataset, which is useful for debugging purposes and
    comparing to the Pytorch DCGAN tutorial.

    Notes:
        * This class can not directly be used together with the BaseTrainer / BaseGan.
          You'll need to make hacky modifications to stop errors and start training. This
          is okay because this is just a class for debugging, not for real use.
        * No render functions are implemented, because again, this class is for debugging.
        * The images are stored in /in1/celeba, but can't be read from there. At least
          for me they loaded unbearably slow. Copy to your home folder.
        * The dataset was downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html,
          specifically the file Img/img_align_celeba.zip.
        * Keep the subfolder in the celeba folder. This is necessary for the ImageFolder
          class to find the images.
    """

    @staticmethod
    def from_config(config: Dict[str, Any], normalization: List[float]):
        return RandomDataset(**config["args"])

    def __init__(self, latent_length: int, render_dims: Tuple[int, int] = (64, 64)):
        assert isinstance(latent_length, int)
        assert latent_length < np.prod(render_dims)
        self.__latent_length = latent_length
        self.__render_dims = render_dims

        # self.__celebs = dset.ImageFolder(
        #     root="/home/a.nieuwland/dcgan test/celeba",
        #     transform=transforms.Compose(
        #         [
        #             transforms.Resize(64),
        #             transforms.CenterCrop(64),
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #         ]
        #     ),
        # )

    def __getitem__(self, idx):
        """Return a vector of random noise as originals / input and a celebrity face as
        targets.
        """
        return torch.randn((self.__latent_length, 1, 1))

    def __len__(self):
        return 1

    # def render(self, inputs: torch.Tensor) -> torch.Tensor:
    #     """Renders a tensor of inputs for visualization.

    #     Parameters
    #     ----------
    #     inputs: torch.Tensor
    #         OTensor to be rendered, shaped as self.__latent_shape.

    #     Returns
    #     -------
    #     torch.Tensor
    #         Renders
    #     """
    #     to_pad = np.prod(self.__render_dims) - self.__latent_length
    #     odd = to_pad % 2 == 1
    #     left = right = to_pad // 2
    #     left = left + 1 if odd else left
    #     inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min()) * 255
    #     inputs = F.pad(inputs.squeeze(), (left, right)).reshape(self.__render_dims)
    #     inputs = torch.stack([inputs.reshape(*self.__render_dims)] * 3)
    #     return inputs
