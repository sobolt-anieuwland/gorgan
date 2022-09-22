import torch
from torch import nn
import pickle
from typing import Tuple, List
from os import path


class LocalPhaseCoherenceLoss(nn.Module):
    """ Adaptation as Loss of Local Phase Coherence (LCP) sharpness index.

    LCP applies several Gabor Filters to the fourier trnsform of a greyscale image, and
    combines the resulting maps into a unique scalar that quantifies the percieved
    sharpness of an image. Below are links to the original paper and the author's matlb
    implementation.

    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.706.1124&rep=rep1&type=pdf
    http://vision.eng.shizuoka.ac.jp/s3/
    """

    __gabor_filters: List[torch.Tensor]

    def __init__(self):
        super(LocalPhaseCoherenceLoss, self).__init__()
        """Initializes LocalPhaseCoherence class, gabor filters are imported as a
        dictionary of tensors from a pickle file.
        """
        self.__gabor_filters = self.__load_filters()

    def __load_filters(self) -> List[torch.Tensor]:
        """Function to load Gabor Filters from external pickle file.

        TODO: create a GaborFilters class to generate custom filters.

        Returns
        -------
        List[torch.Tensor]
            list of Gabor filters casted as tensors.
        """
        # TODO don't load from pickle
        path_file = "/in1/Gabor-Filters/gabor_filters_bank_512.pickle"
        assert path.exists(path_file), f"File {path_file} does not exist!"
        with open(path_file, "rb") as f:
            gabor_filters = pickle.load(f)
        gabor_filters = [torch.tensor(gabor_filters[f"filter_{i}"]) for i in range(24)]
        return gabor_filters

    def __rgb2gray(self, img: torch.Tensor) -> torch.Tensor:
        """Convert RGB image to grayscale, for multiband image the extra bands are just
        discarded.
        Returns
        -------
        torch.Tensor
            grayscale version of input image
        """
        img_gray = (
            (0.2126 * img[0, :, :]) + (0.7152 * img[1, :, :]) + (0.0722 * img[2, :, :])
        ) * 255
        return img_gray

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        """Calculate local phase coherence loss.

        Parameters
        ----------
        generated: torch.Tensor
            A generated input tensor.

        Returns
        -------
        torch.Tensor
            si, the sharpness index of the input image.
        """
        image = generated.squeeze(0)
        image = self.__rgb2gray(image)
        scales = [1, 1.5, 2]
        w = [1, -3, 2]
        c = 2
        beta_k = 1e-4
        n_orientations = 8
        n_scales = len(scales)
        rows, cols = image.shape[0], image.shape[1]
        b = round(min(rows, cols) / 16)

        # Apply 2D Fourier Transformation to grayscale image
        img_fft = torch.rfft(image, 2)
        s_lpc = torch.ones((rows, cols, n_orientations))

        # Apply Gabor filters in the frequency space. Orientation and Scale refers to the
        # type of filer applied.
        energy = torch.zeros(rows, cols, n_orientations)
        filter_ind = 0
        for ind_o in range(n_orientations):
            m = torch.zeros((rows, cols, n_scales))
            for ind_s in range(n_scales):
                gf = self.__gabor_filters[filter_ind].unsqueeze(2).expand(-1, -1, 2)
                gf.to(generated.device)
                gf.requires_grad = False
                m[:, :, ind_s] = torch.ifft(img_fft * gf, 2)
                s_lpc[:, :, ind_o] = s_lpc[:, :, ind_s].clone() * (
                    m[:, :, ind_s].clone() ** w[ind_s]
                )
                filter_ind += 1

            e = torch.abs(m[:, :, 0].clone())
            e_center = e[(b + 1) : (-b), (b + 1) : (-b)].clone()
            e_mean = torch.mean(e_center)
            e_std = torch.std(e_center)
            T = e_mean + (2 * e_std)
            e[(e - T) < 0] = 0
            energy[:, :, ind_o] = e.clone()

        # Compute Local Phase Coherence Map, original implementation use angle() to
        # compute the pixel-wise phase angle of the maps, we approximate it with atan2
        # since torch does not support autograd for torch.angle()
        s_lpc_map = torch.cos(torch.atan2(s_lpc, s_lpc))
        s_lpc_map[s_lpc_map < 0] = 0
        lpc_map = torch.sum((s_lpc_map * energy), 2) / (torch.sum(energy, 2) + c)
        lpc_map_center = lpc_map[b + 1 : -b, b + 1 : -b]

        # Compute final sharpness index score.
        sorted_si, _ = torch.sort(lpc_map_center, descending=True)
        N = len(sorted_si)
        u = torch.exp(-(torch.arange(N) // (N)) // beta_k)
        si = torch.sum(sorted_si * u) / torch.sum(u)
        return 1 / si
