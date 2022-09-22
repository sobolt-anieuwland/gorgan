import os
from typing import List, Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image
import skimage.io
from skimage.transform import resize

from sobolt.gorgan.graphs.factory import graph_factory


class FeatureVisualizer:
    """Generates visualizations of learned features from unactivated convolution layers (
    nn.Module).
    """

    __graph: nn.Module
    __graph_type: str
    __image_tensor: torch.Tensor
    __feature_maps: List[Any]

    @staticmethod
    def from_config(
        config: Dict[str, Any],
        model_path: Optional[str] = None,
        graph_type: str = "generator",
        opt_args: Dict[str, Any] = {},
    ):
        """Creates a Generator and Discriminator instance from a configuration dictionary.

        It does so by using the configuration to determine which graph needs be be
        instantiated and what the optimizer should look like. It might also enable
        or disable certain settings, such as the initialization step for progressive
        training (init_prog_step), or weight setting.

        Parameters
        ----------
        config: Dict[str, Any]
            The configuration file with the relevant parameters for creating
            discriminator and generator instances.
        image_path: str
            The path to any RGB image_path we want to use to inspect the model's learned
            feature maps.
        model_path: Optional[str]
            The path to a graph's weights.
        graph_type: str
            The string name description of the graph. Default is "generator" because
            tends to be of most interest.

        Returns
        -------
        FeatureVisualizer
            Initializes the FeatureVisualizer class.
        """
        # Check whether a path to model weights has been given, no weights is fine too
        model_path = model_path if model_path else config[graph_type].get("weights")
        config[graph_type]["weights"] = model_path  # graph_factory reads this var

        quiet = config.get("quiet", False)

        # Load generator graph
        opt_args = {"init_prog_step": config.get("init_prog_step", 1)}
        generator = graph_factory(
            config["generator"],
            quiet,
            shape_originals=config["shape_originals"],
            **opt_args,
        )

        # Load discriminator graph
        discriminator = graph_factory(
            config["discriminator"], quiet, shape_targets=config["shape_targets"]
        )

        # The graph to conduct feature visualization
        graph = generator if graph_type == "generator" else discriminator

        input_shape = (
            config["shape_originals"][1]
            if graph_type in model_path
            else config["shape_targets"][1]
        )

        bands = config["shape_originals"][0]

        use_gpu = torch.cuda.is_available() and config["use_gpu"]
        device = torch.device("cuda" if use_gpu else "cpu")
        graph.to(device)
        return FeatureVisualizer(graph, device, graph_type, input_shape, bands, quiet)

    def __init__(
        self,
        graph: nn.Module,
        device,
        graph_type: str = "generator",
        input_shape: int = 128,
        bands: int = 3,
        quiet: bool = False,
    ):
        """Initializes the FeatureVisualizer class.

        Parameters
        ----------
        graph: nn.Module
            A generator or discriminator graph we want to extract feature maps from.
        graph_type: str
            The string name description of the graph. Default is "generator" because
            tends to be of most interest.
        input_shape: int
            The shape of the input to pass through the graph. Default is 128.
        bands: int
            The number of bands of the input. Default is 3.
        quiet: bool
            Specify the detailed printing of task information
        """
        # Get graph and related attributes
        self.__graph = graph
        self.__graph_type = graph_type
        self.__device = device

        # Shape of the input image to generate
        self.__input_shape = input_shape
        self.__bands = bands

        self.__quiet = quiet

    @staticmethod
    def process_input_image(
        image_path: str, input_shape: int = 128, bands: int = 3
    ) -> torch.Tensor:
        """We need to process an input image_path from its state_dict before it is passed
        through the network. This includes reading the image_path, and then transforming
        it to the appropriate size and format (a tensor of shape B x C x H x W).

        Parameters
        ----------
        image_path: str
            The path to any RGB image_path we want to use to inspect the model's learned
            feature maps.
        input_shape: int
            The shape of the input to pass through the graph. Default is 128.
        bands: int
            The number of bands of the input. Default is 3.

        Returns
        -------
        torch.Tensor
            A tensor image_path of shape B x C x H x W
        """
        image = skimage.io.imread(image_path)[:, :, :bands]
        image_array = np.array(image)
        __type = image.dtype
        __type = np.iinfo(image_array.dtype)

        image_norm = image / __type.max
        image = resize(
            image_norm, (image.shape[2], input_shape, input_shape), anti_aliasing=True
        )
        image_tensor = torch.from_numpy(image)

        return image_tensor.unsqueeze(0).float()

    def extract_feature_maps(self, graph: nn.Module, convolution_maps: List):
        """This function iterates through a graph to attach to each layer a hook for
        extracting to a dictionary the corresponding weights during a forward pass of the
        network.

        Parameters
        ----------
        graph: nn.Module
            The graph we want to extract feature weights from.
        convolution_maps: List
            The list we are saving the convolution layers to for visualization.
        """

        def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            """Appends the weights ("output") associated with a layer ("module") to a
            list ("convolution_maps") if the layer is a nn.Conv2d.

            Parameters
            ----------
            module: nn.Module
            input: torch.Tensor
            output: torch.Tensor
            """
            if isinstance(module, nn.Conv2d):
                convolution_maps.append(output)

        for layer_name, layer_weights in graph._modules.items():

            # Only register hooks on children modules
            if not isinstance(layer_weights, nn.Sequential) and not isinstance(
                layer_weights, nn.ModuleList
            ):
                layer_weights.register_forward_hook(hook_fn)

            # If come across sequential or module list iterate through children
            self.extract_feature_maps(layer_weights, convolution_maps)

    def generate_visualization(self, out_dir: str, extracted_features: List[Any]):
        """Function uses matplotlib to plot the information extracted from a list of
        convolution layers and then write the visualization-per-layer into a specified
        directory (out_dir).

        Information is defined as, for example, extracted feature maps (floats).
        Matplotlib will generate for each filter in a convolution the corresponding
        feature map (the result of convolving a filter with an image). Thus, the rendering
        of a convolution layer to a file will exhibit a visualization per filer.

        Parameters
        ----------
        out_dir: str
            The directory to write the images to.
        extracted_features: List[Any]
            A list of convolution layer features (type is determined by the kind of
            layer information extraction carried, i.e. feature maps).
        """
        # Directory to save resulting visualizations
        os.makedirs(out_dir, exist_ok=True)

        # Generate feature map from the filters from each convolution layer
        for layer_idx in range(len(extracted_features)):
            plt.figure(figsize=(30, 30))
            layer = extracted_features[layer_idx][0, :, :, :]
            layer = layer.data
            for idx, feature_filter in enumerate(layer):
                if idx == 64:
                    break
                plt.subplot(8, 8, idx + 1)
                plt.imshow(feature_filter.cpu().numpy(), cmap="gray")
                plt.axis("off")
            if not self.__quiet:
                print(f"Saving layer {layer_idx} feature maps...")

            # Save images to directory
            template = os.path.join(out_dir, "{}_featuremap_layer{}.png")
            filename = template.format(self.__graph_type, layer_idx)

            plt.savefig(filename)
            plt.close()

    def render_convolution_layers(self, image_path: str, out_dir: str):
        """Convenience function calling the different methods of feature extraction and
        subsequent visualizations. All visualizations will be written to the out_dir
        specified in visualize_convolution_layer function.

        Parameters
        ----------
        image_path: str
            The path to any RGB image_path we want to use to inspect the model's learned
            feature maps.
        out_dir: str
            The directory to write the images to.
        """
        # Get image_path as tensor
        image_tensor = self.process_input_image(
            image_path, self.__input_shape, self.__bands
        )
        image_tensor_cuda = image_tensor.to(self.__device)

        # Create list to append the layer weights
        convolution_maps: List = []

        # Extract features
        self.extract_feature_maps(self.__graph, convolution_maps)
        output = self.__graph(image_tensor_cuda)

        # Generate visualizationa
        self.generate_visualization(out_dir, convolution_maps)
