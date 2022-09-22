from typing import Dict, Any, Tuple

import torch
import torch.nn.functional as F

from . import Gan


class DummyGenerator:
    def __init__(self, factor: int):
        self.__device = torch.device("cpu")
        self.__factor = factor

    def __call__(self, x) -> Dict[str, torch.Tensor]:
        height = x.shape[-1]
        width = x.shape[-2]
        x = F.interpolate(
            x, size=(height * self.__factor, width * self.__factor), mode="nearest"
        )
        return {"generated": x}


class DummyCycle:
    def __init__(self, factor: int):
        self.__device = torch.device("cpu")
        self.__factor = factor

    def __call__(self, x) -> Dict[str, torch.Tensor]:
        height = x.shape[-1]
        width = x.shape[-2]
        x = F.interpolate(
            x, size=(height // self.__factor, width // self.__factor), mode="nearest"
        )
        return {"generated": x}


class DummyColorTransfer:
    def __init__(self):
        self.__device = torch.device("cpu")

    def __call__(self, x) -> Dict[str, torch.Tensor]:
        return {"generated": x}

    @property
    def graph(self):
        return self

    @property
    def device(self) -> torch.device:
        return self.__device

    def parameters(self):
        return []


class DummyDiscriminator:
    def __call__(self, y) -> Dict[str, torch.Tensor]:
        return {"discriminated": torch.Tensor([1.0] * y.shape[1])}

    @property
    def graph(self):
        return self


class DummyGan(Gan):
    """Implementation of the GAN interface that doesn't do anything. All implemented
    functions, follow the behaviour as specified, but no generator or discriminator is
    actually trained. Calls to train return immediately with a dictionary of dummy
    values. Useful for testing code flow.
    """

    @staticmethod
    def from_config(config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        """Creates all components necessary for a DummyGan training session.

        Parameters
        ----------
        config: Dict[str, Any]
            The config settings for this training run.
        rank: int
            The GPU device this one will train on in the world.
        world_size: int
            The amount of GPUs / devices this training run will have.
        """
        shape_originals = config["shape_originals"]
        shape_targets = config["shape_targets"]
        device = torch.device("cpu")
        device_rank = rank
        num_devices = 0
        factor = config["upsample_factor"]
        generator = DummyGenerator(factor)
        discriminator = DummyDiscriminator()
        cycle = DummyCycle(factor)
        color_transfer = DummyColorTransfer()
        return DummyGan(
            device=device,
            device_rank=device_rank,
            num_devices=num_devices,
            shape_originals=shape_originals,
            shape_targets=shape_targets,
            factor=factor,
            generator=generator,
            discriminator=discriminator,
            cycle=cycle,
            color_transfer=color_transfer,
        )

    __generator: DummyGenerator
    __discriminator: DummyDiscriminator
    __color_transfer: DummyColorTransfer
    __cycle: DummyCycle

    # Multiprocessing variables
    __device: torch.device
    __device_rank: int
    __num_devices: int

    # Size variables
    __shape_originals: Tuple[int, int, int]
    __shape_targets: Tuple[int, int, int]
    __factor: int

    def __init__(
        self,
        device: torch.device,
        device_rank: int,
        num_devices: int,
        factor: int,
        shape_originals: Tuple[int, int, int],
        shape_targets: Tuple[int, int, int],
        generator: DummyGenerator,
        cycle: DummyCycle,
        discriminator: DummyDiscriminator,
        color_transfer: DummyColorTransfer,
    ):
        """Initializes the DummyGan class.

        Parameters
        ----------
        device: torch.device
            The device to put DummyGan training session (CUDA or CPU)
        device_rank: int
            Current device during distributed data parallel training
        num_devices: int
            The total number of devices to train the DummyGan on
        factor: int
            Upsampling factor applied to output
        shape_originals: Tuple[int, int, int]
            Shape of input tensors
        shape_targets: Tuple[int, int, int]
            Shape of generated tensors
        generator: DummyGenerator
            A dummy generator that returns a tensor upsampled with factor
        cycle: DummyCycle
            A dummy cycle that returns a tensor downsampled with factor
        discriminator: DummyDiscriminator
            A dummy discriminator that returns a dummy classification with shape of the
             input
        color_transfer: DummyColorTransfer
            A dummy color transfer that returns the input
        """
        self.__shape_originals = shape_originals
        self.__shape_targets = shape_targets
        self.__device = device
        self.__num_devices = num_devices
        self.__factor = factor
        self.__generator = generator
        self.__discriminator = discriminator

        self.__cycle = cycle
        self.__color_transfer = color_transfer

    def train(self, batch_nr: int, batch_inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Return a dummy train log dictionary
        return {
            "discriminator": {"losses": {"D_loss": 0.0}, "lr": 0},
            "generator": {"losses": {"G_loss": 0.0}, "lr": 0},
        }

    def grow(self):
        print("[dummy] Growing the GAN")

    def to_state_dict(self) -> Dict[str, Any]:
        return {"generator": {}, "discriminator": {}}

    @property
    def generator(self):
        class Container:
            def __init__(self, graph: DummyGenerator):
                self.graph = graph

        return Container(self.__generator)  # type: ignore

    @property
    def discriminator(self):
        class Container:
            def __init__(self, graph: DummyDiscriminator):
                self.graph = graph

        return Container(self.__discriminator)  # type: ignore

    @property
    def cycle(self):
        class Container:
            def __init__(self, graph: DummyCycle):
                self.graph = graph

        return Container(self.__cycle)  # type: ignore

    @property
    def color_transfer_model(self):
        class Container:
            def __init__(self, graph: DummyColorTransfer):
                self.graph = graph

        return Container(self.__color_transfer)  # type: ignore

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def num_devices(self) -> int:
        return self.__num_devices

    @property
    def shape_originals(self) -> Tuple[int, int, int]:
        return self.__shape_originals

    @property
    def shape_targets(self) -> Tuple[int, int, int]:
        return self.__shape_targets

    def __repr__(self) -> str:
        return """
            DUMMYGAN: No layers. This GAN only prints that it trains, showing the data flows through correctly.
        """
