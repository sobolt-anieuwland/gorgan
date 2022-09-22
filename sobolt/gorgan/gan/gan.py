from typing import Dict, Any, Tuple, Iterable, Union

from PIL.Image import Image
import torch


class Gan:
    """Interface to which all GANs must adhere. In other words, interface applies to a
    typical GAN as originally formulated with just a generator and discriminator, but
    also to CycleGAN with a cycle added, or other imaginable GAN types. Any
    implementation GAN can make itself progressively trainable (ProGAN) by
    implementing the grow function.
    """

    def train(self, batch_nr: int, batch_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Train the GAN for one iteration using this batch.

        Parameters
        ----------
        batch_nr: int
            The batch number
        batch_inputs: Dict[str, Any]
            The inputs of this batch, being a dictionary where the key
            matches the name of the input that it should be used for. The
            values under the keys are all those the generator, discriminator
            and other components in the GAN need, such as the images of the
            original and target domain. Typically, the keys are `"originals"` and
            `"targets"`.

        Returns
        -------
        Dict[str, Any]
            A dictionary with logs regarding training available under its keys.
            Specifically `"losses"` is of interest, because it contains all the losses
            logged during this training iteration.
        """
        raise NotImplementedError()

    def validate(
        self, val_set: Iterable[Dict[str, torch.Tensor]], render: bool = True
    ) -> Dict[str, Union[float, Image]]:
        """Validate the GAN on the given batches of validation data."""
        return {}

    def step_scheduler(
        self, logs: Dict[str, float], epoch: int, idx: int, iterations: int, graph: str
    ):
        """
        Calls a learning rate scheduler step function if specified in config,
        which allows for a decrease in learning rate according to the selected
        technique. Schedule choice includes: reduce on plateau, threshold, and cosine
        annealing with warm restart.

        Parameters
        ----------
        logs: Dict[str, float]:
            A logs dictionary containing the value per loss function.
        epoch: int
            The current epoch of training.
        idx: int
            The current training iteration.
        iterations: int
            The total amount of iteration in a given epoch.
        graph: str
            The graph we want to apply a LR schedule to.
        """
        raise NotImplementedError()

    def grow(self):
        """ Tells GAN to grow in the ProGAN sense. """
        pass

    def generate(self, originals: torch.Tensor, detach: bool = True) -> torch.Tensor:
        predictions = self.generator(originals)
        if detach:
            return predictions.detach()
        return predictions

    @property
    def color_transfer_model(self):
        raise NotImplementedError()

    def discriminate(self, candidates: torch.Tensor, detach: bool = True) -> torch.Tensor:
        predictions = self.discriminator(candidates)
        if detach:
            return predictions.detach()
        return predictions

    @property
    def cycle(self):
        """Get the `Cycle` object this GAN trains.

        Returns
        -------
        Generator
            Note that this is not the graph itself, which is `cycle_s2.graph`.
        """
        raise NotImplementedError()

    @property
    def generator(self):
        """Get the `Generator` object this GAN trains.

        Returns
        -------
        Generator
            Note that this is not the graph itself, which is `generator.graph`.
        """
        raise NotImplementedError()

    @property
    def discriminator(self):
        """Get the `Discriminator` object this GAN trains.

        Returns
        -------
        Discriminator
            Note that this is not the graph itself, which is `discriminator.graph`.
        """
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        raise NotImplementedError()

    @property
    def device_rank(self) -> int:
        raise NotImplementedError()

    @property
    def num_devices(self) -> int:
        raise NotImplementedError()

    @property
    def shape_originals(self) -> Tuple[int, int, int]:
        raise NotImplementedError()

    @property
    def shape_targets(self) -> Tuple[int, int, int]:
        raise NotImplementedError()

    def to_state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def export_onnx(self, inputs, onnx_path: str):
        self.generator.export_onnx(inputs, onnx_path)
