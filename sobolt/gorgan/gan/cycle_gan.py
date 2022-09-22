# TODO To continue

from typing import Dict, Any, Tuple, Union, Optional, Iterable, List

import torch
import torch.nn as nn
import torch.optim as optim
from PIL.Image import Image

from sobolt.gorgan.data.render import Renderer, SisrHareRenderer
from sobolt.gorgan.graphs import graph_factory
from sobolt.gorgan.losses import (
    AdversarialGeneratorLoss,
    AdversarialDiscriminatorLoss,
    LeastSquaredErrorDiscriminatorLoss,
    LeastSquaredErrorGeneratorLoss,
    CompositeLoss,
)
from sobolt.gorgan.optim import optimizer_factory
from sobolt.gorgan.validation import ContentValidator
from .gan import Gan


class CycleGan(Gan):
    """Class for training a CycleGan, where both A2B and B2A generators are learned"""

    @staticmethod
    def from_config(config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        """Creates all components necessary for a CycleGan training session.

        Parameters
        ----------
        config: Dict[str, Any]
            The config settings for this training run.
        rank: int
            The GPU device this one will train on in the world.
        world_size: int
            The amount of GPUs / devices this training run will have.
        """
        # Specify for detailed output logging to terminal
        quiet = config.get("quiet", False)

        # Extract arbitrary settings
        shape_originals = config["shape_originals"]
        shape_targets = config["shape_targets"]
        use_gpu = config.get("use_gpu", True)

        g_train_every_n_iters = config["generator"].get("train_every_n_iters", 1)
        d_train_every_n_iters = config["discriminator"].get("train_every_n_iters", 1)

        loss_weights = config.get("loss_weights", {})
        g_adv_weight = loss_weights.get("adversarial_generator", 1.0)
        d_adv_weight = loss_weights.get("adversarial_discriminator", 1.0)

        # Initialize generator and discriminator graphs
        graph_args = {"quiet": quiet, "base_loss": "least-squared-error"}
        discriminator_a = graph_factory(
            config["discriminator"], shape_targets=shape_targets, **graph_args
        )
        discriminator_b = graph_factory(
            config["discriminator"], shape_targets=shape_targets, **graph_args
        )
        generator_a2b = graph_factory(
            config["generator"], shape_originals=shape_originals, **graph_args
        )
        config_b2a = config["cycle"]
        b2a_args = {
            "factor": config["upsample_factor"],
            "shape_originals": shape_originals,
            "quiet": quiet,
            "base_loss": "least-squared-error",
        }
        generator_b2a = graph_factory(config_b2a, **b2a_args)

        # Initialize generator optimizer
        optim_config = config["generator"].get("optimizer", {})
        optimizer_g_a = None
        if "type" in optim_config:
            optim_type = optim_config["type"]
            optim_args = optim_config.get("args", {})
            params = list(generator_a2b.parameters())
            optimizer_g_a = optimizer_factory(params, optim_type, optim_args)

        optim_config = config["cycle"].get("optimizer", {})
        optimizer_g_b = None
        if "type" in optim_config:
            optim_type = optim_config["type"]
            optim_args = optim_config.get("args", {})
            params = list(generator_b2a.parameters())
            optimizer_g_b = optimizer_factory(params, optim_type, optim_args)

        # Initialize discriminator optimizer
        optim_config = config["discriminator"].get("optimizer", {})
        optimizer_d_a = None
        optimizer_d_b = None
        if "type" in optim_config:
            optim_type = optim_config["type"]
            optim_args = optim_config.get("args", {})
            params = list(discriminator_a.parameters())
            optimizer_d_a = optimizer_factory(params, optim_type, optim_args)

            params = list(discriminator_b.parameters())
            optimizer_d_b = optimizer_factory(params, optim_type, optim_args)

        # Initialize composite losses
        o_channels = shape_originals[0]
        t_channels = shape_targets[0]
        composite_loss_a = CompositeLoss.from_config(config, o_channels)
        composite_loss_b = CompositeLoss.from_config(config, t_channels)

        render = SisrHareRenderer(shape_targets)
        return CycleGan(
            shape_originals=shape_originals,
            shape_targets=shape_targets,
            render=render,
            generator_a2b=generator_a2b,
            generator_b2a=generator_b2a,
            discriminator_a=discriminator_a,
            discriminator_b=discriminator_b,
            optimizer_g_a=optimizer_g_a,
            optimizer_g_b=optimizer_g_b,
            optimizer_d_a=optimizer_d_a,
            optimizer_d_b=optimizer_d_b,
            g_train_every_n_iters=g_train_every_n_iters,
            d_train_every_n_iters=d_train_every_n_iters,
            composite_loss_a=composite_loss_a,
            composite_loss_b=composite_loss_b,
            g_adv_weight=g_adv_weight,
            d_adv_weight=d_adv_weight,
            use_gpu=use_gpu,
        )

    # Components that are trained or used during training
    __generator_a2b: nn.Module
    __generator_b2a: nn.Module
    __discriminator_a: nn.Module
    __discriminator_b: nn.Module

    __optimizer_g_a: optim.Optimizer
    __optimizer_g_b: optim.Optimizer
    __optimizer_d_a: optim.Optimizer
    __optimizer_d_b: optim.Optimizer

    __composite_loss_a: nn.Module
    __composite_loss_b: nn.Module
    __adversarial_loss_g: AdversarialGeneratorLoss
    __adversarial_loss_d: AdversarialDiscriminatorLoss

    # Variables influencing the training regime
    __device: torch.device
    __device_rank: int
    __num_devices: int

    __d_train_every_n_iters: int
    __g_train_every_n_iters: int

    # Size variables
    __shape_originals: Tuple[int, int, int]
    __shape_targets: Tuple[int, int, int]

    # Validation
    __content_validator: ContentValidator
    __render: Renderer

    def __init__(
        self,
        shape_originals: Tuple[int, int, int],
        shape_targets: Tuple[int, int, int],
        render: Renderer,
        generator_a2b: nn.Module,
        generator_b2a: nn.Module,
        discriminator_a: nn.Module,
        discriminator_b: nn.Module,
        optimizer_g_a: Optional[optim.Optimizer] = None,
        optimizer_g_b: Optional[optim.Optimizer] = None,
        optimizer_d_a: Optional[optim.Optimizer] = None,
        optimizer_d_b: Optional[optim.Optimizer] = None,
        g_train_every_n_iters: int = 1,
        d_train_every_n_iters: int = 1,
        composite_loss_a: Optional[nn.Module] = None,
        composite_loss_b: Optional[nn.Module] = None,
        g_adv_weight: float = 1.0,
        d_adv_weight: float = 1.0,
        use_gpu: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        """Initializes the CycleGan class.

        Parameters
        ----------
        shape_originals: Tuple[int, int, int]
            Shape of input tensors
        shape_targets: Tuple[int, int, int]
            Shape of generated tensors
        render: Renderer
            Training session rendering class
        generator_a2b: nn.Module
            Generator taking domain A inputs to produce domain B
        generator_b2a: nn.Module
            Generator taking domain B inputs to produce domain A
        discriminator_a: nn.Module
            Discriminator that classifies whether input is from domain A
        discriminator_b: nn.Module
            Discriminator that classifies whether input is from domain B
        optimizer_g_a: optim.Optimizer
            Generator A2B's optimizer, default: ADAM
        optimizer_g_b: optim.Optimizer
            Generator B2A's optimizer, default: ADAM
        optimizer_d_a: optim.Optimizer
            Discriminator A's optimizer, default: ADAM
        optimizer_d_b: optim.Optimizer
            Discriminator B's optimizer, default: ADAM
        g_train_every_n_iters: int
            Specifies how often generator is trained
        d_train_every_n_iters: int
            Specifies how often discriminator is trained
        composite_loss_a: CompositeLoss
            A composite loss that optimizes a domain A, default includes: content,
            SSIM, perceptual loss functions
        composite_loss_b: CompositeLoss
            A composite loss that optimizes a domain B, default includes: content,
            SSIM, perceptual loss functions
        g_adv_weight: float
            Scalar for the generator adversarial loss function
        d_adv_weight: float
            Scalar for the discriminator adversarial loss function
        use_gpu: bool
            Specifies training session with GPU.
        rank: int
            Current device during distributed data parallel training
        world_size: int
            The total number of devices to train the CycleGan on
        """
        # Copy settings influencing the training regime
        use_gpu = torch.cuda.is_available() and use_gpu
        self.__device = torch.device("cuda" if use_gpu else "cpu")
        self.__device_rank = rank
        self.__num_devices = world_size
        self.__num_devices = torch.cuda.device_count() if use_gpu else 1

        self.__g_train_every_n_iters = g_train_every_n_iters
        self.__d_train_every_n_iters = d_train_every_n_iters

        # Shape information
        self.__shape_originals = shape_originals
        self.__shape_targets = shape_targets

        # Networks
        self.__generator_a2b = generator_a2b.to(self.__device)
        self.__generator_b2a = generator_b2a.to(self.__device)
        self.__discriminator_a = discriminator_a.to(self.__device)
        self.__discriminator_b = discriminator_b.to(self.__device)

        # Optimizers
        params = list(self.__generator_a2b.parameters())
        self.__optimizer_g_a = (
            optim.Adam(params, betas=(0.5, 0.999), lr=0.0002)  # Default
            if optimizer_g_a is None
            else optimizer_g_a  # Use configuration settings
        )

        params = list(self.__generator_b2a.parameters())
        self.__optimizer_g_b = (
            optim.Adam(params, betas=(0.5, 0.999), lr=0.0002)  # Default
            if optimizer_g_b is None
            else optimizer_g_b  # Use configuration settings
        )

        params = list(self.__discriminator_a.parameters())
        self.__optimizer_d_a = (
            optim.Adam(params, betas=(0.5, 0.999), lr=0.0002)  # Default
            if optimizer_d_a is None
            else optimizer_d_a  # Use configuration settings
        )

        params = list(self.__discriminator_b.parameters())
        self.__optimizer_d_b = (
            optim.Adam(params, betas=(0.5, 0.999), lr=0.0002)  # Default
            if optimizer_d_b is None
            else optimizer_d_b  # Use configuration settings
        )

        # Loss functions
        self.__adversarial_loss_g = LeastSquaredErrorGeneratorLoss(g_adv_weight)
        self.__adversarial_loss_d = LeastSquaredErrorDiscriminatorLoss(d_adv_weight)

        self.__composite_loss_a = (
            composite_loss_a  # Use given composite loss for domain a
            if composite_loss_a is not None
            else CompositeLoss(
                # Use default composite loss for domain a
                shape_originals[0],
                content=True,
                perceptual=True,
            )
        )
        self.__composite_loss_b = (
            composite_loss_b  # Use given composite loss for domain b
            if composite_loss_b is not None
            else CompositeLoss(
                # Use default composite loss for domain b
                shape_targets[0],
                content=True,
                perceptual=True,
            )
        )

        self.__composite_loss_a = self.__composite_loss_a.to(self.__device)
        self.__composite_loss_b = self.__composite_loss_b.to(self.__device)

        self.__content_validator = ContentValidator(self.__shape_targets[0])
        self.__content_validator = self.__content_validator.to(self.__device)

        self.__render = render

    def train(self, batch_nr: int, batch_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Train the GAN's component for one iteration on the given inputs. For a
        BaseGan that means only a discriminator and a generator are trained.
        The returned dictionary contains losses under the `"losses"` key.

        Parameters
        ----------
        batch_nr: int
            The batch number
        batch_inputs: Dict[str, Any]
            A dictionary of values that the generator and discriminator can
            train on. Typically that means at least the `"originals"`
            and `"targets"` keys contain training data.

        Returns
        -------
        Dict[str, Any]
            A dictionary with output values under`"generator"` and `"discriminator"`.
        """
        # Prepare data
        ##################################################################################
        logs_d: Dict[str, Any] = {"losses": {}, "lr": 0}
        logs_g: Dict[str, Any] = {"losses": {}, "lr": 0}

        domain_a = batch_inputs["originals"].to(self.device)  # Original domain
        domain_b = batch_inputs["targets"].to(self.device)  # Target domain

        if batch_nr % self.__g_train_every_n_iters == 0:
            self.__optimizer_g_a.zero_grad()
            self.__optimizer_g_b.zero_grad()

            b_in_a = self.__generator_b2a(domain_b)["generated"]  # A to B
            a_in_b = self.__generator_a2b(domain_a)["generated"]  # B to A

            cycled_a = self.__generator_b2a(a_in_b)["generated"]  # A2B to A
            cycled_b = self.__generator_a2b(b_in_a)["generated"]  # B2A to B

            identity_a = self.__generator_b2a(domain_a)["generated"]  # A to A
            identity_b = self.__generator_a2b(domain_b)["generated"]  # B to B

            # Train generator
            ##################################################################################
            # Identity losses
            # Is generator_b2a(domain_a) still in A? Is generator_a2b(domain_b) still in B?
            identity_loss_a = self.__composite_loss_a(identity_a, domain_a, postfix="_a")
            identity_loss_b = self.__composite_loss_b(identity_b, domain_b, postfix="_b")
            identity_loss_a = {f"Identity / {k}": v for k, v in identity_loss_a.items()}
            identity_loss_b = {f"Identity / {k}": v for k, v in identity_loss_b.items()}
            logs_g["losses"].update(identity_loss_a)
            logs_g["losses"].update(identity_loss_b)

            # Adversarial losses
            discr_a = self.__discriminator_a(b_in_a)["discriminated"]  # B like A?
            discr_b = self.__discriminator_b(a_in_b)["discriminated"]  # A like B?

            tmp: Dict[str, float] = {}
            self.__adversarial_loss_g(discr_a, losses=tmp, name="_a")
            self.__adversarial_loss_g(discr_b, losses=tmp, name="_b")
            logs_g["losses"] = {f"Generator / {k}": v for k, v in tmp.items()}

            # Cycle consistency losses
            # Is b2a(a2b(a)) indeed back in b? Is a2b(b2a(b)) in b?
            cycle_loss_a = self.__composite_loss_a(cycled_a, domain_a, "_a")  # A again?
            cycle_loss_b = self.__composite_loss_b(cycled_b, domain_b, "_b")  # B again?
            cycle_loss_a = {f"Cycle / {k}": v for k, v in cycle_loss_a.items()}
            cycle_loss_b = {f"Cycle / {k}": v for k, v in cycle_loss_b.items()}
            logs_g["losses"].update(cycle_loss_a)
            logs_g["losses"].update(cycle_loss_b)

            # Backpropagate losses and optimize
            self.__optimizer_g_a.step()
            self.__optimizer_g_b.step()

            # Save LR for tensorboard
            for param_group in self.__optimizer_g_a.param_groups:
                logs_g["losses"]["Generator / LR a"] = param_group["lr"]
            for param_group in self.__optimizer_g_b.param_groups:
                logs_g["losses"]["Generator / LR b"] = param_group["lr"]

        # Train discriminator
        ##################################################################################
        # Apply the discriminator to real and generated data
        if batch_nr % self.__d_train_every_n_iters == 0:
            self.__optimizer_d_a.zero_grad()
            self.__optimizer_d_b.zero_grad()

            a_in_b = self.__generator_a2b(domain_a)["generated"].detach()
            b_in_a = self.__generator_b2a(domain_b)["generated"].detach()

            discr_a_real = self.__discriminator_a(domain_a)["discriminated"]
            discr_a_fake = self.__discriminator_a(b_in_a)["discriminated"]
            discr_b_real = self.__discriminator_b(domain_b)["discriminated"]
            discr_b_fake = self.__discriminator_b(a_in_b)["discriminated"]

            # Calculate losses
            tmp = {}
            self.__adversarial_loss_d(discr_a_real, discr_a_fake, tmp, name="_a")
            self.__adversarial_loss_d(discr_b_real, discr_b_fake, tmp, name="_b")
            logs_d["losses"] = {f"Discriminator / {k}": v for k, v in tmp.items()}

            tpl = "Discriminator / Mean {} original prediction ({})"
            mean_a1 = discr_a_real.mean().cpu().item()
            mean_a2 = discr_a_fake.mean().cpu().item()
            mean_b1 = discr_b_real.mean().cpu().item()
            mean_b2 = discr_b_fake.mean().cpu().item()
            logs_d["losses"][tpl.format("original", "real")] = mean_a1
            logs_d["losses"][tpl.format("original", "generated")] = mean_a2
            logs_d["losses"][tpl.format("target", "real")] = mean_b1
            logs_d["losses"][tpl.format("target", "generated")] = mean_b2

            # Backpropagate losses and optimize
            self.__optimizer_d_a.step()
            self.__optimizer_d_b.step()

            # Save LR for tensorboard
            for param_group in self.__optimizer_d_a.param_groups:
                logs_d["losses"]["Discriminator / LR a"] = param_group["lr"]
            for param_group in self.__optimizer_d_b.param_groups:
                logs_d["losses"]["Discriminator / LR b"] = param_group["lr"]

        # Finish and returns
        ##################################################################################
        logs = {"discriminator": logs_d, "generator": logs_g}
        return logs

    def validate(
        self, val_set: Iterable[Dict[str, torch.Tensor]], render: bool = True
    ) -> Dict[str, Union[float, Image]]:
        # Define basic variables such as the class bin to class label mapping
        logs = {}
        d = self.__adversarial_loss_d
        v = self.__content_validator
        bins = {
            "targets, real vs generated": [1, 0],
            "targets, real vs cycled": [1, 0],
            "originals, real vs generated": [1, 0],
            "originals, real vs cycled": [1, 0],
        }

        # Collect predictions
        renders: Dict[str, Image] = {}
        d_preds_real_a: List[float] = []
        d_preds_fake_a: List[float] = []
        d_preds_cycled_a: List[float] = []
        d_preds_real_b: List[float] = []
        d_preds_fake_b: List[float] = []
        d_preds_cycled_b: List[float] = []
        d.zero_accuracy()  # Ensure we start with an clean slate
        v.zero_metrics()
        with torch.no_grad():
            for batch_idx, cpu_batch in enumerate(val_set, 1):
                batch = {k: v.to(self.__device) for k, v in cpu_batch.items()}

                # Calculate confusions metrics for domain processing domain a into b
                discr = self.__discriminator_b(batch["targets"])["discriminated"]
                d.track_batch_accuracy(discr, 1, "targets, real vs generated")
                d.track_batch_accuracy(discr, 1, "targets, real vs cycled")
                d_preds_real_b.extend(discr.flatten().cpu().tolist())

                generated = self.__generator_a2b(batch["originals"])["generated"]
                discr = self.__discriminator_b(generated)["discriminated"]
                d.track_batch_accuracy(discr, 0, "targets, real vs generated")
                v(generated, batch["originals"], "originals, real vs generated")
                cpu_batch["generated_originals"] = generated.cpu()
                d_preds_fake_b.extend(discr.flatten().cpu().tolist())

                generated = self.__generator_b2a(generated)["generated"]
                discr = self.__discriminator_a(generated)["discriminated"]
                d.track_batch_accuracy(discr, 0, "targets, real vs cycled")
                v(generated, batch["originals"], "originals, real vs cycled")
                cpu_batch["cycled_originals"] = generated.cpu()
                d_preds_cycled_a.extend(discr.flatten().cpu().tolist())

                discr = self.__discriminator_a(batch["originals"])["discriminated"]
                d.track_batch_accuracy(discr, 1, "originals, real vs generated")
                d.track_batch_accuracy(discr, 1, "originals, real vs cycled")
                d_preds_real_a.extend(discr.flatten().cpu().tolist())

                generated = self.__generator_b2a(batch["targets"])["generated"]
                discr = self.__discriminator_b(generated)["discriminated"]
                d.track_batch_accuracy(discr, 0, "originals, real vs generated")
                v(generated, batch["targets"], "targets, real vs generated")
                cpu_batch["generated_targets"] = generated.cpu()
                d_preds_fake_a.extend(discr.flatten().cpu().tolist())

                generated = self.__generator_a2b(generated)["generated"]
                discr = self.__discriminator_b(generated)["discriminated"]
                d.track_batch_accuracy(discr, 0, "originals, real vs cycled")
                v(generated, batch["targets"], "targets, real vs cycled")
                cpu_batch["cycled_targets"] = generated.cpu()
                d_preds_cycled_b.extend(discr.flatten().cpu().tolist())

                if render:
                    renders.update(self.__render(cpu_batch, batch_idx))

        # Formulate mean discriminator prediction logs
        types = {
            ("original", "real"): d_preds_real_a,
            ("original", "generated"): d_preds_fake_a,
            ("original", "cycled"): d_preds_cycled_a,
            ("target", "real"): d_preds_real_b,
            ("target", "generated"): d_preds_fake_b,
            ("target", "cycled"): d_preds_cycled_b,
        }
        calc_mean = lambda predictions: sum(predictions) / len(predictions)
        mean_preds = {
            f"Validation / Mean {domain} prediction ({kind})": calc_mean(predictions)
            for (domain, kind), predictions in types.items()
        }

        # Calculate accuracy metrics
        logs = d.calc_accuracy(bins)
        logs.update(v.calc_metrics())
        logs.update(renders)
        logs.update(mean_preds)
        return logs

    def step_scheduler(
        self,
        logs: Dict[str, float],
        epoch: int,
        idx: int,
        iterations: int,
        graph: str = "generator",
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
        pass

    def grow(self):
        pass

    @property
    def cycle(self):
        """Return this GAN's cycle."""

        class Container:
            def __init__(self, graph: nn.Module):
                self.graph = graph

        return Container(self.__generator_b2a)  # type: ignore

    @property
    def generator(self):
        """Return this GAN's generator."""

        class Container:
            def __init__(self, graph: nn.Module):
                self.graph = graph

        return Container(self.__generator_a2b)  # type: ignore

    @property
    def discriminator(self):
        """Return this GAN's discriminator."""

        class Container:
            def __init__(self, graph: nn.Module):
                self.graph = graph

        return Container(self.__discriminator_b)  # type: ignore

    @property
    def device(self) -> torch.device:
        """Return the device this GAN should be on."""
        return self.__device

    @property
    def device_rank(self) -> int:
        return self.__device_rank

    @property
    def num_devices(self) -> int:
        return self.__num_devices

    @property
    def shape_originals(self) -> Tuple[int, int, int]:
        return self.__shape_originals

    @property
    def shape_targets(self) -> Tuple[int, int, int]:
        return self.__shape_targets

    def to_state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary of pytorch state_dicts."""
        state_dicts = {
            "generator_a2b": self.__generator_a2b.state_dict(),
            "generator_b2a": self.__generator_b2a.state_dict(),
            "discriminator_a": self.__discriminator_a.state_dict(),
            "discriminator_b": self.__discriminator_b.state_dict(),
        }

        return state_dicts

    def __repr__(self) -> str:
        return """
            REALCYCLEGAN: A2B and B2A, both learned
        """
