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
    MinimaxGeneratorLoss,
    WassersteinGeneratorLoss,
    MinimaxDiscriminatorLoss,
    WassersteinCriticLoss,
    LeastSquaredErrorDiscriminatorLoss,
    LeastSquaredErrorGeneratorLoss,
    CompositeLoss,
)
from sobolt.gorgan.optim import scheduler_factory, optimizer_factory, SchedulerAdapter
from sobolt.gorgan.validation import ContentValidator
from .gan import Gan


class BaseGan(Gan):
    """Class for training a BaseGan, includes a generator and discriminator"""

    @staticmethod
    def from_config(config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        """Creates all components necessary for a BaseGan training session.

        Parameters
        ----------
        config: Dict[str, Any]
            Training sessions specification file.
        rank: int
            The GPU device this one will train on in the world.
        world_size
            GPU world size for distributed data parallel.
        """
        use_gpu = torch.cuda.is_available() and config["use_gpu"]
        device = torch.device("cuda" if use_gpu else "cpu")
        device_rank = rank
        # self.num_devices = world_size
        num_devices = torch.cuda.device_count() if use_gpu else 1

        shape_originals = config["shape_originals"]
        shape_targets = config["shape_targets"]

        out = BaseGan.build(config, device)

        content_validator = ContentValidator(shape_targets[0])
        content_validator = content_validator.to(device)

        # Copy some values for easy access and value screening
        d_train_every_n_iters = config["discriminator"].get("train_every_n_iters", 1)
        g_train_every_n_iters = config["generator"].get("train_every_n_iters", 1)

        normalization = config.get("normalization", [0.0, 1.0])
        norm = "sigmoid"
        if normalization == [0.0, 1.0]:
            norm = "sigmoid"
        elif normalization == [-1.0, 1.0]:
            norm = "tanh"
        else:
            norm = "zscore"
        render = SisrHareRenderer(shape_targets, norm=norm)

        return BaseGan(
            device=device,
            num_devices=num_devices,
            device_rank=device_rank,
            render=render,
            content_validator=content_validator,
            shape_originals=shape_originals,
            shape_targets=shape_targets,
            g_train_every_n_iters=g_train_every_n_iters,
            d_train_every_n_iters=d_train_every_n_iters,
            **out,
        )

    @staticmethod
    def build(config: Dict[str, Any], device):
        """Builds the graph by instantiating the generator and discriminator.

        This method is only used by this class' initializer and does not
        need to be called directly.

        Parameters
        ----------
        config: Dict[str, Any]
            Training session specification file.
        device: torch.device
            Cuda or CPU

        Returns
        -------
        Dict[str, Any]
            A dict of all components necessary for a BaseGan training sessions.
        """
        # Specify for detailed output logging to terminal
        quiet = config.get("quiet", False)

        # Initiate generator with config file specifications
        (
            generator_a2b,
            generator_b2a,
            optimizer_g,
            composite_loss,
            adversarial_loss_g,
        ) = BaseGan.build_generator(device, config, config["generator"], quiet)

        # Initiate discriminator with config file specifications
        (discriminator, optimizer_d, adversarial_loss_d) = BaseGan.build_discriminator(
            device, config, config["discriminator"], quiet
        )

        # Initiate additional graph components if specified in config
        scheduler_g = BaseGan.build_lr_scheduler(config["generator"], optimizer_g)
        scheduler_d = BaseGan.build_lr_scheduler(config["discriminator"], optimizer_d)
        return {
            "generator_a2b": generator_a2b,
            "generator_b2a": generator_b2a,
            "discriminator": discriminator,
            "optimizer_g": optimizer_g,
            "optimizer_d": optimizer_d,
            "adversarial_loss_g": adversarial_loss_g,
            "adversarial_loss_d": adversarial_loss_d,
            "composite_loss": composite_loss,
            "scheduler_g": scheduler_g,
            "scheduler_d": scheduler_d,
        }

    @staticmethod
    def build_generator(
        device: torch.device,
        main_config: Dict[str, Any],
        config: Dict[str, Any],
        quiet: bool,
    ) -> Tuple:
        """Creates a Generator instance from a configuration dictionary.

        It does so by using the configuration to determine which graph needs be be
        instantiated and what the optimizer should look like. It might also enable
        or disable extra behaviour depending on the settings, such as auxiliary
        classification.
        Parameters
        ----------
        device: torch.device
            Specify Cuda or CPU
        main_config: Dict[str, Any]
            Training session config.
        config: Dict[str, Any]
            A generator specific config.
        quiet: bool
            Specify the silencing of detailed output logging to the terminal.

        Returns
        -------
        Tuple
            Tuple including two generators, an optimizer and respective losses.
        """
        # Extract and propagate general GAN configuration arguments
        opt_args = {
            "auxiliary": main_config.get("aux_gan", False),
            "use_attention": main_config.get("attention", False),
            "use_progressive": main_config.get("progressive_gan", False),
            "aux_num_classes": main_config.get("aux_num_classes", 0),
            "base_loss": main_config.get("base_loss", "least-squared-error"),
            "upsample_factor": main_config.get("upsample_factor", 1),
            "use_condition": main_config.get("conditional_gan", False),
            "conditional_mask_indices": main_config.get("conditional_mask_indices", None),
            "init_prog_step": main_config.get("init_prog_step", 1),
            "hue": main_config.get("hue", False),
            "content": main_config.get("content", False),
            "perceptual": main_config.get("perceptual", False),
            "total_variation": main_config.get("total_variation", False),
            "lp_coherence": main_config.get("lp_coherence", False),
            "ssim": main_config.get("ssim", False),
        }
        # Extract arguments specific to this kind of generator graph
        graph_args = main_config.get("generator", {}).get("reconstructor", {})

        # Build graphs for generators
        generator_a2b = graph_factory(
            config,
            quiet=quiet,
            shape_originals=main_config["shape_originals"],
            shape_targets=main_config["shape_targets"],
            **opt_args,
            **graph_args,
        )
        generator_a2b.to(device)

        generator_b2a = nn.Identity().to(device)

        # Build optimizer for generators
        params = list(generator_a2b.parameters())
        params += list(generator_b2a.parameters())
        optimizer_g = optimizer_factory(
            params, config["optimizer"]["type"], config["optimizer"]["args"]
        )

        # Initialize losses
        adversarial_loss_g: Union[
            MinimaxGeneratorLoss, WassersteinGeneratorLoss, LeastSquaredErrorGeneratorLoss
        ]
        if opt_args["base_loss"] == "minimax":
            adversarial_loss_g = MinimaxGeneratorLoss()
        elif opt_args["base_loss"] == "wasserstein":
            adversarial_loss_g = WassersteinGeneratorLoss()
        elif opt_args["base_loss"] == "least-squared-error":
            adversarial_loss_g = LeastSquaredErrorGeneratorLoss()
        else:
            raise ValueError("Invalid loss chosen: {}".format(opt_args["base_loss"]))

        opt_args.pop("base_loss")
        opt_args.pop("upsample_factor")
        opt_args.pop("use_attention")
        opt_args.pop("use_condition")
        opt_args.pop("aux_num_classes")
        opt_args.pop("use_progressive")
        opt_args.pop("init_prog_step")
        opt_args.pop("conditional_mask_indices")

        num_o_channels = main_config["shape_originals"][0]
        composite_loss = CompositeLoss.from_config(main_config, num_o_channels)
        composite_loss = composite_loss.to(device)

        return (
            generator_a2b,
            generator_b2a,
            optimizer_g,
            composite_loss,
            adversarial_loss_g,
        )

    @staticmethod
    def build_discriminator(
        device: torch.device,
        main_config: Dict[str, Any],
        config: Dict[str, Any],
        quiet: bool,
    ) -> Tuple[nn.Module, optim.Optimizer, AdversarialDiscriminatorLoss]:
        """Creates a Discriminator instance from a configuration dictionary.

        It does so by using the configuration to determine which graph needs be be
        instantiated and what the optimizer should look like. It might also enable
        or disable certain settings, such as auxiliary classification.

        Parameters
        ----------
        device: torch.device
            Specify Cuda or CPU
        main_config: Dict[str, Any]
            Training session config.
        config: Dict[str, Any]
            A generator specific config.
        quiet: bool
            Specify the silencing of detailed output logging to the terminal.

        Returns
        -------
        Tuple[nn.Module, optim.Optimizer, AdversarialDiscriminatorLoss]
            Tuple including 1 discriminators, an optimizer and respective loss function.
        """
        # Flags for the graph
        opt_args = {
            "use_auxiliary": main_config.get("aux_gan", False),
            "use_attention": main_config.get("attention", False),
            "use_condition": main_config.get("conditional_gan", False),
            "aux_num_classes": main_config.get("aux_num_classes", -1),
            "base_loss": main_config.get("base_loss", "least-squared-error"),
        }

        # Build graph for discriminator
        discriminator = graph_factory(
            config, quiet=quiet, shape_targets=main_config["shape_targets"], **opt_args
        )
        discriminator.to(device)

        # Build optimizer
        optimizer_d = optimizer_factory(
            discriminator.parameters(),
            config["optimizer"]["type"],
            config["optimizer"]["args"],
        )

        # Initialize losses
        adversarial_loss_d: Union[
            MinimaxDiscriminatorLoss,
            WassersteinCriticLoss,
            LeastSquaredErrorDiscriminatorLoss,
        ]
        if opt_args["base_loss"] == "minimax":
            adversarial_loss_d = MinimaxDiscriminatorLoss()
        elif opt_args["base_loss"] == "wasserstein":
            use_gp = config.get("gradient_penalty", True)
            adversarial_loss_d = WassersteinCriticLoss(
                discriminator, gradient_penalty=use_gp
            )
        elif opt_args["base_loss"] == "least-squared-error":
            adversarial_loss_d = LeastSquaredErrorDiscriminatorLoss()
        else:
            raise ValueError("Invalid loss chosen: {}".format(opt_args["base_loss"]))

        return discriminator, optimizer_d, adversarial_loss_d

    @staticmethod
    def build_lr_scheduler(
        config: Dict[str, Any], optimizer: optim.Optimizer
    ) -> Optional[SchedulerAdapter]:
        """Initialize LR scheduler if specified in config file

        Parameters
        ----------
        config: Dict[str, Any]
            File containing a GAN training session specifications
        optimizer: optim.Optimizer
            A graph's optimizer

        Returns
        -------
        Optional[SchedulerAdapter]
         The schedule we want to set for decreasing the learning rate during training.
        """
        lr_scheduler = config.get("lr_scheduler", {"type": "none"})
        factor_decay = lr_scheduler.get("factor_decay", 1.0)
        scheduler_type = lr_scheduler.get("type", "none")
        scheduler = scheduler_factory(scheduler_type, optimizer, factor_decay)
        return scheduler

    __generator_a2b: nn.Module
    # __generator_b2a: nn.Module  # TODO Reimplement
    __discriminator: nn.Module

    __optimizer_g: optim.Optimizer
    __optimizer_d: optim.Optimizer

    __adversarial_loss_g: AdversarialGeneratorLoss
    __adversarial_loss_d: AdversarialDiscriminatorLoss
    __composite_loss: nn.Module

    # Values copied from __config for quick and easy access
    __d_train_every_n_iters: int
    __g_train_every_n_iters: int
    __scheduler_g: Optional[SchedulerAdapter]
    __scheduler_d: Optional[SchedulerAdapter]

    # Multiprocessing variables
    __device: torch.device
    __device_rank: int
    __num_devices: int

    # Size variables
    __shape_originals: Tuple[int, int, int]
    __shape_targets: Tuple[int, int, int]

    # Validation
    __render: Renderer
    __content_validator: ContentValidator

    def __init__(
        self,
        device: torch.device,
        device_rank: int,
        num_devices: int,
        render: Renderer,
        content_validator: ContentValidator,
        shape_originals: Tuple[int, int, int],
        shape_targets: Tuple[int, int, int],
        generator_a2b: nn.Module,
        generator_b2a: nn.Module,
        discriminator: nn.Module,
        optimizer_g: optim.Optimizer,
        optimizer_d: optim.Optimizer,
        adversarial_loss_g: AdversarialGeneratorLoss,
        adversarial_loss_d: AdversarialDiscriminatorLoss,
        composite_loss: nn.Module,
        scheduler_g: Optional[SchedulerAdapter],
        scheduler_d: Optional[SchedulerAdapter],
        g_train_every_n_iters: int = 1,
        d_train_every_n_iters: int = 1,
    ):
        """Initializes a BaseGan class.

        Parameters
        ----------
        device: torch.device
            The device to put BaseGan training session (CUDA or CPU)
        device_rank: int
            Current device during distributed data parallel training
        num_devices: int
            The total number of devices to train the BaseGan on
        render: Renderer
            Training session rendering class
        content_validator: ContentValidator
            Training session content validation class
        shape_originals: Tuple[int, int, int]
            Shape of input tensors
        shape_targets: Tuple[int, int, int]
            Shape of generated tensors
        generator_a2b: nn.Module
            Generator taking domain A inputs to produce domain B
        generator_b2a: nn.Module
            Generator taking domain B inputs to produce domain A
        discriminator: nn.Module
            Discriminator that classifies whether input is from a target domain
        optimizer_g: optim.Optimizer
            A generator's optimizer, default: ADAM
        optimizer_d: optim.Optimizer
            A discriminator's optimizer, default: ADAM
        adversarial_loss_g: AdversarialGeneratorLoss
            A generator's adversarial loss function
        adversarial_loss_d: AdversarialDiscriminatorLoss
            A discriminator's adversarial loss function
        composite_loss: CompositeLoss
            A composite loss that optimizes for the target domain, default includes:
            content, SSIM, perceptual loss functions
        scheduler_g: Optional[SchedulerAdapter]
            A generator's learning rate scheduler (ex. CosineAnnealing, MultiStep)
        scheduler_d: Optional[SchedulerAdapter]
            A discriminator's learning rate scheduler (ex. CosineAnnealing, MultiStep)
        g_train_every_n_iters: int
            Specifies how often generator is trained
        d_train_every_n_iters: int
            Specifies how often discriminator is trained
        """
        self.__device = device
        self.__device_rank = device_rank
        # self.__num_devices = world_size
        self.__num_devices = num_devices

        self.__content_validator = content_validator
        self.__render = render

        # training components
        self.__generator_a2b = generator_a2b
        self.__generator_b2a = generator_b2a
        self.__discriminator = discriminator

        self.__optimizer_g = optimizer_g
        self.__optimizer_d = optimizer_d

        self.__adversarial_loss_g = adversarial_loss_g
        self.__adversarial_loss_d = adversarial_loss_d

        self.__composite_loss = composite_loss

        # Training parameters
        self.__shape_originals = shape_originals
        self.__shape_targets = shape_targets

        self.__d_train_every_n_iters = d_train_every_n_iters
        self.__g_train_every_n_iters = g_train_every_n_iters

        self.__scheduler_g = scheduler_g
        self.__scheduler_d = scheduler_d

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

        domain_a = batch_inputs["originals"].to(self.device)
        domain_b = batch_inputs["targets"].to(self.device)

        if batch_nr % self.__g_train_every_n_iters == 0:
            self.__optimizer_g.zero_grad()
            a_in_b = self.__generator_a2b(domain_a)["generated"]
            # cycled_a = self.__generator_b2a(a_in_b)["generated"]
            # identity_b = self.__generator_a2b(domain_b)["generated"]

            # Train generator
            ##################################################################################
            # Adversarial losses
            # Is the a2b(domain a) indeed in domain b?
            tmp: Dict[str, float] = {}
            discriminated = self.__discriminator(a_in_b)["discriminated"]
            self.__adversarial_loss_g(discriminated, losses=tmp)
            logs_g["losses"].update({f"Generator / {k}": v for k, v in tmp.items()})

            # Cycle consistency losses
            # Leaving it in for to easily re-implement it in the future
            # cycle_loss_a = self.__composite_loss(cycled_a, domain_a, "_a")
            # cycle_loss_a = {f"Cycle / {k}": v for k, v in cycle_loss_a.items()}
            # logs_g["losses"].update(cycle_loss_a)

            # Identity loss
            # Leaving it in for to easily re-implement it in the future
            # identity_loss_b = self.__composite_loss(identity, domain_b, "_id")

            # Backpropagate losses and optimize
            self.__optimizer_g.step()

            # Save LR for tensorboard
            for param_group in self.__optimizer_g.param_groups:
                logs_g["lr"] = param_group["lr"]

        # Train discriminator
        ##################################################################################
        # Apply the discriminator to real and generated data
        if batch_nr % self.__d_train_every_n_iters == 0:
            self.__optimizer_d.zero_grad()

            a_in_b = self.__generator_a2b(domain_a)["generated"].detach()

            discr_b_real = self.__discriminator(domain_b)["discriminated"]
            discr_b_fake = self.__discriminator(a_in_b)["discriminated"]

            # Calculate losses
            tmp = {}
            d_params = [discr_b_real, discr_b_fake, tmp]
            if isinstance(self.__adversarial_loss_d, WassersteinCriticLoss):
                d_params = [domain_b, domain_a] + d_params
            self.__adversarial_loss_d(*d_params)
            logs_d["losses"] = {f"Discriminator / {k}": v for k, v in tmp.items()}

            tpl = "Discriminator / Mean {} prediction ({})"
            mean_real = discr_b_real.mean().cpu().item()
            mean_fake = discr_b_fake.mean().cpu().item()
            logs_d["losses"][tpl.format("target", "real")] = mean_real
            logs_d["losses"][tpl.format("target", "generated")] = mean_fake

            # Backpropagate losses and optimize
            self.__optimizer_d.step()

            # Save LR for tensorboard
            for param_group in self.__optimizer_d.param_groups:
                logs_d["lr"] = param_group["lr"]

        # Finish and returns
        ##################################################################################
        logs = {"discriminator": logs_d, "generator": logs_g}
        return logs

    def validate(
        self, val_set: Iterable[Dict[str, torch.Tensor]], render: bool = True
    ) -> Dict[str, Union[float, Image]]:
        """Validate the GAN on the given batches of validation data."""
        # Define basic variables such as the mapping of class bin to class label
        logs = {}
        d = self.__adversarial_loss_d
        v = self.__content_validator
        bins = {"targets, real vs generated": [1, 0]}
        # Collect predictions
        d.zero_accuracy()  # Ensure we start with an clean slate
        v.zero_metrics()
        renders: Dict[str, Image] = {}
        d_preds_real: List[float] = []
        d_preds_fake: List[float] = []
        with torch.no_grad():
            for batch_idx, cpu_batch in enumerate(val_set):
                batch = {k: v.to(self.__device) for k, v in cpu_batch.items()}

                # Calculate confusions metrics for processing generated and target
                discriminated = self.__discriminator(batch["targets"])["discriminated"]
                d.track_batch_accuracy(discriminated, 1, "targets, real vs generated")
                d_preds_real.extend(discriminated.flatten().cpu().tolist())

                generated = self.__generator_a2b(batch["originals"])["generated"]
                discriminated = self.__discriminator(generated)["discriminated"]
                d.track_batch_accuracy(discriminated, 0, "targets, real vs generated")
                cpu_batch["generated_originals"] = generated.cpu()
                d_preds_fake.extend(discriminated.flatten().cpu().tolist())

                if render:
                    renders.update(self.__render(cpu_batch, batch_idx))

        # Formulate mean discriminator prediction logs
        types = {("target", "real"): d_preds_real, ("target", "generated"): d_preds_fake}
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
        if graph == "generator" and self.__scheduler_g:
            self.__scheduler_g.step(logs, epoch, idx, iterations)
        if graph == "discriminator" and self.__scheduler_d:
            self.__scheduler_d.step(logs, epoch, idx, iterations)

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

        return Container(self.__discriminator)  # type: ignore

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
            "discriminator": self.__discriminator.state_dict(),
        }

        return state_dicts

    def __repr__(self) -> str:
        return """
            BaseGan: A GAN with a generator and discriminator
        """
