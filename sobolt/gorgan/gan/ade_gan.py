from typing import Dict, Any, Tuple, Union, Optional, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL.Image import Image
from skimage.exposure import match_histograms

from sobolt.gorgan.data.render import Renderer, SisrHareRenderer
from sobolt.gorgan.graphs import graph_factory
from sobolt.gorgan.graphs.generators.unet import UNet
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
from sobolt.gorgan.nn import ConvertToGrey
from sobolt.gorgan.optim import scheduler_factory, optimizer_factory, SchedulerAdapter
from sobolt.gorgan.validation import ContentValidator
from .gan import Gan


class AdeGan(Gan):
    """Class for training the GAN developed for the Artificial Data Enhancement
    project(s)."""

    @staticmethod
    def from_config(config: Dict[str, Any], rank: int = 0, world_size: int = 1):
        """Creates all components necessary for a AdeGan training session.

        Parameters
        ----------
        config: Dict[str, Any]
            The config settings for this training run.
        rank: int
            The GPU device this one will train on in the world.
        world_size: int
            The amount of GPUs / devices this training run will have.
        """
        use_gpu = torch.cuda.is_available() and config["use_gpu"]
        device = torch.device("cuda" if use_gpu else "cpu")
        device_rank = rank
        # self.num_devices = world_size
        num_devices = torch.cuda.device_count() if use_gpu else 1

        multi_d = config.get("multi_d_gan", False)
        color_match = config.get("color_match", False)
        color_transfer = config.get("color_transfer", False)

        shape_originals = config["shape_originals"]
        shape_targets = config["shape_targets"]

        out = AdeGan.build(config, device, multi_d, color_transfer)

        content_validator = ContentValidator(shape_targets[0])
        content_validator = content_validator.to(device)

        # Copy some values for easy access and value screening
        d_train_every_n_iters = config["discriminator"].get("train_every_n_iters", 1)
        g_train_every_n_iters = config["generator"].get("train_every_n_iters", 1)

        render = SisrHareRenderer(shape_targets)

        return AdeGan(
            device=device,
            num_devices=num_devices,
            device_rank=device_rank,
            render=render,
            content_validator=content_validator,
            shape_originals=shape_originals,
            shape_targets=shape_targets,
            g_train_every_n_iters=g_train_every_n_iters,
            d_train_every_n_iters=d_train_every_n_iters,
            color_match=color_match,
            multi_d=multi_d,
            **out,
        )

    @staticmethod
    def build(
        config: Dict[str, Any],
        device: torch.device,
        multi_d: bool = False,
        color_transfer: bool = False,
    ) -> Dict[str, Any]:
        """Builds the graph by instantiating the generator and discriminator.

        This method is only used by this class' initializer and does not
        need to be called directly.

        Parameters
        ----------
        config: Dict[str, Any]
            Training session specification file.
        device: torch.device
            Cuda or CPU
        multi_d: bool
            Specify a multi-discriminator set up.
        color_transfer: bool
            Specify applying learned color transfer from the original to the target
            domain.

        Returns
        -------
        Dict[str, Any]
            A dict of all components necessary for a AdeGan training sessions.
        """
        # Specify for detailed output logging to terminal
        quiet = config.get("quiet", False)

        # Specify optional arguments
        unet_color_transfer = None
        discriminator_texture = None
        optimizer_d_texture = None
        to_grey = None
        adversarial_loss_d_t = None
        adversarial_loss_g_t = None
        scheduler_d_t = None
        dt_train_every_n_iters = 1

        # Initiate generator with config file specifications
        gs = AdeGan.build_generator(device, config, config["generator"], quiet)
        (
            generator_a2b,
            generator_b2a,
            optimizer_g,
            composite_loss_a,
            composite_loss_b,
            adversarial_loss_g,
        ) = gs

        # Initiate discriminator with config file specifications
        (discriminator_b, optimizer_d, adversarial_loss_d) = AdeGan.build_discriminator(
            device, config, config["discriminator"], quiet
        )

        # Initiate additional graph components if specified in config
        if multi_d:
            to_grey = ConvertToGrey().to(device)

            texture_cfg = config["multi_discriminator"]["texture"]
            ds = AdeGan.build_discriminator(
                device, texture_cfg, texture_cfg["discriminator"], quiet
            )
            (discriminator_texture, optimizer_d_texture, adversarial_loss_d_t) = ds
            adversarial_loss_g_t = LeastSquaredErrorGeneratorLoss()

            scheduler_d_t = AdeGan.build_lr_scheduler(
                texture_cfg["discriminator"], optimizer_d_texture
            )
            texture_cfg = texture_cfg["discriminator"]
            dt_train_every_n_iters = texture_cfg.get("train_every_n_iters", 1)

        if color_transfer:
            weights_color_transfer_path = (
                "/in1/experiments/reference/2021-05-29-"
                "experiment14-style_transfer-cross-sensor-sv_to_s2-512x512-full_cycle_gan"
                "/checkpoints/generator_a2b_iteration26516_checkpoint.pth"
            )
            weights_color_transfer = torch.load(weights_color_transfer_path)
            unet_color_transfer = UNet(4, 4).to(device)
            unet_color_transfer.load_state_dict(weights_color_transfer)

        scheduler_g = AdeGan.build_lr_scheduler(config["generator"], optimizer_g)
        scheduler_d = AdeGan.build_lr_scheduler(config["discriminator"], optimizer_d)

        return {
            "generator_a2b": generator_a2b,
            "generator_b2a": generator_b2a,
            "unet_color_transfer": unet_color_transfer,
            "discriminator_b": discriminator_b,
            "discriminator_texture": discriminator_texture,
            "optimizer_g": optimizer_g,
            "optimizer_d": optimizer_d,
            "optimizer_d_texture": optimizer_d_texture,
            "adversarial_loss_g": adversarial_loss_g,
            "adversarial_loss_g_t": adversarial_loss_g_t,
            "adversarial_loss_d": adversarial_loss_d,
            "adversarial_loss_d_t": adversarial_loss_d_t,
            "composite_loss_a": composite_loss_a,
            "composite_loss_b": composite_loss_b,
            "scheduler_g": scheduler_g,
            "scheduler_d": scheduler_d,
            "scheduler_d_t": scheduler_d_t,
            "to_grey": to_grey,
            "dt_train_every_n_iters": dt_train_every_n_iters,
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
            quiet,
            shape_originals=main_config["shape_originals"],
            shape_targets=main_config["shape_targets"],
            **opt_args,
            **graph_args,
        )
        generator_a2b.to(device)

        config_b2a = main_config["cycle"]
        b2a_args = {
            "factor": main_config["upsample_factor"],
            "shape_originals": main_config["shape_originals"],
        }
        generator_b2a = graph_factory(config_b2a, quiet, **b2a_args)
        generator_b2a.to(device)

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
        num_t_channels = main_config["shape_targets"][0]
        composite_loss_a = CompositeLoss.from_config(main_config, num_o_channels)
        composite_loss_b = CompositeLoss.from_config(main_config, num_t_channels)
        composite_loss_a = composite_loss_a.to(device)
        composite_loss_b = composite_loss_b.to(device)

        return (
            generator_a2b,
            generator_b2a,
            optimizer_g,
            composite_loss_a,
            composite_loss_b,
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
        discriminator_b = graph_factory(
            config, quiet, shape_targets=main_config["shape_targets"], **opt_args
        )
        discriminator_b.to(device)

        # Build optimizer
        optimizer_d = optimizer_factory(
            discriminator_b.parameters(),
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
                discriminator_b, gradient_penalty=use_gp
            )
        elif opt_args["base_loss"] == "least-squared-error":
            adversarial_loss_d = LeastSquaredErrorDiscriminatorLoss()
        else:
            raise ValueError("Invalid loss chosen: {}".format(opt_args["base_loss"]))

        return discriminator_b, optimizer_d, adversarial_loss_d

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

    # Components that are trained or used during training
    __generator_a2b: nn.Module
    __generator_b2a: nn.Module
    __unet_color_transfer: nn.Module
    __discriminator_a: nn.Module
    __discriminator_b: nn.Module
    __discriminator_texture: nn.Module

    __optimizer_g: optim.Optimizer
    __optimizer_d: optim.Optimizer
    __optimizer_d_texture: optim.Optimizer

    __adversarial_loss_g: AdversarialGeneratorLoss
    __adversarial_loss_g_t: AdversarialGeneratorLoss
    __adversarial_loss_d: AdversarialDiscriminatorLoss
    __adversarial_loss_d_t: AdversarialDiscriminatorLoss
    __composite_loss_a: nn.Module
    __composite_loss_b: nn.Module

    # Values copied from __config for quick and easy access
    __d_train_every_n_iters: int
    __g_train_every_n_iters: int
    __dt_train_every_n_iters: int
    __scheduler_g: Optional[SchedulerAdapter]
    __scheduler_d: Optional[SchedulerAdapter]
    __scheduler_d_t: Optional[SchedulerAdapter]
    __color_match: bool
    __color_transfer: bool
    __multi_d: bool
    __to_grey: ConvertToGrey

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
        unet_color_transfer: nn.Module,
        discriminator_b: nn.Module,
        discriminator_texture: nn.Module,
        optimizer_g: optim.Optimizer,
        optimizer_d: optim.Optimizer,
        optimizer_d_texture: optim.Optimizer,
        adversarial_loss_g: AdversarialGeneratorLoss,
        adversarial_loss_g_t: AdversarialGeneratorLoss,
        adversarial_loss_d: AdversarialDiscriminatorLoss,
        adversarial_loss_d_t: AdversarialDiscriminatorLoss,
        composite_loss_a: nn.Module,
        composite_loss_b: nn.Module,
        scheduler_g: Optional[SchedulerAdapter],
        scheduler_d: Optional[SchedulerAdapter],
        scheduler_d_t: Optional[SchedulerAdapter],
        to_grey: ConvertToGrey,
        g_train_every_n_iters: int = 1,
        d_train_every_n_iters: int = 1,
        dt_train_every_n_iters: int = 1,
        color_match: bool = False,
        color_transfer: bool = False,
        multi_d: bool = False,
    ):
        """Initializes the components necessary for a Artificial Data Enhancement GAN.

        Parameters
        ----------
        device: torch.device
            The device to put AdeGan training session (CUDA or CPU)
        device_rank: int
            Current device during distributed data parallel training
        num_devices: int
            The total number of devices to train the AdeGan on
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
        unet_color_transfer: nn.Module
            Trained U-Net to generate SuperView data with the spectral values from
            Sentinel-2
        discriminator_b: nn.Module
            Discriminator that classifies whether input is from domain B
        discriminator_texture: nn.Module
            Discriminator that classifies whether input has textures from domain B
        optimizer_g: optim.Optimizer
            A generator's optimizer, default: ADAM
        optimizer_d: optim.Optimizer
            A discriminator's optimizer, default: ADAM
        optimizer_d_texture: optim.Optimizer
            A texture discriminator's optimizer, default: ADAM
        adversarial_loss_g: AdversarialGeneratorLoss
            A generator's adversarial loss function
        adversarial_loss_g_t: AdversarialGeneratorLoss
            A generator trained with a texture discriminator's adversarial loss function
        adversarial_loss_d: AdversarialDiscriminatorLoss
            A discriminator's adversarial loss function
        adversarial_loss_d_t: AdversarialDiscriminatorLoss
            A texture discriminator's adversarial loss function
        composite_loss_a: CompositeLoss
            A composite loss that optimizes a domain A, default includes: content,
            SSIM, perceptual loss functions
        composite_loss_b: CompositeLoss
            A composite loss that optimizes a domain B, default includes: content,
            SSIM, perceptual loss functions
        scheduler_g: Optional[SchedulerAdapter]
            A generator's learning rate scheduler (ex. CosineAnnealing, MultiStep)
        scheduler_d: Optional[SchedulerAdapter]
            A discriminator's learning rate scheduler (ex. CosineAnnealing, MultiStep)
        scheduler_d_t: Optional[SchedulerAdapter
            A texture discriminator's learning rate scheduler (ex. CosineAnnealing,
            MultiStep)
        to_grey: ConvertToGrey
            Class to greyscale inputs, used for input processing of texture discriminator
        g_train_every_n_iters: int
            Specifies how often generator is trained
        d_train_every_n_iters: int
            Specifies how often discriminator is trained
        dt_train_every_n_iters: int
            Specifies how often texture discriminator is trained
        color_match: bool
            Matches domain B's spectral histogram to the spectral histogram of domain A
        color_transfer: bool
            Specifies training session that processes domain B inputs with the UNet
            color transfer model
        multi_d: bool
            Specifies training session with texture discriminator in addition to
            default discriminator for domain B
        """
        self.__device = device
        self.__device_rank = device_rank
        # self.__num_devices = world_size
        self.__num_devices = num_devices
        self.__multi_d = multi_d
        self.__color_match = color_match
        self.__color_transfer = color_transfer
        self.__content_validator = content_validator
        self.__render = render

        # training components
        self.__generator_a2b = generator_a2b
        self.__generator_b2a = generator_b2a
        self.__discriminator_b = discriminator_b
        self.__discriminator_texture = discriminator_texture
        self.__unet_color_transfer = unet_color_transfer

        self.__optimizer_g = optimizer_g
        self.__optimizer_d = optimizer_d
        self.__optimizer_d_texture = optimizer_d_texture

        self.__adversarial_loss_g = adversarial_loss_g
        self.__adversarial_loss_g_t = adversarial_loss_g_t
        self.__adversarial_loss_d = adversarial_loss_d
        self.__adversarial_loss_d_t = adversarial_loss_d_t

        self.__composite_loss_b = composite_loss_b
        self.__composite_loss_a = composite_loss_a

        # Training parameters
        self.__shape_originals = shape_originals
        self.__shape_targets = shape_targets

        self.__d_train_every_n_iters = d_train_every_n_iters
        self.__g_train_every_n_iters = g_train_every_n_iters
        self.__dt_train_every_n_iters = dt_train_every_n_iters

        self.__scheduler_g = scheduler_g
        self.__scheduler_d = scheduler_d
        self.__scheduler_d_t = scheduler_d_t

        # Optional
        self.__to_grey = to_grey

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

        # Transfer color from domain a to domain b
        if self.__color_transfer:
            batch_inputs["targets"] = self.__unet_color_transfer(
                batch_inputs["targets"].to(self.__device)
            )["generated"].detach()

        # Match color histogram of domain b to domain a
        if self.__color_match:
            batch_inputs = self.match_spectral(batch_inputs)

        domain_a = batch_inputs["originals"].to(self.device)  # low res
        domain_b = batch_inputs["targets"].to(self.device)  # high res

        if batch_nr % self.__g_train_every_n_iters == 0:
            self.__optimizer_g.zero_grad()
            b_in_a = self.__generator_b2a(domain_b)["generated"]  # HR to LR
            a_in_b = self.__generator_a2b(domain_a)["generated"]  # LR to HR

            cycled_a = self.__generator_b2a(a_in_b)["generated"]  # LR2HR to LR
            cycled_b = self.__generator_a2b(b_in_a)["generated"]  # HR2LR to HR

            # Train generator
            ##################################################################################
            # Adversarial losses
            # Is the a2b(LR) indeed in domain b (HR)?
            tmp: Dict[str, float] = {}
            discriminated_b = self.__discriminator_b(a_in_b)["discriminated"]
            self.__adversarial_loss_g(discriminated_b, losses=tmp)
            logs_g["losses"].update({f"Generator / {k}": v for k, v in tmp.items()})

            # Cycle consistency losses
            cycle_loss_a = self.__composite_loss_a(cycled_a, domain_a, "_a")  # LR again?
            cycle_loss_b = self.__composite_loss_b(cycled_b, domain_b, "_b")  # HR again?
            cycle_loss_a = {f"Cycle / {k}": v for k, v in cycle_loss_a.items()}
            cycle_loss_b = {f"Cycle / {k}": v for k, v in cycle_loss_b.items()}
            logs_g["losses"].update(cycle_loss_a)
            logs_g["losses"].update(cycle_loss_b)

            # Multi-discriminators losses
            train_dt = batch_nr % self.__dt_train_every_n_iters == 0
            if self.__multi_d and train_dt:
                grey_a_in_b = self.__to_grey(a_in_b)
                discriminated_b_texture = self.__discriminator_texture(grey_a_in_b)[
                    "discriminated"
                ]
                tmp = {}
                self.__adversarial_loss_g_t(discriminated_b_texture, tmp, name="texture")
                logs_g["losses"].update({f"Generator / {k}": v for k, v in tmp.items()})

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

            discr_b_real = self.__discriminator_b(domain_b)["discriminated"]
            discr_b_fake = self.__discriminator_b(a_in_b)["discriminated"]

            # Calculate losses
            tmp = {}
            self.__adversarial_loss_d(discr_b_real, discr_b_fake, tmp)
            logs_d["losses"] = {f"Discriminator / {k}": v for k, v in tmp.items()}

            mean1 = discr_b_real.mean().cpu().item()
            mean2 = discr_b_fake.mean().cpu().item()
            logs_d["losses"]["Discriminator / Mean target prediction (real)"] = mean1
            logs_d["losses"]["Discriminator / Mean target prediction (generated)"] = mean2

            train_dt = batch_nr % self.__dt_train_every_n_iters == 0
            if self.__multi_d and train_dt:
                self.train_texture_d(a_in_b, domain_b, logs_d)

            # Backpropagate losses and optimize
            self.__optimizer_d.step()

            # Save LR for tensorboard
            for param_group in self.__optimizer_d.param_groups:
                logs_d["lr"] = param_group["lr"]

        # Finish and returns
        ##################################################################################
        logs = {"discriminator": logs_d, "generator": logs_g}
        return logs

    def train_texture_d(
        self, a_in_b: torch.Tensor, domain_b: torch.Tensor, logs_d: Dict[str, Any]
    ):
        """Trains a texture discriminator.

        Parameters
        ----------
        a_in_b: torch.Tensor
            Generated high resolution tensor from low resolution tensor.
        domain_b: torch.Tensor
            Target high resolution tensor.
        logs_d: Dict[str, Any]
            The loss dictionary composed of loss name as string and loss value as float.

        Returns
        -------
        logs_d: Dict[str, Any]
            An updated loss dictionary containing the loss values computed from
            training texture discriminator.
        """
        self.__optimizer_d_texture.zero_grad()
        grey_domain_b = self.__to_grey(domain_b)
        grey_a_in_b = self.__to_grey(a_in_b)
        discriminated_b_real_texture = self.__discriminator_texture(grey_domain_b)[
            "discriminated"
        ]
        discriminated_b_fake_texture = self.__discriminator_texture(grey_a_in_b)[
            "discriminated"
        ]

        tmp: Dict[str, float] = {}
        self.__adversarial_loss_d_t(
            discriminated_b_real_texture,
            discriminated_b_fake_texture,
            tmp,
            name="texture",
        )
        logs_d["losses"].update({f"Discriminator / {k}": v for k, v in tmp.items()})
        self.__optimizer_d_texture.step()
        return logs_d

    def validate(
        self, val_set: Iterable[Dict[str, torch.Tensor]], render: bool = True
    ) -> Dict[str, Union[float, Image]]:
        """Validate the GAN on the given batches of validation data."""
        # Define basic variables such as the mapping of class bin to class label
        logs = {}
        d = self.__adversarial_loss_d
        v = self.__content_validator
        bins = {"targets, real vs generated": [1, 0], "targets, real vs cycled": [1, 0]}
        renamings = {"generated_targets": "degraded_targets"}

        # Collect predictions
        renders: Dict[str, Image] = {}
        d_preds_real: List[float] = []
        d_preds_fake: List[float] = []
        d_preds_cycled: List[float] = []
        d.zero_accuracy()  # Ensure we start with an clean slate
        v.zero_metrics()
        with torch.no_grad():
            for batch_idx, cpu_batch in enumerate(val_set):
                batch = {k: v.to(self.__device) for k, v in cpu_batch.items()}

                # Calculate confusions metrics for processing domain a into b
                discr = self.__discriminator_b(batch["targets"])["discriminated"]
                d.track_batch_accuracy(discr, 1, "targets, real vs generated")
                d.track_batch_accuracy(discr, 1, "targets, real vs cycled")
                d_preds_real.extend(discr.flatten().cpu().tolist())

                generated = self.__generator_a2b(batch["originals"])["generated"]
                discr = self.__discriminator_b(generated)["discriminated"]
                d.track_batch_accuracy(discr, 0, "targets, real vs generated")
                cpu_batch["generated_originals"] = generated.cpu()
                d_preds_fake.extend(discr.flatten().cpu().tolist())

                # Calculate content validation metrics of generated vs originals
                # and cycled originals vs originals
                v(generated, batch["originals"], "originals, real vs generated")
                generated = self.__generator_b2a(generated)["generated"]
                v(generated, batch["originals"], "originals, real vs cycled")
                cpu_batch["cycled_originals"] = generated.cpu()

                # Calculate confusion metrics for processing domain b back into b
                generated = self.__generator_b2a(batch["targets"])["generated"]
                cpu_batch["generated_targets"] = generated.cpu()
                generated = self.__generator_a2b(generated)["generated"]
                discr = self.__discriminator_b(generated)["discriminated"]
                d.track_batch_accuracy(discr, 0, "targets, real vs cycled")
                cpu_batch["cycled_targets"] = generated.cpu()
                d_preds_cycled.extend(discr.flatten().cpu().tolist())

                # Calculate content validation metrics of cycled targets vs targets
                v(generated, batch["targets"], "targets, real vs cycled")
                renders.update(self.__render(cpu_batch, batch_idx, renamings))

        # Formulate mean discriminator prediction logs
        types = {
            ("target", "real"): d_preds_real,
            ("target", "generated"): d_preds_fake,
            ("target", "cycled"): d_preds_cycled,
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
        if graph == "generator" and self.__scheduler_g:
            self.__scheduler_g.step(logs, epoch, idx, iterations)
        if graph == "discriminator" and self.__scheduler_d:
            self.__scheduler_d.step(logs, epoch, idx, iterations)

    def grow(self):
        pass

    def match_spectral(
        self, batch_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Match spectral values from one sensor domain to another using histogram
        color matching.

        Parameters
        ----------
        batch_inputs: Dict[str, torch.Tensor]
            A dictionary of tensors and their corresponding name (i.e. originals, targets)

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary where the spectral values of target tensors are matched to
            original tensors.
        """
        # Apply histogram matching between targets and inputs
        matched = match_histograms(
            batch_inputs["targets"].cpu().numpy(), batch_inputs["originals"].cpu().numpy()
        )
        batch_inputs["targets"] = torch.from_numpy(matched.astype(np.float32)).to(
            self.device
        )
        return batch_inputs

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
    def color_transfer_model(self):
        """Return UNet used for targets color mapping"""

        class Container:
            def __init__(self, graph: nn.Module):
                self.graph = graph

        return Container(self.__unet_color_transfer)  # type: ignore

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
            "discriminator_b": self.__discriminator_b.state_dict(),
        }

        if self.__multi_d:
            state_dicts.update(
                {"discriminator_texture": self.__discriminator_texture.state_dict()}
            )
        return state_dicts

    def __repr__(self) -> str:
        return """
            ADEGAN: The GAN developed for the Artificial Data Enhancement project(s)
        """
