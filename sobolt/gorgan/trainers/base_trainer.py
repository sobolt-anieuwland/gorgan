import math
import os
import random
from datetime import datetime as dt
from typing import Dict, Optional, List, Any, Tuple, Union
import warnings

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from . import Trainer
from sobolt.gorgan.data import Dataset, ProgressiveUpsamplingDecorator
from sobolt.gorgan.gan import Gan, gan_factory
from sobolt.gorgan.nn import convolution_layer_extractor


class BaseTrainer(Trainer):
    @staticmethod
    def from_config(
        config: Dict[str, Any],
        quiet: bool = False,
        device_rank: int = 0,
        num_devices: int = 1,
    ):
        """Initializes the trainer. Uses the given config dictionary to construct the
        right GAN.

        Parameters
        ----------
        config: Dict[str, Any]
            The GAN's configuration. Is used to construct the right GAN type, the
            generator's and discriminator's graphs/optimizer, etc.
        quiet: bool (Default False)
            If set to `True`, will not print output to the console.
        device_rank: int (Default 0)
            The id number of the device. The device may be a CPU or GPU.
        num_devices: int (Default 1)
            The number of devices. The devices meant are either CPUs or GPUs and
            therefore is always atleast 1. This variables different from the `"use_gpu"`
            key in the config dictionary in that it might also mean CPUs.
        """
        # Set random seed for reproducibility
        seed = config.get("seed", False)
        if not seed:
            seed = random.randint(1, 10000)
        config["seed"] = seed
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        date = dt.strftime(dt.now(), "%Y-%m-%d")
        time = dt.strftime(dt.now(), "%H.%M")

        name = config.get("name", "unnamedrun")
        train_dir = config.get("train_dir", "in1_train")
        out_dir = os.path.join(train_dir, f"{date}")
        out_dir = os.path.join(out_dir, f"{time}_{name}")

        gan = gan_factory(config)

        # GAN training session components
        quiet = quiet
        use_progressive = config.get("progressive_gan", False)
        use_auxiliary = config.get("aux_gan", False)
        tensorboard = SummaryWriter(log_dir=f"{out_dir}")

        log_every_n_iters = config.get("log_every_n_iters", 100)
        save_every_n_iters = config.get("save_every_n_iters", 1000)
        render_every_n_iters = config.get("render_every_n_iters", 1000)

        init_prog_step = config.get("init_prog_step", 1)
        skip_init_val = config.get("skip_init_val", False)

        batch_size = config.get("batch_size", 1)
        data_workers = config.get("data_workers", 6)
        num_epochs = config.get("num_epochs", 10)
        num_batches_val = config.get("num_batches_val", 5)
        shape_originals = config["shape_originals"]
        upsample_factor = config.get("upsample_factor", 1)

        scheduler_g = config["generator"]["lr_scheduler"].get("type", "none")
        scheduler_d = config["discriminator"]["lr_scheduler"].get("type", "none")

        config_out = os.path.join(out_dir, f"config.{name}.yaml")
        with open(config_out, "w") as handle:
            yaml.dump(config, handle)

        if not quiet:
            print(gan)
            print()

        return BaseTrainer(
            gan=gan,
            tensorboard=tensorboard,
            name=name,
            out_dir=out_dir,
            quiet=quiet,
            batch_size=batch_size,
            data_workers=data_workers,
            num_epochs=num_epochs,
            num_batches_val=num_batches_val,
            shape_originals=shape_originals,
            upsample_factor=upsample_factor,
            use_auxiliary=use_auxiliary,
            use_progressive=use_progressive,
            log_every_n_iters=log_every_n_iters,
            save_every_n_iters=save_every_n_iters,
            render_every_n_iters=render_every_n_iters,
            init_prog_step=init_prog_step,
            skip_init_val=skip_init_val,
            scheduler_d=scheduler_d,
            scheduler_g=scheduler_g,
        )

    __gan: Gan
    __quiet: bool
    __print_prefix = ""
    __name: str
    __out_dir: str
    __tensorboard: SummaryWriter

    __log_every_n_iters: int
    __save_every_n_iters: int
    __render_every_n_iters: int

    __num_batches_train: int
    __num_batches_val: int
    __batch_size: int
    __data_workers: int
    __num_epochs: int
    __shape_originals: Tuple[int, int, int]
    __upsample_factor: int

    __init_prog_step: int
    __skip_init_val: bool

    __scheduler_g: str
    __scheduler_d: str

    __use_auxiliary: bool
    __use_progressive: bool

    def __init__(
        self,
        gan: Gan,
        tensorboard: SummaryWriter,
        name: str,
        out_dir: str,
        quiet: bool,
        batch_size: int,
        data_workers: int,
        num_epochs: int,
        num_batches_val: int,
        shape_originals: Tuple[int, int, int],
        upsample_factor: int,
        use_auxiliary: bool,
        use_progressive: bool,
        log_every_n_iters: int,
        save_every_n_iters: int,
        render_every_n_iters: int,
        init_prog_step: int,
        skip_init_val: bool,
        scheduler_d: str,
        scheduler_g: str,
    ):
        """Initializes the trainer. Uses the given config dictionary to construct the
        right GAN.

        Parameters
        ----------
        gan: Gan
            A GAN to train (ex. AdeGan, BaseGan, CycleGan)
        tensorboard: SummaryWriter:
            Tensorboard for validation logging
        name: str
            Name of a training session
        out_dir: str
            Directory validation output is saved to
        quiet: bool
            Specifies silencing of detailed output logging to terminal
        batch_size: int
            Training batch size
        data_workers: int
            Number of workers used during CPU multiprocessing
        num_epochs: int
            Number of epochs for a training session
        num_batches_val: int
            Validation batch size
        shape_originals: Tuple[int, int, int]
            Shape of input tensors
        upsample_factor: int
            Factor by which input will be upsampled to (default: 4)
        use_auxiliary: bool
            Specifies using auxiliary task (default: land cover classification) during
            training session
        use_progressive: bool
            Specifies a progressively trained GAN
        log_every_n_iters: int
            Frequency of running validation metrics for tensorboard logging
        save_every_n_iters: int
            Frequency of saving network weights as checkpoints
        render_every_n_iters: int
            Frequency of input rendering
        init_prog_step: int
            Setting for progressive training, progressive step to begin traing for
        skip_init_val: bool
            Skips initial validation step of untrained network during a GAN training
            session
        scheduler_g: Optional[SchedulerAdapter]
            A generator's learning rate scheduler (ex. CosineAnnealing, MultiStep)
        scheduler_d: Optional[SchedulerAdapter]
            A discriminator's learning rate scheduler (ex. CosineAnnealing, MultiStep)
        """
        super().__init__()
        self.__gan = gan

        # GAN training session components
        self.__name = name
        self.__out_dir = out_dir
        self.__quiet = quiet
        self.__use_auxiliary = use_auxiliary
        self.__tensorboard = tensorboard

        self.__log_every_n_iters = log_every_n_iters
        self.__save_every_n_iters = save_every_n_iters
        self.__render_every_n_iters = render_every_n_iters

        self.__init_prog_step = init_prog_step
        self.__skip_init_val = skip_init_val

        self.__batch_size = batch_size
        self.__data_workers = data_workers
        self.__num_epochs = num_epochs
        self.__num_batches_val = num_batches_val
        self.__shape_originals = shape_originals
        self.__upsample_factor = upsample_factor

        self.__scheduler_g = scheduler_g
        self.__scheduler_d = scheduler_d

        if use_progressive:
            self.train = self.__train_progressively  # type: ignore
        else:
            self.train = self.__train  # type: ignore

    def train(self, train_set: Dataset, val_set: Optional[Dataset], init_epoch: int = 1):
        # At runtime, `__init__()` should replace this method with the function we
        # actually want to use. This allows us to switch between progressively training
        # or normal training.
        raise ValueError("`BaseTrainer.train()` method not correctly overwritten.")

    def __train(
        self,
        train_set: Dataset,
        val_set: Optional[Dataset],
        init_epoch: int = 1,  # Start at one to be more human readable
    ):
        """Trains the GAN. See `Trainer.train()` for details."""
        self.__train_set = train_set
        self.__val_set = val_set

        # Prepare datasets
        train_loader = train_set.as_dataloader(self.__batch_size, self.__data_workers)
        self.__num_batches_train = len(train_loader)
        val_loader = None
        self.__num_batches_val = max(self.__num_batches_val, 1)
        if val_set and len(val_set) != 0:
            val_loader = val_set.as_dataloader(
                self.__batch_size, self.__data_workers, shuffle=False
            )
        else:
            val_loader = self.__random_training_images()
        self.__num_batches_val = len(val_loader)

        # Do a validation round before training anything to see how it performs.
        # In the first epoch, apply validation to training data to see how the
        # generator performs before any training
        if not self.__skip_init_val:
            self.__validate(0, val_loader, "untrained", tb_step=0, render=True)

        # Epochs loops: both training and validation.
        # We substract 1 in final_epoch because we want the printed epoch range to be
        # to be [init_epoch, final_epoch]. However, because `range()` works as
        # [start, end) however, we add 1 there again.
        torch.backends.cudnn.benchmark = True  # Optimize cuda use
        final_epoch = init_epoch + self.__num_epochs - 1
        for epoch in range(init_epoch, final_epoch + 1):
            if not self.__quiet:
                print(f"{self.__print_prefix}[{epoch}/{final_epoch}] Starting new epoch")

            # Training
            self.__train_epoch(epoch, final_epoch, train_loader, val_loader)

            if not self.__quiet:
                print(f"{self.__print_prefix}[{epoch}/{final_epoch}] Finished epoch")

            # Validation
            if val_loader and self.__val_set and epoch % self.__save_every_n_iters == 0:
                iteration = epoch * self.__num_batches_train
                self.__validate(epoch, val_loader, "end", tb_step=iteration, render=True)

            # Exporting weights
            if not self.__quiet:
                print(
                    f"{self.__print_prefix}[{epoch}/{final_epoch}] Exporting checkpoint"
                )

            if not self.__quiet:
                print()

    def __train_progressively(
        self, train_set: Dataset, val_set: Optional[Dataset], init_epoch: int = 1
    ):
        """Trains the configured GAN according to ProGAN principles.

        It basically wraps a normal training run in a loop of progressive training
        steps. After each step we grow the GAN; the GAN's components decide themselves
        how they are grown. The given datasets are wrapped in a decorator dataset that
        adjust the target data to that progressive step's domain (for example, by
        downsampling).
        """
        # Validate the configuration before we start training
        valid_factor = (
            self.__upsample_factor > 0 and math.log2(self.__upsample_factor).is_integer()
        )
        assert (
            valid_factor
        ), f"Invalid upsample_factor {self.__upsample_factor}, not power of 2."
        num_prog_steps = int(math.log2(self.__upsample_factor))
        if (
            not self.__quiet
            and (self.__init_prog_step is not None)
            and (self.__init_prog_step > num_prog_steps)
        ):
            msg = (
                f"WARNING: init_prog_step {self.__init_prog_step} greater than "
                f"num_prog_steps {num_prog_steps}, should be other way around. "
                f"To resolve, check that the upsample_factor is high enough: "
                f"init_prog_step < log2(upsample_factor)."
            )
            print(msg)

        # For each progresive step, calculate the first epoch and otherwise do a normal
        # training routine. Afterwards, grow the GAN.
        # upsample_Factor 4: range(1, 2log(4) + 1) range(1,3)
        for step in range(self.__init_prog_step, num_prog_steps + 1):
            if not self.__quiet:
                self.__print_prefix = f"[prog{step}/{num_prog_steps}]"
                print(f"{self.__print_prefix} Starting progressive training step {step}")

            # Create datasets that output progressively resampled data
            # In progressive step 0, we train to the same size as the original images,
            # in step 1, to twice the size; step 2, four times the size, etc.
            prog_train_set = ProgressiveUpsamplingDecorator(train_set, 2 ** step)
            prog_val_set = (
                ProgressiveUpsamplingDecorator(val_set, 2 ** step)
                if val_set is not None
                else val_set
            )

            # Train a progressive stepping, counting epochs from the last one `__train()`
            # finished.
            self.__train(prog_train_set, prog_val_set, init_epoch)
            init_epoch += self.__num_epochs
            if step <= num_prog_steps:
                self.__gan.grow()  # Don't grow GAN last iteration

    def export_checkpoint(self, iteration: int, dirpath: str):
        """Exports the current GAN to a pytorch state dict and writes that out as a
        checkpoint.
        """
        state_dict = self.__gan.to_state_dict()
        for name, sub_state_dict in state_dict.items():
            path = os.path.join(dirpath, f"{name}_iteration{iteration}_checkpoint.pth")
            torch.save(sub_state_dict, path)

    def export_onnx(self, iteration: int, dirpath: str):
        """Convert the generator (and generator only) to an onnx."""
        np_inputs = np.array([0.1] * np.prod(self.__shape_originals))
        np_inputs = np_inputs.reshape(self.__shape_originals)
        torch_inputs = torch.tensor(np_inputs).float().unsqueeze(0)
        path = os.path.join(dirpath, f"generator_iteration{iteration}.onnx")
        self.__gan.export_onnx(torch_inputs, path)

    def __train_epoch(
        self, epoch: int, final_epoch: int, data_train: DataLoader, data_val: DataLoader
    ):
        """Train the GAN for a whole epoch on the given data set.

        Also prints a log in case the trainer is not set to be quiet.

        Parameters
        ----------
        epoch: int
            The epoch this is. Is expected to start at 1, not 0.
        final_epoch: int
            The last epoch. Used for printing where the training is at.
        data_train: DataLoader
            A pytorch DataLoader with the training data. Each iteration should
            yield a dictionary with typically `"originals"` and `"targets"` filled.
            Additional training data, such as auxiliary classes, might exist under
            other keys.
        """
        # Initialize some variables
        num_batches = self.__num_batches_train
        checkpoints_dir = os.path.join(f"{self.__out_dir}", "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        # For each batch in the dataloader, do a training iteration
        batch_losses: Dict[str, List[float]] = {}
        batch_lrs: Dict[str, List[float]] = {}
        for batch_nr, batch_data in enumerate(data_train, 1):
            iteration = (epoch - 1) * num_batches + batch_nr

            # Train an iteration and print results
            if not self.__quiet:
                msg = f"{self.__print_prefix}\
                        [{epoch}/{final_epoch}][{batch_nr}/{num_batches}]\t "
                print(msg, end="")
            logs = self.__train_iteration(epoch, batch_nr, batch_data)

            # Gather all losses & learning rates for logging to tensorboard
            logs["losses"] = logs["generator"]["losses"]
            logs["losses"].update(logs["discriminator"]["losses"])

            if "cycle" in logs.keys():
                logs["losses"].update(logs["cycle"]["losses"])

            logs["learning_rates"] = {
                "lr_g": logs["generator"]["lr"],
                "lr_d": logs["discriminator"]["lr"],
            }

            if self.__scheduler_g != "none":
                self.__gan.step_scheduler(
                    logs["generator"]["losses"], epoch, batch_nr, num_batches, "generator"
                )

            if self.__scheduler_d != "none":
                self.__gan.step_scheduler(
                    logs["discriminator"]["losses"],
                    epoch,
                    batch_nr,
                    num_batches,
                    "discriminator",
                )

            if not self.__quiet:
                msg = _losses_to_str(logs["losses"])
                print(msg)

            # Save losses for graphs
            for loss, value in logs["losses"].items():
                if loss not in batch_losses:
                    batch_losses[loss] = []
                batch_losses[loss].append(value)

            # Save LRs for tensorboard
            for learning_rate, lr_value in logs["learning_rates"].items():
                if learning_rate not in batch_lrs:
                    batch_lrs[learning_rate] = []
                batch_lrs[learning_rate].append(lr_value)

            # Write to tensorboard and validate
            stats = batch_nr == 1 or batch_nr % self.__log_every_n_iters == 0
            render = batch_nr % self.__render_every_n_iters == 0
            if stats or render:
                if not self.__quiet:
                    msg = (
                        f"{self.__print_prefix}\
                        [{epoch}/{final_epoch}][{batch_nr}/{num_batches}]"
                        f"\tLogging and/or validating for iteration {iteration} and "
                        f"validating on training images"
                    )
                    print(msg)

                # Log losses to tensorboard
                for loss, values in batch_losses.items():
                    mean_value = np.mean(values)
                    self.__tensorboard.add_scalar(loss, mean_value, iteration)

                # Log LRs to tensorboard
                for learning_rate, lr_values in batch_lrs.items():
                    mean_lrs = np.mean(lr_values)
                    self.__tensorboard.add_scalar(learning_rate, mean_lrs, iteration)

                # Add weight histograms for layers of interests
                # layers_dict = self.get_layers_dict()
                # for name, layers in layers_dict.items():
                #     self.__tensorboard.add_histogram(name, layers, iteration)

                note = "iter" + str(iteration)
                self.__validate(epoch, data_val, note, tb_step=iteration, render=render)

            # Exception save-every-n not used
            if num_batches < self.__save_every_n_iters and batch_nr % num_batches == 0:
                if not self.__quiet:
                    msg = (
                        f"{self.__print_prefix}\
                        [{epoch}/{final_epoch}][{batch_nr}/{num_batches}]"
                        f"\tSaving model at iteration {iteration}"
                    )
                    print(msg)
                self.export_checkpoint(iteration, checkpoints_dir)
                warnings.warn(
                    "Warning: the save-every-n-iters specified is not being used. Instead the max number of iterations in the epoch was used"
                )

            # Save models every so often
            if batch_nr % self.__save_every_n_iters == 0:
                if not self.__quiet:
                    msg = (
                        f"{self.__print_prefix}\
                        [{epoch}/{final_epoch}][{batch_nr}/{num_batches}]"
                        f"\tSaving model at iteration {iteration}"
                    )
                    print(msg)
                self.export_checkpoint(iteration, checkpoints_dir)
                # self.export_onnx(epoch, checkpoints_dir)

    def __train_iteration(self, epoch: int, batch_nr: int, batch_data: Dict[str, Any]):
        """Train the GAN a single iteration.

        Parameters
        ----------
        epoch: int
            The epoch we are at. Is expected to start at 1, not 0.
        batch_nr: int
            Which batch out of the total this is. Is expected to start at 1, not 0.
        batch_data: Dict[str, Any]
            The actual data to train the GAN on this iteration.

        Returns
        -------
        Dict[str, Any]
            The dictionary the GAN returns when it trains, containing among other
            things the loss values.
        """
        return self.__gan.train(batch_nr, batch_data)

    def __validate(
        self,
        epoch: int,
        data_val: DataLoader,
        note: str = "end",
        tb_step: Optional[int] = None,
        render: bool = False,
    ):
        """Validate the current generator using the given validation set.

        Parameters
        ----------
        epoch: int
            This epoch's number
        data_val: DataLoader
            The dataloader with validation data. Is expected to yield dictionaries
            with the keys `"originals"` and `"targets"`.
        note: str (Default "end")
            A custom string incorporated into output written to disk, which can be
            used for incorporating the iteration number or similar.
        tb_step: Optional[int] (Default None)
        render: bool (Default False)
            Whether or not to renders GAN output and write them out.
        """
        metrics = self.__gan.validate(data_val, render=render)
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.__tensorboard.add_scalar(metric, value, tb_step)
            elif render and isinstance(value, Image.Image):
                img_dir = os.path.join(self.__out_dir, "val_imgs")
                img_dir = os.path.join(img_dir, f"epoch{epoch}_iter{tb_step}")
                os.makedirs(img_dir, exist_ok=True)
                o = os.path.join(img_dir, metric)
                value.save(o)

    def __random_training_images(self, amount: Optional[int] = None) -> DataLoader:
        """Randomly takes the given amount from the training images set and creates a
        new DataLoader with it. Useful for seeing how the GAN progresses on its
        training data.

        Parameters
        ----------
        amount: Optional[int]
            Optional. The amount of images to take from the training set. If `None`,
            the amount equal to the validation set it taken.

        Returns
        -------
        DataLoader
            A pytorch DataLoader of training images, initialized in the same way
        """
        fallback_len = (
            len(self.__val_set)
            if self.__val_set
            else self.__num_batches_val * self.__batch_size
        )
        amount = amount if amount else fallback_len
        random_indices = random.sample(range(len(self.__train_set)), amount)
        subset = Subset(self.__train_set, random_indices)
        return DataLoader(subset, self.__batch_size, shuffle=False, num_workers=2)

    def get_layers_dict(self) -> Dict[str, torch.Tensor]:
        """We use the state dict of the graph to be able to extract layer names and
        corresponding weights, specifically convolution layers, to better understand
        how our model is learning throughout training.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary consisting of a layer name and the corresponding weights. This
            is then used in __train_epoch() to log in tensorboard the
            convolution layer weights as histograms to better understand model
            performance.
        """
        state_dict = self.__gan.to_state_dict()
        return convolution_layer_extractor(state_dict)


def _losses_to_str(losses: Dict[str, float]) -> str:
    """Helper function to reduce a dictionary of str-float pairs to a single,
    printable string.
    """
    loss_strs = [f"{name}: {val:.4f}" for name, val in losses.items()]
    loss_strs.sort()
    return "" + "  ".join(loss_strs)
