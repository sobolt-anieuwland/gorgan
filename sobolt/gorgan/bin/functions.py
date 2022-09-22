import os
from typing import Dict, Any
from datetime import datetime as dt
from pprint import pprint

import torch
import yaml

from sobolt.gorgan.trainers import Trainer
from sobolt.gorgan.data import dataset_factory
from sobolt.gorgan.bin.batch_processor import process_batches


def train_in1(args):
    """ Loads datasets and start a GAN training session """

    # Read config and update with settings from CLI
    config = read_config(args.config)
    update_config(args, config)

    # Specify for detailed output logging to terminal
    quiet = config.get("quiet", False)
    if not quiet:
        pprint(config)
        print()

    # Dataset independent variable
    normalization = config.get("normalization", [0.0, 1.0])

    # Load train and val datasets based on the config file, with modifications
    # given on the cli
    num_images_train = (
        config["batch_size"] * config["num_batches_train"]
        if config.get("num_batches_train")
        else None
    )
    num_images_val = (
        config["batch_size"] * config["num_batches_val"]
        if config.get("num_batches_val")
        else None
    )

    train_config = config["datasets"]["train"]
    val_config = config["datasets"].get("validation", {})

    # Setting conditional GAN
    conditional_gan = config.get("conditional_gan", False)
    conditional_mask_indices = config.get("conditional_mask_indices", None)

    # Setting auxiliary GAN
    auxiliary_gan = config.get("aux_gan", False)

    train_set = dataset_factory(
        train_config,
        config,
        num_images_train,
        0,
        conditional_gan,
        conditional_mask_indices,
        auxiliary_gan,
        normalization,
    )
    # Use train settings is no path is specified in the config file
    num_val_skip = val_config.get("offset", 0)
    val_set = (
        None
        if val_config == {}
        else dataset_factory(
            val_config,
            config,
            num_images_val,
            num_val_skip,
            conditional_gan,
            conditional_mask_indices,
            auxiliary_gan,
            normalization,
        )
    )

    if not quiet:
        print()

    num_devices = config["use_gpu"]

    go(0, 1, config, train_set, val_set, quiet)


def go(device_rank, num_devices, config, train_set, val_set, quiet):
    from sobolt.gorgan import trainer_factory

    # Variables device_rank and num_devices are currently not / barely used, but they
    # allow implementing GIL-less parallel training in the future.

    trainer: Trainer = trainer_factory(config, device_rank, num_devices, quiet)
    trainer.train(train_set, val_set)


def test_in1(args):
    """ Loads datasets and starts a generator testing session """
    from sobolt.gorgan.validation import FeatureVisualizer
    from sobolt.gorgan.validation.validator import Validator
    from sobolt.gorgan.data import In1Dataset

    config = read_config(args.config)

    # Get graph variables
    opt_args = {
        "init_prog_step": config.get("init_prog_step", 1),
        "upsample_factor": config.get("upsample_factor", 1),
        "use_auxiliary": config.get("aux_gan", False),
        "use_attention": config.get("attention", False),
        "use_progressive": config.get("progressive_gan", False),
        "use_condition": config.get("conditional_gan", False),
    }

    feature_visualizer = FeatureVisualizer.from_config(
        config, model_path=args.weights, graph_type=args.graph_type, opt_args=opt_args
    )

    validator = Validator.from_config(
        config, model_path=args.weights, band_order=args.band_order, opt_args=opt_args
    )

    # Specify for detailed output logging to terminal
    quiet = config.get("quiet", False)
    if not quiet:
        print()

    val_config = config["datasets"]["validation"]
    num_images_train = config["batch_size"] * config["num_batches_train"]
    num_images_val = config["batch_size"] * config["num_batches_val"]
    num_images_train = 0 if not val_config.get("offset", True) else num_images_train
    val_set = In1Dataset.from_config(val_config, config, num_images_val, num_images_train)
    if not quiet:
        print()

    name = retrieve(args.name, str, "name", "unnamed", config)
    date = dt.strftime(dt.now(), "%Y-%m-%d")
    time = dt.strftime(dt.now(), "%H.%M")

    test_dir = retrieve(args.test_dir, str, "test_dir", "in1_test", config)
    out_dir = os.path.join(test_dir, f"{date}")
    out_dir = os.path.join(out_dir, f"{time}_{name}")

    val_imgs_dir = os.path.join(f"{out_dir}", "val_imgs")
    feature_imgs_dir = os.path.join(f"{out_dir}", "feature_imgs")
    batch_size = config.get("batch_size", 1)
    data_workers = config.get("data_workers", 1)

    validator.validate(
        val_set.as_dataloader(batch_size, data_workers, shuffle=False),
        set_object=val_set,
        test=True,
        directory=val_imgs_dir,
    )
    feature_visualizer.render_convolution_layers(args.image, out_dir=feature_imgs_dir)


# def infer_in1(args):
#     """ Apply graph to given input raster, save processed raster to given output path """

#     config = read_config(args.config)
#     # set default values if not specified from flags
#     overlap = args.crop_overlap if args.crop_overlap is not None else 10
#     tiles_size = args.crop_size if args.crop_size is not None else 256
#     padding = args.crop_padding if args.crop_padding is not None else 7
#     bgr = False if args.band_order in ["RGB", "RGBI", "RGBI+"] else True
#     sensor = args.sensor if args.sensor is not None else "sentinel2"

#     if args.weights:
#         config["weights"] = args.weights

#     file_in = retrieve(args.raster_in, str, "raster_in", "unnamed", config)
#     file_out = retrieve(args.raster_out, str, "raster_out", "unnamed", config)

#     process_batches(file_in, file_out, config, overlap, tiles_size, padding, bgr, sensor)


def read_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
        if "base" in config:
            file_name = config.pop("base")
            path = os.path.dirname(os.path.abspath(config_path))
            file_path = os.path.join(path, file_name)

            if os.path.isfile(file_path):
                base_config = read_config(file_path)
                config = update_nested_dict(base_config, config)
            else:
                msg = f"Specified base config file {file_name} does not exist"
                raise ValueError(msg)
    return config


def retrieve(val, val_type, val_name, default, config):
    return val if isinstance(val, val_type) else config.get(val_name, default)


def update_config(args, config):
    """Updates configuration from file in-place with arguments from the command line

    Parameters
    ----------
    args: argparse result object
        An argparse name space with the CLI flags / settings. These values update the
        config file.
    config: Dict[str, Any]
        The base configuration to update. Is updated in-place.
    """
    num_gpu = torch.cuda.device_count()

    # TODO Process members of the args namespace without needing to specify them here
    config["quiet"] = retrieve(args.quiet, bool, "quiet", False, config)
    config["name"] = retrieve(args.name, str, "name", "unnamed", config)
    config["train_dir"] = retrieve(args.train_dir, str, "train_dir", "in1_train", config)
    config["use_gpu"] = retrieve(args.use_gpu, int, "use_gpu", False, config)
    config["batch_size"] = retrieve(args.batch_size, int, "batch_size", 2, config)
    if config["use_gpu"] and torch.cuda.device_count() > 0:
        config["batch_size"] *= torch.cuda.device_count()
    config["save_epochs"] = retrieve(args.save_epochs, int, "save_epochs", 1, config)
    config["num_epochs"] = retrieve(args.num_epochs, int, "num_epochs", 10, config)
    config["num_batches_train"] = retrieve(
        args.num_batches_train, int, "num_batches_train", 10, config
    )
    config["num_batches_val"] = retrieve(
        args.num_batches_val, int, "num_batches_val", 10, config
    )
    config["data_workers"] = retrieve(args.data_workers, int, "data_workers", 4, config)
    config["log_every_n_iters"] = retrieve(
        args.log_every_n_iters, int, "log_every_n_iters", 100, config
    )
    config["render_every_n_iters"] = retrieve(
        args.render_every_n_iters, int, "render_every_n_iters", 1000, config
    )
    config["save_every_n_iters"] = retrieve(
        args.save_every_n_iters, int, "save_every_n_iters", 1000, config
    )
    # config["attention"] = retrieve(args.attention, bool, "attention", False, config)
    # config["conditional_gan"] = retrieve(
    #     args.conditional, bool, "conditional_gan", False, config
    # )
    # config["aux_gan"] = retrieve(args.auxiliary, bool, "aux_gan", False, config)
    # config["dummy_gan"] = retrieve(args.dummy, bool, "dummy_gan", False, config)
    # config["progressive_gan"] = retrieve(
    #     args.progressive, bool, "progressive_gan", False, config
    # )
    config["attention"] = False
    config["aux_gan"] = False
    config["conditional_gan"] = False
    config["dummy_gan"] = False
    config["progressive_gan"] = False
    config["cycle_gan"] = retrieve(args.cycle, bool, "cycle_gan", False, config)
    config["skip_init_val"] = retrieve(
        args.skip_init_val, bool, "skip_init_val", False, config
    )
    # config["init_prog_step"] = retrieve(
    #     args.init_prog_step, int, "init_prog_step", 1, config
    # )
    config["init_prog_step"] = 1
    config["upsample_factor"] = retrieve(
        args.upsample_factor, int, "upsample_factor", 4, config
    )
    config["base_loss"] = retrieve(args.base_loss, str, "base_loss", "minimax", config)
    config["total_variation"] = retrieve(
        args.total_variation, bool, "total_variation", False, config
    )

    # Generator specific settings
    config["generator"]["weights"] = (
        args.generator_weights
        if isinstance(args.generator_weights, str)
        else config["generator"].get("weights", "random")
    )

    config["generator"]["optimizer"]["args"]["lr"] = (
        args.generator_learning_rate
        if isinstance(args.generator_learning_rate, float)
        else config["generator"]["optimizer"]["args"].get("lr", 0.0002)
    )

    config["generator"]["train_every_n_iters"] = (
        args.generator_train_every_n_iters
        if isinstance(args.generator_train_every_n_iters, int)
        else config["generator"].get("train_every_n_iters", 1)
    )

    config["generator"]["lr_scheduler"] = config["generator"].get("lr_scheduler", {})
    config["generator"]["lr_scheduler"]["type"] = (
        args.generator_scheduler_type
        if isinstance(args.generator_scheduler_type, str)
        else config["generator"].get("lr_scheduler", {}).get("type", "none")
    )

    # Discriminator specific settings
    config["discriminator"]["weights"] = (
        args.discriminator_weights
        if isinstance(args.discriminator_weights, str)
        else config["discriminator"].get("weights", "random")
    )

    config["discriminator"]["optimizer"]["args"]["lr"] = (
        args.discriminator_learning_rate
        if isinstance(args.discriminator_learning_rate, float)
        else config["discriminator"]["optimizer"]["args"].get("lr", 0.0002)
    )

    config["discriminator"]["train_every_n_iters"] = (
        args.discriminator_train_every_n_iters
        if isinstance(args.discriminator_train_every_n_iters, int)
        else config["discriminator"].get("train_every_n_iters", 1)
    )

    config["discriminator"]["lr_scheduler"] = config["discriminator"].get(
        "lr_scheduler", {}
    )
    config["discriminator"]["lr_scheduler"]["type"] = (
        args.discriminator_scheduler_type
        if isinstance(args.discriminator_scheduler_type, str)
        else config["discriminator"].get("lr_scheduler", {}).get("type", "none")
    )

    # Cycle specific settings
    config["cycle"] = config.get("cycle", {})
    config["cycle"]["optimizer"] = config["cycle"].get("optimizer", {})
    config["cycle"]["optimizer"]["args"] = config["cycle"]["optimizer"].get("args", {})
    config["cycle"]["lr_scheduler"] = config["cycle"].get("lr_scheduler", {})
    config["cycle"]["optimizer"]["args"]["lr"] = (
        args.cycle_learning_rate
        if isinstance(args.cycle_learning_rate, float)
        else config["cycle"]["optimizer"]["args"].get("lr", 0.0002)
    )

    config["cycle"]["train_every_n_iters"] = (
        args.cycle_train_every_n_iters
        if isinstance(args.cycle_train_every_n_iters, int)
        else config["cycle"].get("train_every_n_iters", 1)
    )

    config["cycle"]["lr_scheduler"]["type"] = (
        args.cycle_scheduler_type
        if isinstance(args.cycle_scheduler_type, str)
        else config["cycle"]["lr_scheduler"].get("type", "none")
    )


def onnxify(args):
    from sobolt.gorgan.graphs import graph_factory
    from sobolt.gorgan.gan import Generator, DummyGan

    config = read_config(args.config)
    if args.generator_weights:
        config["generator"]["weights"] = args.generator_weights
    if not args.out_onnx:
        name = os.path.basename(args.generator_weights)
        name = os.path.splitext(name)[0] + ".onnx"
        args.out_onnx = name

    generator = Generator.from_config(config, DummyGan(config))
    graph = generator.graph

    dummy_input_size = [1] + [nr for nr in config["shape_originals"]]
    dummy_input = torch.randn(dummy_input_size)

    torch.onnx.export(graph, dummy_input, args.out_onnx)


def update_nested_dict(d: Dict[Any, Any], u: Dict[Any, Any]) -> Dict[Any, Any]:
    """Takes two dictionaries and does an equivalent operation to `dict1.update(dict2)`,
    but for nested dictionaries. In other words, if a value under a key is a
    dictionary, it will keep that dictionary's values and update them with values in
    the other dictionary. A normal `update()` operation would simply overwrite that
    dictionary with whatever value is in the other dictionary.

    Parameters
    ----------
    d: Dict[Any, Any]
        The dictionary to update
    u: Dict[Any, Any]
        The second dictionary whose values we want to include or use to overwrite the
        first dictionary

    Returns
    -------
    Dict[Any, Any]
        A dictionary based on `d` with `u`'s values included, or used to overwrite
        values already in `d`.
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d_v = d.get(k, {})
            if isinstance(d_v, dict):
                d[k] = update_nested_dict(d_v, v)
            else:
                d[k] = v
        else:
            d[k] = v
    return d
