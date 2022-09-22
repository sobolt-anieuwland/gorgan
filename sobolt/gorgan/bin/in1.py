#!/usr/bin/env python3
import pathlib

from .functions import train_in1, test_in1, onnxify


def main():
    args = parse_arguments()
    execute(args)


def execute(args):
    # Dictionary containing functions
    commands = {
        "train": train_in1,
        "test": test_in1,
        # "infer": infer_in1,
        "onnxify": onnxify,
    }

    to_execute = commands[args.command]
    to_execute(args)


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()

    commands_parser = parser.add_subparsers()
    commands_parser.required = True
    commands_parser.dest = "command"

    # Define available commands
    train_parser = commands_parser.add_parser("train")
    test_parser = commands_parser.add_parser("test")
    # infer_parser = commands_parser.add_parser("infer")
    # integrate_gradients_parser = commands_parser.add_parser("integrate-gradients")
    onnxify_parser = commands_parser.add_parser("onnxify")

    # Set up the subparsers
    ## train parser
    ## Only required option is --config
    train_parser.add_argument(
        "-c", "--config", required=True, help="path of training config file."
    )

    ## Optional top-level flag that change the config
    train_parser.add_argument("-q", "--quiet", action="store_true")
    train_parser.add_argument("-n", "--name", help="name of training results folder.")
    train_parser.add_argument(
        "-dir", "--train-dir", help="path to directory where results folder is created."
    )
    train_parser.add_argument(
        "--batch-size", type=int, help="number of images to load in each train batch."
    )
    train_parser.add_argument(
        "--upsample-factor", type=int, help="upsampling factor for super resolution."
    )
    train_parser.add_argument(
        "--save-epochs",
        type=int,
        help="log losses and render images at the end of every epoch.",
    )
    train_parser.add_argument(
        "--num-epochs", type=int, help="max number of training epochs."
    )
    train_parser.add_argument(
        "--num-batches-train", type=int, help="max number of training batches."
    )
    train_parser.add_argument(
        "--num-batches-val",
        type=int,
        help="number of validation images to save for every training batch.",
    )
    train_parser.add_argument(
        "--data-workers", type=int, help="max number of data worker to use."
    )
    train_parser.add_argument(
        "--log-every-n-iters",
        type=int,
        help="number of iteration after which losses are logged for tensorboard.",
    )
    train_parser.add_argument(
        "--save-every-n-iters",
        type=int,
        help="number of iterations after which checkpoints are exported.",
    )
    train_parser.add_argument(
        "--render-every-n-iters",
        type=int,
        help="number of iterations after which validation images are rendered and saved.",
    )
    # train_parser.add_argument(
    #     "--attention",
    #     type=bool,
    #     help="add or not attention layer to the model graph.",
    #     choices=[True, False],
    # )
    # train_parser.add_argument(
    #     "--auxiliary",
    #     type=bool,
    #     help="use or not auxiliary classes during training.",
    #     choices=[True, False],
    # )
    # train_parser.add_argument(
    #     "--dummy", type=bool, help="use or not dummy gan.", choices=[True, False]
    # )
    # train_parser.add_argument(
    #     "--progressive",
    #     type=bool,
    #     help="use or not progressive training.",
    #     choices=[True, False],
    # )
    # train_parser.add_argument(
    #     "--conditional",
    #     type=bool,
    #     help="use or not conditional information during training.",
    #     choices=[True, False],
    # )
    train_parser.add_argument(
        "--cycle",
        type=bool,
        help="wheter to use or not cycle gan during training.",
        choices=[True, False],
    )
    train_parser.add_argument(
        "--skip-init-val",
        type=bool,
        help="skip or not initial rendering of untrained generated images.",
        choices=[True, False],
    )
    # train_parser.add_argument(
    #     "--init-prog-step",
    #     type=int,
    #     help="initial progressive step, i.e. initial fraction of rrdb blocks to train.",
    # )
    train_parser.add_argument(
        "-l",
        "--base-loss",
        required=False,
        choices=["minimax", "wasserstein", "least-squared-error"],
        help="adversarial loss to use during training.",
    )
    train_parser.add_argument(
        "-tva",
        "--total-variation",
        type=bool,
        help="compute or not total variational loss during training.",
        choices=[True, False],
    )
    train_parser.add_argument(
        "-lpc",
        "--lp-coherence",
        type=bool,
        help="compute or not local phase coherence loss during training.",
        choices=[True, False],
    )
    train_parser.add_argument(
        "--use_gpu",
        type=int,
        help="Number of GPUs (not id number of the GPU to use). For CPU, set to 0",
    )
    # Optional generator-level flags that change generator settings
    train_parser.add_argument(
        "-g:w",
        "--generator-weights",
        type=str,
        help="Distribution or checkpoint to use to initialize generator weights.",
    )
    train_parser.add_argument(
        "-g:lr", "--generator-learning-rate", type=float, help="Generator learning rate."
    )
    train_parser.add_argument(
        "-g:iters",
        "--generator-train-every-n-iters",
        type=int,
        help="number of iteration after which the generator weights are updated.",
    )
    train_parser.add_argument(
        "-g:s",
        "--generator-scheduler-type",
        type=str,
        help="Type of learning rate scheduler. If None costant learning rate is used.",
        choices=["plateau", "threshold", "cosine", "multistep", "None"],
    )

    # Optional discriminator-level flags that change discriminator settings
    train_parser.add_argument(
        "-d:w",
        "--discriminator-weights",
        type=str,
        help="Distribution or checkpoint to use to initialize generator weights.",
    )
    train_parser.add_argument(
        "-d:lr",
        "--discriminator-learning-rate",
        type=float,
        help="Discriminator learning rate.",
    )
    train_parser.add_argument(
        "-d:iters",
        "--discriminator-train-every-n-iters",
        type=int,
        help="number of iteration after which the discriminator weights are updated.",
    )
    train_parser.add_argument(
        "-d:s",
        "--discriminator-scheduler-type",
        type=str,
        help="Type of learning rate scheduler. If None costant learning rate is used.",
        choices=["plateau", "threshold", "cosine", "multistep", "None"],
    )

    # Optional discriminator-level flags that change discriminator settings
    train_parser.add_argument(
        "-c:lr", "--cycle-learning-rate", type=float, help="Cycle learning rate."
    )
    train_parser.add_argument(
        "-c:iters",
        "--cycle-train-every-n-iters",
        type=int,
        help="number of iteration after which the cycle weights are updated.",
    )
    train_parser.add_argument(
        "-c:s",
        "--cycle-scheduler-type",
        type=str,
        help="Type of learning rate scheduler. If None costant learning rate is used.",
        choices=["plateau", "threshold", "cosine", "None"],
    )

    ## test parser
    test_parser.add_argument(
        "-c", "--config", required=True, help="path of config file used for training."
    )
    test_parser.add_argument(
        "-i",
        "--image",
        required=True,
        help="path to dummy input image to use to compute quality metrics.",
    )
    test_parser.add_argument(
        "-gt",
        "--graph-type",
        required=True,
        help="gan component to test.",
        choices=["generator", "discriminator"],
    )
    test_parser.add_argument(
        "-n", "--name", required=False, help="name of test results folder."
    )
    test_parser.add_argument(
        "-dir",
        "--test-dir",
        required=False,
        help="path to directory where results folder is created.",
    )
    test_parser.add_argument(
        "-w",
        "--weights",
        required=False,
        help="path to graph weights checkpoint to load.",
    )  # Can be in config too
    test_parser.add_argument(
        "-b",
        "--band-order",
        required=False,
        help="input image format.",
        choices=["RGB", "BGR", "RGBI", "BGRI", "RGBI+", "BGRI+"],
    )

    ## infer parser
    # infer_parser.add_argument(
    #     "-c",
    #     "--config",
    #     required=True,
    #     help="path of config file used for training. Basic config files can be found in \
    #           configs/inference/. Weights path can be overwriten using flag -w.",
    # )
    # infer_parser.add_argument(
    #     "-i", "--raster-in", required=True, help="path to input raster to process."
    # )
    # infer_parser.add_argument(
    #     "-o", "--raster-out", required=True, help="path to output raster to save."
    # )
    # infer_parser.add_argument(
    #     "-w",
    #     "--weights",
    #     required=False,
    #     help="path to graph weights checkpoint to load. A path to pretrained weights must \
    #     be provided either using this flag or in a config file.",
    # )  # Can be in config too
    # infer_parser.add_argument(
    #     "-b",
    #     "--band-order",
    #     required=False,
    #     help="input raster format. Default first channel is considered to be Blue.",
    #     choices=["RGB", "BGR", "RGBI", "BGRI", "RGBI+", "BGRI+"],
    # )
    # infer_parser.add_argument(
    #     "-c:o",
    #     "--crop-overlap",
    #     required=False,
    #     help="overlap pixel size to use for batch cropping.",
    # )
    # infer_parser.add_argument(
    #     "-c:s",
    #     "--crop-size",
    #     required=False,
    #     help="pixel size to use for batch cropping.",
    # )
    # infer_parser.add_argument(
    #     "-c:p",
    #     "--crop-padding",
    #     required=False,
    #     help="padding pixel size to use for batch cropping.",
    # )
    # infer_parser.add_argument(
    #     "-s",
    #     "--sensor",
    #     required=False,
    #     help="Name of sensor / satellite product of input raster. Required to choose the \
    #     correct normalization and denormalization functions. Default is sentinel2.",
    # )

    ## Export as ONNX parser
    onnxify_parser.add_argument(
        "-c", "--config", required=True, help="path of config file used for training."
    )
    onnxify_parser.add_argument(
        "-g:w",
        "--generator-weights",
        required=False,
        help="path to generator weights checkpoint to load.",
    )
    onnxify_parser.add_argument(
        "-o", "--out-onnx", required=False, help="path where output onnx will be stored."
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
