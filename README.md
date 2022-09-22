GorGAN: A pytorch framework for GAN experiments
=============================================
[Website](https://in1.ai) | [Installation](#installation) | [Usage](#usage) | [Features](#features) | [Configuration](#configuration) | [Contact](#contact) | [License and contributions](#license-and-contributions)

GorGAN is a GAN framework for reproducibly, systematically experimenting with GAN tasks. It was originally developed by [Sobolt](http://sobolt.com) for researching and training [in1](https://in1.ai), a single image super resolution service for Sentinel-2 imagery. To that end, it supports a variety of generators, discriminators, adversarial losses, auxiliary losses and data types.

## What's in a name?

The GorGAN framework is named after the gorgon creatures from Greek mythology. The snakes they have for hair represent the variety of components the framework supports. Of course, the pun also brings joy.

## Installation
Installing is a matter of cloning the repository and registering the package with pip. This also installs an executable with which training commands can be executed.

1. `git clone https://github.com/sobolt-anieuwland/gorgan`
2. `python3.8 -m venv gorgan/venv`
3. `source gorgan/venv/bin/activate`
4. `pip install -e gorgan`

## Usage
Download the Celeb-A faces dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or via this [direct link](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ). Extract it such that the files are located as `celeba/img_alignb_celeba/*.jpg` under the repo root. With this data, the three example configs can be run (`configs/dcgan.yaml`, `configs/esrgan.yaml`, `configs/cycle.yaml`).

To launch a DCGAN face generation training run, execute the below command:

```bash
gorgan train -c configs/dcgan.yaml
```

To use the framework with your own data, modify the `datasets` section to point to your own data.
Modify the GAN section and loss components to run your own tasks. See `configs/train_config_commented_example.yaml` for a commented config. More flags for the training command can be found as such:

```bash
gorgan train --help
```

## Features

* Train a variety of GAN computer vision tasks: image generation, style transfer, super resolution, super resolution as style transfer
* A variety of generators: DCGAN, ESRGAN, SPADE, UNet
* A variety discriminators: DCGAN, ESRGAN, PatchGAN, Wasserstein Critic
* Three adversarial losses: minimax, wasserstein, least squared error
* Several additional losses: SSIM, total variation, perceptual, local phase coherence,
* Quantitative validation via Tensorboard output
* Qualitative validation via rendering and saving of generated data
* Reproducible experimentation: the output directory stores the used config file and all trained weights
* Able to deal with non-RGB data, such as infrared or class maps
* Extendible for new tasks and custom data
* A special downsampling method using physical properties of sallite imagery, specifically Sentinel-2 and SuperView

## Configuration

Training runs are driven by and documented with the training configuration files. Some samples can be found in `configs/`. Specifically see `configs/train_config_commented_example.yaml` for a commented config.

## Contact

Please use the issues page to get in touch regarding the code: comments, questions or feedback. For general inquiries into the in1 service - but not the GAN framework - please e-mail us via [info@sobolt.com](info@sobolt.com).

## License and contributions
This GAN framework is licensed under the MIT license, as described in LICENSE.txt.

Importantly, it is released in the hope, but not he guarantee that it will be useful to someone. It was built for Sobolt to experiment with Single Image Super Resolution and other satellite style transfer tasks. This has led to the framework including some commands and features that are very specific to Sobolt's needs. Furthermore, gorgan is not under active development. While contributions are welcome in the form of PRs, there will be limited to no bugfixes and support from Sobolt.

## Selection of implemented literature

The following papers are some of the successfully implemented components:

* K. FLIEGEL, [Modeling and Measurement of Image Sensor Characteristics](https://www.radioeng.cz/fulltexts/2004/04_04_27_34.pdf).
* I. GULRAJANI e.a., [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf).
* J. JOHNSON e.a., [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf).
* X. MAO e.a., [Least Squares Generative Adversarial Networks](https://arxiv.org/pdf/1611.04076).
* X. WANG e.a., [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/pdf/1809.00219).
* J. ZHU e.a., [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf).

## What's in a name?
The GorGAN framework is named after the [gorgon creatures](https://en.wikipedia.org/wiki/Gorgon) from Greek mythology. The snakes they have for hair represent the variety of components the framework supports. Importantly however, the name can be written as a pun.

## Interesting command flags
### General
```bash
  -h, --help            Show this help message and exit.
  -n NAME, --name NAME  Name of training results folder.
  --batch-size BATCH_SIZE
                        Number of images to load in each train batch.
  -dir TRAIN_DIR, --train-dir TRAIN_DIR
                        Path to directory where results folder is created.
  --use_gpu USE_GPU     Number of GPUs (not id number of the GPU to use). For CPU, set to 0
```

### Losses
```bash
  -l {minimax,wasserstein,least-squared-error}, --base-loss {minimax,wasserstein,least-squared-error}
                        Adversarial loss to use during training.
  -tva {True,False}, --total-variation {True,False}
                        Compute or not total variational loss during training.
  -lpc {True,False}, --lp-coherence {True,False}
```

### Generator and discriminator

```bash
  -g:w GENERATOR_WEIGHTS, --generator-weights GENERATOR_WEIGHTS
                        Distribution or checkpoint to use to initialize generator weights.
  -g:lr GENERATOR_LEARNING_RATE, --generator-learning-rate GENERATOR_LEARNING_RATE
                        Generator learning rate.
  -g:iters GENERATOR_TRAIN_EVERY_N_ITERS, --generator-train-every-n-iters GENERATOR_TRAIN_EVERY_N_ITERS
                        Number of iteration after which the generator weights are updated.
  -g:s {plateau,threshold,cosine,multistep,None}, --generator-scheduler-type {plateau,threshold,cosine,multistep,None}
                        Type of learning rate scheduler. If None costant learning rate is used.
  -d:w DISCRIMINATOR_WEIGHTS, --discriminator-weights DISCRIMINATOR_WEIGHTS
                        Distribution or checkpoint to use to initialize generator weights.
  -d:lr DISCRIMINATOR_LEARNING_RATE, --discriminator-learning-rate DISCRIMINATOR_LEARNING_RATE
                        Discriminator learning rate.
  -d:iters DISCRIMINATOR_TRAIN_EVERY_N_ITERS, --discriminator-train-every-n-iters DISCRIMINATOR_TRAIN_EVERY_N_ITERS
                        Number of iteration after which the discriminator weights are updated.
  -d:s {plateau,threshold,cosine,multistep,None}, --discriminator-scheduler-type {plateau,threshold,cosine,multistep,None}
                        Type of learning rate scheduler. If None costant learning rate is used.
  -c:lr CYCLE_LEARNING_RATE, --cycle-learning-rate CYCLE_LEARNING_RATE
                        Cycle learning rate.
  -c:iters CYCLE_TRAIN_EVERY_N_ITERS, --cycle-train-every-n-iters CYCLE_TRAIN_EVERY_N_ITERS
                        Number of iteration after which the cycle weights are updated.
  -c:s {plateau,threshold,cosine,None}, --cycle-scheduler-type {plateau,threshold,cosine,None}
                        Type of learning rate scheduler. If None costant learning rate is used.
```
