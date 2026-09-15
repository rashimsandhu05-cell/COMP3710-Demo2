# COMP3710 Lab Demonstration 2

This repository contains my work for Lab Demonstration 2 for **COMP3710 Pattern Analysis**.

The project covers DFT, Eigenfaces, CNNs, DAWNBench and the three OASIS recognition tasks (VAE, U-Net and GAN).

## Project Structure

### Notebooks

- `DFT.ipynb` — Discrete Fourier Transform
- `EigenFaces.ipynb` — Eigenfaces / PCA
- `CNN.ipynb` — Convolutional Neural Network
- `DAWNBench.ipynb` — DAWNBench CIFAR-10 experiment

### DAWNBench

- `dawnbench_train.py` — main training implementation
- `dawnbench_job.sh` — Slurm training job
- `dawnbench_demo.py` — one-epoch demonstration run
- `dawnbench_demo_job.sh` — Slurm job for the demonstration run

The DAWNBench implementation uses a custom CIFAR-style ResNet-18 with data augmentation, SGD with momentum, cosine learning-rate scheduling and mixed-precision training.

Recorded DAWNBench results:

| Configuration | Accuracy | Training time | Hardware |
|---|---:|---:|---|
| Higher-accuracy configuration | **95.21%** | **394.44 s** | NVIDIA A100 |
| Faster configuration | **94.56%** | **370.82 s** | NVIDIA A100 |

The 95.21% run achieved the highest recorded accuracy, while the 94.56% run provided a faster training time. Both results were obtained on an NVIDIA A100 GPU.

## OASIS Recognition Tasks

### Task 1 — VAE

The VAE was trained using the OASIS brain MRI training data.

Configuration:

- Image size: 64 × 64
- Latent dimension: 2
- Batch size: 128
- Epochs: 30
- Optimiser: Adam
- Learning rate: 0.001

The trained model was used to generate reconstructions and samples from the 2D latent space.

The final training used 30 epochs and 9664 training images. The training loss decreased from approximately 1276.45 to 1047.85.

Files:

- `oasis_vae.py`
- `oasis_vae_job.sh`
- `oasis_vae_visualize.py`

Results are available in the `results/` folder.

### Task 2 — U-Net

A U-Net model was trained for multi-class segmentation of the OASIS MRI images.

The segmentation contains four classes:

- Background
- Label 1
- Label 2
- Label 3

Configuration:

- Image size: 128 × 128
- Batch size: 32
- Epochs: 40
- Optimiser: Adam
- Learning rate: 0.001
- Output classes: 4

Best validation result:

| Class | DSC |
|---|---:|
| Background | 0.9996 |
| Label 1 | 0.9404 |
| Label 2 | 0.9528 |
| Label 3 | 0.9724 |
| **Mean DSC** | **0.9663** |

Files:

- `oasis_unet.py`
- `oasis_unet_job.sh`
- `oasis_unet_visualize.py`

The test prediction visualisation is included in:

`results/unet_test_predictions.png`

### Task 3 — GAN

A DCGAN-style model was trained to generate OASIS brain MRI images.

Configuration:

- Image size: 64 × 64
- Latent dimension: 100
- Batch size: 128
- Epochs: 50
- Optimiser: Adam
- Learning rate: 0.0002
- Hinge GAN loss

The generator produced brain MRI-like images with variation between samples. Samples were saved throughout training to check the progression of the model and look for mode collapse.

The GAN training took approximately 210.34 seconds.

Files:

- `oasis_gan.py`
- `oasis_gan_job.sh`
- `gan_progression.py`

Results:

- `results/gan_final_samples.png`
- `results/gan_training_progression.png`

## Results

The `results/` directory contains selected visual evidence from the OASIS tasks:

- `vae_reconstructions.png` — VAE reconstruction results
- `vae_latent_space.png` — VAE latent-space visualisation
- `vae_latent_samples.png` — samples generated from the 2D latent space
- `unet_test_predictions.png` — U-Net segmentation predictions
- `gan_final_samples.png` — final GAN-generated samples
- `gan_training_progression.png` — GAN training progression

## Running on Rangpur

The OASIS training jobs were run on the UQ Rangpur HPC system using Slurm and an NVIDIA A100 GPU.

The Slurm scripts in this repository contain the configurations used for the training runs.

## GitHub Development

The project was developed using Git with separate commits for the main stages of development, including:

- notebooks
- DAWNBench scripts
- OASIS training scripts
- OASIS model implementations
- visualisation scripts
- result evidence
- documentation

The dataset itself is excluded from the repository using `.gitignore`.

## AI Use

AI tools were used as a support tool during development for understanding concepts, code suggestions, debugging, Rangpur/Slurm assistance and documentation.

More details are provided in:

- `AI_DECLARATION.md`
- `PROMPT_HISTORY_AND_DEVELOPMENT_RECORD.txt`

The code and results were tested by me and I am responsible for the submitted work.