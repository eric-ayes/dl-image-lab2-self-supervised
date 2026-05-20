# Lab 2: Rotation-Based Self-Supervised Learning

Master's course project — Deep Learning for Image Processing (UAM)  
Authors: Eric Ayestaran Guillorme · Unax Murua Urizarbarrena

## Overview

This lab studies the effect of rotation-based self-supervised pretraining on downstream image classification. A ResNet-14 (and ResNet-18 in extended experiments) is first trained to predict image rotations (0°, 90°, 180°, 270°) as a pretext task, and then fine-tuned for supervised classification. The goal is to analyse how self-supervised initialization affects convergence speed, final accuracy, and feature representation quality compared to training from scratch.

## Contents

| Notebook | Description |
|----------|-------------|
| `lab2-imagen.ipynb` | Main notebook: full pipeline |
| `lab2-imagen (1-7).ipynb` | Experimental variants (dataset size, learning rate, architecture, augmentation, combined config) |

## Experimental Configurations

- **Baseline** — ResNet-14, 45 epochs, 15×480 dataset
- **Full dataset** — 50×480 classes to study the effect of data diversity
- **Learning rate variation** — lr=0.01 vs default
- **Model depth** — ResNet-14 vs ResNet-18
- **Data augmentation** — padding, random cropping, horizontal flip, ColorJitter
- **Combined** — ResNet-18 + lr=0.005 + weight decay 5e-4 + full augmentation

## Key Results

### Pretext Task (Rotation Prediction)
- Baseline (ResNet-14, 15×480): **70.8% val accuracy**
- Full dataset (50×480): **78.8% val accuracy** — more data → less overfitting
- Data augmentation has the strongest impact on generalization; architecture depth alone is insufficient

### Classification Task (Scratch vs. Self-Supervised Init)

| Config | Scratch | Self-Supervised |
|--------|---------|-----------------|
| Baseline (15×480) | 65.6% | **73.1%** |
| Full dataset (50×480) | 65.1% | **70.4%** |
| ResNet-14 + Augmentation | 68.7% | **76.8%** |
| ResNet-18 + Augmentation | 67.6% | **69.6%** |
| Combined (best config) | 65.0% | **71.2%** |

Self-supervised initialization consistently yields faster convergence and smoother validation curves across all configurations.

### t-SNE Analysis
Fine-tuned models show tighter, better-separated feature clusters than scratch-trained models, confirming that rotation pretraining captures transferable visual structure even without labels.

## Main Conclusions

- Self-supervised pretraining is most beneficial when labelled data are limited
- Data diversity and augmentation matter more than model depth
- Higher pretext accuracy does not guarantee better downstream performance — representation quality is what transfers

## Tech Stack

- Python · PyTorch · torchvision
- scikit-learn (t-SNE)
- matplotlib · Google Colab

## Author

Eric Ayestaran — MSc Deep Learning in Audio, Video and Image Signal Processing, UAM
