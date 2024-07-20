# UNet Architecture

This project contains my code walking through the UNet paper and coding demo found here:
[U-Net Coding Demo](https://nn.labml.ai/unet/index.html)

Link to the paper can be found here: 
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597v1)

This paper builds on top of the ResNet, a convolutional neural network that employs a skip connection to ensure the gradient doesn't vanish and the data is maintained through training. 

The U-Net guesses image masks while also utilizing skip connections to ensure the original image is not lost. The most applicable feature is the ability to track certain features in medical scans depending on what the user has trained it on.

## Overview

UNet is a convolutional neural network architecture designed for biomedical image segmentation. This repository contains an implementation of the UNet model, along with a detailed walkthrough based on the UNet paper and an accompanying coding demo.

## Contents

- `U-Net.py`: The main UNet model implementation.
- `main.py`: Script for training the UNet model on a dataset as well as loading the dataset.
- `utils.py`: Helper functions used throughout the project.
- `README.md`: This file, providing an overview of the project and its contents.

