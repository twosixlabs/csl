#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.utils.py
Standard helper methods
"""
import os
import logging
import coloredlogs
import torchvision

import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger("utils")
coloredlogs.install(level="info", logger=log)


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_xticklabels("")
    ax.set_yticklabels("")

    return ax


def visualize_batch(batch, plot_title: str = None, grid_dims: int = 4):
    batch = torchvision.utils.make_grid(batch).numpy()
    batch = np.transpose(batch, (1, 2, 0))
    plt.figure(figsize=(grid_dims, grid_dims))
    plt.imshow(batch, cmap="Greys_r")
    if plot_title is not None:
        title = plot_title
    else:
        title = "Batch visualization"
    plt.title(title)
    plt.show()


def confirm_directory(directory: os.PathLike):
    if not os.path.isdir(directory):
        log.info(f"The given directory '{directory}' does not exist. Creating it!")
        os.makedirs(directory)
    else:
        log.info(f"The directory '{directory}' exists. Ready to use.")
