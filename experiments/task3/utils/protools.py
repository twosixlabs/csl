#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protools.py prototyping tools and helper methods.

ImageFolder datasets:
    + imagenette: https://github.com/fastai/imagenette
    + imagenet

@author: carlos.torres <carlos.torres@twosixlabs.com>
"""
import os
import logging
import coloredlogs
import numpy as np

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from torchvision.datasets import (
    MNIST,
    FashionMNIST,
    CIFAR10,
    STL10,
    ImageFolder,
)

log = logging.getLogger("protools")
coloredlogs.install(level="warning", logger=log)
# ROOT_DIR = "/persist/datasets"
ROOT_DIR = "~/Documents/twosix/datasets"

DATASETS = {
    "mnist": MNIST,
    "fashion-mnist": FashionMNIST,
    "cifar10": CIFAR10,
    "stl10": STL10,
    "imagenet": ImageFolder,
    "imagenette": ImageFolder,
}


def build_sampler(dataset, class_idx, num_samples: int = None):
    """DATA LOADER HELPER. Enables dataloader with  given by class_idx
    uses the buit-in weighted random sampler, and returns a sampler object
    with num_samples.
    inputs:
        dataset: torchvision dataset, object - preloaded dataset.
        class_idx: int, index representative of the desired class.
        num_samples: int, number of samples in dataloder.
    output:
        sampler: torch weighted random sampler object.
    """
    # unique labels and their frequency counts
    unique_labels, label_counts = torch.unique(torch.as_tensor(dataset.targets), return_counts=True)
    n_samples = label_counts[class_idx].item()
    label = unique_labels[class_idx]
    label_weight = 1.0/n_samples

    # n_classes = len(dataset.classes)

    if hasattr(dataset, "targets"):
        sample_weights = [label_weight if val==label else 0 for val in dataset.targets]
    elif hasattr(dataset, "labels"):
        sample_weights = [label_weight if val==label else 0 for val in dataset.labels]


    # NOTE: num_samples <= label_counts[class_idx] -- MUST
    num_samples = label_counts[class_idx] if (num_samples is None) else num_samples
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=int(num_samples),  #
        replacement=True,
    )
    return sampler


def create_dataloaders(dataset_name: str, batch_size: int, input_size: int = 32, num_workers: int = 0, class_idx: int = None, num_samples: int = None):
    """
    A list of Pytorch supported datasets:
        https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    train_sampler, test_sampler = None, None

    # DFAULT DATA TRANSFORMATIONS:
    train_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        # # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor(),
    ])

    if dataset_name not in DATASETS:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' -- not implemented"
        )

    # DATASET HANDLER
    dataset = DATASETS[dataset_name]

    # check dataset structure (slightly different methods)
    if "imagenet" in dataset_name:
        print(f"Processing '{dataset_name}' ImageFolder structure")

        train_dataset = ImageFolder(
            root=f"{ROOT_DIR}/{dataset_name}/train/",
            transform=train_transforms,
        )
        test_dataset = ImageFolder(
            root=f"{ROOT_DIR}/{dataset_name}/val/",
            transform=test_transforms,
        )

    else:
        print(f"Processing '{dataset_name}' torch.vision built-in structure")
        train_dataset = dataset(
            root=f"{ROOT_DIR}/{dataset_name.upper()}_data/",
            download=True,
            train=True,
            transform=train_transforms,
        )
        test_dataset = dataset(
            root=f"{ROOT_DIR}/{dataset_name.upper()}_data/",
            download=True,
            train=False,
            transform=test_transforms,
        )

    if class_idx is not None:
        aux1 = "all" if num_samples is None else num_samples
        print(f" > Extracting {aux1} (max) samples for {class_idx} class.")
        train_sampler = build_sampler(train_dataset, class_idx)
        test_sampler = build_sampler(test_dataset, class_idx)

    # DATA LOADERS
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=num_workers,
        sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        # shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler,
    )

    return train_loader, test_loader


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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def visualize_batch(batch, plot_title: str = None, grid_dims: int = 4):
    batch = torchvision.utils.make_grid(batch).numpy()
    batch = np.transpose(batch, (1, 2, 0))
    plt.figure(figsize=(grid_dims, grid_dims))
    plt.imshow(batch, cmap='Greys_r')
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
        log.info(f"The directory '{directory}' exists. Ready to store data.")


def run_basic_sample(dataset_name: str = "cifar10"):
    """ Demonstration. Basic use.
    Data loaders with all classes all avail samples from each class.
    """
    image_dim = 32
    batch_size = 64
    num_workers = 2
    train_loader, test_loader = create_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        class_idx=None,
        input_size=image_dim,
    )
    # simple verification
    batch, labels = iter(test_loader).next()
    print(batch.shape)  # ([batch_size, L, H, W])

    # display all batches using a grid
    for b, (batch, labels) in enumerate(test_loader):
        plot_title = f"{b}-th batch from {dataset_name} dataset"
        visualize_batch(batch, plot_title=plot_title, grid_dims=8)


def run_subset_sample(dataset_name: str = "mnist", class_idx: int = 0):
    """ Demonstration. Basic use.
    Data loades with all samples from a selected class (by class_idx).
    """
    image_dim = 32
    batch_size = 64
    num_workers = 2

    train_loader, test_loader = create_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        input_size=image_dim,
        num_workers=num_workers,
        class_idx=class_idx,
        num_samples=None,  # int
    )
    batch_data, batch_labels = iter(train_loader).next()

    # display all batches using a grid
    for b, (batch, labels) in enumerate(test_loader):
        plot_title = f"{b}-th batch from {dataset_name} dataset ({class_idx} index)"
        visualize_batch(batch, plot_title=plot_title, grid_dims=8)


if __name__ == "__main__":
    dataset_name = "imagenette"
    class_idx = 3
    run_basic_sample(dataset_name)
    run_subset_sample(dataset_name, class_idx)
