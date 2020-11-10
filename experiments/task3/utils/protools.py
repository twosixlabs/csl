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
import pickle

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
from torch.utils.data import Subset, ConcatDataset

from sklearn.model_selection import train_test_split


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
    # synthetic datasets
    "mnist_vae": ImageFolder,
    "mnist_cvae": ImageFolder,
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
    label_weight = 1.0 / n_samples

    # n_classes = len(dataset.classes)

    if hasattr(dataset, "targets"):
        sample_weights = [label_weight if val==label else 0 for val in dataset.targets]
    elif hasattr(dataset, "labels"):
        sample_weights = [label_weight if val==label else 0 for val in dataset.labels]


    # NOTE: num_samples <= label_counts[class_idx] -- MUST
    num_samples = label_counts[class_idx] if (num_samples is None) else num_samples
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=int(num_samples),
        replacement=True,
    )
    return sampler


def split_dataset(dataset: torchvision.datasets, split: float = 0.5, seed: int = None):
    """ Helper. Split dataset
    """
    idxs_1, idxs_2 = train_test_split(list(range(len(dataset))), train_size=split, random_state=seed)
    dataset_part_1 = Subset(dataset, idxs_1)
    dataset_part_2 = Subset(dataset, idxs_2)
    return dataset_part_1, dataset_part_2


def fetch_datasets(dataset_name: str,
                   input_size: int,
                   train_transforms: torchvision.transforms=None,
                   test_transforms: torchvision.transforms=None
) -> (torchvision.datasets, torchvision.datasets):

    # check dataset structure (slightly different methods)
    if dataset_name not in DATASETS:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' -- not implemented"
        )
    else:
        dataset = DATASETS[dataset_name]


    # DFAULT DATA TRANSFORMATIONS:
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ])

    if "imagenet" in dataset_name or "vae" in dataset_name:
        log.info(f"Processing '{dataset_name}' ImageFolder structure")

        train_dataset = ImageFolder(
            root=f"{ROOT_DIR}/{dataset_name}/train/",
            transform=train_transforms,
        )
        test_dataset = ImageFolder(
            root=f"{ROOT_DIR}/{dataset_name}/val/",
            transform=test_transforms,
        )

    else:
        log.info(f"Processing '{dataset_name}' torch.vision built-in structure")
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
    return train_dataset, test_dataset


def create_dataloaders(dataset_name: str, batch_size: int,
                       input_size: int = 32, num_workers: int = 0,
                       class_idx: int = None, num_samples: int = None,
                       train_transforms: torchvision.transforms = None,
                       test_transforms: torchvision.transforms=None,
                       train_dataset: torchvision.datasets=None,
                       test_dataset: torchvision.datasets=None,

):
    """
    A list of Pytorch supported datasets:
        https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    train_sampler, test_sampler = None, None


    if (train_dataset is None) and (test_dataset is None):
        # Fetch the datasets: train and test
        train_dataset, test_dataset = fetch_datasets(
            dataset_name,
            input_size,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
        )

    if class_idx is not None:
        aux1 = "all" if num_samples is None else num_samples
        log.info(f" > Extracting {aux1} (max) samples for {class_idx} class.")
        train_sampler = build_sampler(train_dataset, class_idx)
        test_sampler = build_sampler(test_dataset, class_idx)

    shuffle_train_loader = True if train_sampler is None else None

    # DATA LOADERS
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train_loader,
        num_workers=num_workers,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
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


def save_model(model, dataset_name: str, model_name: str, model_dir=None, method_name='tuned'):
    """
    save a pytorch model to local directory
    save_model(model, 'coco', 'subclass','resnet152', model_dir="../models/", method_name='trained')
    """
    if model_dir is None:
        model_file_name = f'{model_name}_{dataset_name}_{method_name}_model.pt'
    else:
        model_file_name = f'{model_dir}{model_name}_{dataset_name}_{method_name}_model.pt'
    torch.save(model, model_file_name)
    log.info(f"Saved {model_name} to \'{model_dir}\' directory")

    return


def get_last_checkpoint(model_name, checkpoint_dir: os.PathLike=f'../model_files/checkpoints/'):
    "Check for existing checkpoints and load the most recent/latest"
    checkpoint_path = f"{checkpoint_dir}/"

    # see if checkpoints exist
    import glob
    from natsort import natsorted, ns

    last_checkpoint_name = None
    checkpoints = [f for f in glob.glob(f"{checkpoint_dir}/{model_name}*.pt")]

    if len(checkpoints) > 0:
        sorted_checkpoints = natsorted(checkpoints, alg=ns.IGNORECASE)  # or alg=ns.IC
        last_checkpoint_name = sorted_checkpoints[-1]
        checkpoint_model = torch.load(last_checkpoint_name)
        val_acc_history_name = last_checkpoint_name.replace('.pt', '_hist.pkl')
        with open(val_acc_history_name, 'rb') as pfile:
            val_acc_history = pickle.load(pfile)
    else:
        log.info("No previous checkpoint")
        checkpoint_model = None
        val_acc_history = None

    return checkpoint_model, val_acc_history, last_checkpoint_name


def save_checkpoint(model, val_acc_history: dict, checkpoint_name: str,
                    checkpoint_dir='model_files/checkpoints/'):
    """
    save a pytorch model to local directory
    save_checkpoint(model, model_name='inception_imagenet-small-subclass_temp_epoch-x'
    check_point_dir="../checkpoints")
    """
    confirm_directory(checkpoint_dir)

    if not checkpoint_name.endswith('.pt'):
        checkpoint_name = f"{checkpoint_name}.pt"

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(model, checkpoint_path)

    val_acc_history_name = checkpoint_path.replace(".pt","_hist.pkl")
    with open(val_acc_history_name, 'wb') as pfile:
        pickle.dump(val_acc_history, pfile)

    log.info(f"Saved {checkpoint_name}  and train_val_history to \'{checkpoint_path}\' directory")

    return


def load_checkpoint(checkpoint_name:str, checkpoint_dir: str = 'model_files/checkpoints/'):
    """
    save a pytorch model to local directory
    save_checkpoint(model, model_name='inception_imagenet-small-subclass_temp_epoch-x'
    check_point_dir="../checkpoints")
    """
    if not checkpoint_name.endswith('.pt'):
        checkpoint_name = f"{checkpoint_name}.pt"

    if checkpoint_dir not in checkpoint_name:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    else:
        checkpoint_path = checkpoint_dir
    # check and load
    if os.path.isfile(checkpoint_path):
        model = torch.load(checkpoint_path)
        log.info(
            f"Loaded checkpoint {checkpoint_dir.split('/')[-1]} from "
            f"'{checkpoint_dir}' directory"
        )
        return model
    else:
        log.error(f"Unable to find the checkpoint file from: \n {checkpoint_path}")
        model = None

    return model


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
    print(f"Batch shape: {batch.shape}")  # ([batch_size, L, H, W])

    # display all batches using a grid
    for b, (batch, labels) in enumerate(test_loader):
        plot_title = f"{b}-th batch from {dataset_name} dataset"
        visualize_batch(batch, plot_title=plot_title, grid_dims=8)


def run_subset_sample(dataset_name: str = "mnist", class_idx: int = 0):
    """ Demonstration. Basic use.
    Data loaders with all samples from a selected class (by class_idx).
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


def run_hybrid_dataloader_construction(hybrid_portion: float = .5):
    """Demonstration. Basic use.
    Data loaders with 50% original mnist and 50% synthetic mnist. It uses
    all classes and 'portion' of samples.
    """
    # create a hybrid dataset: half original + half synthetic
    seed = 42
    image_dim = 28
    batch_size = 64
    num_workers = 4
    class_idx = None
    original_dataset_name = "mnist"
    synthetic_dataset_name = "mnist_vae"

    # step 1. get the original train dataset and dump half of it
    mnist_train_original, mnist_test_original = fetch_datasets(original_dataset_name, input_size=image_dim)
    mnist_train_original, _ = split_dataset(mnist_train_original, split=.5, seed=seed)
    mnist_test_original_p1, _ = split_dataset(mnist_test_original, split=.5, seed=seed)

    # step 2. get the synthetic train dataset and dump half of it
    data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_dim),
            transforms.CenterCrop(image_dim),
            transforms.ToTensor(),
    ])

    mnist_train_synthetic, mnist_test_synthetic = fetch_datasets(
        synthetic_dataset_name,
        input_size=image_dim,
        train_transforms=data_transforms,
        test_transforms=data_transforms,
    )
    mnist_train_synthetic, _ =  split_dataset(mnist_train_synthetic, split=.25, seed=seed)
    mnist_test_synthetic_p1, _ =  split_dataset(mnist_test_synthetic, split=.5, seed=seed)

    # step 3. combine the two parts into a complete dataset
    train_dataset = ConcatDataset([mnist_train_original, mnist_train_synthetic])
    test_dataset = ConcatDataset([mnist_test_original_p1, mnist_test_synthetic_p1])

    # setp 4. Create the dataloader
    train_loader, test_loader = create_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        input_size=image_dim,
        num_workers=num_workers,
        class_idx=class_idx,
        num_samples=None,  # int
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        # test_dataset = mnist_test_synthetic_p1,
    )

    # step 5. Display all batches using a grid
    for b, (batch, labels) in enumerate(train_loader):
        plot_title = f"{b}-th batch from {dataset_name} dataset ({class_idx} index)"
        visualize_batch(batch, plot_title=plot_title, grid_dims=8)


if __name__ == "__main__":
    dataset_name = "mnist_vae"
    class_idx = 3
    # run_basic_sample(dataset_name)
    # run_subset_sample(dataset_name, class_idx)
    # run_hybrid_dataloader_construction
    seed = 42
    image_dim = 28
    batch_size = 64
    num_workers = 4
    class_idx = None
    original_dataset_name = "mnist"
    synthetic_dataset_name = "mnist_cvae"

    # step 1. get the original train dataset and dump half of it
    mnist_train_original, mnist_test_original = fetch_datasets(
        original_dataset_name,
        input_size=image_dim,
    )
    mnist_train_original, _ = split_dataset(mnist_train_original, split=.5, seed=seed)
    mnist_test_original_p1, _ = split_dataset(mnist_test_original, split=.5, seed=seed)

    # step 2. get the synthetic train dataset and dump half of it
    data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_dim),
            transforms.CenterCrop(image_dim),
            transforms.ToTensor(),
    ])

    mnist_train_synthetic, mnist_test_synthetic = fetch_datasets(
        synthetic_dataset_name,
        input_size=image_dim,
        train_transforms=data_transforms,
        test_transforms=data_transforms,
    )
    mnist_train_synthetic, _ = split_dataset(mnist_train_synthetic, split=.25, seed=seed)
    mnist_test_synthetic_p1, _ = split_dataset(mnist_test_synthetic, split=.5, seed=seed)

    # step 3. combine the two parts into a complete dataset
    train_dataset = ConcatDataset([mnist_train_original, mnist_train_synthetic])
    test_dataset = ConcatDataset([mnist_test_original_p1, mnist_test_synthetic_p1])

    # setp 4. Create the dataloader
    train_loader, test_loader = create_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        input_size=image_dim,
        num_workers=num_workers,
        class_idx=class_idx,
        num_samples=None,  # int
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        # test_dataset = mnist_test_synthetic_p1,
    )

    # step 5. Display all batches using a grid
    for b, (batch, labels) in enumerate(test_loader):
        plot_title = f"{b}-th batch from {dataset_name} dataset ({class_idx} index)"
        visualize_batch(batch, plot_title=plot_title, grid_dims=8)

