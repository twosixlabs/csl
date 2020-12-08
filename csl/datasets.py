#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Module to load and manipulate original and synthetic versions of datasets.

Usage:

import csl.datasets as dset
# example 1. load the complete mnist dataset
# a. original mnist:
train_loader, test_loader = dset.load("mnist")

# b. synthetic mnist -- generated using dcgan:
train_loader, test_loader = dset.load("mnist_dcgan")

# example 2. train and test dataloaders containing all samples class_index=0
# from cifar10:
train_loader, test_loader = dset.get_dataloaders(
    "cifar10",
    batch_size=16,
    num_workers=4,
    class_idx=0,
    num_samples="all",
)

# example 3. train and test dataloaders containing samples with 50% original
# cifar10 and 50% synthetic samples generated using vae:
train_loader, test_loader = dset.get_hybrid_dataloaders(
    "cifar10",
    "cifar10_vae"
    batch_size=16,
    original_portion=0.5,
    synthetic_portion=0.5,
    num_workers=4,
)
"""
import os
import logging
import coloredlogs
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

import pdb
from torch.utils.data import Subset, ConcatDataset
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
coloredlogs.install(level="info", logger=log)

DATA_DIR = "~/Documents/twosix/datasets"
# DATA_DIR = "/persist/datasets"

# supported datasets
DATASETS = {
    "cifar10": CIFAR10,
    "fashion-mnist": FashionMNIST,
    "imagenet": ImageFolder,
    "imagenette": ImageFolder,
    "mnist": MNIST,
    "stl10": STL10,
    # synthetic datasets
    "cifar10_vae": ImageFolder,
    "cifar10_cvae": ImageFolder,
    "cifar10_dcgan": ImageFolder,
    "fashion-mnist_vae": ImageFolder,
    "mnist_vae": ImageFolder,
    "mnist_cvae": ImageFolder,
    # PENDING:
    # "fashion-mnist_cvae": ImageFolder,
    # "fashion-mnist_dcgan": ImageFolder,
    # imagenet_vae: ImageFolder,
    # imagenet_cvae: ImageFolder,
    # imagenet_dcgan: ImageFolder,
    # imagenette_vae: ImageFolder,
    # imagenette_cvae: ImageFolder,
    # imagenette_dcgan: ImageFolder,
    # "mnist_dcgan": ImageFolder,
}

# DATASETS AND DIMENSIONS TESTED
IMAGE_DIMS = {
    "imagenet": 32,  # 224
    "imagenette": 28,  # 224
    "mnist": 28,
    "fashion-mnist": 28,
    "cifar10": 28,
    # "stl10": 32,  # TODO
    "fashion-mnist_vae": 28,
}


def load(
    dataset_name: str, image_size: int = None, data_directory: os.PathLike = None
) -> torchvision.datasets:
    """Helper method for datasets.load
    run:
        train_ds, test_ds = load('mnist')
    """
    if image_size is None:
        # revert to dataset size
        image_size = IMAGE_DIMS[dataset_name]
    return Dataset.load(dataset_name, image_size, data_directory)


def get_dataloaders(
    dataset_name: str,
    image_size: int = None,
    batch_size: int = 16,
    num_workers: int = 2,
    class_idx: int = "all",
    num_samples: int = "all",
    train_transforms: torchvision.transforms = None,
    test_transforms: torchvision.transforms = None,
    data_directory: os.PathLike = None,
) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """Helper method for datasets.load
    run:
        train_ds, test_ds = load('mnist')
    """
    kwargs = {
        "dataset_name": dataset_name,
        "input_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "class_idx": class_idx,
        "num_samples": num_samples,
        "data_directory": data_directory,
    }
    return Dataset.create_dataloaders(**kwargs)


def get_hybrid_dataloaders(
    dataset_name: str,
    image_size: int = None,
    method: str = "vae",
    batch_size: int = 16,
    original_portion: float = 0.5,
    synthetic_portion: float = 0.5,
    num_workers: int = 2,
    data_directory: os.PathLike = None,
) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """Helper method for datasets.load
    run:
        train_ds, test_ds = load('mnist')
    """
    synthetic_name = f"{dataset_name}_{method}"
    kwargs = {
        "dataset_name": dataset_name,
        "input_size": image_size,
        "synthetic_dataset_name": synthetic_name,
        "batch_size": batch_size,
        "original_portion": original_portion,
        "synthetic_portion": synthetic_portion,
        "num_workers": num_workers,
        "data_directory": data_directory,
    }

    return Dataset.create_hybrid_dataloaders(**kwargs)


class Dataset(object):
    """Dataset class that can be used to instantiate built-in pytorch datasets
    or synthesized datasets and dataloaders.
    """

    def __init__(self):
        self.dataset_name = None
        self.synthetic_dataset_name = None

    @classmethod
    def load(
        self,
        dataset_name: str,
        input_size: int = None,
        train_transforms: torchvision.transforms = None,
        test_transforms: torchvision.transforms = None,
        image_folder_structure: bool = False,
        data_directory: os.PathLike = None,
    ) -> (torchvision.datasets, torchvision.datasets):
        """Conveninence method for loading datasets by name.
        Optional args: trainsformations for train and test
        """
        # import pdb; pdb.set_trace()
        self.dataset_name = dataset_name

        # check dataset structure (slightly different methods)
        if self.dataset_name not in DATASETS:
            raise NotImplementedError(
                f"Dataset '{dataset_name}' -- not implemented. "
                "It can be added if ImageFolder structure is followed."
            )
        else:
            dataset = DATASETS[self.dataset_name]

        if input_size is None:
            self.input_size = IMAGE_DIMS[dataset_name]
        else:
            self.input_size = input_size

        if data_directory is None:
            data_directory = DATA_DIR

        # single channel vs. rgb (3-channel) images
        normalize_strategy = (
            transforms.Normalize((0.5), (0.5))
            if "mnist" in dataset_name
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )

        # DFAULT DATA TRANSFORMATIONS:
        if train_transforms is None:
            train_transforms = transforms.Compose(
                [
                    transforms.Resize(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize_strategy,  # rgb vs. bw
                ]
            )
        if test_transforms is None:
            test_transforms = transforms.Compose(
                [
                    transforms.Resize(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    normalize_strategy,  # rgb vs. bw
                ]
            )

        # Check for the special "ImageFolder" structure
        if (
            image_folder_structure
            or "imagenet" in dataset_name
            or "vae" in dataset_name
            or "gan" in dataset_name
        ):
            log.info(f"Processing '{dataset_name}' ImageFolder structure")

            train_dataset = ImageFolder(
                root=f"{data_directory}/{dataset_name}/train/",
                transform=train_transforms,
            )
            test_dataset = ImageFolder(
                root=f"{data_directory}/{dataset_name}/val/", transform=test_transforms,
            )

        else:
            log.info(f"Processing '{dataset_name}' torch.vision built-in structure")
            train_dataset = dataset(
                root=f"{data_directory}/{dataset_name}/",
                download=True,
                train=True,
                transform=train_transforms,
            )
            test_dataset = dataset(
                root=f"{data_directory}/{dataset_name}/",
                download=True,
                train=False,
                transform=test_transforms,
            )
        return train_dataset, test_dataset

    @classmethod
    def create_dataloaders(
        self,
        dataset_name: str,
        input_size: int,
        batch_size: int,
        num_workers: int = 0,
        class_idx: int = None,
        num_samples: int = None,
        train_transforms: torchvision.transforms = None,
        test_transforms: torchvision.transforms = None,
        train_dataset: torchvision.datasets = None,
        test_dataset: torchvision.datasets = None,
        data_directory: os.PathLike = None,
    ) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        """Convenience method for a customized and subsampled dataloading.
        Dataloaders for the complete datasets or for a specific class index.
        E.g., To get all 0-digit samples from mnist dataset  use
            class_idx = 0
            num_samples = None
        Options: pass your own dataset and your own image transformations
        (must adhere to PyTorch dataset standards -- not currently
         checking for those)
        A list of Pytorch supported datasets:
            https://pytorch.org/docs/stable/torchvision/datasets.html
        """
        # pdb.set_trace()

        # which dataset
        self.dataset_name = dataset_name
        # dataset image dimensions
        if input_size is None:
            self.input_size = IMAGE_DIMS[dataset_name]
        else:
            self.input_size = input_size

        # Datasets: train and test
        if (train_dataset is None) and (test_dataset is None):
            train_dataset, test_dataset = self.load(
                self.dataset_name,
                self.input_size,
                train_transforms=train_transforms,
                test_transforms=test_transforms,
                data_directory=data_directory,
            )

        # how many samples
        if (num_samples is None) or (num_samples == "all"):
            num_samples = "all"

        # which classes
        if (class_idx is None) or (class_idx == "all"):
            train_sampler, test_sampler = None, None
        else:
            log.info(
                f" > Extracting {num_samples} (max) samples for {class_idx} class(es)."
            )
            train_sampler = self._build_sampler(train_dataset, class_idx, num_samples)
            test_sampler = self._build_sampler(test_dataset, class_idx, num_samples)

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

    @classmethod
    def create_hybrid_dataloaders(
        self,
        dataset_name: str,
        synthetic_dataset_name: str,
        batch_size: int,
        original_portion: float,
        synthetic_portion: float,
        seed: int = None,
        num_workers: int = 2,
        input_size: int = None,
        data_directory: os.PathLike = None,
    ) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
        """Convenience method. Combines datasets (original and synthetic).
        mixes the datasets (i.e., hybrid) and returns a hybrid
        training_dataloader and an original test_dataloader.
        """
        self.dataset_name = dataset_name
        self.synthetic_dataset_name = synthetic_dataset_name

        # dataset image dimensions
        if input_size is None:
            self.input_size = IMAGE_DIMS[dataset_name]
        else:
            self.input_size = input_size

        if "mnist" in dataset_name:  # grayscale
            n_channels = 1
        else:  # color
            n_channels = 3

        if n_channels == 1:
            data_transforms = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                ]
            )
        elif n_channels == 3:
            data_transforms = transforms.Compose(
                [
                    transforms.Resize(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            raise NotImplementedError(
                f"The provided n_channels ({n_channels}) is not supported."
            )
        train_len = 0
        datasets = []
        # ORIGINAL DATASET
        # step 1. get the original train dataset and keep 'split' portion of it.
        train_original, test_dataset = self.load(
            self.dataset_name,
            input_size=self.input_size,
            train_transforms=data_transforms,
            test_transforms=data_transforms,
            data_directory=data_directory,
        )
        log.info(
            f"Loaded complete original '{self.dataset_name}' dataset with "
            f"{len(train_original)} training data and "
            f"{len(test_dataset)} test data."
        )

        if original_portion == 1.0:
            log.info(
                f" > No-Split synthetic '{self.dataset_name}' dataset with "
                f"{len(train_original)} training data and "
                f"{len(test_dataset)} test data."
            )

        elif (original_portion > 0.0) and (original_portion < 1):
            train_original, _ = self._split_dataset(
                train_original, split=original_portion, seed=seed,
            )
            log.info(
                f" > Splitted original '{self.dataset_name}' dataset with "
                f"{len(train_original)} training data and "
                f"{len(test_dataset)} test data."
            )
        elif original_portion <= 0.0:
            train_original = None

        else:
            raise NotImplementedError(
                f"Portion Selection '{original_portion}' not supported."
            )

        if train_original is not None:
            datasets.append(train_original)
            train_len += len(train_original)

        # SYNTHETIC DATASET
        if synthetic_portion > 0.0:
            train_synthetic, test_synthetic = self.load(
                synthetic_dataset_name,
                input_size=self.input_size,
                train_transforms=data_transforms,
                test_transforms=data_transforms,
                data_directory=data_directory,
            )
            log.info(
                f" Loaded complete synthetic '{synthetic_dataset_name}' "
                f" dataset with {len(train_synthetic)} training data and "
                f"{len(test_synthetic)} test data."
            )

        if synthetic_portion == 1.0:
            log.info(
                f" > No-Split synthetic '{synthetic_dataset_name}' "
                f"dataset with {len(train_synthetic)} training data and "
                f"{len(test_synthetic)} test data."
            )
        elif (synthetic_portion > 0.0) and (synthetic_portion < 1.0):
            # step 2. get the synthetic train dataset and dump some of it
            train_synthetic, _ = self._split_dataset(
                train_synthetic, split=synthetic_portion, seed=seed,
            )

            log.info(
                f" > Splitted synthetic '{synthetic_dataset_name}' "
                f"dataset with {len(train_synthetic)} training data and "
                f"{len(test_synthetic)} test data."
            )

        elif synthetic_portion <= 0.0:
            train_synthetic = None

        else:
            raise NotImplementedError(
                f"Portion Selection '{synthetic_portion}' not supported."
            )

        if train_synthetic is not None:
            datasets.append(train_synthetic)
            train_len += len(train_synthetic)

        # step 3. combine the two parts into a complete dataset
        if len(datasets) == 1.0:
            train_dataset = datasets[0]
        elif len(datasets) > 1.0:
            train_dataset = ConcatDataset(datasets)
        else:
            raise ValueError(
                "No data/requested found (hint: look at requested "
                "synthetic and/or original portions!)"
            )

        # test_dataset = ConcatDataset([mnist_test_original, mnist_test_synthetic])
        log.info(
            f"Constructed a hybrid '{self.dataset_name}' dataset with "
            f"{original_portion:.2f}-original and "
            f"{synthetic_portion:.2f}-hybrid portions "
            f"to produce {train_len} training and "
            f"{len(test_dataset)} test datasets."
        )

        # setp 4. Create the dataloader
        train_loader, test_loader = self.create_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            input_size=input_size,
            num_workers=num_workers,
            # class_idx=None,
            # num_samples=None,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            # test_dataset = mnist_test_synthetic_p1,
        )

        return train_loader, test_loader

    def _build_sampler(dataset, class_idx, num_samples: int = "all"):
        """Helper. Enables dataloader with a given by class_idx
        uses the buit-in weighted random sampler, and returns a sampler object
        with num_samples.
        inputs:
            dataset: torchvision dataset, object - preloaded dataset.
            class_idx: int, index representative of the desired class.
            num_samples: int, number of samples in dataloder.
        output:
            sampler: torch weighted random sampler object.
        """
        # pdb.set_trace()
        # unique labels and their frequency counts
        unique_labels, label_counts = torch.unique(
            torch.as_tensor(dataset.targets), return_counts=True
        )
        n_samples = label_counts[class_idx].item()
        label = unique_labels[class_idx]
        label_weight = 1.0 / n_samples

        # n_classes = len(dataset.classes)

        if hasattr(dataset, "targets"):
            sample_weights = [
                label_weight if val == label else 0 for val in dataset.targets
            ]
        elif hasattr(dataset, "labels"):
            sample_weights = [
                label_weight if val == label else 0 for val in dataset.labels
            ]

        # NOTE: num_samples <= label_counts[class_idx] -- MUST
        num_samples = label_counts[class_idx] if num_samples == "all" else num_samples
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=int(num_samples), replacement=True,
        )
        return sampler

    def _split_dataset(
        dataset: torchvision.datasets, split: float = 0.5, seed: int = None
    ):
        """Helper. Split dataset
        """
        idxs_1, idxs_2 = train_test_split(
            list(range(len(dataset))), train_size=split, random_state=seed
        )
        dataset_part_1 = Subset(dataset, idxs_1)
        dataset_part_2 = Subset(dataset, idxs_2)
        return dataset_part_1, dataset_part_2


def simple_test(name: str):
    train_data, test_data = Dataset.load(name)
    log.info(
        f"[TEST PASSED] loaded '{name}' dataset (train and val sets) with "
        f"'label-symbol: index' map: \n\n{train_data.class_to_idx}."
    )


if __name__ == "__main__":
    name = "mnist"
    simple_test(name)
