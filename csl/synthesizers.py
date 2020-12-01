#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run:
# mimic the mnist dataset by training a vae for each class (all) using 10 epochs
    python3 synthesizers.py \
        --method vae \
        --dataset_name mnist \
        --num_samples all \
        --class_index all \
        --num_epochs 10 \
        --task mimic \
"""
# standard
import os
import logging
import coloredlogs
import argparse

# torch
import torch
from torchvision import transforms

# custom
from data_generators.vae import vaes
from data_generators.gan import gans
import datasets as dsets
from utils.utils import confirm_directory

# storage
MODELS_DIR = "~/Documents/twosix/models"
DATA_DIR = "~/Documents/twosix/datasets"

# supported generative methods
METHODS = {
    "vae": vaes.VAE,
    "cvae": vaes.CVAE,
    "dcgan": gans.DCGAN,
}

# datasets
DATASETS = dsets.DATASETS
IMAGE_DIMS = dsets.IMAGE_DIMS

log = logging.getLogger(__name__)
coloredlogs.install(level="info", logger=log)


def parse_args():
    """
    Parse CLI arguments for model training
    """
    parser = argparse.ArgumentParser(description="Run CLS image Synthesizer")
    parser.add_argument(
        "--method",
        help=(
            "Name of the method used to generate synthetic images "
            "(options: vae, cvae, dcgan) --required"
        ),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        help=(
            "str, name of the original dataset used to train the "
            "synthesizer (options: mnist, fashion-mnist, cifar10, "
            "imagenet, imagenette) --required"
        ),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--class_index",
        help="int, class index of the class to be simulated (default='all')",
        # type=int,
        default="all",
        required=False,
    )

    parser.add_argument(
        "--num_samples",
        help="int, number of sampled to generate (default='all')",
        # type=int,
        default="all",
        required=False,
    )
    parser.add_argument(
        "--num_workers",
        help="int, number of cpu cores/workers (default=4)",
        type=int,
        default=4,
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        help="int, batch_size (default=16)",
        type=int,
        default=16,
        required=False,
    )
    parser.add_argument(
        "--num_epochs",
        help="int, number of epochs to fit the synthesizer (default=10)",
        default=10,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--load_model_path",
        help="Path to a pretrained checkpoint and name (bypasses training)",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model_save_dir",
        default="temp/models/",
        help=(
            "Path to a directory where the final model.pth is to be saved "
            "(default=temp/models/)"
        ),
        type=str,
        required=False,
    )
    parser.add_argument(
        "--data_save_dir",
        default="temp/datasets/",
        help=(
            "Path to a directory where the synthesized images will be "
            "saved (default=temp/datasets/)"
        ),
        type=str,
        required=False,
    )
    parser.add_argument(
        "--task",
        help="str, name of the task (train, val, or mimic (tran and val))",
        default="train",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--image_dim",
        help=(
            "Square dimensions (h=w=image_dim) of the synthesized images "
            "(default = None, produces images with a pre-defined "
            "dataset dimension)"
        ),
        default=None,
        type=int,
        required=False,
    )

    return parser.parse_args()


def get_label_counts(dataset_name, class_idx):

    train_loader, test_loader = dsets.create_dataloaders(
        dataset_name, batch_size=16, input_size=28, num_workers=1, class_idx=class_idx,
    )
    train_unique_labels, train_label_counts = torch.unique(
        torch.as_tensor(train_loader.dataset.targets), return_counts=True
    )
    test_unique_labels, test_label_counts = torch.unique(
        torch.as_tensor(test_loader.dataset.targets), return_counts=True
    )
    if (class_idx is not None) or (not class_idx == "all"):

        label_info = {
            "train_labels": (train_unique_labels, train_label_counts),
            "val_labels": (test_unique_labels, test_label_counts),
        }
    else:
        label_info = {
            "train_labels": (
                train_unique_labels[class_idx],
                train_label_counts[class_idx],
            ),
            "val_labels": (test_unique_labels[class_idx], test_label_counts[class_idx]),
        }

    return label_info


def synthesize_using_pretrained(
    method: str,
    dataset_name: str,
    load_model_path: os.PathLike,
    num_workers: int,
    class_idx: int,
    tasks: list = ["train"],
    num_samples: int = "all",
    save_dir: os.PathLike = "temp/datasets/",
):
    """
    loads a pretained model and synthesizes images that are saved to
    the given save_dir location
    """
    # load the model
    model = METHODS[method].load(load_model_path)
    # model = model.load_model(args.load_model_path)

    # generate based on task
    for task in tasks:
        if (num_samples is not None) or (not num_samples == "all"):
            label_info = get_label_counts(args.dataset_name, args.class_index)
            num_samples = label_info[f"{task}_labels"][1]
        data_save_dir = f"{save_dir}{dataset_name}_{method}/{task}/{class_idx}"
        confirm_directory(images_path)
        model.generate_images(num_samples, data_save_dir)


def train_and_synthesize(
    method: str,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    class_idx: int,
    num_epochs: int,
    image_dim: int = None,
    tasks: list = ["train"],
    num_samples: int = "all",
    data_save_dir: os.PathLike = "temp/datasets/",
    model_save_dir: os.PathLike = "temp/models/",
):

    normalize_strategy = (
        transforms.Normalize((0.5), (0.5))
        if "mnist" in dataset_name
        else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )

    # train data transforms:
    train_transforms = transforms.Compose(
        [
            transforms.Resize(image_dim),
            transforms.CenterCrop(image_dim),
            transforms.ToTensor(),
            normalize_strategy,
        ]
    )

    # test data transforms:
    test_transforms = transforms.Compose(
        [
            transforms.Resize(image_dim),
            transforms.CenterCrop(image_dim),
            transforms.ToTensor(),
            normalize_strategy,
        ]
    )

    label_info = get_label_counts(args.dataset_name, class_idx)

    if (num_samples is None) or (num_samples == "all"):
        num_samples = "all"

    # generate based on task
    for task in tasks:
        if (class_idx is not None) or (not class_idx == "all"):
            unique_labels = label_info[f"{task}_labels"][0]
        else:
            unique_labels = [args.class_index]

        for idx in unique_labels:  # all class indices
            # dataloaders for the specific class
            train_loader, test_loader = dsets.create_dataloaders(
                args.dataset_name,
                batch_size=batch_size,
                input_size=image_dim,
                num_workers=num_workers,
                class_idx=idx,
                train_transforms=train_transforms,
                test_transforms=test_transforms,
            )
        if task == "train":
            model = METHODS[method]
            model.train(train_loader, test_loader, num_epochs=num_epochs)
            model_save_dir = (
                f"{model_save_dir}{dataset_name}_{method}/{task}/{class_idx}"
            )
            model.save_model(model_path=model_save_dir)

        num_samples = (
            label_info[f"{task}_labels"][1] if num_samples == "all" else num_samples
        )
        data_save_dir = f"{data_save_dir}{dataset_name}_{method}/{task}/{class_idx}"
        confirm_directory(data_save_dir)
        model.generate_images(num_samples, data_save_dir)


def prep_args(args, approach) -> dict:
    # standard things
    kwargs = {
        "method": args.method,
        "num_samples": args.num_samples,
        "dataset_name": args.dataset_name,
        "data_save_dir": args.data_save_dir,
    }
    # train before generating images
    if approach == "train_model":
        if task == "mimic":
            tasks = ["train", "val"]
        elif task == "train":
            tasks = ["train"]
        elif task == "val":
            raise NotImplementedError(
                "Cannot create a validation dataset without an existing model."
            )

        kwargs.update(
            {
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "class_idx": args.class_index,
                "num_epochs": args.num_epochs,
            }
        )
    # generate images by loading a pretrained model
    elif approach == "load_model":
        if args.task == "mimic":
            tasks = ["train", "val"]
        elif args.task == "train" or args.task == "val":
            tasks = ["train"]

        kwargs.update(
            {"load_model_path": args.load_model_path,}
        )

    else:
        raise NotImplementedError(
            f"Approach '{approach}' is not supported. "
            "The supported options are: 'train_model' or 'load_model'."
        )

    if args.image_dim is None:
        kwargs["image_dim"] = IMAGE_DIMS[dataset_name]
        kwargs["tasks"] = tasks

    return kwargs


def synthesize(args):
    """Command Line Convenience (cli) method to parse args and generate synthetic
    data samples.
    It can be used to load pre-trained model or train a model from scrath.
    The task class_idx and num_samples argmuments can be used to mimic
    a dataset class label distribution (sample instance count).
    If training and mimicking the method trains a new synthesizer for each
    class in the dataset.
    """
    # verify the main args
    if args.method not in METHODS:
        raise NotImplementedError(
            f"The method '{args.method}' is not supported. "
            "Options: vae, cvae, and dcgan."
        )

    if args.dataset_name not in DATASETS:
        raise NotImplementedError(
            f"The dataset '{args.dataset_name}' is not currently supported "
            f"(can be added easily iff data follow the ImageFolder structure)"
        )

    # load a trained model
    if args.load_model_path is not None:
        kwargs = prep_args(args, "train_model")
        synthesize_using_pretrained(**kwargs)
    else:
        # cannot handle "val" task alone -- must throw an error.
        kwargs = prep_args(args, "load_model")
        train_and_synthesize(**kwargs)


if __name__ == "__main__":
    args = parse_args()
    synthesize(args)
