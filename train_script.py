#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:56:53 2020

GETTING STARTED WITH CSL DATA SYNTHESIZERS.

@author: carlos-torres
"""
import torch
import torchvision
import sys
import os

# sys.path.append("/persist/carlos_folder/csl/")
# sys.path.append("../csl/")
import csl.synthesizers as syn

# VARS
METHOD = "dcgan"
DATASET_NAME = "imagenette"
MODELS_DIR = "/home/ben.gelman/debug/private_models/"
DATA_DIR = "/home/ben.gelman/debug/private_datasets/"
TASK = ["train"]  # "val"]  # successively mimics the train and validation sets


EPSILONS, ALPHAS = [None], [None]  # COMPUTE-CUDA0 -- running

# EPSILONS, ALPHAS = [1e6, 5e5, 2e5, 1e5], [20, 2]  # COMPUTE-CUDA
# EPSILONS, ALPHAS = [10000, 1000, 100, 10], [20, 2]  # COMPUTE-CUDA
# EPSILONS, ALPHAS = [1, 0.1, 0.01], [20, 2]  # COMPUTE-CUDA
# EPSILONS, ALPHAS = [10, 1], [20]  # COMPUTE-CUDA
# EPSILONS, ALPHAS = [0.1, 0.01], [20]  # COMPUTE-CUDA
# EPSILONS, ALPHAS = [1e5, 1e4], [20]
# EPSILONS, ALPHAS = [1e6, 2e5, 1e5, 1e4], [20]

EPSILONS, ALPHAS, THROW_OUTS = [1e6, 2e5], [20], [10]

for ALPHA in ALPHAS:
    if ALPHA is None:
        N_EPOCHS = 500
    else:
        N_EPOCHS = 300
    for EPSILON in EPSILONS:
        for THROW_OUT in THROW_OUTS:
            args = {
                "method": METHOD,
                "dataset_name": DATASET_NAME,
                "batch_size": 32,
                "num_workers": 4,
                "class_index": "all",
                "num_epochs": N_EPOCHS,  # 250
                "image_dim": 64,
                "tasks": ["train"],  # , "val"],
                "num_samples": "all",
                "data_save_dir": DATA_DIR,
                "model_save_dir": MODELS_DIR,
                "alpha": ALPHA,
                "epsilon": EPSILON,
                "throw_out": THROW_OUT}
            syn.train_and_synthesize(**args)

            # LOAD THE PRETRAINED MODELS
            # adjust data destination and source model locations
            # based on epsilon and alpha parameter values
            task = "val"
            model_dir = f"/home/ben.gelman/debug/private_models/{DATASET_NAME}_{METHOD}"
            data_dir = f"/home/ben.gelman/debug/private_datasets/{DATASET_NAME}_{METHOD}"
            if (ALPHA is not None) and (EPSILON is not None):
                model_dir += f"_{ALPHA}a_{EPSILON}e"
                data_dir += f"_{ALPHA}a_{EPSILON}e"

            # the ability to throw out data is independent of using the dp training,
            # so we need to adjust the directory name separately.
            if THROW_OUT is not None:
                model_dir += f"_{THROW_OUT}t"
                data_dir += f"_{THROW_OUT}t"

            model_dir += "/train/"
            data_dir += f"/{task}/"

            print(f"Loading checkpoint from {model_dir}")
            print(f"Synthetic dataset destination {data_dir}")

            for class_index in range(10):
                model_path = f"{model_dir}{class_index}/"
                data_path = f"{data_dir}{class_index}/"
                args = {
                    "method": METHOD,
                    "dataset_name": DATASET_NAME,
                    "num_workers": 4,
                    "class_index": class_index,  # "all",
                    "tasks": [task],  # , "train"],
                    "num_samples": "all",
                    "data_save_dir": data_path,
                    "load_model_path": model_path,
                }
                syn.synthesize_using_pretrained(**args)

    print(
        f"\n\n===\n{DATASET_NAME}-{METHOD} DATA GENERATION USING "
        f"{ALPHA}a and {EPSILON}e IS COMPLETE\n===\n\n"
    )
