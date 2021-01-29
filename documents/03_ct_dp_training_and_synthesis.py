#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GETTING STARTED WITH CSL DIFFERENTIALLY PRIVATE DATA SYNTHESIZERS.
"""
import sys

# sys.path.append("/persist/carlos_folder/csl/")
sys.path.append("../csl/")
import csl.synthesizers as syn

ROOT_DIR = "/persist/"
DATA_DIR = f"{ROOT_DIR}datasets/"
MODELS_DIR = f"{ROOT_DIR}models/"

# VARS
METHOD = "dcgan"
DATASET_NAME = "mnist"

TASK = ["train"]  # "val"]  # successively mimics the train and validation sets
N_EPOCHS = 100
NUM_WORKERS = 8
BATCH_SIZE = 32
CLASS_INDEX = "all"  # 0, 1, 2... etc
ALPHAS = [None, 20]
for ALPHA in ALPHAS:
    EPSILONS = (
        [None]
        if ALPHA is None
        else [1e6, 5e5, 2e5, 1e5, 1e4, 1e3, 1e2, 10, 1, 0.1, 0.01]
    )
    for EPSILON in EPSILONS:
        args = {
            "method": METHOD,
            "dataset_name": DATASET_NAME,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "class_index": CLASS_INDEX,
            "num_epochs": N_EPOCHS,
            "image_dim": 64,
            "tasks": ["train"],  # , "val"],
            "num_samples": "all",
            "data_save_dir": DATA_DIR,
            "model_save_dir": MODELS_DIR,
            "alpha": ALPHA,
            "epsilon": EPSILON,
        }
        syn.train_and_synthesize(**args)

        # LOAD THE PRETRAINED MODELS
        # adjust data destination and source model locations based on epsilon and alpha parameter
        task = "val"
        if (ALPHA is not None) and (EPSILON is not None):
            model_dir = (
                f"{MODELS_DIR}{DATASET_NAME}_{METHOD}_dp_{ALPHA}a_{EPSILON}e/train/"
            )
            data_dir = (
                f"{DATA_DIR}{DATASET_NAME}_{METHOD}_dp_{ALPHA}a_{EPSILON}e/{task}/"
            )
        else:
            model_dir = f"{MODELS_DIR}{DATASET_NAME}_{METHOD}/train/"
            data_dir = f"{DATA_DIR}{DATASET_NAME}_{METHOD}/{task}/"
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
        f"{ALPHA}-alpha and {EPSILON}-epsilon IS COMPLETE\n===\n\n"
    )
