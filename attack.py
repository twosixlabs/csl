"""
Usage: to perform an attack on your models at /path/to/models/MNIST-Baseline at every 100th checkpoint up to 1,000
and to compute the FID between your real data in /path/to/data/ and the generated samples, execute

python -m attack \
--generate_samples \
--compute_fid \
--save \
--data_dir /path/to/data \
--model_dir /path/to/models \
--model_name MNIST-Baseline \
--checkpoint_min 100 \
--checkpoint_step 100 \
--checkpoint_max 1000 
"""
import argparse
import itertools
import json
import logging
import os
import shutil
import sys
sys.path.append("csl-gan/") # This assumes that the csl-gan repo is a subdirectory of the current working directory.
sys.path.append("csl-gan/opacus/") # This assumes that the Opacus fork is a subdirectory of ./csl-gan/.
import uuid
from typing import *


import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torchvision
from scipy.stats import chi2
from sklearn.model_selection import train_test_split


import init_util as csl_init_util
import mnist_models as csl_baseline_mnist_models
import old_utils as csl_old_init_util
import options as csl_options
import util as csl_utils
from pytorch_fid import fid_score


def attack(attack_values_train: List[float], 
           attack_values_nontrain: List[float], 
           data_prop: float = 0.1) -> float:
    """
    Performs the membership inference attack as presented in Hayes et al., 2018, framed as follows:
    "given a dataset only 10% of which is training data, identify that 10% by sorting on attack value."
    The attack is successful if the attack value is generally higher for training samples than for
    non-training samples. Otherwise, the attack success rate should be within a few points of data_prop.

    args:
        attack_values_train (List[float]): A list of attack values associated to training samples.
        attack_values_nontrain (List[float]): A list of attack values associated to non-training samples.
        data_prop (float): The proportion of the adversary's dataset they believe are training samples.
            Default = 0.1.

    Returns:
        The attack success rate given random subsets of training and non-training samples.
    """
    N, M = len(attack_values_train), len(attack_values_nontrain)
    n = int(1000 * data_prop)
    m = int(1000 * (1 - data_prop))
    attack_subset_train = _get_random_subset(attack_values_train, n)
    attack_subset_nontrain = _get_random_subset(attack_values_nontrain, m)
    attack_pairs_train = [(v, 1) for v in attack_subset_train]
    attack_pairs_nontrain = [(v, 0) for v in attack_subset_nontrain]
    attack_pairs_all = attack_pairs_train + attack_pairs_nontrain
    attack_pairs_sorted = sorted(attack_pairs_all, key=lambda pair: pair[0], reverse=True)
    attack_pairs_best = attack_pairs_sorted[:n]
    _, training_indicators = zip(*attack_pairs_best)
    attack_success_rate = np.mean(training_indicators)
    return attack_success_rate


def _get_random_subset(collection: List, k: int) -> List:
    N = len(collection)
    random_subset_indices = np.random.choice(range(N), size=k, replace=False)
    random_subset = np.array(collection)[random_subset_indices].tolist()
    return random_subset


def apply_mnist_discriminator(dataloader: torch.utils.data.DataLoader, 
                              device: torch.device, 
                              G: torch.nn.Module,
                              D: torch.nn.Module,
                              options: Optional[argparse.Namespace] = None,
                              **kwargs) -> List:
    D_values = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            _, D_values_cond_batch = D(x_batch, y=y_batch)
            D_values_batch = torch.nn.functional.softmax(D_values_cond_batch, 1).max(1)[0]
            D_values_batch = D_values_batch.detach().cpu().numpy().flatten().tolist()
            D_values.extend(D_values_batch)
    D_values = np.array(D_values)
    return D_values


def apply_celeba_discriminator(dataloader: torch.utils.data.DataLoader, 
                               device: torch.device, 
                               G: torch.nn.Module,
                               D: torch.nn.Module,
                               options: Optional[argparse.Namespace] = None,
                               **kwargs) -> List:
    D_values = []
    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            D_values_batch, _ = D(x_batch)
            D_values_batch = D_values_batch.detach().cpu().numpy().flatten().tolist()
            D_values.extend(D_values_batch)
    D_values = np.array(D_values)
    return D_values


def get_model_filepath(model_dir: Text, model_name: Text, model_filename: Text) -> Text:
    model_filepath = os.path.join(model_dir, model_name, model_filename)
    return model_filepath


def load_weights(model_dir: Text, 
                 model_name: Text, 
                 G: torch.nn.Module, 
                 D: torch.nn.Module, 
                 device: torch.device, 
                 checkpoint: Optional[int] = None) -> None:
    model_savepath = get_model_filepath(model_dir, model_name, "saves/")
    weights_filenames = os.listdir(model_savepath)
    if checkpoint is not None and f"G-{checkpoint}" in weights_filenames and f"D-{checkpoint}" in weights_filenames:
        G_weights_filepath = os.path.join(model_savepath, f"G-{checkpoint}")
        D_weights_filepath = os.path.join(model_savepath, f"D-{checkpoint}")
    else:
        G_weights_filenames = filter(lambda filename: filename.startswith("G-"), weights_filenames)
        sorted_G_weights_filenames = sorted(G_weights_filenames, key=lambda filename: int(filename.split("-")[1]))
        G_weights_filename = sorted_G_weights_filenames[-1]
        G_weights_filepath = os.path.join(model_savepath, G_weights_filename)
        D_weights_filenames = filter(lambda filename: filename.startswith("D-"), weights_filenames)
        sorted_D_weights_filenames = sorted(D_weights_filenames, key=lambda filename: int(filename.split("-")[1]))
        D_weights_filename = sorted_D_weights_filenames[-1]
        D_weights_filepath = os.path.join(model_savepath, D_weights_filename)
    csl_utils.load_model(G_weights_filepath, G, device)
    csl_utils.load_model(D_weights_filepath, D, device)
    return None


def validate_checkpoints(model_dir: Text, model_name: Text, checkpoints: List[int]) -> None:
    model_savepath = get_model_filepath(model_dir, model_name, "saves/")
    try:
        existing_checkpoints = os.listdir(model_savepath)
    except FileNotFoundError as e:
        raise ValueError(f"Directory does not exist: {model_savepath}.")
    for checkpoint in checkpoints:
        g_checkpoint = f"G-{checkpoint}"
        if g_checkpoint not in existing_checkpoints:
            raise ValueError(f"{g_checkpoint} is not in {model_savepath}.")
        d_checkpoint = f"D-{checkpoint}"
        if d_checkpoint not in existing_checkpoints:
            raise ValueError(f"{d_checkpoint} is not in {model_savepath}.")
    return None


def save_data_as_pngs(dataset: torch.utils.data.Dataset, directory: Text) -> None:
    num_digits = len(str(len(dataset)))
    for i, (x, y) in enumerate(dataset):
        png_filename = f"{str(i).zfill(num_digits)}.png"
        torchvision.utils.save_image(x, os.path.join(directory, png_filename))
    return None


if __name__ == "__main__":


    logging.basicConfig(format="%(asctime)s %(user)-8s %(message)s",
                    level=logging.INFO)
    coloredlogs.install(level=logging.INFO)


    run_id = uuid.uuid4().hex


    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_iters", type=int, default=10000,
        help="<Optional> (int) The number of subset pairs to choose to compute attack success rate. Default = 10000.")
    parser.add_argument("--batch_size", type=int, default=1000,
        help="<Optional> (int) The batch size to use when applying the discriminator to the data. Default = 1000.", required=False)
    parser.add_argument("--compute_fid", default=False, action="store_true",
        help="<Optional> Whether or not to compute the FID between the training data and the generated samples. Default = False.", required=False)
    parser.add_argument("--data_dir", type=Text, default="/home/reed.gordon-sarney/work/csl/data/",
        help="<Optional> (Text) The root directory to find the dataset. Default = /home/reed.gordon-sarney/work/csl/data/.", required=False)
    parser.add_argument("--data_prop", type=float, default=0.1,
        help="<Optional> (float) The proportion of training data the adversary has access to. Default = 0.1.", required=False)
    parser.add_argument("--fid_dir", type=Text, default="fid/",
        help="<Optional> (Text) The directory to save and load the FID values. Default = 'fid/'.", required=False)
    parser.add_argument("--generate_samples", default=False, action="store_true",
        help="<Optional> Whether or not to generate and save synthetic data. Default = False.", required=False)
    parser.add_argument("--gpu", type=int, choices=range(-1, torch.cuda.device_count()), default=-1, 
        help="<Optional> (int) The GPU device number to use. Default = -1 (CPU).", required=False)
    parser.add_argument("--checkpoint_max", type=int, default=None,
        help="<Optional> (int) The maximum model checkpoint to evaluate. Default = None.", required=False)
    parser.add_argument("--checkpoint_min", type=int, default=None,
        help="<Optional> (int) The minimum model checkpoint to evaluate. Default = None.", required=False)
    parser.add_argument("--checkpoint_step", type=int, default=None,
        help="<Optional> (int) The step to take between model checkpoints to evaluate. Default = None.", required=False)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None,
        help="<Optional> (int) The checkpoint(s) in number of epochs (e.g., 5000 <-> G-5000 and D-5000) to load. Default = None.", required=False)
    parser.add_argument("--model_dir", type=Text, default="/home/reed.gordon-sarney/work/csl/models/",
        help="<Optional> (Text) The root directory to find the models. Default = /home/reed.gordon-sarney/work/csl/models/.", required=False)
    parser.add_argument("--model_name", type=Text,
        help="<Required> (Text) The name of the model as written by the training script.", required=True)
    parser.add_argument("--num_generated_samples", type=int, default=2048,
        help="<Optional> (int) The number of synthetic images to generate for each GAN loaded. Default = 2048.", required=False)
    parser.add_argument("--outputs_dir", type=Text, default="outputs/",
        help="<Optional> (Text) The directory to save any attack outputs. Default = 'outputs/'.", required=False)
    parser.add_argument("--public_set_size", type=int, default=10000,
        help="<Optional> (int) The number of non-training samples to draw from for the attack. Default = 10000.", required=False)
    parser.add_argument("--real_samples_dir", type=Text, default=f"real_samples_dir/",
        help="<Optional> (Text) The temporary directory to store the real training samples as images for FID computation. Default = 'tmp/real_samples_dir/'.", required=False)
    parser.add_argument("--samples_dir", type=Text, default="samples/",
        help="<Optional> (Text) The directory to save the generated samples. Default = 'samples/'.", required=False)
    parser.add_argument("--save", default=False, action="store_true",
        help="<Optional> Whether or not to save the performance artifacts. Default = False.")
    parser.add_argument("--tmp_dir", type=Text, default="tmp/",
        help="<Optional> (Text) The directory to save temporary files. Default = 'tmp/'.", required=False)
    parser.add_argument("--train_set_size", type=int, default=None,
        help="<Optional> (int) The number of training samples to draw from for the attack. Default = None.", required=False)
    parser.add_argument("--values_dir", type=Text, default="values/",
        help="<Optional> (Text) The directory to save and/or load the discriminator values. Default = 'values/'.", required=False)
    args = parser.parse_args()


    if all(arg is not None for arg in [args.checkpoint_max, args.checkpoint_min, args.checkpoint_step]) and \
            (args.checkpoint_max > args.checkpoint_min > 0):
        args.checkpoints = list(range(args.checkpoint_min, args.checkpoint_max + args.checkpoint_step, args.checkpoint_step))

    logging.info(f"Model: {args.model_name}")
    logging.info("Parsing the options.")
    validate_checkpoints(args.model_dir, args.model_name, args.checkpoints)
    options_filepath = get_model_filepath(args.model_dir, args.model_name, "opt.txt")
    if not os.path.exists(options_filepath):
        raise ValueError(f"Invalid filepath: {options_filepath}")
    if "MNIST" in args.model_name.upper():
        get_values = apply_mnist_discriminator
        data_path = os.path.join(args.data_dir, "mnist")
        label_path = None
        public_set_size = args.public_set_size
    elif "CELEBA" in args.model_name.upper():
        get_values = apply_celeba_discriminator
        data_path = os.path.join(args.data_dir, "celeba/imgs/")
        label_path = os.path.join(args.data_dir, "celeba/list_attr_celeba.csv")
        public_set_size = args.public_set_size
    else:
        raise ValueError(f"Invalid model name: {args.model_name}.")
    my_arguments = {"data_path": data_path,
                    "label_path": label_path,
                    "num_workers": 1,
                    "public_set_size": public_set_size,
                    "weights_seed": 42}
    if "CelebA-Baseline" in args.model_name:
        my_arguments["per_sample_grad"] = False
    if args.train_set_size is not None:
        my_arguments["train_set_size"] = args.train_set_size
    options = csl_options.load_opt(options_filepath)
    if not hasattr(options, "unconditional") and hasattr(options, "conditional"):
        my_arguments["unconditional"] = not options.conditional
    if hasattr(options, "unconditional") and not hasattr(options, "conditional"):
        my_arguments["conditional"] = not options.unconditional
    if not hasattr(options, "label_attr"):
        my_arguments["label_attr"] = "Male"
    for key, value in my_arguments.items():
        setattr(options, key, value)
    logging.info("Parsed the options.")


    logging.info("Loading the data.")
    train_dataset, train_dataloader, nontrain_dataset, nontrain_dataloader = csl_init_util.init_data(options)
    logging.info("Loaded the data.")

    
    if args.compute_fid:
        if "MNIST" in args.model_name.upper():
            dataset_label = "mnist"
            if not os.path.exists(args.tmp_dir):
                os.mkdir(args.tmp_dir)
            if not os.path.exists(os.path.join(args.tmp_dir, args.real_samples_dir)):
                os.mkdir(os.path.join(args.tmp_dir, args.real_samples_dir))
            if not os.path.exists(os.path.join(args.tmp_dir, args.real_samples_dir, dataset_label)):
                os.mkdir(os.path.join(args.tmp_dir, args.real_samples_dir, dataset_label))
            real_samples_dir_w_id = os.path.join(args.tmp_dir, args.real_samples_dir, dataset_label)
            if len(os.listdir(real_samples_dir_w_id)) == 0:
                logging.info(f"Saving the training data as .pngs to {real_samples_dir_w_id}.")
                save_data_as_pngs(train_dataset, real_samples_dir_w_id)
                logging.info("Saved the training data as .pngs.")
        elif "CELEBA" in args.model_name.upper():
            dataset_label = "celeba"
            real_samples_dir_w_id = data_path
            logging.info("CelebA .pngs already exist.")


    logging.info("Instantiating the GAN architecture.")
    device = torch.device("cpu")
    if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    if args.model_name == "MNIST-Baseline":
        G = csl_baseline_mnist_models.MNISTVanillaG(options.g_latent_dim, 
                                                    unconditional=not options.conditional,
                                                    num_classes=options.n_classes)
        D = csl_baseline_mnist_models.MNISTVanillaD(unconditional=not options.conditional,
                                                    num_classes=options.n_classes)
    else:
        try:
            G, D = csl_init_util.init_models(options)
        except AttributeError as e:
            options.g_latent_dim = 130
            from calvin_utils import init_old_celeba_models
            G, D = init_old_celeba_models(options)
    G, D = G.to(device), D.to(device)
    logging.info(f"Instantiated the GAN architecture on {device}.")


    logging.info(f"Computing ASR with training data proportion {args.data_prop:.2%}.")


    json_filename = f"{args.model_name}.json"
    json_savepath = os.path.join(args.outputs_dir, json_filename)
    if os.path.exists(json_savepath):
        with open(json_savepath, "r") as file:
            checkpoint_stats = json.load(file)
        logging.info("Checkpoint stats loaded.")
    else:
        checkpoint_stats = {}
    checkpoints = list(filter(lambda checkpoint: str(checkpoint) not in checkpoint_stats,
                              args.checkpoints))                              
    for checkpoint in checkpoints:

        
        checkpoint_stats[checkpoint] = {}
        g_name, d_name = f"G-{checkpoint}", f"D-{checkpoint}"
        load_weights(args.model_dir, args.model_name, G, D, device, checkpoint)
        logging.info(f"Loaded {g_name} and {d_name}.")


        attack_values_dir = os.path.join(args.values_dir, args.model_name, f"checkpoint-{checkpoint}")
        attack_values_dir_split = attack_values_dir.split("/")
        for k in range(1, len(attack_values_dir_split)+1):
            attack_values_dir_k = os.path.join(*attack_values_dir_split[:k])
            if not os.path.exists(attack_values_dir_k):
                os.mkdir(attack_values_dir_k)
        attack_values_train_filepath = f"{attack_values_dir}/attack_values_train.npy"
        attack_values_nontrain_filepath = f"{attack_values_dir}/attack_values_nontrain.npy"
        if os.path.exists(attack_values_train_filepath):
            attack_values_train = np.load(attack_values_train_filepath)
            logging.info(len(list(filter(lambda x: x > 0, attack_values_train))))
            logging.info(f"{len(attack_values_train)} training attack values loaded.")
        else:
            logging.info(f"Getting the training attack values.")
            attack_values_train = get_values(train_dataloader, device, G, D, options)
            np.save(attack_values_train_filepath, attack_values_train)
            logging.info(f"{len(attack_values_train)} training attack values generated and saved.")
        if os.path.exists(attack_values_nontrain_filepath):
            attack_values_nontrain = np.load(attack_values_nontrain_filepath)
            logging.info(len(list(filter(lambda x: x > 0, attack_values_nontrain))))
            logging.info(f"{len(attack_values_nontrain)} non-training attack values loaded.")
        else:
            logging.info(f"Getting the non-training attack values.")
            attack_values_nontrain = get_values(nontrain_dataloader, device, G, D, options)
            np.save(attack_values_nontrain_filepath, attack_values_nontrain)
            logging.info(f"{len(attack_values_nontrain)} non-training attack values generated and saved.")


        attack_success_rates = [attack(attack_values_train, attack_values_nontrain, args.data_prop) for _ in range(args.asr_iters)]
        asr = np.mean(attack_success_rates)
        checkpoint_stats[checkpoint]["asr"] = asr
        logging.info(f"ASR on {args.model_name}-{checkpoint}: {asr:.2%}")


        if args.generate_samples:
            logging.info(f"Generating samples from {g_name}.")
            with torch.no_grad():
                if options.conditional:
                    true_num_generated_samples = options.n_classes * ((args.num_generated_samples // options.n_classes) + 1)
                    y = []
                    for label in range(options.n_classes):
                        y_label = label * torch.ones(true_num_generated_samples // options.n_classes, dtype=torch.int64, device=device)
                        y.append(y_label)
                    y = torch.cat(y, 0)
                else:
                    true_num_generated_samples = args.num_generated_samples
                    y = None
                z = torch.empty(true_num_generated_samples, options.g_latent_dim, device=device).normal_(0.0, 1.0)
                if args.model_name == "MNIST-Baseline":
                    y = torch.nn.functional.one_hot(y, num_classes=options.n_classes)
                generated_images = G(z, y).cpu()
                if options.conditional and y is not None:
                    generated_targets = y.cpu()
                    generated_dataset = torch.utils.data.TensorDataset(generated_images, generated_targets)
                else:
                    generated_dataset = torch.utils.data.TensorDataset(generated_images)
                generated_dataloader = torch.utils.data.DataLoader(dataset=generated_dataset,
                                                                   batch_size=args.batch_size,
                                                                   shuffle=True)
            logging.info(f"Generated {len(generated_images)} samples.")


            logging.info("Saving the generated examples.")
            if "CelebA" in args.model_name:
                generated_images = csl_utils.denorm_celeba(generated_images)
            fake_samples_dir_w_id = os.path.join(args.samples_dir, args.model_name, g_name, run_id)
            fake_samples_dir_w_id_split = fake_samples_dir_w_id.split("/")
            for k in range(1, len(fake_samples_dir_w_id_split)+1):
                fake_samples_dir_w_id_k = os.path.join(*fake_samples_dir_w_id_split[:k])
                if not os.path.exists(fake_samples_dir_w_id_k):
                    os.mkdir(fake_samples_dir_w_id_k)
            for i, generated_image_i in enumerate(generated_images):
                filename_i = f"{str(i).zfill(4)}.png"
                filepath_i = f"{fake_samples_dir_w_id}/{filename_i}"
                with open(filepath_i, "wb") as file:
                    torchvision.utils.save_image(generated_image_i, file)
            logging.info("Saved the generated examples.")


        if args.compute_fid and not args.generate_samples:
            logging.info("Loading FID.")
            fid_filepath = f"{os.path.join(args.values_dir, args.fid_dir, args.model_name, g_name)}/fid.txt"
            if not os.path.exists(fid_filepath):
                raise FileNotFoundError(f"{fid_filepath} does not exist.")
            with open(fid_filepath, "r") as file:
                fid = float(file.read())
            checkpoint_stats[checkpoint]["fid"] = fid
            logging.info(f"Loaded FID: {fid:.2f}.")
        elif args.compute_fid:
            logging.info("Computing FID.")
            fid = fid_score.calculate_fid_given_paths((real_samples_dir_w_id, fake_samples_dir_w_id), 50, device, 2048, 1)
            checkpoint_stats[checkpoint]["fid"] = fid
            logging.info(f"Computed FID: {fid:.2f}.")
            fid_filedir = os.path.join(args.values_dir, args.fid_dir, args.model_name, g_name)
            fid_filedir_split = fid_filedir.split("/")
            for k in range(1, len(fid_filedir_split)+1):
                fid_filedir_k = os.path.join(*fid_filedir_split[:k])
                if not os.path.exists(fid_filedir_k):
                    os.mkdir(fid_filedir_k)
            fid_filepath = f"{fid_filedir}/fid.txt"
            with open(fid_filepath, "w") as file:
                file.write(str(fid))
            logging.info(f"FID written to {fid_filepath}.")


        if args.generate_samples:
            logging.info(f"Deleting the checkpoint samples directory {fake_samples_dir_w_id}.")
            assert not fake_samples_dir_w_id.startswith("/")
            shutil.rmtree(fake_samples_dir_w_id, ignore_errors=True)
            logging.info("Deleted the checkpoint samples directory.")


    logging.info(json.dumps(checkpoint_stats, indent=4))


    if args.save:
        if not os.path.exists(args.outputs_dir):
            os.mkdir(args.outputs_dir)
        logging.info("Saving the checkpoint measurements.")
        json_filename = f"{args.model_name}.json"
        json_savepath = os.path.join(args.outputs_dir, json_filename)
        with open(json_savepath, "w") as file:
            json.dump(checkpoint_stats, file)
        logging.info("Saved the checkpoint measurements.")
