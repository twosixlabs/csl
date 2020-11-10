#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 16:57:24 2020

# Image synthesizer (given a set of sample, create
        their vae-synthesized version)
    ref:
        https://github.com/bvezilic/Variational-autoencoder
        https://github.com/topics/mnist-generation

    Initialize weights
    https://github.com/screddy1313/VAE/blob/master/vae_mnist_cnn.ipynb

python vae-synthesizer
@author: carlos-torres
"""
import os
import math
import logging
import coloredlogs
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid
# import torch.nn.functional as F
from torch.autograd import Variable

from synthesizers.vaes import VAE, CVAE, VQ_CVAE
from utils.protools import create_dataloaders, confirm_directory

log = logging.getLogger("vae_synth")
coloredlogs.install(level="info", logger=log)
DEVICE = "cuda:0"  # "cpu"
torch.cuda.set_device(0)

# DATASETS AND DIMENSIONS
IMAGE_DIMS = {
    "imagenet": 32,  # 224
    "imagenette": 28,  # 224
    "mnist": 28,
    "fashion-mnist": 28,
    "cifar10": 32,
    # "stl10": 32,  # TODO
}

# THE DATASET
DATASET_NAME = "imagenette"
BATCH_SIZE = 32
# CLASS_IDX = 1
FIDELITY = 1  # percent ratio for reconstruction (between: > 0.0 and <= 1.0)

# VAE PARAMS -- EXAMPLE
DIM_LATENT = int(IMAGE_DIMS[DATASET_NAME]**2 * FIDELITY)  # using 20% of the info: 205 ~ 32 * 32 * .2
N_EPOCHS = 3
H_DIM1 = 512
H_DIM2 = 256
VAE_NAME = "cvae"

save_every_epochs = 10
num_workers = 4
DISP = False

USE_PRETRAINED = False
DATA_SRC = "train"
# DATA_SRC = "val"

def train(vae, epoch, train_loader, optimizer, class_idx):
    """
    Parameters
    ----------
    vae : pth.model
        the in-training variational auto-encoder.
    epoch : int
        epoch index number.
    train_loader : pth.dataloder
        iterable containing (data, labels).

    Returns
    -------
    vae : pth.model
        the in-training variational auto-encoder
        after the given epoch.
    """
    # set to train mode (enable grad)
    vae.train()
    train_loss = 0
    ttotal = len(train_loader)
    # for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, (data, _) in zip(tqdm(range(ttotal), desc=f"Train Batch {epoch} of {class_idx}-class"), train_loader):
    # for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data)
        loss = vae.loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        x = loss.item()
        if np.isnan(x):
            log.debug(f"Epoch (before adj): {epoch}; batch_idx: {batch_idx} loss: {loss}")
            # x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x = 0
            log.debug(f"    epoch (adfter adj): {epoch}; batch_idx: {batch_idx} loss: {loss}")
        train_loss += x

        optimizer.step()

        if batch_idx % 100 == 0:
            log.debug(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)} / {len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]"
                f"\t loss: {loss.item() / len(data):.6f}"
            )
    epoch_loss = train_loss / len(train_loader.dataset)
    log.info(
        f"====> Epoch: {epoch} Average Train loss: {epoch_loss: .4f}"
    )
    return vae, epoch_loss


def test(vae, test_loader) -> None:
    """
    Parameters
    ----------
    vae : pth.model
        the in-training variational auto-encoder.
    epoch : int
        epoch index number.
    test_loader : pth.dataloder
        iterable containing (data, labels).

    Returns
    -------
    None.

    """
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            # data = data.cuda()
            data = data.cuda()
            recon, mu, log_var = vae(data)

            # sum up batch loss
            test_loss += vae.loss_function(recon, data, mu, log_var).item()

    test_loss /= len(test_loader.dataset)
    log.info(f"====> Test set loss: {test_loss: .4f}\n")
    return test_loss


def init_xavier(m):
    """ xavier weight initialization
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        # TODO: evealuate this effect
        m.bias.data.fill_(0.01)


def init_kaiming(m):
    """kaiming weight initialization
    """
    if type(m) == nn.Linear:
        weights, bias = m.named_parameters()
        m.weight = torch.nn.Parameter(
            torch.randn(
                weights[1].shape[0],
                weights[1].shape[1]
            ) * math.sqrt(2./weights[1].shape[0])
        )
        m.bias.data.fill_(0)


def visualize_batch(batch, plot_title: str = None, grid_dims: int = 4):
    batch = torchvision.utils.make_grid(batch).cpu().numpy()
    batch = np.transpose(batch, (1, 2, 0))
    plt.figure(figsize=(grid_dims, grid_dims))
    plt.imshow(batch, cmap='Greys_r')
    if plot_title is not None:
        title = plot_title
    else:
        title = "Batch visualization"
    plt.title(title)
    plt.show()


def show(img, plot_title: str = None):
    batch = img.numpy()
    batch = np.transpose(batch, (1, 2, 0))
    plt.imshow(batch, cmap='Greys_r')
    if plot_title is not None:
        title = plot_title
    else:
        title = "Single sample visualization"
    plt.title(title)
    plt.show()


def encode_data(vae, data_iterator, dim_latent: int, dataset_name: str, save: bool = True, disp: bool = True):
    """
    Parameters
    ----------
    data_iterator : pth dataloader
        DESCRIPTION. iterable containing (batch, batch_labels)
    dim_latent : int, number of latent variables
        DESCRIPTION. mnist : dim_latent=2.

    Returns
    -------
    means, variances, data_labels, batch_latents, z

    Dims
    -------
        mean.shape = (d * batch_size, DIM_LATENT)
        log_var.shape = (batch_size * d, DIM_LATENT)
        var = log_var.exp()
        recon.shape = (batch_size * d, w * h)
        reconstructed_batch.shape = (batch_size * d, w * h)
        z.shape = (batch_size * d, DIM_LATENT)
    """
    vae.eval()

    #mnist
    means = np.zeros((1, dim_latent))
    variances = np.zeros((1, dim_latent))
    batch_latents = []
    data_labels = []

    ttotal = len(data_iterator)

    with torch.no_grad():
        for b, (batch, labels) in zip(tqdm(range(ttotal)), data_iterator):
            labels = list(labels.numpy())
            data_labels.extend(labels)
            log.debug(f" > Processing the {b}-th batch with {len(labels)}-data")

            batch = batch.cuda()
            recon, mean, log_var = vae(batch)

            batch_latents.append([mean, log_var])
            z = vae.reparameterize(mean, log_var)
            reconstructed_batch = vae.decode(z).cuda()

            # create original and reconstructed image-grids for
            d = batch.shape[1]
            w = batch.shape[2]
            h = batch.shape[3]
            orig_img = batch.view(len(batch), d, w, h)
            recons_img = reconstructed_batch.view(len(batch), d, w, h)

            if save:
                directory = f"test/data/{VAE_NAME}/{dataset_name}/{class_idx}/{class_idx}_grid_batches/"
                confirm_directory(directory)
                log.debug(f"Synthesizing the {b}-th random-sampled batch")
                canvas = torch.cat([orig_img, recons_img], 0)
                canvas = canvas.cpu().data
                # save_image(orig_img, f'{directory}orig_grid_sample_{b}.png')
                canvas_grid = make_grid(canvas, nrow=BATCH_SIZE, range=(-1, 1), normalize=True)
                save_image(canvas, f'{directory}recons_grid_sample_{b}.png')

            if disp:
                visualize_batch(batch, plot_title=f"Visualized {b}-Batch", grid_dims=4)
                visualize_batch(recons_img, plot_title=f"Reconstructed {b}-Batch", grid_dims=8)
                # visualize_batch(canvas, plot_title=f"Reconstructed {b}-Batch", grid_dims=8)

            mean = mean.cpu().numpy()
            var = log_var.cpu().numpy()
            means = np.vstack((means, mean))
            variances = np.vstack((variances, var))

    means = means[1:]
    variances = variances[1:]

    # z = np.vstack((means, variances))
    z = torch.tensor(np.vstack((means, variances))).cuda()
    log.debug(f" FINAL z_shape: {z.shape}")

    return means, variances, data_labels, batch_latents, z


def reconstruct(vae, batch_latents: list, data_labels: list,
                d: int, w: int, h: int,
                dataset_name: str, save=False, disp=True,
                n_samples: int = None, task: str = "test"):
    """
    Dims
    -------
        mu.shape = (d, DIM_LATENT)
        log_var.shape = (d, DIM_LATENT)
        len(label) = 1
        z.shape = (d, DIM_LATENT)
        sample.shape = (d, h*w)
    """
    vae.eval()

    sample_counter = 0
    with torch.no_grad():
        for b in range(len(batch_latents)):
            batch = batch_latents[b]

            # Batches can have diff sizes, specially at the end.
            starts = [r for r in range(0, len(batch[0]) - d, d)]
            ends = [
                r + d
                for r in range(0, len(batch[0]) - d, d)
            ]

            for j, (start, end) in enumerate(zip(starts, ends)):
                mu = batch[0][start: end]
                log_var = batch[1][start: end]
                label = data_labels[sample_counter]
                log.debug(
                    f"Synthesizing the ({j}-th item from the {b}-th batch "
                    f"of {len(batch[0])} batches) {sample_counter}-sample "
                    f"for '{label}' label"
                )
                mu = mu.cuda()
                log_var.cuda()
                z = vae.reparameterize(mu, log_var)
                sample = vae.decode(z)
                image = (sample.view(d, w, h) + 1) / 2

                if save:
                    # directory=f"test/data/{dataset_name}/{CLASS_IDX}/{task}/{label}/"
                    directory=f"test/data/{VAE_NAME}/{dataset_name}/{class_idx}/{class_idx}_reconstructed/"
                    confirm_directory(directory)
                    recon_image_path = f'{directory}label_{label}_sample_{sample_counter}.png'
                    save_image(image.cpu(), recon_image_path)
                if disp:
                    show(
                        image.cpu(),
                        plot_title=f"{sample_counter}-th reconstructed datum. label: {label}"
                    )

                if (n_samples is not None) and sample_counter > n_samples:
                    log.info(f" >> Done reconstructing limited {n_samples} samples")
                    return

                sample_counter += 1


def generate_samples(vae, n_samples, n: int, d: int, directory: str, disp: bool = False, save: bool = False):
    """
    # PENDING
    """
    vae.cuda()
    vae.eval()

    mu = Variable(torch.randn(n_samples * d, DIM_LATENT))
    log_var = Variable(torch.randn(n_samples * d, DIM_LATENT))

    with torch.no_grad():

        # vae.cpu()
        z = vae.reparemetrize(mu, log_var)
        z = z.cuda()

        samples = vae.sample(n_samples, d, seed=0)
        samples = vae.decode(z)
        samples = samples.view(-1, d, h, w).cpu()
    if disp:
        plot_title = f"Random-sampled batch {DATASET_NAME} dataset"
        visualize_batch(samples.detach(), plot_title=plot_title, grid_dims=8)

    if save:
        directory=f"test/data/{DATASET_NAME}/{class_idx}/{class_idx}_sampled_grids/"
        save_path=f"{directory}grid_{n}.png"
        save_image(samples, save_path)


def run_example(model_dir):
    num_workers = 4
    batch_size = BATCH_SIZE
    confirm_directory(model_dir)
    image_dim = IMAGE_DIMS[DATASET_NAME]

    # dataloaders
    train_loader, test_loader = create_dataloaders(
        DATASET_NAME,
        batch_size=batch_size,
        input_size=image_dim,
        num_workers=num_workers,
        class_idx=class_idx,
    )
    batch, labels = iter(train_loader).next()

    # reconstruction: (d)epth, (w)idth, (h)eight
    d = batch.shape[1]
    w = batch.shape[2]
    h = batch.shape[3]
    x_dim = w * h

    # == THE VAE-MODEL
    # FITTING/TRAINNG
    vae = VAE(x_dim=x_dim, h_dim1=H_DIM1, h_dim2=H_DIM2, z_dim=DIM_LATENT)
    # weight initialization
    vae.apply(init_kaiming)  # init_xavier,
    vae = vae.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    # optimizer = optim.RMSprop(vae.parameters(), lr=0.001, centered=True)
    # optimizer = optim.SGD(vae.parameters(), lr=0.01, momentum=0.9)
    train_hist = []
    val_hist = []
    for epoch in range(1, N_EPOCHS + 1):
        vae, train_loss = train(vae, epoch, train_loader, optimizer)
        val_loss = test(vae, test_loader)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if epoch % 10 == 0:
            MODEL_PATH = f"{model_dir}{DATASET_NAME}_{DIM_LATENT}components_{epoch}epoch.pth"
            torch.save(vae.state_dict(), MODEL_PATH)
    model_dir = "test/models/vae/"
    MODEL_PATH = f"{model_dir}{DATASET_NAME}_{class_idx}class_{DIM_LATENT}components_{epoch}epoch-FINAL.pth"
    torch.save(vae.state_dict(), MODEL_PATH)

    # plot the losses
    plt.plot(range(1, N_EPOCHS + 1), train_hist, label="Trained", marker="x")
    plt.plot(range(1, N_EPOCHS + 1), val_hist, label="Validated", marker=".")

    # Add Figure information
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    # plt.ylim((0, 1.))
    plt.xticks(np.arange(1, N_EPOCHS + 1, 1.0))
    plt.legend()
    plt.show()
    log.info(f">>> Training, and Evaluation processes for: \
    \n\t VAE using {DATASET_NAME}\n are complete")

    # == Example use #1: USE THE MODEL TO COMPRESS and RECONSTRUCT IMAGES
    # random_data_synthesizer(vae, n_digits=3, dim_latent=dim_latent, selection=None)
    means, variances, data_labels, batch_latents, z = encode_data(
        vae,
        test_loader,
        DIM_LATENT,
        DATASET_NAME,
        save=True,
        disp=False
    )

    n_test_samples = 10  # test with small sample of size n_digits
    n_test_samples = None  # all

    reconstruct(
        vae,
        batch_latents,
        data_labels,
        d,
        w,
        h,
        DATASET_NAME,
        save=True,
        disp=False,
        n_samples=n_test_samples
    )

    # means.shape = (len(dataset)* d, DIM_LATENT)
    # variances.shape = (len(dataset) * d, DIM_LATENT)
    starts = [r for r in range(0, len(means) - d, d)]
    ends = [r + d for r in range(0, len(variances) - d, d)]

    # single sample
    start = starts[0]
    end = ends[0]

    vae.cuda()
    vae.eval()
    with torch.no_grad():
        # reconstruction example:
        mu = torch.as_tensor(means[start:end, :]).cuda().view(d, -1)
        var = torch.as_tensor(variances[start:end, :]).cuda().view(d, -1)
        z = vae.sampling(mu, var)

        sample = vae.decoder(z.float())

    #
    sample = sample.detach()
    image_tensor = sample.view(d, w, h).cpu()
    show(image_tensor, plot_title="Reconstructed sample.")


def fit_vae(train_loader, test_loader, class_idx):
    """
    """
    confirm_directory(f"{model_dir}/temps/")
    batch, labels = iter(train_loader).next()

    # reconstruction: (d)epth, (w)idth, (h)eight
    d = batch.shape[1]
    w = batch.shape[2]
    h = batch.shape[3]
    x_dim = w * h

    # == THE VAE-MODEL
    # FITTING/TRAINNG
    if VAE_NAME == "vae":
        vae = VAE(x_dim=x_dim, h_dim1=H_DIM1, h_dim2=H_DIM2, z_dim=DIM_LATENT)
    elif VAE_NAME == "cvae":
        vae = CVAE(x_dim, DIM_LATENT, d)
    elif VAE_NAME == "vqvae":
        vae = VQ_CVAE(128, DIM_LATENT)
    else:
        raise NotImplementedError(f"'{VAE_NAME}' not currently supported.")

    # weight initialization
    vae.apply(init_kaiming)  # init_xavier,
    vae = vae.cuda()
    lr = 2e-4

    optimizer = optim.Adam(vae.parameters(), lr=lr, amsgrad=True)
    # optimizer = optim.Adamax(vae.parameters(), lr=lr)
    # optimizer = optim.RMSprop(vae.parameters(), lr=lr, centered=True)
    # optimizer = optim.SGD(vae.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if DATASET_NAME in ['imagenette', 'imagenet'] else 30, 0.5,)

    samples_directory = f"test/data/{VAE_NAME}/{DATASET_NAME}/{DATA_SRC}_grids/{class_idx}/"
    confirm_directory(samples_directory)

    train_hist = []
    val_hist = []
    n_samples = 64
    # import pdb; pdb.set_trace()
    for epoch in range(1, N_EPOCHS + 1):
        vae, train_loss = train(vae, epoch, train_loader, optimizer, class_idx)
        val_loss = test(vae, test_loader)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        # sample, visualize, and save
        if epoch % 10 == 0:
            with torch.no_grad():
                samples = vae.sample(n_samples, d, seed=0)
            tensor_image = samples.view(-1, d, h, w).cpu().data

            # save_image(orig_img, f'{directory}orig_grid_sample_{b}.png')
            grid = make_grid(tensor_image, nrow=int(n_samples**(.5)), range=(0, 1), normalize=True)

            plot_title = f"Random-sampled batch {DATASET_NAME} and {epoch} epoch"
            visualize_batch(grid.detach(), plot_title=plot_title, grid_dims=int(n_samples**(.5)))
            save_path = f"{samples_directory}grid_{epoch}.png"
            save_image(grid, save_path)
        # if epoch == 27:
            # import pdb; pdb.set_trace()
        if epoch == 1:
            best_vae = vae
            best_loss = val_loss
        else:
            if val_loss < best_loss:
                print(f"\n\n ===> {epoch}-epoch. Updating best (with {val_loss:.3f}), which is less than previous ({best_loss:.3f}) best_loss")
                best_vae = vae
                best_epoch = epoch
                best_loss = val_loss
                if DATA_SRC == "train":
                    model_path = f"{model_dir}{DATASET_NAME}_{CLASS_IDX}class_{DIM_LATENT}components_BEST.pth"
                    torch.save(best_vae.state_dict(), model_path)
        scheduler.step()

    temp_model_path = f"{model_dir}temps/{DATASET_NAME}_{DIM_LATENT}components_{epoch}epoch_FINAL.pth"
    if DATA_SRC == "train":
        torch.save(vae.state_dict(), temp_model_path)

    # plot the losses: train & val
    plt.plot(range(1, N_EPOCHS + 1), train_hist, label="Trained", marker="x")
    plt.plot(range(1, N_EPOCHS + 1), val_hist, label="Validated", marker=".")

    # Add Figure information
    plt.title(f"Loss as a function Epochs (best epoch={best_epoch}, loss={best_loss:.3f})")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    # plt.ylim((0, 1.))
    plt.xticks(np.arange(1, N_EPOCHS + 1, 1.0))
    plt.legend()
    plot_name = f"{model_dir}{DATASET_NAME}_{CLASS_IDX}class_{DIM_LATENT}components.png"
    plt.savefig(plot_name)
    # plt.show()
    log.info(f">>> Training, and Evaluation processes for: \
    \n\t VAE using {DATASET_NAME}\n are complete")

    return best_vae


def load_existing_vae(class_idx):

        # FITTING/TRAINNG
    if VAE_NAME == "vae":
        vae = VAE(x_dim=x_dim, h_dim1=H_DIM1, h_dim2=H_DIM2, z_dim=DIM_LATENT)
    elif VAE_NAME == "cvae":
        vae = CVAE(x_dim, DIM_LATENT, d)
    elif VAE_NAME == "vqvae":
        vae = VQ_CVAE(128, DIM_LATENT)
    else:
        raise NotImplementedError(f"'{VAE_NAME}' not currently supported.")

    model_dir = f"test/checkpoints/{VAE_NAME}/{DATASET_NAME}/{class_idx}"
    chkp_name = f"{DATASET_NAME}_{class_idx}class_{DIM_LATENT}components_BEST.pth"
    chkp_path = f"{model_dir}/{chkp_name}"
    vae.load_state_dict(torch.load(chkp_path))
    return vae



    #%%
    # == Example use #1: USE THE MODEL TO COMPRESS and RECONSTRUCT IMAGES
    # random_data_synthesizer(vae, n_digits=3, dim_latent=dim_latent, selection=None)


if __name__=="__main__":
    # save the torch model
    # for CLASS_IDX in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for CLASS_IDX in [0, 1]:

        class_idx = "all" if CLASS_IDX is None else CLASS_IDX

        model_dir = f"test/checkpoints/{VAE_NAME}/{DATASET_NAME}/{class_idx}/"
        confirm_directory(model_dir)
        # run_example(model_dir)

        batch_size = BATCH_SIZE
        image_dim = IMAGE_DIMS[DATASET_NAME]
        # dataloaders
        train_loader, test_loader = create_dataloaders(
            DATASET_NAME,
            batch_size=batch_size,
            input_size=image_dim,
            num_workers=num_workers,
            class_idx=CLASS_IDX,
        )

        batch, labels = iter(train_loader).next()

        # reconstruction: (d)epth, (w)idth, (h)eight
        d = batch.shape[1]
        w = batch.shape[2]
        h = batch.shape[3]
        x_dim = w * h

        #%%
        pth_directory = f"{model_dir}temps/"
        confirm_directory(pth_directory)
        pth_paths = sorted([os.path.join(pth_directory, file) for file in os.listdir(pth_directory) if file.endswith(".pth")])[:1]

        train_unique_labels, train_label_counts = torch.unique(torch.as_tensor(train_loader.dataset.targets), return_counts=True)
        test_unique_labels, test_label_counts = torch.unique(torch.as_tensor(test_loader.dataset.targets), return_counts=True)

        if USE_PRETRAINED:
            vae = load_existing_vae(class_idx)
        else:
            vae = fit_vae(train_loader, train_loader, class_idx)
            # select to train and derive synthetic samples from train or validation datasets
        if DATA_SRC == "train":
            unique_labels, label_counts = train_unique_labels, train_label_counts
        elif DATA_SRC == "val":
            # vae = fit_vae(test_loader, train_loader, class_idx)
            unique_labels, label_counts = test_unique_labels, test_label_counts
        else:
            raise NotImplementedError(f"The '{DATA_SRC}' is not currently supported")

        # #%% ALTERNATIVELY extract samples without predefining the tensors (mu and log_var)
        # for pth_path in pth_paths:
        #     n = pth_path.split("/")[-1].split("_")[-1].split(".")[0]
        #     vae.load_state_dict(torch.load(pth_path))

        #%% Create Synthetic samples
        directory = f"test/data/{VAE_NAME}/{DATASET_NAME}/{DATA_SRC}/{class_idx}/"
        confirm_directory(directory)

        # unique labels and their frequency counts
        n_samples = label_counts[CLASS_IDX].item()
        label = unique_labels[class_idx].item()

        vae.cuda()
        vae.eval()
        with torch.no_grad():
            samples = vae.sample(n_samples, d, seed=0)
        tensor_image = samples.view(-1, d, h, w).cpu()

        for i, img in enumerate(tensor_image):
            img_dst_path = f"{directory}sample_{i}.png"

            save_image(img, img_dst_path, normalize=True)

        if DISP:
            plot_title = f"Random-sampled batch {DATASET_NAME} dataset"
            visualize_batch(tensor_image.detach(), plot_title=plot_title, grid_dims=8)

            # save_path = f"{directory}grid_{i}.png"
            # save_image(tensor_image.cpu(), save_path)


