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
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math

from utils.protools import create_dataloaders, confirm_directory

import logging
import coloredlogs
from tqdm import tqdm

log = logging.getLogger("vae_synth")
coloredlogs.install(level="info", logger=log)
DEVICE = "cuda:0"  # "cpu"
torch.cuda.set_device(0)

# DATASETS AND DIMENSIONS
IMAGE_DIMS = {
    "imagenet": 32,  # 224
    "imagenette": 32,  # 224
    "mnist": 28,
    "fashion-mnist": 28,
    "cifar10": 32,
    "stl10": 32,
}

# THE DATASET
DATASET_NAME = "cifar10"
BATCH_SIZE = 64
CLASS_IDX = 0

# VAE PARAMS
DIM_LATENT = 204
N_EPOCHS = 15
H_DIM1 = 512
H_DIM2 = 256

# save the torch model
MODEL_DIR = f"test/models/vae/temps/{DATASET_NAME}/{CLASS_IDX}"

class VAE(nn.Module):

    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)

        # decoder
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x) -> (float, float):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # -> (mu, log_var)

    def sampling(self, mu, log_var) -> float:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # -> z sample

    def decoder(self, z) -> float:
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x) -> (float, float, float):
        mu, log_var = self.encoder(x.view(-1, x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def loss_function(recon_x, x, mu, log_var, eta=0.1) -> float:
    """
    Parameters
    ----------
    recon_x : pth.tensor
        reconstructed data.
    x : pth.tensor
        source data.
    mu : pth.tensor
        estimated distribution mean.
    log_var : TYPE
        estimated distribution log_variance.

    Returns
    -------
    bce + kld
        Compute recontruction error (bce) & KL divergence (kld).
    """
    # import pdb; pdb.set_trace()
    bce = F.binary_cross_entropy(recon_x, x.view(-1, recon_x.shape[1]), reduction="sum")
    # bce = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, recon_x.shape[1]), reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + eta * kld

def train(vae, epoch, train_loader):
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
    for batch_idx, (data, _) in zip(tqdm(range(ttotal), desc=f"Train Batch {epoch}"), train_loader):
    # for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
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
            log.info(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)} / {len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]"
                f"\t loss: {loss.item() / len(data):.6f}"
            )
    epoch_loss = train_loss / len(train_loader.dataset)
    log.info(
        f"====> Epoch: {epoch} Average loss: {epoch_loss: .4f}"
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
            test_loss += loss_function(recon, data, mu, log_var).item()

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
            if disp:
                visualize_batch(batch, plot_title=f"Visualized {b}-Batch", grid_dims=4)
            batch = batch.cuda()
            recon, mean, log_var = vae(batch)

            batch_latents.append([mean, log_var])
            z = vae.sampling(mean, log_var)
            reconstructed_batch = vae.decoder(z).cuda()

            # create original and reconstructed image-grids for
            d = batch.shape[1]
            w = batch.shape[2]
            h = batch.shape[3]
            orig_img = batch.view(len(batch), d, w, h)
            recons_img = reconstructed_batch.view(len(batch), d, w, h)

            if save:
                directory = f"test/data/{dataset_name}/{CLASS_IDX}/grid_batches/"
                confirm_directory(directory)
                log.info(f"Synthesizing the {b}-th random-sampled batch")
                save_image(orig_img, f'{directory}orig_grid_sample_{b}.png')
                save_image(recons_img, f'{directory}recons_grid_sample_{b}.png')

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


def reconstruct(vae, batch_latents: list, d: int, w: int, h: int,
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
                log.info(
                    f"Synthesizing the ({j}-th item from the {b}-th batch "
                    f"of {len(batch[0])} batches) {sample_counter}-sample "
                    f"for '{label}' label"
                )
                mu = mu.cuda()
                log_var.cuda()
                z = vae.sampling(mu, log_var)

                sample = vae.decoder(z)
                image = sample.view(d, w, h)

                if save:
                    # directory=f"test/data/{dataset_name}/{CLASS_IDX}/{task}/{label}/"
                    directory=f"test/data/{dataset_name}/{CLASS_IDX}/samples/"
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


#%%

if __name__=="__main__":
    num_workers = 4
    batch_size = BATCH_SIZE
    confirm_directory(MODEL_DIR)
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

    # == THE VAE-MODEL
    # FITTING/TRAINNG
    vae = VAE(x_dim=x_dim, h_dim1=H_DIM1, h_dim2=H_DIM2, z_dim=DIM_LATENT)
    # weight initialization
    vae.apply(init_kaiming)  # init_xavier,
    vae = vae.cuda()
    # print(vae)

    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    # optimizer = optim.RMSprop(vae.parameters(), lr=0.001, centered=True)
    # optimizer = optim.SGD(vae.parameters(), lr=0.01, momentum=0.9)
    train_hist = []
    val_hist = []
    for epoch in range(1, N_EPOCHS + 1):
        vae, train_loss = train(vae, epoch, train_loader)
        val_loss = test(vae, test_loader)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if epoch % 10 == 0:
            MODEL_PATH = f"{MODEL_DIR}{DATASET_NAME}_{DIM_LATENT}components_{epoch}epoch.pth"
            torch.save(vae.state_dict(), MODEL_PATH)
    MODEL_DIR = "test/models/vae/"
    MODEL_PATH = f"{MODEL_DIR}{DATASET_NAME}_{CLASS_IDX}class_{DIM_LATENT}components_{epoch}epoch-FINAL.pth"
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

    # == USE THE MODEL TO RECONSTRUCT MAGES
    # random_data_synthesizer(vae, n_digits=3, dim_latent=dim_latent, selection=None)
    means, variances, data_labels, batch_latents, z = encode_data(
        vae,
        test_loader,
        DIM_LATENT,
        DATASET_NAME,
        save=True,
        disp=False
    )
    #%%
    # test with small sample of size n_digits
    n_test_samples = 10
    subset_means = means[:n_test_samples]  # assumes single channel images
    subset_variances = variances[:n_test_samples]
    subset_labels = data_labels[:n_test_samples]
    # decode_data(vae, subset_means, subset_variances, subset_labels, d, w, h, DATASET_NAME, save=True, disp=True)
    # alternatively do all
    #%% decode_data(vae, means, variances, data_labels, w, h, save=True, disp=True)
    reconstruct(vae, batch_latents, d, w, h, DATASET_NAME, save=True, disp=False, n_samples=None)

    #%%
    # means.shape = (len(dataset), DIM_LATENT)
    # variances.shape = (len(dataset), DIM_LATENT)

    starts = [r for r in range(0, len(subset_means) - d, d)]
    ends = [r + d for r in range(0, len(subset_variances) - d, d)]

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

    #%%
    sample = sample.detach()
    image_tensor = sample.view(d, w, h).cpu()
    show(image_tensor, plot_title="Reconstructed sample.")
