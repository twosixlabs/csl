#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Autoencoder Sythesizers:
  VAE
  CVAE

TODO: VQ-VAE
    Needs latent-prior training
"""
import os
import random
import math
import numpy as np
from tqdm import tqdm
import logging
import coloredlogs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.optim as optim
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
coloredlogs.install(level="info", logger=log)

# Set random seed for reproducibility
MANUAL_SEED = 999

# manualSeed = random.randint(1, 10000) # use if you want new results
if MANUAL_SEED is not None:
    log.debug("Setting The Random Seed: ", MANUAL_SEED)
    random.seed(MANUAL_SEED)
    torch.manual_seed(MANUAL_SEED)

# Decide which device we want to run on
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VariationalAutoEncoder(nn.Module):
    """ Prototype.
    """

    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()

    def encode(self, x):
        return

    def decode(self, z):
        return

    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    def sample(self, n_samples: int, n_channels: int, seed: int):
        """sample new tansors from model"""
        return

    def generate_images(self, **kwargs):
        """sample new images from model"""
        return

    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as
        returned from forward()"""
        return

    def latest_losses(self):
        """returns the latest losses in a dictionary.
        Useful for logging."""
        return


class VAE_ARCHITECTURE(nn.Module):
    """Varaitional Auto-Encoder
    refs:
        https://github.com/bvezilic/Variational-autoencoder
        https://github.com/topics/mnist-generation
    """

    def __init__(
        self, x_dim: int, h_dim1: int = 512, h_dim2: int = 256, z_dim: int = 2
    ):
        super(VAE_ARCHITECTURE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        # encoder
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)

        # decoder
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encode(self, x) -> (float, float):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # -> (mu, log_var)

    def reparameterize(self, mu, log_var) -> float:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # -> z-latent

    def decode(self, z) -> float:
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        # return torch.tanh(self.fc6(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x) -> (float, float, float):
        mu, log_var = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def sample(self, n_samples: int = 1, n_channels: int = 1, seed: int = None):
        """Helper.
        Return a vae generated tensor represenative of synthetic images. The tensor
        requires reshaping.
        """
        if seed is not None:
            torch.manual_seed(seed)
        sample = torch.randn(n_samples * n_channels, self.z_dim)
        if self.cuda():
            sample = (sample.cuda() + 1) / 2
        sample = self.decode(sample).cpu()
        return sample

    def loss_function(self, recon_x, x, mu, log_var, eta=0.1) -> float:
        """
        bce + kld
            Compute recontruction error via binary cross-entropy (bce) &
            KL divergence (kld).
        """
        bce = F.binary_cross_entropy(
            recon_x, x.view(-1, recon_x.shape[1]), reduction="sum"
        )
        # bce = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, recon_x.shape[1]), reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return bce + eta * kld

    def latest_losses(self):
        return {"bce": self.bce, "kl": self.kld}


class CVAE_ARCHITECTURE(nn.Module):
    """
    https://github.com/coolvision/vae_conv/blob/master/mvae_conv_model.py
    """

    def __init__(
        self,
        x_dim: int = None,
        nz: int = 2,
        nc: int = None,
        ngf: int = 64,
        ndf: int = 64,
    ):
        super(CVAE_ARCHITECTURE, self).__init__()

        self.have_cuda = False
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc  # num_channels from, input image
        self.x_dim = x_dim

        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, self.x_dim, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.x_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

        self.fc1 = nn.Linear(self.x_dim, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, self.x_dim)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x)
        h1 = self.fc1(conv.view(-1, self.w * self.h))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(-1, self.x_dim, 1, 1)
        return self.decoder(deconv_input)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        # batch_size, channels, height, width
        self.b, self.d, self.h, self.w = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, n_samples: int = 1, n_channels: int = 1, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        # sample = torch.randn(n_samples * n_channels, self.nz)
        sample = torch.randn(n_samples, self.nz)
        if self.cuda():
            sample = sample.cuda()
        sample = self.decode(sample).cpu()
        return sample

    def loss_function(self, x, recon_x, mu, logvar, eta=0.1):
        self.mse = F.mse_loss(recon_x, x)
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalize by same number of elements as in reconstruction
        self.kl_loss /= self.b * self.d * self.w * self.h
        return self.mse + eta * self.kl_loss

    def generate_images(
        self,
        n_samples: int = 1,
        image_height: int = None,
        image_width: int = None,
        image_channels: int = 1,
        seed: int = None,
        save_path: os.PathLike = None,
    ):
        """sample and generate new images from generator model"""
        mu = Variable(torch.randn(n_samples * image_channels, self.z_dim))
        log_var = Variable(torch.randn(n_samples * image_channels, self.z_dim))

        with torch.no_grad():
            # vae.cpu()
            z = self.reparemetrize(mu, log_var)
            z = z.cuda()

            samples = self.sample(n_samples, image_channels, seed=seed)
            samples = self.decode(z)
            tensor_image = samples.view(
                -1, image_channels, image_height, image_width
            ).cpu()

        if save_path:
            if not os.path.isdir(save_path):
                log.info(
                    f"The given directory '{save_path}' does not exist. Creating it!"
                )
                os.makedirs(save_path)
            for image_counter, img in enumerate(tensor_image):
                img_dst_path = f"{save_path}sample_{image_counter}.png"
                vutils.save_image(img, img_dst_path, normalize=True)
        log.info(f"Generated {image_counter} samples")


class VAE(object):
    """Object.
    wraps around VAE_ARCHITECTURE to enable convenient training, testing, and
    synthetic sample generation.
    """

    def __init__(self):
        self.model = None
        self.h_dim1 = 512
        self.h_dim2 = 256
        self.fidelity = 0.5  # range: (0.0, 1.0]
        self.z_dim = None
        # self._set_vae()

    def _train_one_epoch(self, epoch, train_loader):
        # set to train mode (enable grad)
        self.model.train()
        train_loss = 0
        ttotal = len(train_loader)
        # for batch_idx, (data, _) in enumerate(train_loader):
        for batch_idx, (data, _) in zip(
            tqdm(range(ttotal), desc=f"Train Batch {epoch}"), train_loader,
        ):
            # for batch_idx, (data, _) in enumerate(train_loader):
            data = data.cuda()
            self.optimizer.zero_grad()

            recon_batch, mu, log_var = self.model(data)
            loss = self.model.loss_function(recon_batch, data, mu, log_var)
            loss.backward()
            x = loss.item()
            if np.isnan(x):
                log.debug(
                    f"Epoch (before adj): {epoch}; batch_idx: {batch_idx} loss: {loss}"
                )
                # x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
                x = 0
                log.debug(
                    f"    epoch (adfter adj): {epoch}; batch_idx: {batch_idx} loss: {loss}"
                )
            train_loss += x

            self.optimizer.step()

            if batch_idx % 100 == 0:
                log.debug(
                    f"Train Epoch: {epoch} "
                    f"[{batch_idx * len(data)} / {len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]"
                    f"\t loss: {loss.item() / len(data):.6f}"
                )
        epoch_loss = train_loss / len(train_loader.dataset)
        log.info(f"====> Epoch: {epoch} Average Train loss: {epoch_loss: .4f}")
        return epoch_loss

    def test(self, test_loader):
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
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                # data = data.cuda()
                data = data.cuda()
                recon, mu, log_var = self.model(data)

                # sum up batch loss
                test_loss += self.model.loss_function(recon, data, mu, log_var).item()

        test_loss /= len(test_loader.dataset)
        log.info(f"====> Test set loss: {test_loss: .4f}\n")
        return test_loss

    def _set_model(self):
        self.model = VAE_ARCHITECTURE(
            x_dim=self.x_dim, h_dim1=self.h_dim1, h_dim2=self.h_dim2, z_dim=self.z_dim
        )

    def train(self, train_loader, test_loader, num_epochs: int = 3):
        """Convenience.
        Training loop that runs and produces a trained model.
        """
        # training epochs
        self.num_epochs = num_epochs
        # Learning rate for optimizers
        self.lr = 2e-4
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True)

        batch, labels = iter(train_loader).next()

        # reconstruction: (nc)hannels, (w)idth, (h)eight
        self.nc = batch.shape[1]
        self.w = batch.shape[2]
        self.h = batch.shape[3]
        self.x_dim = self.w * self.h
        self.z_dim = self.x_dim * self.fidelity

        # setup the model
        self._set_model()

        # weight initialization
        self.model.apply(init_kaiming)  # init_xavier,
        self.model = self.model.to(DEVICE)

        self.train_hist = []
        self.test_hist = []

        for epoch in range(1, self.num_epochs + 1):
            self.train_loss = self._train_one_epoch(self, epoch, train_loader)
            self.val_loss = self.test(test_loader)
            self.train_hist.append(self.train_loss)
            self.val_hist.append(self.val_loss)

            if epoch == 1:
                self.best_model = self.model
                self.best_loss = self.val_loss
            else:
                if self.val_loss < self.best_loss:
                    print(
                        f"\n\n ===> {epoch}-epoch. Updating best (with "
                        f"{self.val_loss:.3f}), which is less than previous "
                        f"({self.best_loss:.3f}) best_loss"
                    )
                    self.best_model = self.model
                    self.best_epoch = epoch
                    self.best_loss = self.val_loss
        log.info(
            f">>> Completed training and evaluation of '{self.model}'. "
            "Ready to generate samples."
        )

        return self

    def plot_losses(self, ptitle: str = None, save_plot_dir: os.PathLike = None):
        # plot the losses: train & val
        plt.plot(
            range(1, self.num_epochs + 1), self.train_hist, label="Trained", marker="x"
        )
        plt.plot(
            range(1, self.num_epochs + 1), self.val_hist, label="Validated", marker="."
        )

        # Add Figure information
        if ptitle is None:
            plt.title(
                f"Loss as a function Epochs (best epoch={self.best_epoch}, "
                f"loss={self.best_loss:.3f})"
            )
        else:
            plt.title(ptitle)

        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        # plt.ylim((0, 1.))
        plt.xticks(np.arange(1, self.num_epochs + 1, 1.0))
        plt.legend()
        if save_plot_dir is not None:
            plot_name = f"{save_plot_dir}{ptitle}_{self.z_dim}components.png"
            plt.savefig(plot_name)
        plt.show()

    def save_model(self, model_path: os.PathLike = "temp/models/"):
        """Helper.
        Saves the generator and the discriminator.
        """
        self.model_path = f"{model_path}vae_{self.z_dim}components_BEST.pth"
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self, model_path: os.PathLike = "temp/models/"):
        """Helper.
        Loads the generator and the discriminator.
        """
        # set the modelarchitecture

        self.model = VAE(
            x_dim=self.x_dim, h_dim1=self.h_dim1, h_dim2=self.h_dim2, z_dim=self.z_dim
        )

        # self.model_path = f"{model_save_path}_{self.z_dim}components_BEST_vae.pth"
        self.model_path = f"{model_path}vae_{self.z_dim}components_BEST.pth"
        return self

    def generate_images(
        self, num_samples: int, save_directory: os.PathLike, seed: int = 0
    ):
        """
        # confirm to ImageFolder structure:
        save_directory = f"/data/{SYNTHESIZER_NAME}/{DATASET_NAME}/{DATA_SRC}/{class_idx}/"
        example:
        save_directory = f"/data/mnist_vae/train/0/"
        save_directory = f"/data/mnist_vae/val/0/"
        """
        with torch.no_grad():
            samples = self.model.sample(num_samples, self.nc, seed=seed)
        tensor_image = samples.view(-1, self.nc, self.h, self.w).cpu()

        # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        for i, img in enumerate(tensor_image):
            img_dst_path = f"{save_directory}sample_{i}.png"
            vutils.save_image(img, img_dst_path, normalize=True)


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
            torch.randn(weights[1].shape[0], weights[1].shape[1])
            * math.sqrt(2.0 / weights[1].shape[0])
        )
        m.bias.data.fill_(0)


class CVAE(VAE):
    """Object.
    Inherits from VAE and wraps around CVAE_ARCHITECTURE to enable convenient
    training, testing, and synthetic sample generation.
    """

    def __init__(self):
        self.model = None
        # variables set in .train()
        self.x_dim = None
        self.nc = None
        self.z_dim = None
        # architecture defaults
        self.ngf = 64
        self.ndf = 64
        self.h_dim2 = 256
        self.fidelity = 0.5  # range: (0.0, 1.0]

    def _set_model(self):
        self.model = CVAE_ARCHITECTURE(
            x_dim=self.x_dim, nz=self.z_dim, nc=self.nc, ngf=self.ngf, ndf=self.ndf
        )


if __name__ == "__main__":
    vae = VAE()
    cvae = CVAE()
