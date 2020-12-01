#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generative Adversarial Net Sythesizers:
  DCGAN

TODO: BIG-GAN
PENDING: STYLE-GAN
"""
import os
import torch
import random
import logging
import coloredlogs

import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.parallel
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


# Generator code: DCGAN
class Generator(nn.Module):
    def __init__(
        self,
        ngpu: int,
        nc: int,  # 3: color images; 1: grayscale
        nz: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        beta1: float = 0.5,
        bias: bool = False,
    ):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Discriminator code: DCGAN
class Discriminator(nn.Module):
    def __init__(
        self,
        ngpu: int,
        nc: int,  # 3: color images; 1: grayscale
        nz: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        beta1: float = 0.5,
        bias: bool = False,
    ):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class DCGAN_ARCHITECTURE:
    def __init__(self, ngpu: int = 1, nc: int = None, lr: float = 2e-4):
        super(DCGAN_ARCHITECTURE, self).__init__()
        # # Spatial size of training images. All images will be resized to this
        # #   size using a transformer.
        # image_size = 64
        #
        # # Number of channels in the training images. For color images this is 3
        self.nc = nc
        #
        # Number of training epochs
        # self.num_epochs = None
        #
        self.lr = lr
        # Number of GPUs available. Use 0 for CPU mode.
        self.ngpu = ngpu

        # Size of z latent vector (i.e. size of generator input)
        self.nz = 100
        # Size of feature maps in generator
        self.ngf = 64
        # Size of feature maps in discriminator
        self.ndf = 64
        #
        # set the generator
        self._set_generator()
        #
        # set the disriminator
        self._set_discriminator()

    def _set_generator(self):
        # the generator
        self.netG = Generator(self.ngpu).to(DEVICE)
        # Handle multi-gpu if desired
        if (DEVICE.type == "cuda") and (self.ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))
        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netG.apply(weights_init)

    def _set_discriminator(self):
        # the Discriminator
        self.netD = Discriminator(self.ngpu).to(DEVICE)
        # Handle multi-gpu if desired
        if (DEVICE.type == "cuda") and (self.ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))
        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netD.apply(weights_init)


class DCGAN(object):
    """Object.
    wraps around DCGAN_ARCHITECTURE to enable convenient training, testing, and
    synthetic sample generation.
    """

    def __init__(self, ngpu: int = 1):
        self.nc = None
        self.ngpu = 1
        self.lr = 2e-4

    def _set_model(self):
        self.model = DCGAN_ARCHITECTURE(nc=self.nc, lr=self.lr, ngpu=self.ngpu)

    def _train_one_epoch(self, epoch, train_loader):
        for i, data in enumerate(train_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            self.model.netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full(
                (b_size,), self.real_label, dtype=torch.float, device=DEVICE
            )
            # Forward pass real batch through D
            output = self.modelnetD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, self.nz, 1, 1, device=DEVICE)
            # Generate fake image batch with G
            fake = self.model.netG(noise)
            label.fill_(self.fake_label)
            # Classify all fake batch with D
            output = self.model.netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.model.netG.zero_grad()
            label.fill_(self.real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.model.netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            self.optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                log.info(
                    f"Epoch: [{epoch}/{self.num_epochs}]:: "
                    f"iter: [{i}/{len(train_loader)}]"
                    f"\tLoss_D: {errD.item():.4f} \tLoss_G: "
                    f"{errG.item():.4f}"
                    f"\tD(x): {D_x:.4f} "
                    f"\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
                )
            # Save Losses for plotting later
            self.G_losses.append(errG.item())
            self.D_losses.append(errD.item())

            if (i == 0) and (epoch == 0):
                self.best_iter = 0
                self.best_epoch = epoch
                self.best_netG = self.model.netG
                best_G_loss = self.G_losses[0]
                self.best_netD = self.model.netD
                best_D_loss = self.D_losses[0]
            else:
                # keep the best Generator-Discriminator Pair
                if (errG.item() < best_G_loss) and (errD.item() < best_D_loss):
                    self.best_iter = i
                    self.best_epoch = epoch
                    self.best_netG = self.model.netG
                    best_G_loss = errG.item()
                    self.best_netD = self.model.netD
                    best_D_loss = errD.item()

            # Check how the generator is doing by saving G's output on
            # fixed_noise image_list tracks a visual/image progression of
            # the generator.
            if (self.iters % 500 == 0) or (
                (epoch == self.num_epochs - 1) and (i == len(train_loader) - 1)
            ):
                with torch.no_grad():
                    fake = self.best_netG(self.fixed_noise).detach().cpu()
                self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            self.iters += 1
        # finished the training and update

    def train(self, train_loader, num_epochs: int = 3):
        """Convenience methog. Trains the model's generator and discriminator.
        """
        batch, labels = iter(train_loader).next()

        # reconstruction: (nc)hannels, (w)idth, (h)eight
        self.nc = batch.shape[1]
        self.w = batch.shape[2]
        self.h = batch.shape[3]
        # self.x_dim = self.w * self.h
        # self.z_dim = self.x_dim * self.fidelity

        # training epochs
        self.num_epochs = num_epochs
        # Learning rate for optimizers
        self.lr = 0.0002
        # Beta1 hyperparam for Adam optimizers
        self.beta1 = 0.5
        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Establish convention for real and fake labels during training
        self.real_label = 1.0
        self.fake_label = 0.0

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(
            self.model.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        self.optimizerG = optim.Adam(
            self.model.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )

        # Lists to keep track of progress
        self.img_list = []
        self.G_losses = []
        self.D_losses = []

        # num_samples
        num_samples = 64
        self.fixed_noise = torch.randn(num_samples, self.nz, 1, 1, device=DEVICE)

        # training loop
        log.debug("Starting Training Loop...")
        self.iters = 0
        # For each epoch
        for epoch in range(self.num_epochs):
            self._train_one_epoch(epoch, train_loader)
        self.model.netG = self.best_netG
        self.model.netD = self.best_netD

        log.info(
            f"Generator and Discriminator Loss During Training "
            f"(over {self.num_epochs} Epochs; best epoch: {self.best_epoch} "
            f"& best_iter:{self.best_iter})"
        )

    def plot_losses(self, ptitle: str = None, save_plot_dir: os.PathLike = None):
        """Helper.
        Plots the generator and the discriminator losses higlighting the best iteration and best epoch.
        """
        plt.figure(figsize=(10, 5))
        if ptitle is None:
            plt.title(
                f"Generator and Discriminator Loss During Training (over {self.num_epochs} "
                f"Epochs; best epoch: {self.best_epoch} & best_iter:{self.best_iter})"
            )
        else:
            plt.title(ptitle)
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        if save_plot_dir is not None:
            plot_name = f"{save_plot_dir}{ptitle}_{self.z_dim}components.png"
            plt.savefig(plot_name)

    def generate_images(self, num_samples: int, save_directory: os.PathLike):
        """
        # confirm to ImageFolder structure:
        save_directory = f"/data/{SYNTHESIZER_NAME}/{DATASET_NAME}/{DATA_SRC}/{class_idx}/"
        example:
        save_directory = f"/data/mnist_dcgan/train/0/"
        save_directory = f"/data/mnist_dcgan/val/0/"
        """
        fixed_noise = torch.randn(num_samples, self.nz, 1, 1, device=DEVICE)
        with torch.no_grad():
            fake = self.netG(fixed_noise).detach().cpu()

        # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        for i, img in enumerate(fake):
            img_dst_path = f"{save_directory}sample_{i}.png"
            vutils.save_image(img, img_dst_path, normalize=True)

    def save_model(self, model_path: os.PathLike = "temp"):
        """Helper.
        Saves the generator and the discriminator.
        """
        self.netG_path = f"{model_path}_{self.nz}components_BEST_dcgan_G.pth"
        torch.save(self.model.netG.state_dict(), self.netG_path)
        self.netD_path = f"{model_path}_{self.nz}components_BEST_dcgan_D.pth"
        torch.save(self.model.netD.state_dict(), self.netD_path)

    def load_model(self, netG_path: os.PathLike, netD_path: os.PathLike):
        """Helper.
        Loads the generator and the discriminator.
        """
        self._set_model()
        # set the generator and dsicriminator architectures
        self._set_generator()
        self._set_discriminator()
        # self.netG_path = f"{model_save_path}_{self.nz}components_BEST_dcgan_G.pth"
        self.netG_path = netG_path
        self.netD_path = netD_path
        # load the pre-trained existing weights
        self.model.netG.load_state_dict(torch.load(self.netG_path))
        self.model.netD.load_state_dict(torch.load(self.netD_path))
        return self


def weights_init(m):
    """Custom weights initialization called on netG and netD"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    dcgan = DCGAN()
