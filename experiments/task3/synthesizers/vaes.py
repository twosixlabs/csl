#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:26:22 2020

Sythesizers
@author: carlos-torres
"""
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.nearest_embed import NearestEmbed


class AbstractAutoEncoder(nn.Module):
    """ Prototype.
        https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sampling(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class VAE(nn.Module):
    """Varaitional Auto-Encoder
    refs:
        https://github.com/bvezilic/Variational-autoencoder
        https://github.com/topics/mnist-generation
    """

    def __init__(self, x_dim: int, h_dim1: int = 512, h_dim2: int = 256, z_dim: int = 2):
        super(VAE, self).__init__()
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
        return torch.sigmoid(self.fc6(h))
        # return torch.tanh(self.fc6(h))

    def forward(self, x) -> (float, float, float):
        mu, log_var = self.encode(x.view(-1, self.x_dim))
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def sample(self, n_samples: int = 1, n_channels: int = 1, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        bce = F.binary_cross_entropy(recon_x, x.view(-1, recon_x.shape[1]), reduction="sum")
        # bce = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, recon_x.shape[1]), reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return bce + eta * kld

    def latest_losses(self):
        return {'bce': self.bce, 'kl': self.kld}


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)




class CVAE(nn.Module):
    """
    https://github.com/coolvision/vae_conv/blob/master/mvae_conv_model.py
    """
    def __init__(self, x_dim: int, nz: int, nc: int, ngf: int = 64, ndf: int = 64):
        super(CVAE, self).__init__()

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
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
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
        # batch_size, d, h, w = x.shape
        # self.hw = h * w  # = 1024
        conv = self.encoder(x);
        # print("encode conv", conv.size())
        # h1 = self.fc1(conv.view(-1, 1024))
        h1 = self.fc1(conv.view(-1, self.w * self.h))
        # h1 = self.fc1(conv.view(-1, self.hw))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1, self.x_dim, 1, 1)
        # print("deconv_input", deconv_input.size())
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

        # print("x", x.size())
        mu, logvar = self.encode(x)
        # print("mu, logvar", mu.size(), logvar.size())
        z = self.reparameterize(mu, logvar)
        # print("z", z.size())
        # self.decoded = self.decode(z)
        # print("decoded", decoded.size())
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
        # import pdb; pdb.set_trace()
        self.mse = F.mse_loss(recon_x, x)
        # batch_size = x.size(0)
        # d = x.size(1)
        # h = x.size(2)
        # w = x.size(3)
        # batch_size, d, h, w = x.shape
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        self.kl_loss /= self.b * self.d * self.w * self.h

        # return mse
        return self.mse + eta * self.kl_loss


# class CVAE(nn.Module):
#     """
#     https://github.com/coolvision/vae_conv/blob/master/mvae_conv_model.py
#     """
#     def __init__(self, nz: int, nc: int, ngf: int = 64, ndf: int = 64):
#         super(CVAE, self).__init__()

#         self.have_cuda = False
#         self.nz = nz
#         self.ngf = ngf
#         self.ndf = ndf
#         self.nc = nc  # num_channels from, input image

#         self.encoder = nn.Sequential(
#             # input is (nc) x 28 x 28
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 14 x 14
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 7 x 7
#             nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 4 x 4
#             nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
#             # nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             # nn.Sigmoid()
#         )

#         self.decoder = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
#             # nn.BatchNorm2d(ngf),
#             # nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
#             # nn.Tanh()
#             nn.Sigmoid()
#             # state size. (nc) x 64 x 64
#         )

#         self.fc1 = nn.Linear(1024, 512)
#         self.fc21 = nn.Linear(512, nz)
#         self.fc22 = nn.Linear(512, nz)

#         self.fc3 = nn.Linear(nz, 512)
#         self.fc4 = nn.Linear(512, 1024)

#         self.lrelu = nn.LeakyReLU()
#         self.relu = nn.ReLU()
#         # self.sigmoid = nn.Sigmoid()

#     def encode(self, x):
#         # batch_size, d, h, w = x.shape
#         # self.hw = h * w  # = 1024
#         conv = self.encoder(x);
#         # print("encode conv", conv.size())
#         # h1 = self.fc1(conv.view(-1, 1024))
#         h1 = self.fc1(conv.view(-1, self.w * self.h))
#         # h1 = self.fc1(conv.view(-1, self.hw))
#         # print("encode h1", h1.size())
#         return self.fc21(h1), self.fc22(h1)

#     def decode(self, z):
#         h3 = self.relu(self.fc3(z))
#         deconv_input = self.fc4(h3)
#         # print("deconv_input", deconv_input.size())
#         deconv_input = deconv_input.view(-1, 1024, 1, 1)
#         # print("deconv_input", deconv_input.size())
#         return self.decoder(deconv_input)

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = std.new(std.size()).normal_()
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def forward(self, x):
#         # batch_size, channels, height, width
#         self.b, self.d, self.h, self.w = x.shape

#         # print("x", x.size())
#         mu, logvar = self.encode(x)
#         # print("mu, logvar", mu.size(), logvar.size())
#         z = self.reparameterize(mu, logvar)
#         # print("z", z.size())
#         decoded = self.decode(z)
#         # print("decoded", decoded.size())
#         return decoded, mu, logvar

#     def sample(self, n_samples: int = 1, n_channels: int = 1, seed: int = None):
#         if seed is not None:
#             torch.manual_seed(seed)
#         sample = torch.randn(n_samples * n_channels, self.nz)
#         if self.cuda():
#             sample = sample.cuda()
#         sample = self.decode(sample).cpu()
#         return sample

#     def loss_function(self, x, recon_x, mu, logvar, eta=0.1):
#         import pdb; pdb.set_trace()
#         self.mse = F.mse_loss(recon_x, x)
#         # batch_size = x.size(0)
#         # d = x.size(1)
#         # h = x.size(2)
#         # w = x.size(3)
#         batch_size, d, h, w = x.shape
#         # see Appendix B from VAE paper:
#         # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#         # https://arxiv.org/abs/1312.6114
#         # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#         self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         # Normalise by same number of elements as in reconstruction
#         self.kl_loss /= batch_size * d * w * h

#         # return mse
#         return self.mse + eta * self.kl_loss



class VQ_CVAE(nn.Module):
    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
        super(VQ_CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn),
            nn.BatchNorm2d(d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        self.d = d
        self.emb = NearestEmbed(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb, argmin

    def sample(self, size, seed: int = None):

        sample = torch.randn(size, self.d, self.f, self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        self.commit_loss = torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))

        return self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}

    # def print_atom_hist(self, argmin):

    #     argmin = argmin.detach().cpu().numpy()
    #     unique, counts = np.unique(argmin, return_counts=True)
    #     logging.info(counts)
    #     logging.info(unique)