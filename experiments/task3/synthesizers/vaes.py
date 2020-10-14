#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 12:26:22 2020

Sythesizers
@author: carlos-torres
"""
import torch
import torch.nn as nn
import torch.nn.functional as F



class VAE(nn.Module):
    """Varaitional Auto-Encoder
    refs:
        https://github.com/bvezilic/Variational-autoencoder
        https://github.com/topics/mnist-generation
    """

    def __init__(self, x_dim: int, h_dim1: int = 512, h_dim2: int = 256, z_dim: int = 2):
        super(VAE, self).__init__()
        self.x_dim = x_dim

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
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var