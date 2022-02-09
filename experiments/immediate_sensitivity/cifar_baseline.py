import numpy as np
import torch
import pickle
import experiment_runner as er
from torch import nn
import math

import torch.nn.functional as F
import cifar_classifier as cc

import torchvision.datasets as datasets
import torchvision

from sklearn.model_selection import train_test_split


cifar_train = datasets.CIFAR10(root='../../inputs', train=True, download=True, transform=torchvision.transforms.ToTensor()) #target_transform=one_hot_label)
cifar_test = datasets.CIFAR10(root='../../inputs', train=False, download=True, transform=torchvision.transforms.ToTensor())

_, cifar_train = train_test_split(cifar_train,test_size=.18, shuffle=True, random_state=7)


    
epsilons = [0, 1000000, 10000000, 100000000]
epsilons = [1000000, 100000, 10000, 1000, 100, 10, 1]
epsilons = [100, 10, 1]
#epsilons = [1000000000, 10000000000, 100000000000] + epsilons
#epsilons = [0]
#throw_outs = [1, 10, 20, 100] # , 1, 1.5, 2, 2.5, 3]

throw_outs = [5, 100, 10, 1] #, .5, 1]
batch_sizes = [128] #, 256, 512]
for e in epsilons:
    for t in throw_outs:
        for batch_size in batch_sizes:
            infos = []
            for i in range(5):
                print(f"model:, {e}, {t}, {batch_size} begin")
                model = cc.Cifar_Classifier()
                info = er.opacus_experiment(model,
                                                cifar_train,
                                                cifar_test,
                                                epsilon=e,
                                                alpha=2,
                                                epochs=50,
                                                add_noise=True,
                                                C = t,
                                                batch_size=batch_size,
                                                lf=torch.nn.NLLLoss,
                                                print_rate=1,
                                                idx=i)
                infos.append(info)
            pickle.dump(infos, open(f"../../data/cifar/cifar_mb_{e}_{t}_{batch_size}.b", 'wb'))
