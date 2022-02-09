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

    
    
epsilons = [0, 1000, 10000, 50000, 100000]
epsilons = epsilons + [1, 10, 100]
#epsilons = epsilons + [25, 50, 75, 500]
#epsilons = [3000]
throw_outs = [0]# , 1, 1.5, 2, 2.5, 3]
batch_size = 128

for e in epsilons:
    for t in throw_outs:
        infos = []
        for i in range(5):
            print(f"model:, {e}, {t}, {batch_size} begin")
            model = cc.Cifar_Classifier()
            info, _ = er.run_experiment(model,
                                            cifar_train,
                                            cifar_test,
                                            epsilon=e,
                                            alpha=2,
                                            epochs=50,
                                            add_noise=True,
                                            throw_out_threshold=False,
                                            throw_out_std=t,
                                            batch_size=batch_size,
                                            lf=torch.nn.NLLLoss,
                                            print_rate=1)
            infos.append(info)
        pickle.dump(infos, open(f"../../data/cifar/cifar_m_{e}_{t}_{batch_size}.p", 'wb'))
