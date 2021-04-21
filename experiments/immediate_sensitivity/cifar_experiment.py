import numpy as np
import torch
import pickle
import experiment_runner as er
from torch import nn


import torchvision.datasets as datasets
import torchvision

from sklearn.model_selection import train_test_split


cifar_train = datasets.CIFAR100(root='../../inputs', train=True, download=True, transform=torchvision.transforms.ToTensor()) #target_transform=one_hot_label)
cifar_test = datasets.CIFAR100(root='../../inputs', train=False, download=True, transform=torchvision.transforms.ToTensor())

_, cifar_train = train_test_split(cifar_train,test_size=.18, shuffle=True)


print("test 2")

class Cifar_Classifier(nn.Module):
    def __init__(self):
        super(Cifar_Classifier, self).__init__()
        self.c1 = nn.Conv2d(3, 32, kernel_size=(3,3))
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, kernel_size=(3,3))
        self.bn2 = nn.BatchNorm2d(32)
        self.p1 = nn.MaxPool2d((2,2))
            
        self.c3 = nn.Conv2d(32, 64, kernel_size=(3,3))
        self.bn3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(64,64, kernel_size=(3,3))
        self.bn4 = nn.BatchNorm2d(64)
        self.p2 = nn.MaxPool2d((2,2))
            
        self.c5 = nn.Conv2d(64, 128, kernel_size=(3,3))
        self.bn5 = nn.BatchNorm2d(128)
        self.c6 = nn.Conv2d(128,128, kernel_size=(3,3))
        self.bn6 = nn.BatchNorm2d(128)
        self.p3 = nn.MaxPool2d((2,2))

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 100)
        

    def forward(self, x):
        x = self.c1(x)
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = self.c2(x)
        x = nn.ReLU()(x)
        x = self.bn2(x)
        x = self.p1(x)
        
        x = self.c3(x)
        x = nn.ReLU()(x)
        x = self.bn3(x)
        x = self.c4(x)
        x = nn.ReLU()(x)
        x = self.bn4(x)
        x = self.p2(x)
        
        x = self.c5(x)
        x = nn.ReLU()(x)
        x = self.bn5(x)
        x = self.c6(x)
        x = nn.ReLU()(x)
        x = self.bn6(x)
        #x = self.p3(x)
       
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.log_softmax(x,dim=1)
        
        
        
        

    print("test 3")
    
    
epsilons = [1000, 10000, 50000, 100000]
throw_outs = [0]# , 1, 1.5, 2, 2.5, 3]

for e in epsilons:
    for t in throw_outs:
        print(f"model:, {e}, {t}, {64} begin")
        model = Cifar_Classifier()
        info, mode = er.run_experiment(model,
                                        cifar_train,
                                        cifar_test,
                                        epsilon=e,
                                        alpha=2,
                                        epochs=20,
                                        add_noise=True,
                                        throw_out_threshold=False,
                                        throw_out_std=t,
                                        batch_size=64,
                                        lf=torch.nn.NLLLoss,
                                        print_rate=1)
        pickle.dump(info, open(f"../../data/cifar/cifar_{e}_{t}_{64}.b", 'wb'))
