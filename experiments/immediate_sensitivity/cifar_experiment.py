import numpy as np
import torch
import pickle
import experiment_runner as er
from torch import nn
import math

import torch.nn.functional as F


import torchvision.datasets as datasets
import torchvision

from sklearn.model_selection import train_test_split


cifar_train = datasets.CIFAR100(root='../../inputs', train=True, download=True, transform=torchvision.transforms.ToTensor()) #target_transform=one_hot_label)
cifar_test = datasets.CIFAR100(root='../../inputs', train=False, download=True, transform=torchvision.transforms.ToTensor())

#_, cifar_train = train_test_split(cifar_train,test_size=.18, shuffle=True)


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



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
        
        
        
        

    print("test 3")
    
    
epsilons = [0, 1000, 10000, 50000, 100000]
throw_outs = [0]# , 1, 1.5, 2, 2.5, 3]
batch_size = 64

for e in epsilons:
    for t in throw_outs:
        print(f"model:, {e}, {t}, {batch_size} begin")
        model = DenseNet3(40, 100)
        info, mode = er.run_experiment(model,
                                        cifar_train,
                                        cifar_test,
                                        epsilon=e,
                                        alpha=2,
                                        epochs=20,
                                        add_noise=True,
                                        throw_out_threshold=False,
                                        throw_out_std=t,
                                        batch_size=batch_size,
                                        lf=torch.nn.CrossEntropyLoss,
                                        print_rate=1)
        pickle.dump(info, open(f"../../data/cifar/cifar_f_{e}_{t}_{batch_size}.b", 'wb'))
