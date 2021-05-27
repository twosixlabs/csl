import torch
from torch import nn
import torch.nn.functional as F

class Cifar_Classifier(nn.Module):
    def __init__(self):
        super(Cifar_Classifier, self).__init__()
        self.c1 = nn.Conv2d(3, 32, kernel_size=(3,3))
        #self.bn1 = nn.GroupNorm(8, 32)
        self.c2 = nn.Conv2d(32, 32, kernel_size=(3,3))
        #self.bn2 = nn.GroupNorm(8, 32)
        self.p1 = nn.MaxPool2d((2,2))
            
        self.c3 = nn.Conv2d(32, 64, kernel_size=(3,3))
        #self.bn3 = nn.GroupNorm(16,64)
        self.c4 = nn.Conv2d(64,64, kernel_size=(3,3))
        #self.bn4 = nn.GroupNorm(16, 64)
        self.p2 = nn.MaxPool2d((2,2))
            
        self.c5 = nn.Conv2d(64, 128, kernel_size=(3,3))
        #self.bn5 = nn.GroupNorm(32, 128)
        self.c6 = nn.Conv2d(128,128, kernel_size=(3,3))
        #self.bn6 = nn.GroupNorm(32, 128)
        self.p3 = nn.MaxPool2d((2,2))

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)
        

    def forward(self, x):
        x = self.c1(x)
        x = nn.ReLU()(x)
        #x = self.bn1(x)
        x = self.c2(x)
        x = nn.ReLU()(x)
        #x = self.bn2(x)
        x = self.p1(x)
        
        x = self.c3(x)
        x = nn.ReLU()(x)
        #x = self.bn3(x)
        x = self.c4(x)
        x = nn.ReLU()(x)
        #x = self.bn4(x)
        x = self.p2(x)
        
        x = self.c5(x)
        x = nn.ReLU()(x)
        #x = self.bn5(x)
        x = self.c6(x)
        x = nn.ReLU()(x)
        #x = self.bn6(x)
        #x = self.p3(x)
       
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.log_softmax(x,dim=1)

class Opacus_Classifier(nn.Module):
    def __init__(self):
        super(Opacus_Classifier,self).__init__()
        self.network = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, 10, bias=True),
        )


    def forward(self, x):
        x = self.network(x)
        return torch.log_softmax(x,dim=1)
