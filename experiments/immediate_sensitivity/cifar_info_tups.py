import torch
from torch import nn
import pickle
import membership_inference as mi
from experiment_runner import loader_accuracy
import cifar_classifier as cc
from collections import defaultdict
import glob
from os import path
from os import stat
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets

cifar_train = datasets.CIFAR10(root='../../inputs', train=True, download=True, transform=torchvision.transforms.ToTensor()) #target_transform=one_hot_label)
cifar_test = datasets.CIFAR10(root='../../inputs', train=False, download=True, transform=torchvision.transforms.ToTensor())

_, cifar_train = train_test_split(cifar_train,test_size=.18, shuffle=True, random_state=7)



def create_info(model, avg_train_l, train_loader, test_loader):
    with torch.no_grad():
        model.to(torch.cuda.current_device())
        avg_test_acc, avg_test_l = loader_accuracy(model, test_loader, lf=nn.NLLLoss())
        
        tpr = mi.run_yeom_loader(model, avg_train_l, train_loader, lf=nn.NLLLoss)
        fpr = mi.run_yeom_loader(model, avg_train_l, test_loader, lf=nn.NLLLoss)
        adv = tpr-fpr
        return avg_test_acc, avg_test_l, tpr, fpr

def process_model_files(str_epsilon, str_clips, batch, train_losses, train_loader, test_loader):
    info = []
    for j, (losses, eps) in enumerate(train_losses): 
        infos = defaultdict(list)
        clips = float(str_clips)
        epsilon = int(str_epsilon)
        for i, l in enumerate(losses):
            model  = cc.Cifar_Classifier()
            fpath = f'../../data/cifar/cifar_model_{j}_{batch}_{epsilon}_{str_clips}_{i}.torch'
            model.load_state_dict(torch.load(fpath))
            
            test_acc, test_l, tpr, fpr = create_info(model, l, train_loader, test_loader)
            infos['train_l'].append(l) 
            infos['test_l'].append(test_l) 
            infos['acc'].append(test_acc) 
            infos['yeom_tpr'].append(tpr) 
            infos['yeom_fpr'].append(fpr) 
            print(epsilon, i, test_acc, tpr - fpr)
        info.append(infos)
    return info 
        



for f in glob.glob('../../data/cifar/cifar_mb_*.b'):
    fname = f.split('/')[-1][:-2]
    params = fname.split('_')
    ds, method, str_epsilon, str_clips, batch = params
    batch = int(batch)
    if batch == 64:
        continue
    train_losses = pickle.load(open(f, 'rb'))
    train_loader = DataLoader(cifar_train, batch_size=batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(cifar_test, batch_size=batch, shuffle=False, drop_last=True)
    p_file = f[:-2] + '.p'
    if not (path.exists(p_file) and stat(p_file).st_mtime > stat(f).st_mtime):
        infos = process_model_files(str_epsilon, str_clips, batch, train_losses, train_loader, test_loader)
        pickle.dump(infos, open(p_file, 'wb'))

