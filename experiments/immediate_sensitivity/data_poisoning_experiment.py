import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from IPython import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from collections import defaultdict

import sklearn.datasets

torch.manual_seed(1)
np.random.seed(7)
#sns.set(style="white", palette="muted", color_codes=True, context="talk")


class Classifier(nn.Module):
    def __init__(self, n_features, n_hidden=256):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.network(x)


def grad_immediate_sensitivity(model, criterion, inputs, labels, epoch):
    inp = Variable(inputs, requires_grad=True)
    
    outputs = model.forward(inp)
#     print('outputs', outputs)
#     print('labels', labels)
    loss = criterion(torch.squeeze(outputs), labels)
#     print(loss)
    
    # (1) first-order gradient (wrt parameters)
    first_order_grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    
    # (2) L2 norm of the gradient from (1)
    grad_l2_norm = torch.norm(torch.cat([x.view(-1) for x in first_order_grads]), p = 2)
    
    # (3) Gradient (wrt inputs) of the L2 norm of the gradient from (2)
    sensitivity_vec = torch.autograd.grad(grad_l2_norm, inp, retain_graph=True)[0]
    
    # (4) L2 norm of (3) - "immediate sensitivity"
    s = [torch.norm(v, p=2).numpy().item() for v in sensitivity_vec]

    loss.backward()
    return loss, s

def accuracy(model, X, y):
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    outputs = model(Xt)
    values, indices = outputs.max(dim=1)
    y_hat = indices.detach().numpy()
    accuracy = np.sum(y_hat == y) / len(y)
    return accuracy

def run_experiment(train_loader, epsilon, epochs, add_noise=False, throw_out_threshold=False, logging=True):
    # reset the model
    model = Classifier(n_features=n_features)
    model_criterion = nn.NLLLoss() 
    model_optimizer = optim.Adam(model.parameters(),lr=0.001)
    alpha = 25
    epsilon_iter = epsilon / epochs

    info = defaultdict(lambda: [])
    train_accs = []
    test_accs = []
    advs = []

    
    for epoch in range(epochs):
        for x_batch_train, y_batch_train in train_loader:
            plz_update = True

            model_optimizer.zero_grad()
            loss, batch_sensitivities = grad_immediate_sensitivity(model, model_criterion, x_batch_train, y_batch_train,epoch)
            batch_sensitivity = np.max(batch_sensitivities) / BATCH_SIZE

            sigma = np.sqrt((batch_sensitivity**2 * alpha) / (2 * epsilon_iter))

            if add_noise:
                with torch.no_grad():
                    for p in model.parameters():
                        p.grad += (sigma * torch.randn(1).float())

            if plz_update:
                model_optimizer.step()

        lossfn = model_criterion
        
        if epoch % 10 == 0 and logging:
            acc = accuracy(model, nobkd_trn_x, nobkd_trn_y)
            print(f'Epoch {epoch}: acc {acc}')

    return info, model

#info, model = run_experiment(train_loader_bkd, 0.1, 21, add_noise=True, throw_out_threshold=False)

def backdoor(model, bkd_x, bkd_y):

    _, predsw  = model(torch.from_numpy(bkd_x).float()).max(dim=1)
    _, predswo = model(torch.from_numpy(np.zeros_like(bkd_x)).float()).max(dim=1)

    diff = (predsw - predswo).detach().numpy()
    pred = np.multiply(bkd_y, diff).sum()
    return pred

def backdoor_experiment(train_loader, x, y, epsilon=1.0, epochs=10):
    info, model = run_experiment(train_loader, epsilon, epochs, add_noise=True, 
                                 throw_out_threshold=False, logging=False)
    
    n_backdoor = backdoor(model, x, y)
    return n_backdoor

from scipy import stats
def clopper_pearson(count, trials, conf):
    count, trials, conf = np.array(count), np.array(trials), np.array(conf)
    q = count / trials
    ci_low = stats.beta.ppf(conf / 2., count, trials - count + 1)
    ci_upp = stats.beta.isf(conf / 2., count + 1, trials - count)

    if np.ndim(ci_low) > 0:
        ci_low[q == 0] = 0
        ci_upp[q == 1] = 1
    else:
        ci_low = ci_low if (q != 0) else 0
        ci_upp = ci_upp if (q != 1) else 1
    return ci_low, ci_upp

def get_eps_thresh(nobkd_arr, bkd_arr, thresh):
    #thresh = 2800
    poisoning_size = 8
    
    bkd_ct = (bkd_arr >= thresh).sum()
    nobkd_ct = (nobkd_arr >= thresh).sum()

    nobkd_lb, nobkd_ub = clopper_pearson(nobkd_ct, nobkd_arr.shape[0], .1)
    bkd_lb, bkd_ub = clopper_pearson(bkd_ct, bkd_arr.shape[0], .1)


    if nobkd_ub + bkd_lb > 1:
        corr_ratio = (1-nobkd_ub)/(1-bkd_lb)
    else:
        corr_ratio = bkd_lb/nobkd_ub

    corr_eps = np.log(corr_ratio)/poisoning_size

    return corr_eps

def get_eps(nobkd_arr, bkd_arr):
    all_arr = np.concatenate((nobkd_arr, bkd_arr)).ravel()
    all_threshs = np.unique(all_arr)
    all_threshs = all_threshs[all_threshs > 0]
    all_epsilons = [(thresh, get_eps_thresh(nobkd_arr, bkd_arr, thresh)) for thresh in all_threshs]

    thresh, corr_eps = max(all_epsilons, key = lambda x: x[1])
    return thresh, corr_eps


import argparse

parser = argparse.ArgumentParser(description='Run the data poisoning experiment.')

parser.add_argument('epsilon', type=float,
                    help='privacy parameter')

parser.add_argument('epochs', type=int,
                    help='number of epochs to train')

parser.add_argument('trials', type=int,
                    help='number of trials to run')

parser.add_argument('distance', type=int,
                    help='distance between poisoned and non-poisoned data')

parser.add_argument('filename', type=str,
                    help='filename of the data to run on')

# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

config = parser.parse_args()
print(config.epsilon)

# /home/jnear/co/temp/auditing-dpsgd/datasets/fmnist/clipbkd-new-8.npy
print('Loading data file', config.filename)

(nobkd_trn_x, nobkd_trn_y), (bkd_trn_x, bkd_trn_y), _, _ = np.load(config.filename, allow_pickle=True)
nobkd_trn_x = nobkd_trn_x.reshape(nobkd_trn_x.shape[0], -1)
bkd_trn_x = bkd_trn_x.reshape(bkd_trn_x.shape[0], -1)
print('Training data shape, no backdoor:', nobkd_trn_x.shape)
print('Training data shape, backdoor:', bkd_trn_x.shape)

n_features = nobkd_trn_x.shape[1]
n_classes = 2

BATCH_SIZE = 64

training_dataset_nobkd = TensorDataset(torch.from_numpy(nobkd_trn_x).float(), 
                                       torch.from_numpy(nobkd_trn_y).long())
train_loader_nobkd = DataLoader(training_dataset_nobkd, batch_size=BATCH_SIZE,
                                shuffle=True, drop_last=True)

training_dataset_bkd = TensorDataset(torch.from_numpy(bkd_trn_x).float(), 
                                       torch.from_numpy(bkd_trn_y).long())
train_loader_bkd = DataLoader(training_dataset_bkd, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)


print('nobkd experiments, epsilon', config.epsilon)
nobkd_results = [backdoor_experiment(train_loader_nobkd, nobkd_trn_x, nobkd_trn_y, 
                                     epsilon=config.epsilon,
                                     epochs=config.epochs) for _ in range(config.trials)]

print('bkd experiments, epsilon', config.epsilon)
bkd_results = [backdoor_experiment(train_loader_bkd, bkd_trn_x, bkd_trn_y, 
                                   epsilon=config.epsilon,
                                   epochs=config.epochs) for _ in range(config.trials)]
    
#all_backdoor_results[epsilon] = (nobkd_results, bkd_results)
# thresh, calculated_eps = get_eps(np.array(nobkd_results), np.array(bkd_results))
# #all_backdoor_epsilons[epsilon] = calculated_eps
# print('for epsilon', config.epsilon, 'calculated epsilon was', calculated_eps)

f1 = open(f'results/nobkd_results_{config.distance}_{config.epsilon}.txt', "a+")
for r in nobkd_results:
    f1.write(str(r) + '\n')
f1.close()

f2 = open(f'results/bkd_results_{config.distance}_{config.epsilon}.txt', "a+")
for r in bkd_results:
    f2.write(str(r) + '\n')
f2.close()
