import torch.nn as nn
from torch.autograd import Variable
import torch
from immediate_sensitivity_primitives import grad_immediate_sensitivity
import numpy as np



# returns counts rather than counts over thresh
def merlin(model, inputs, labels, lf):
    if type(inputs) == np.ndarray:
        inputs = torch.from_numpy(inputs).float()
    if type(labels) == np.ndarray:
        lables = torch.from_numpy(labels).float()
    sigma = .01
    T = 100
    loss_function = lf(reduction='none')
    inp = Variable(inputs, requires_grad=True)
    outputs = model.forward(inp)
    loss = loss_function(torch.squeeze(outputs), torch.squeeze(labels))

    # probably a better way to create a bytetensor of zeros, but this works
    counts = torch.zeros(len(loss))

    for i in range(T):
        noisy_inputs = inputs + (sigma * torch.randn(1).float())

        noisy_inp = Variable(noisy_inputs, requires_grad=True)

        noisy_outputs = model.forward(noisy_inp)

        noisy_loss = loss_function(torch.squeeze(noisy_outputs), torch.squeeze(labels))

        gt = noisy_loss > loss
        counts += gt
    return counts, [float(l) for l in loss]


def run_merlin(model, thresh, X_target, y_target, lf=nn.MSELoss):
    counts, train_loss = merlin(model,
                                  X_target,
                                  y_target, lf)
    pass_inf = counts > thresh
    return sum(pass_inf) / len(pass_inf)

def merlin_optimal_thresh(model, train_loader, test_loader, lf=nn.MSELoss):
    train_counts = np.zeros(len(train_loader.dataset))
    test_counts = np.zeros(len(test_loader.dataset))
    train_bs = train_loader.batch_size
    test_bs = test_loader.batch_size
    for i, (inputs, labels) in enumerate(train_loader):
        idx = i * train_bs
        counts, _ = merlin(model, inputs, labels, lf)
        train_counts[idx : (idx + len(labels))] = counts
    
    for i, (inputs, labels) in enumerate(test_loader):
        idx = i * test_bs
        counts, _ = merlin(model, inputs, labels, lf)
        test_counts[idx : (idx + len(labels))] = counts
    
    max_thresh = 0
    max_adv = 0
    for i in range(0, 100):
        train_pos = train_counts > i
        test_pos = test_counts > i
        train_rat = sum(train_pos)/len(train_pos)
        test_rat = sum(test_pos)/len(test_pos)
        adv = train_rat - test_rat
        if adv > max_adv:
            max_thresh = i
            max_adv = adv
    return max_adv, max_thresh
    
    

def run_merlin_loader(model, thresh, loader, lf=nn.MSELoss):
    ratios = []
    for inputs, labels in loader:
        ratios.append(run_merlin(model, thresh, inputs, labels, lf))
    return sum(ratios)/len(ratios)


def gaussian_pdf(sd, x):
    if sd <= 0:
        raise ValueError('standard deviation must be positive but is {}'.format(sd))
    else:  # sd > 0
        return np.e ** (-0.5 * (x / sd) ** 2) / sd


def membership_inf(model, avg_train_loss, inputs, labels, lf=nn.MSELoss):
    inp = Variable(inputs, requires_grad=True)

    outputs = model.forward(inp)
    loss = lf(reduction='none')(torch.squeeze(outputs), labels)
    pass_inf = [1 if abs(l) < avg_train_loss else 0 for l in loss]

    return pass_inf, [float(l) for l in loss]


def run_membership_inference_attack(model, avg_train_l, X_target, y_target, lf=nn.MSELoss):
    _, sensitivities = grad_immediate_sensitivity(model,
                                                      lf(),
                                                      torch.from_numpy(X_target).float(),
                                                      torch.from_numpy(y_target).float(),
                                                      None)
    max_sen = max(sensitivities)
    norm_sen = [s/max_sen for s in sensitivities]
    min_exp = min([np.log(s) for s in sensitivities if s != 0])
    log_sen = [np.log(s) if s !=0 else min_exp - 1 for s in sensitivities]
    paws = [s - (min_exp - 1) for s in log_sen]

    pass_inf, train_loss = membership_inf(model,
                                          avg_train_l,
                                          torch.from_numpy(X_target).float(),
                                          torch.from_numpy(y_target).float(), lf)
    #plt.scatter(paws, pass_inf)
    #print('positive ratio:',sum(pass_inf)/len(pass_inf))
    return sum(pass_inf)/len(pass_inf)