import immediate_sensitivity_primitives as isp
import membership_inference as mi
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import defaultdict
import numpy as np
from autograd_hacks import *


def loader_accuracy(model, test_loader, lf=nn.NLLLoss()):
    correct = 0
    num_data = 0
    lossies = []

    #grab a batch from the test loader
    for examples, labels in test_loader:
        examples = examples.to(torch.cuda.current_device())
        gpu_lab = labels.to(torch.cuda.current_device())
        outputs = model.forward(examples)
        lossies.append(lf(torch.squeeze(outputs), torch.squeeze(gpu_lab)))
        
        #for each output in the batch, check if the label is correct
        for i, output in enumerate(outputs):
            num_data += 1
            output = output.cpu()
            max_i = np.argmax(output.detach().numpy())
            if max_i == labels[i]:
                correct += 1

    acc = float(correct)/num_data
    loss = sum(lossies)/len(lossies)
    
    return acc, loss

def run_experiment(model, train_set, test_set, epsilon=1, alpha=25, epochs=10, add_noise=False, throw_out_threshold=False, batch_size=32, lf=nn.NLLLoss, print_rate=1):
    if epsilon==0:
        add_noise=False
    # reset the model
    model.to(torch.cuda.current_device())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    model_criterion = lf() 
    model_optimizer = optim.Adam(model.parameters(),lr=0.001)
    epsilon_iter = epsilon / epochs

    info = defaultdict(list)
    train_accs = []
    test_accs = []
    advs = []
    
    train_losses = []
    
    for epoch in range(epochs):
        for x_batch_train, y_batch_train in train_loader:
            plz_update = True
            train_losses = []

            model_optimizer.zero_grad()
            loss, batch_sensitivities = isp.grad_immediate_sensitivity(model, model_criterion, x_batch_train, y_batch_train, epoch)
            batch_sensitivity = np.max(batch_sensitivities) / batch_size
            
            train_losses.append(loss)
            if throw_out_threshold:
                # delete gradients?
                #with torch.no_grad():
                #    for p in model.parameters():
                #        p.grad = None

                # throw out "bad" examples
                good_idxs = np.array(batch_sensitivities) < throw_out_threshold
                #print(len(x_batch_train[good_idxs]), 'out of', len(x_batch_train))

                # re-do the gradients
                good_xs = x_batch_train[good_idxs]
                good_ys = y_batch_train[good_idxs]
                
                if len(good_xs) / len(x_batch_train) < 0.5:
                    plz_update = False
                else:
                    good_xs = good_xs.to(torch.cuda.current_device())
                    good_ys = good_ys.to(torch.cuda.current_device())
                    outputs = model.forward(good_xs)
                    loss = model_criterion(torch.squeeze(outputs), good_ys)

                    loss.backward()
                
                batch_sensitivity = min(throw_out_threshold, batch_sensitivity)


            if add_noise:
                sigma = np.sqrt((batch_sensitivity**2 * alpha) / (2 * epsilon_iter))
                with torch.no_grad():
                    for p in model.parameters():
                        p.grad += (sigma * torch.randn(1,device=torch.cuda.current_device()).float())

            if plz_update:
                model_optimizer.step()


        
        avg_test_acc, avg_test_l = loader_accuracy(model, test_loader, lf=model_criterion)
        avg_train_l = sum(train_losses)/len(train_losses)
            
        tpr = mi.run_yeom_loader(model, avg_train_l, train_loader, lf=lf)
        fpr = mi.run_yeom_loader(model, avg_train_l, test_loader, lf=lf)
        adv = tpr-fpr

        info['train_l'].append(avg_train_l.item())
        info['test_l'].append(avg_test_l.item())
        info['adv'].append(adv)
        info['acc'] = avg_test_acc
        info['merlin_adv'].append(mi.merlin_optimal_thresh(model, train_loader, test_loader, lf=lf, num_batches=20))
        
        if epoch % print_rate == 0:
            acc = avg_test_acc
            print(f'Epoch {epoch}: train loss {avg_train_l}, test loss {avg_test_l}, adv {adv}, acc {acc}')

    return info, model


def baseline_experiment(model, train_set, test_set, epsilon=1, alpha=25, C=2 epochs=10, add_noise=False, batch_size=32, lf=nn.NLLLoss, print_rate=1):
    if epsilon==0:
        add_noise=False
    # reset the model
    model.to(torch.cuda.current_device())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    model_criterion = lf() 
    model_optimizer = optim.Adam(model.parameters(),lr=0.001)
    autograd_acks.add_hooks(model)
    
    epsilon_iter = epsilon / epochs

    info = defaultdict(list)
    train_accs = []
    test_accs = []
    advs = []
    
    train_losses = []
    
    for epoch in range(epochs):
        for x_batch_train, y_batch_train in train_loader:
            train_losses = []

            model_optimizer.zero_grad()
            inp = Variable(x_batch_train, requires_grad=True)
            inp = inp.to(torch.cuda.current_device())
            outputs = model.forward(inp)
            y_batch_train = y_batch_train.to(torch.cuda.current_device())
            loss = model_criterion(outputs, y_batch_train)
            loss.backward()
            autograd_hacks.compute_grad1(model)
            clipper, mn = clipped_autograd(model, C)
            max_norms.append(mn)
            autograd_hacks.clear_backprops(model)
            train_losses.append(loss) 
            
            if add_noise:
                sigma = np.sqrt((C**2 * alpha) / (2 * epsilon_iter))
                with torch.no_grad():
                    for p in model.parameters():
                        p.grad += (sigma * torch.randn(1,device=torch.cuda.current_device()).float())

            model_optimizer.step()


        
        avg_test_acc, avg_test_l = loader_accuracy(model, test_loader, lf=model_criterion)
        avg_train_l = sum(train_losses)/len(train_losses)
            
        tpr = mi.run_yeom_loader(model, avg_train_l, train_loader, lf=lf)
        fpr = mi.run_yeom_loader(model, avg_train_l, test_loader, lf=lf)
        adv = tpr-fpr

        info['train_l'].append(avg_train_l.item())
        info['test_l'].append(avg_test_l.item())
        info['adv'].append(adv)
        info['acc'] = avg_test_acc
        info['merlin_adv'].append(mi.merlin_optimal_thresh(model, train_loader, test_loader, lf=lf, num_batches=20))
        
        if epoch % print_rate == 0:
            acc = avg_test_acc
            print(f'Epoch {epoch}: train loss {avg_train_l}, test loss {avg_test_l}, adv {adv}, acc {acc}')

    return info, model