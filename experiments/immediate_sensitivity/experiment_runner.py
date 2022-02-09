import immediate_sensitivity_primitives as isp
import membership_inference as mi
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from collections import defaultdict
import numpy as np
import autograd_hacks
import opacus

def loader_accuracy(model, test_loader, lf=nn.NLLLoss()):
    lossies = []
    accs = []
    #grab a batch from the test loader
    with torch.no_grad():
        for examples, labels in test_loader:
            torch.cuda.empty_cache()
            examples = examples.to(torch.cuda.current_device())
            gpu_lab = labels.to(torch.cuda.current_device())
            outputs = model.forward(examples)
            lossies.append(lf(torch.squeeze(outputs), torch.squeeze(gpu_lab)).item())
            
            #for each output in the batch, check if the label is correct
            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()
            accuracy = (preds == labels).mean()
            accs.append(accuracy)

    loss = sum(lossies)/len(lossies)
    acc = sum(accs)/len(accs)
    
    return acc, loss

def run_experiment(model, train_set, test_set, epsilon=1, alpha=25, epochs=10, add_noise=False, throw_out_threshold=False, throw_out_std=0, batch_size=32, lf=nn.NLLLoss, print_rate=1):
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
        train_losses = []
        for x_batch_train, y_batch_train in train_loader:
            torch.cuda.empty_cache()
            x_batch_train =  x_batch_train.to(torch.cuda.current_device())
            y_batch_train =  y_batch_train.to(torch.cuda.current_device())
            plz_update = True

            model_optimizer.zero_grad()
            loss, batch_sensitivities = isp.grad_immediate_sensitivity(model, model_criterion, x_batch_train, y_batch_train, epoch)
            batch_sensitivity = np.max(batch_sensitivities) / batch_size
            ms = np.mean(batch_sensitivities)
            std = np.std(batch_sensitivities)
            info['mean_sen'].append(ms) 
            info['min_sen'].append(np.min(batch_sensitivities)) 
            info['max_sen'].append(np.max(batch_sensitivities)) 
            info['median_sen'].append(np.median(batch_sensitivities)) 
            info['std_sen'].append(std) 
            train_losses.append(loss.item())
            if throw_out_threshold or throw_out_std:
                # delete gradients?
                #with torch.no_grad():
                #    for p in model.parameters():
                #        p.grad = None

                # throw out "bad" examples
                if throw_out_std:
                    throw_out_threshold = ms + (throw_out_std * std)
                good_idxs = np.array(batch_sensitivities) < throw_out_threshold
                #print(len(x_batch_train[good_idxs]), 'out of', len(x_batch_train))
                # re-do the gradients
                good_xs = x_batch_train[good_idxs]
                good_ys = y_batch_train[good_idxs]
                
                if len(good_xs) / len(x_batch_train) < 0.5:
                    plz_update = False
                else:
                    #good_xs = good_xs.to(torch.cuda.current_device())
                    #good_ys = good_ys.to(torch.cuda.current_device())
                    outputs = model.forward(good_xs)
                    loss = model_criterion(torch.squeeze(outputs), good_ys)

                    loss.backward()
               	    throw_out_bs = np.max(np.array(batch_sensitivities)[good_idxs]) 
                    avg_sen = throw_out_bs / len(good_xs)
                    batch_sensitivity = min(avg_sen, batch_sensitivity)
            else:
                loss.backward()


            if plz_update and add_noise:
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
        
        madv, mopt_round, mtpr, mfpr = mi.merlin_optimal_thresh(model, train_loader, test_loader, lf=lf, num_batches=20, tpr=True)

        info['train_l'].append(avg_train_l)
        info['test_l'].append(avg_test_l)
        info['yeom_tpr'].append(tpr)
        info['yeom_fpr'].append(fpr)
        info['acc'].append(avg_test_acc)
        info['merlin_tpr'].append(mtpr)
        info['merlin_fpr'].append(mfpr)
        
        if epoch % print_rate == 0:
            acc = avg_test_acc
            print(f'Epoch {epoch}: train loss {avg_train_l}, test loss {avg_test_l}, adv {adv}, acc {acc}')

    return info, model


def baseline_experiment(model, train_set, test_set, epsilon=1, alpha=25, C=2, epochs=10, add_noise=False, batch_size=32, lf=nn.NLLLoss, print_rate=1):
    if epsilon==0:
        add_noise=False
    # reset the model
    model.to(torch.cuda.current_device())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    model_criterion = lf() 
    model_optimizer = optim.Adam(model.parameters(),lr=0.001)
    autograd_hacks.add_hooks(model)
    
    epsilon_iter = epsilon / epochs

    info = defaultdict(list)
    train_accs = []
    test_accs = []
    advs = []

    
    train_losses = []
    
    for epoch in range(epochs):
        train_losses = []
        for x_batch_train, y_batch_train in train_loader:

            model_optimizer.zero_grad()
            inp = x_batch_train.to(torch.cuda.current_device())
            outputs = model.forward(inp)
            y_batch_train = y_batch_train.to(torch.cuda.current_device())
            loss = model_criterion(outputs, y_batch_train)
            loss.backward()
            autograd_hacks.compute_grad1(model)
            mn = isp.clipped_autograd(model, C)
            train_losses.append(loss.item()) 
            
            if add_noise:
                sigma = np.sqrt((C**2 * alpha) / (2 * epsilon_iter))
            else:
                sigma = 0
            with torch.no_grad():
                for p in model.parameters():
                    p.grad += (sigma * torch.randn(p.grad.shape,device=torch.cuda.current_device()).float())

            model_optimizer.step()
            autograd_hacks.clear_backprops(model)


        
        avg_test_acc, avg_test_l = loader_accuracy(model, test_loader, lf=model_criterion)
        avg_train_l = sum(train_losses)/len(train_losses)
            
        tpr = mi.run_yeom_loader(model, avg_train_l, train_loader, lf=lf)
        fpr = mi.run_yeom_loader(model, avg_train_l, test_loader, lf=lf)
        adv = tpr-fpr

        madv, mopt_round, mtpr, mfpr = mi.merlin_optimal_thresh(model, train_loader, test_loader, lf=lf, num_batches=20, tpr=True)

        info['train_l'].append(avg_train_l)
        info['test_l'].append(avg_test_l)
        info['yeom_tpr'].append(tpr)
        info['yeom_fpr'].append(fpr)
        info['acc'].append(avg_test_acc)
        info['merlin_tpr'].append(mtpr)
        info['merlin_fpr'].append(mfpr)
        
        if epoch % print_rate == 0:
            acc = avg_test_acc
            print(f'Epoch {epoch}: train loss {avg_train_l}, test loss {avg_test_l}, adv {adv}, acc {acc}')

    return info, model


def weight_experiment(model, train_set, test_set, epsilon=1, alpha=25, epochs=10, add_noise=False, batch_size=32, lf=nn.NLLLoss, print_rate=1):
    if epsilon==0:
        add_noise=False
    # reset the model
    model.to(torch.cuda.current_device())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    model_criterion = lf() 
    model_optimizer = optim.Adam(model.parameters(),lr=0.001)
    
    epsilon_iter = epsilon / epochs
    num_batches = len(train_set) // batch_size

    info = defaultdict(list)
    train_accs = []
    test_accs = []
    advs = []
    sens = .01 * 3.1622 # learning rate x 1 - beta1 / root(1 - beta2)

    
    train_losses = []
    
    for epoch in range(epochs):
        train_losses = []
        for x_batch_train, y_batch_train in train_loader:

            model_optimizer.zero_grad()
            inp = Variable(x_batch_train, requires_grad=True)
            inp = inp.to(torch.cuda.current_device())
            outputs = model.forward(inp)
            y_batch_train = y_batch_train.to(torch.cuda.current_device())
            loss = model_criterion(outputs, y_batch_train)
            loss.backward()
            train_losses.append(loss) 
            

            model_optimizer.step()

        if add_noise:
            sigma = np.sqrt((sens**2 * alpha) / (2 * epsilon_iter))
            with torch.no_grad():
                for p in model.parameters():
                    p += (sigma * torch.randn(p.shape,device=torch.cuda.current_device()).float())

        
        avg_test_acc, avg_test_l = loader_accuracy(model, test_loader, lf=model_criterion)
        avg_train_l = sum(train_losses)/len(train_losses)
            
        tpr = mi.run_yeom_loader(model, avg_train_l, train_loader, lf=lf)
        fpr = mi.run_yeom_loader(model, avg_train_l, test_loader, lf=lf)
        adv = tpr-fpr

        madv, mopt_round, mtpr, mfpr = mi.merlin_optimal_thresh(model, train_loader, test_loader, lf=lf, num_batches=20, tpr=True)

        info['train_l'].append(avg_train_l.item())
        info['test_l'].append(avg_test_l.item())
        info['yeom_tpr'].append(tpr)
        info['yeom_fpr'].append(fpr)
        info['acc'].append(avg_test_acc)
        info['merlin_tpr'].append(mtpr)
        info['merlin_fpr'].append(mfpr)
        
        if epoch % print_rate == 0:
            acc = avg_test_acc
            print(f'Epoch {epoch}: train loss {avg_train_l}, test loss {avg_test_l}, adv {adv}, acc {acc}')

    return info, model



def opacus_experiment(model, train_set, test_set, epsilon=1, alpha=25, C=2, epochs=10, add_noise=False, batch_size=32, lf=nn.NLLLoss, print_rate=1, learning_rate=.001, idx=0):
    if epsilon==0:
        add_noise=False
    # reset the model
    virtual_bs = 64
    n_vb = batch_size / virtual_bs
    assert batch_size % virtual_bs == 0
    
  
    model.to(torch.cuda.current_device())
    train_loader = DataLoader(train_set, batch_size=virtual_bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=virtual_bs, shuffle=False, drop_last=True)
    model_criterion = lf() 
    model_optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    
    if epsilon != 0: 
        epsilon_iter = epsilon / epochs
        sigma = np.sqrt((1**2 * alpha) / (2 * epsilon_iter))
    else:
        sigma = 0

    privacy_engine = opacus.PrivacyEngine(
        model,
        alphas=[alpha],
        max_grad_norm=C,
        target_epsilon=epsilon,
        target_delta=1e-5,
        epochs=epochs,
        batch_size=batch_size,
        sample_size=len(train_set),
        noise_multiplier=sigma
    )
    privacy_engine.attach(model_optimizer)
    privacy_engine.to(torch.cuda.current_device())
    print(sigma, privacy_engine.noise_multiplier)
    

    info = defaultdict(list)
    train_accs = []
    test_accs = []
    avg_train_ls = []
    advs = []

    
    train_losses = []
    true_budgets = []
    
    for epoch in range(epochs):
        train_losses = []
        for i, (x_batch_train, y_batch_train) in enumerate(train_loader):

            x_batch_train = x_batch_train.to(torch.cuda.current_device())
            outputs = model.forward(x_batch_train)
            y_batch_train = y_batch_train.to(torch.cuda.current_device())
            loss = model_criterion(outputs, y_batch_train)
            loss.backward()
            train_losses.append(loss.item()) 
            if i % n_vb == 0:
                model_optimizer.step()
                model_optimizer.zero_grad()
            else:
                model_optimizer.virtual_step()


        print(privacy_engine.get_privacy_spent()) 
        torch.save(model.state_dict(), f'../../data/cifar/cifar_model_{idx}_{batch_size}_{epsilon}_{C}_{epoch}.torch')
        avg_train_l = sum(train_losses)/len(train_losses)
        print(f'Epoch {epoch}: train loss {avg_train_l}: cifar_model_{epsilon}_{C}_{epoch}.torch saved') # , test loss {avg_test_l}, adv {adv}, acc {acc}')
        #avg_test_acc, avg_test_l = loader_accuracy(model, test_loader, lf=model_criterion)
        #    
        #tpr = mi.run_yeom_loader(model, avg_train_l, train_loader, lf=lf)
        #fpr = mi.run_yeom_loader(model, avg_train_l, test_loader, lf=lf)
        #adv = tpr-fpr

        ##madv, mopt_round, mtpr, mfpr = mi.merlin_optimal_thresh(model, train_loader, test_loader, lf=lf, num_batches=20, tpr=True)

        #info['train_l'].append(avg_train_l)
        #info['test_l'].append(avg_test_l)
        #info['yeom_tpr'].append(tpr)
        #info['yeom_fpr'].append(fpr)
        #info['acc'].append(avg_test_acc)
        ##info['merlin_tpr'].append(mtpr)
        ##info['merlin_fpr'].append(mfpr)
        avg_train_ls.append(avg_train_l)
        true_budgets.append(privacy_engine.get_privacy_spent())
    return avg_train_ls, true_budgets
        


def opacus_vanilla(model, train_set, test_set, epsilon=1, alpha=25, C=2, epochs=10, add_noise=False, batch_size=32, lf=nn.NLLLoss, print_rate=1):
    if epsilon==0:
        add_noise=False
    # reset the model
    model.to(torch.cuda.current_device())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)
    model_criterion = lf() 
    model_optimizer = optim.Adam(model.parameters(),lr=0.001)
    if add_noise:
        privacy_engine = opacus.PrivacyEngine(
            model,
            alphas=[alpha],
            max_grad_norm=5,
            target_epsilon=epsilon,
            target_delta=1e-5,
            epochs=epochs,
            sample_rate=1
        )
        privacy_engine.attach(model_optimizer)
        privacy_engine.to(torch.cuda.current_device())
    

    info = defaultdict(list)
    train_accs = []
    test_accs = []
    advs = []

    
    train_losses = []
    
    for epoch in range(epochs):
        train_losses = []
        for x_batch_train, y_batch_train in train_loader:

            model_optimizer.zero_grad()
            x_batch_train = x_batch_train.to(torch.cuda.current_device())
            outputs = model.forward(x_batch_train)
            y_batch_train = y_batch_train.to(torch.cuda.current_device())
            loss = model_criterion(outputs, y_batch_train)
            loss.backward()
            train_losses.append(loss) 
            
            model_optimizer.step()


        
        avg_test_acc, avg_test_l = loader_accuracy(model, test_loader, lf=model_criterion)
        avg_train_l = sum(train_losses)/len(train_losses)
            


        
        if epoch % print_rate == 0:
            acc = avg_test_acc
            print(f'Epoch {epoch}: train loss {avg_train_l}, test loss {avg_test_l}, acc {acc}')

    return info, model
