from autograd_hacks import *
import torch
import numpy as np
from torch.autograd import Variable

def flatten(grads):
    shapes = [g.shape for g in grads]
    flats = [x.view(-1) for x in grads]
    lens = [len(f) for f in flats]
    view = torch.cat(flats)
    return view, shapes, lens

def reshape(view, shapes, lens):
    i = 0
    tensors = []
    for s, l in zip(shapes, lens):
        flat = view[i: i + l]
        tensors.append(flat.view(s))
        i += l
    return tensors

def clipped_autograd(model, C):
    max_norms = []
    for param in model.parameters():
        t1 = torch.transpose(param.grad1, 0, 1)
        d = tuple(range(1,len(param.grad.shape) + 1))
        if len(param.grad.shape) == 1:
            d = 1
            t = t1
        else:
            t = torch.transpose(t1, 1, 2)
        norms = param.grad1.data.norm(dim=d)
        torch.equal(norms, param.grad1.norm(dim=d))
        max_norms.append(torch.max(norms))
        c_norms = C / norms
	
        clips_with_ones = torch.minimum(c_norms, torch.ones(c_norms.shape, device=torch.cuda.current_device()))
        print(param.grad.shape)
        param.grad = torch.matmul(t, clips_with_ones)

    return max(max_norms)


def grad_immediate_sensitivity(model, criterion, inputs, labels, epoch):
    inp = Variable(inputs, requires_grad=True)
    outputs = model.forward(inp)
    loss = criterion(torch.squeeze(outputs), torch.squeeze(labels))

    # (1) first-order gradient (wrt parameters)
    first_order_grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)

    # (2) L2 norm of the gradient from (1)
    grad_l2_norm = torch.norm(torch.cat([x.view(-1) for x in first_order_grads]), p=2)

    # (3) Gradient (wrt inputs) of the L2 norm of the gradient from (2)
    sensitivity_vec = torch.autograd.grad(grad_l2_norm, inp, retain_graph=True)[0]

    # (4) L2 norm of (3) - "immediate sensitivity"
    s = [torch.norm(v, p=2).cpu().numpy().item() for v in sensitivity_vec]

    return loss, s

def per_param_immediate_sensitivity(model, criterion, inputs, labels, epoch):
    inp = Variable(inputs, requires_grad=True)
    inp = inp.to(torch.cuda.current_device())
    outputs = model.forward(inp)
    labels = labels.to(torch.cuda.current_device())
    loss = criterion(torch.squeeze(outputs), torch.squeeze(labels))

    # (1) first-order gradient (wrt parameters)
    first_order_grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)

    # (2) L2 norm of the gradient from (1)
    print([f.shape for f in first_order_grads])
    grad_l2_norm = torch.norm(torch.cat([x.view(-1) for x in first_order_grads]), p=2)

    # (3) Gradient (wrt inputs) of the L2 norm of the gradient from (2)
    shaped_grads = torch.cat([x.view(-1) for x in first_order_grads])
    print(shaped_grads.shape)
    print(grad_l2_norm.shape)
    print(inp.shape)
    sensitivity_vec = [torch.autograd.grad(g, inp, retain_graph=True)[0] for g in shaped_grads]
    print(sensitivity_vec.size())

    # (4) L2 norm of (3) - "immediate sensitivity"
    s = [torch.norm(v, p=2).cpu().numpy().item() for v in sensitivity_vec]
    print(len(s), s[0].shape())

    return loss, s


def clipped_grad(model, loss, C):
    # do gradient for each element in loss
    first_order_grads = [torch.autograd.grad([l], model.parameters(), retain_graph=True, create_graph=True) for l in
                         loss]
    shapes = []
    lens = []
    views = []
    # flatten views out per sample
    for f in first_order_grads:
        v, shapes, lens = flatten(f)
        views.append(v)
    # views = [torch.cat([x.view(-1) for x in f]) for f in first_order_grads]

    # a norm for every sample
    grad_l2_norms = [torch.norm(v, p=2) for v in views]

    # divisors turned out to be recipricol multiplication
    divisors = [C / norm.item() if norm.item() > C else 1 for norm in grad_l2_norms]

    # the part where we clip
    clipped_grads = [v * d for v, d in zip(views, divisors)]

    # sum of gradients
    cg_sum = torch.stack(clipped_grads, dim=0).sum(dim=0)

    # reshape per model shape
    cgs = reshape(cg_sum, shapes, lens)
    return cgs



