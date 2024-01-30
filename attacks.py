from utils import *
import torch.nn.functional as F
import numpy as np

def cw_attack(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None, prompt=None, a_lam=0, detection=False, acw=False):
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            if prompt is not None:
                output = model(X + delta, prompt, deep=args.deep_prompt)
            else:
                output = model(X + delta)
            loss = CW_loss(output, y, a_lam=a_lam, detection=detection) if not acw else ACW_loss(output, y)

            grad = torch.autograd.grad(loss, delta)[0].detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta = delta.detach()
    return delta

def simul_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt, length):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            loss = 0
            num = prompt.size(1) // length
            for k in range(num):
                output = model(X + delta, prompt[:,i*length:,:])
                loss += F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[:, :, :, :], upper_limit - X[:, :, :, :])
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        delta = delta.detach()
        output = output.detach()
        if prompt is not None:
            all_loss = F.cross_entropy(model(X+delta, prompt), y, reduction='none').detach()
        else:
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, tar=None, prompt=None, a_lam=0):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            if prompt is not None:
                output = model(X + delta, prompt)
            else:
                output = model(X + delta)
            if tar is None:
                loss = F.cross_entropy(output, y)
            elif tar is not None:
                loss = -F.cross_entropy(output, tar)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[:, :, :, :], upper_limit - X[:, :, :, :])
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
        delta = delta.detach()
        # output = output.detach()
        if prompt is not None:
            all_loss = F.cross_entropy(model(X+delta, prompt), y, reduction='none').detach()
        else:
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta




def attack_cw(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None, prompt=None, a_lam=-1, detection=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            if prompt is not None:
                output = model(X + delta, prompt)
            else:
                output = model(X + delta)
            loss = CW_loss(output, y, a_lam=a_lam, detection=detection)

            grad = torch.autograd.grad(loss, delta)[0].detach()

            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta = delta.detach()
    return delta

def split_vote(args, X, y, model, prompt, length):
    with torch.no_grad():
        outs = []
        labs = F.one_hot(y, 10)
        votes = torch.zeros_like(labs)
        num = prompt.size(1)//length
        for i in range(num):
            out = model(X, prompt[:,i*length:,:])
            outs.append(out.max(1)[1].detach())
            votes += F.one_hot(out.max(1)[1], 10)
        # print(votes)
        return (votes.max(1)[1] == y).float().sum().item(), outs





def CW_loss(x, y, reduction=True, num_cls=10, threshold=10, a_lam=-1, detection=False):
    batch_size = x.shape[0]
    x_cut = x[:, :num_cls]
    x_sorted, ind_sorted = x_cut.sort(dim=1)
    # print(ind_sorted)
    ind = (ind_sorted[:, -1] == y).float()
    logit_mc = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1. - ind)
    logit_gt = x[np.arange(batch_size), y]
    loss_value_ori = -(logit_gt - logit_mc)
    loss_value = torch.maximum(loss_value_ori, torch.tensor(-threshold).cuda())

    if reduction:
        return loss_value.mean()
    else:
        return loss_value

def RCW_loss(x, y, reduction=True, num_cls=10, threshold = 10):
    batch_size = x.shape[0]
    x_cut = x[:, :num_cls]
    inds = torch.randint(0, num_cls, size=(batch_size,)).cuda()
    while (sum(inds == y).item() > 0):
        inds[inds==y] = torch.randint(0, num_cls, size=(sum(inds==y).item(),)).cuda()
    logit_r = x_cut[np.arange(batch_size), inds]
    logit_c = x_cut[np.arange(batch_size), y]
    loss_value_ori = -(logit_c - logit_r)
    loss_value = torch.maximum(loss_value_ori, torch.tensor(-threshold).cuda())
    if reduction:
        return loss_value.mean()
    else:
        return loss_value

def ACW_loss(x, y, reduction=True, num_cls=10, threshold = 10):
    batch_size = x.shape[0]
    logit_l = x[:, -1]
    logit_c = x[np.arange(batch_size), y]
    x_cut = x[:, :-1]
    x_sorted, ind_sorted = x_cut.sort(dim=1)
    # print(x_cut.size())
    ind = (ind_sorted[:, -1] == y).float()
    logit_mc = x_sorted[:, -2] * ind + x_sorted[:, -1] * (1. - ind)
    diffl = torch.maximum(logit_mc - logit_l, torch.tensor(-threshold).cuda())
    diffc = torch.maximum(logit_mc - logit_c, torch.tensor(-threshold).cuda())
    loss_value = (diffl + diffc)/2
    if reduction:
        return loss_value.mean()
    else:
        return loss_value
