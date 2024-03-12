from attacks import attack_pgd
import torch.nn.functional as F
from utils import *
import torch
import torch.nn as nn
from hsic import hsic_normalized 


def mu_std(args):   
    if args.dataset == 'cifar':
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    elif args.dataset == 'cifar100':
        mu = torch.tensor(cifar100_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar100_std).view(3, 1, 1).cuda()
    elif args.dataset == 'imagenette':
        mu = torch.tensor(imagenet_mean).view(3, 1, 1).cuda()
        std = torch.tensor(imagenet_std).view(3, 1, 1).cuda()
    return mu, std

def nat_loss(model, prompt, X, y, args):
    out = model(X, prompt)
    loss = F.cross_entropy(out, y)

    return loss, out

def AT_loss(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    delta = attack_pgd(model, X, y, epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit).detach()
    out = model(X + delta, prompt, deep=args.deep_p)
    loss = F.cross_entropy(out, y)
    return loss, out

def TRADES_loss(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    beta = args.beta
    batch_size = len(X)
    epsilon = epsilon_base.cuda()
    
    if args.delta_init == 'random':
        delta = 0.001 * torch.randn(X.shape).cuda()
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()

    delta.requires_grad = True

    for _ in range(args.attack_iters):
        loss_kl = F.kl_div(F.log_softmax(model(X + delta), dim=1),
                                F.softmax(model(X), dim=1), reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, [delta])[0]
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)

    delta = delta.detach()

    outc = model(X, prompt, deep=args.deep_p)
    outa = model(X + delta, prompt, deep=args.deep_p)

    loss_natural = F.cross_entropy(outc, y)
    loss_robust = F.kl_div(F.log_softmax(outa, dim=1),
                                                    F.softmax(outc, dim=1), reduction='batchmean')
    loss = loss_natural + beta * loss_robust
    return loss, outa

def PAT_loss(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    delta = attack_pgd(model, X, y, epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt).detach()

    outc = model(X, prompt, deep=args.deep_p)
    outa = model(X + delta, prompt, deep=args.deep_p)
    loss = F.cross_entropy(outc, y) + args.beta * F.cross_entropy(outa, y)
    return loss, outa

def PKL_loss(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    beta = args.beta
    batch_size = len(X)
    epsilon = epsilon_base.cuda()
    
    delta = attack_pgd(model, X, y, epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt)

    delta = delta.detach()

    outc = model(X, prompt, deep=args.deep_p)
    outa = model(X + delta, prompt, deep=args.deep_p)

    loss_natural = F.cross_entropy(outc, y)
    loss_robust = F.kl_div(F.log_softmax(outa, dim=1),
                                                    F.softmax(outc, dim=1), reduction='batchmean')
    loss = loss_natural + beta * loss_robust
    return loss, outa


def PTRADES_loss(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    beta = args.beta
    batch_size = len(X)
    epsilon = epsilon_base.cuda()
    
    if args.delta_init == 'random':
        delta = 0.001 * torch.randn(X.shape).cuda()
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()

    delta.requires_grad = True
    
    for _ in range(args.attack_iters):
        loss_kl = F.kl_div(F.log_softmax(model(X + delta, prompt=prompt), dim=1),
                                F.softmax(model(X, prompt=prompt), dim=1), reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, [delta])[0]
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)

    delta = delta.detach()

    outc = model(X, prompt, deep=args.deep_p)
    outa = model(X + delta, prompt, deep=args.deep_p)

    loss_natural = F.cross_entropy(outc, y)
    loss_robust = F.kl_div(F.log_softmax(outa, dim=1),
                                                    F.softmax(outc, dim=1), reduction='batchmean')
    loss = loss_natural + beta * loss_robust
    return loss, outa

def PSIM_loss(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    epsilon = epsilon_base.cuda()

    delta = attack_pgd(model, X, y, epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt).detach()


    outc, fc = model(X, prompt, get_fs=True, deep=args.deep_p)
    outa, fa = model(X + delta, prompt, get_fs=True, deep=args.deep_p)
    loss_cosim = -hsic_normalized(fc, fa)
    # wandb.log({
    #     'cos_sim': abs(loss_cosim.item())
    # })
    loss = F.cross_entropy(outc, y) + args.beta * loss_cosim

    return loss, outa.detach()