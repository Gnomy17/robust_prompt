from attacks import attack_pgd
import torch.nn.functional as F
from utils import *
import torch
import torch.nn as nn
from torch.autograd import Variable

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

def natural(model, prompt, X, y, args):
    out = model(X, prompt)
    loss = F.cross_entropy(out, y)

    return loss, out

def AT(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    delta = attack_pgd(model, X, y, epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit).detach()
    out = model(X + delta, prompt)
    loss = F.cross_entropy(out, y)
    return loss, out

def TRADES(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    beta = args.beta
    epsilon = epsilon_base.cuda()
    
    if args.delta_init == 'random':
        delta = 0.001 * torch.randn(X.shape).cuda()
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
    model.eval()

    delta.requires_grad = True

    for _ in range(args.attack_iters):
        loss_kl = F.kl_div(F.log_softmax(model(X + delta), dim=1),
                                F.softmax(model(X), dim=1), reduction='batchmean')
        grad = torch.autograd.grad(loss_kl, [delta])[0]
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)

    delta = delta.detach()

    outc = model(X, prompt)
    outa = model(X + delta, prompt)

    loss_natural = F.cross_entropy(outc, y)
    loss_robust = F.kl_div(F.log_softmax(outa, dim=1),
                                                    F.softmax(outc, dim=1), reduction='batchmean')
    loss = loss_natural + beta * loss_robust
    return loss, outa

def NFGSM(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    eta = torch.zeros_like(X).cuda()
    for j in range(len(epsilon)):
        eta[:, j, :, :].uniform_(-2.0*epsilon[j][0][0].item(), 2.0*epsilon[j][0][0].item())
    eta = torch.clamp(eta, lower_limit - X, upper_limit - X)
    eta.requires_grad = True

    output = model(X + eta)
    loss = F.cross_entropy(output, y)
    grad = torch.autograd.grad(loss, eta)[0]
    grad = grad.detach()
    # Compute perturbation based on sign of gradient
    delta = eta + alpha * torch.sign(grad)

    delta = torch.clamp(delta, lower_limit - X, upper_limit - X)
    delta = delta.detach()
    
    output = model(X + delta, prompt)
    loss = F.cross_entropy(output, y)
    return loss, output

def MART(model, prompt, X, y, args, distance='l_inf'):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = X.size(0)
    beta = args.beta
    # generate adversarial example
    x_adv = X.detach() + 0.001 * torch.randn(X.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(args.attack_iters):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, X - epsilon_base), X + epsilon_base)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    logits = model(X, prompt)

    logits_adv = model(x_adv, prompt)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss, logits_adv

def ADAPT_CE(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    ## Adaptive Attack
    delta = attack_pgd(model, X, y, epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt).detach()

    ## Prompted model output
    outc = model(X, prompt)
    outa = model(X + delta, prompt)

    ## loss
    loss = F.cross_entropy(outc, y) + args.beta * F.cross_entropy(outa, y)
    return loss, outa

def ADAPT_KL(model, prompt, X, y, args):
    mu, std = mu_std(args)
    upper_limit = ((1 - mu) / std).cuda()
    lower_limit = ((0 - mu) / std).cuda()
    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    beta = args.beta
    
    ## Adaptive Attack
    delta = attack_pgd(model, X, y, epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt)

    delta = delta.detach()

    ## Prompted Output
    outc = model(X, prompt)
    outa = model(X + delta, prompt)

    ## Loss
    loss_natural = F.cross_entropy(outc, y)
    loss_robust = F.kl_div(F.log_softmax(outa, dim=1),
                                                    F.softmax(outc, dim=1), reduction='batchmean')
    loss = loss_natural + beta * loss_robust
    return loss, outa
