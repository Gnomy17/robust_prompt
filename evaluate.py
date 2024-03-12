import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import numpy as np
from attacks import CW_loss, attack_cw, attack_pgd
from torchvision import datasets, transforms
from autoattack import AutoAttack
# from utils import normalize
# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack



def evaluate_aa(args, model,test_loader,log_path,aa_batch=128, prompt=None):
    if args.dataset=="cifar":
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    if args.dataset=='cifar100':
        mu = torch.tensor(cifar100_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar100_std).view(3,1,1).cuda()
    if args.dataset=="imagenette" or args.dataset=="imagenet":
        mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(imagenet_std).view(3,1,1).cuda()
    epsilon = (args.epsilon / 255.) / std.mean().item()
    model.eval()
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    class normalize_model():
        def __init__(self, model, prompt=None, deep=False):
            self.model_test = model
            self.prompt = prompt
            self.deep = deep
        def __call__(self, x):
            return self.model_test(x, self.prompt, deep=self.deep)
    new_model = normalize_model(model, prompt, args.deep_p)
    adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard',log_path=log_path)
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=aa_batch)


def evaluate_natural(args, model, test_loader, logger, prompt=None):
    model.eval()
    with torch.no_grad():
        test_loss = test_acc = test_n = 0
        for step, (X_batch, y_batch) in enumerate(test_loader):
            X, y = X_batch.cuda(), y_batch.cuda()
            output = model(X, prompt, deep=args.deep_p)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).float().mean() * y.size(0)
            test_n += y.size(0)
    return test_loss/test_n, test_acc/test_n

def evaluate_pgd(args, model, test_loader, eval_steps=None, prompt=None, unadapt=False):
    attack_iters = args.eval_iters # 50
    restarts = args.eval_restarts # 10
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    print('Evaluating with PGD {} steps and {} restarts'.format(attack_iters, restarts))
    if args.dataset=="cifar":
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    if args.dataset=='cifar100':
        mu = torch.tensor(cifar100_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar100_std).view(3,1,1).cuda()
    if args.dataset=="imagenette" or args.dataset=="imagenet":
        mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(imagenet_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, 
                prompt=prompt if not unadapt else None, deep=args.deep_p).detach()
        with torch.no_grad():
            output = model(X + pgd_delta, prompt, deep=args.deep_p)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        if step + 1 == eval_steps:
            break
        if (step + 1) % 10 == 0 or step + 1 == len(test_loader):
            print('{}/{}'.format(step+1, len(test_loader)), 
                pgd_loss/n, pgd_acc/n)
    return pgd_loss/n, pgd_acc/n

def evaluate_CW(args, model, test_loader, eval_steps=None, prompt=None, unadapt=False):
    attack_iters = 20
    restarts = 1
    cw_loss = 0
    cw_acc = 0
    n = 0
    model.eval()
    print('Evaluating with CW {} steps and {} restarts'.format(attack_iters, restarts))
    if args.dataset=="cifar":
        num_cls = 10
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    if args.dataset=="imagenette" or args.dataset=="imagenet":
        num_cls = 10 if args.dataset=="imagenette" else 1000 
        mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(imagenet_std).view(3,1,1).cuda()
    if args.dataset=='cifar100':
        num_cls = 100
        mu = torch.tensor(cifar100_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar100_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        delta = attack_cw(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit,
                 prompt=prompt if not unadapt else None, deep=args.deep_p, num_cls=num_cls)
        with torch.no_grad():
            output = model(X + delta, prompt, deep=args.deep_p)
            loss = CW_loss(output, y)
            cw_loss += loss.item() * y.size(0)
            cw_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

        if (step + 1) % 10 == 0 or step + 1 == len(test_loader):
            print('{}/{}'.format(step+1, len(test_loader)),
                cw_loss/n, cw_acc/n)
    return cw_loss/n, cw_acc/n