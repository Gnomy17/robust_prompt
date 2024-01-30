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


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)



def normalize(args, X):
    if args.dataset=="cifar":
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    elif args.dataset=="imagenette" or args.dataset=="imagenet" :
        mu = torch.tensor(imagenet_mean).view(3, 1, 1).cuda()
        std = torch.tensor(imagenet_std).view(3, 1, 1).cuda()
    return (X - mu) / std



def evaluate_aa(args, model,test_loader,log_path,aa_batch=128, prompt=None):
    # if args.dataset=="cifar":
    #     test_transform_nonorm = transforms.Compose([
    #         transforms.Resize([args.resize, args.resize]),
    #         transforms.ToTensor()
    #     ])
    #     test_dataset_nonorm = datasets.CIFAR10(
    #     args.data_dir, train=False, transform=test_transform_nonorm, download=True)
    # if args.dataset=="imagenette" or args.dataset=="imagenet" :
    #     test_transform_nonorm = transforms.Compose([
    #         transforms.Resize([args.resize, args.resize]),
    #         transforms.ToTensor()
    #     ])
    #     test_dataset_nonorm = datasets.ImageFolder(args.data_dir+"val/",test_transform_nonorm)
    # test_loader_nonorm = torch.utils.data.DataLoader(
    #     dataset=test_dataset_nonorm,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=4,
    # )
    model.eval()
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    # print(x_test.size())
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    class normalize_model():
        def __init__(self, model, prompt=None):
            self.model_test = model
            self.prompt = prompt
        def __call__(self, x):
            x_norm = normalize(args, x)
            return self.model_test(x, self.prompt)
    new_model = normalize_model(model, prompt)
    epsilon = args.epsilon / 255.
    adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard',log_path=log_path)
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=aa_batch)


def evaluate_natural(args, model, test_loader, verbose=False, prompt=None):
    model.eval()
    with torch.no_grad():
        meter = MultiAverageMeter()
        test_loss = test_acc = test_n = 0
        def test_step(step, X_batch, y_batch):
            X, y = X_batch.cuda(), y_batch.cuda()
            if prompt is not None:
                output = model(X, prompt, deep=args.deep_prompt)
            else:
                output = model(X)
            loss = F.cross_entropy(output, y)
            meter.update('test_loss', loss.item(), y.size(0))
            meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))
        for step, (X_batch, y_batch) in enumerate(test_loader):
            test_step(step, X_batch, y_batch)
        logger.info('Evaluation {}'.format(meter))

def evaluate_pgd(args, model, test_loader, eval_steps=None, prompt=None, a_lam=0):
    attack_iters = args.eval_iters # 50
    restarts = args.eval_restarts # 10
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    print('Evaluating with PGD {} steps and {} restarts'.format(attack_iters, restarts))
    if a_lam >= 0:
        print('Using adaptive loss with lambda {:.4f} to avoid detection'.format(a_lam))
        detect_acc = 0
    if args.dataset=="cifar":
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    if args.dataset=="imagenette" or args.dataset=="imagenet" :
        mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(imagenet_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit).detach()
        with torch.no_grad():
            if prompt is not None:
                output = model(X + pgd_delta, prompt)
            else:
                output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            if a_lam >= 0:
                a_label = torch.ones_like(y) * (output.size(1) - 1)
                loss += a_lam * F.cross_entropy(output, a_label)
                detect_acc += (output.max(1)[1] == a_label).sum().item()
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output[:, :10].max(1)[1] == y).sum().item()
            n += y.size(0)
        if step + 1 == eval_steps:
            break
        if (step + 1) % 10 == 0 or step + 1 == len(test_loader):
            print('{}/{}'.format(step+1, len(test_loader)), 
                pgd_loss/n, pgd_acc/n, detect_acc/n if a_lam >= 0 else '')
    return pgd_loss/n, pgd_acc/n

def evaluate_CW(args, model, test_loader, eval_steps=None, prompt=None, a_lam=0, detection=False):
    attack_iters = args.eval_iters # 50
    restarts = args.eval_restarts # 10
    cw_loss = 0
    cw_acc = 0
    n = 0
    model.eval()
    print('Evaluating with CW {} steps and {} restarts'.format(attack_iters, restarts))
    if a_lam >= 0:
        print('Using adaptive loss with lambda {:.4f} to avoid detection'.format(a_lam))
        detect_acc = 0
    if args.dataset=="cifar":
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
    if args.dataset=="imagenette" or args.dataset=="imagenet":
        mu = torch.tensor(imagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(imagenet_std).view(3,1,1).cuda()
    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    for step, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_cw(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=prompt)
        with torch.no_grad():
            if prompt is not None:
                output = model(X + pgd_delta, prompt)
            else:
                output = model(X + pgd_delta)
            loss = CW_loss(output, y)
            cw_loss += loss.item() * y.size(0)
            cw_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        if step + 1 == eval_steps:
            break
        if (step + 1) % 10 == 0 or step + 1 == len(test_loader):
            print('{}/{}'.format(step+1, len(test_loader)),
                cw_loss/n, cw_acc/n, detect_acc/n if a_lam >= 0 else '')
    return cw_loss/n, cw_acc/n