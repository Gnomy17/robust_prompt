import numpy as np
import tqdm
import random
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import  SoftTargetCrossEntropy
from timm.data import Mixup
from parser_cifar import get_args
from auto_LiRPA.utils import MultiAverageMeter
from utils import *
from torch.autograd import Variable
from pgd import evaluate_pgd,evaluate_CW,attack_pgd, attack_cw
from evaluate import evaluate_aa
from auto_LiRPA.utils import logger
import matplotlib.pyplot as plt
from buffer import Buffer

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

args = get_args()

args.out_dir = args.out_dir+"_eval_"+args.dataset+"_"+args.model+"_"+args.method
args.out_dir = args.out_dir +"/seed"+str(args.seed)


print(args.out_dir)
os.makedirs(args.out_dir,exist_ok=True)
logfile = os.path.join(args.out_dir, 'log_{:.4f}.log'.format(args.weight_decay))

file_handler = logging.FileHandler(logfile)
file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
logger.addHandler(file_handler)

logger.info(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

resize_size = args.resize
crop_size = args.crop
train_loader, test_loader= get_loaders(args)

def sepdet_atk(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, disc=None, pd=None, pc=None, a_lam=0.5):
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            outc = model(X + delta, pc)
            fsadv = model(X +delta, pd, get_fs=True)[1]
            sadv = disc(fsadv[:, pd.size(1)])
            # print(sadv.size(), torch.ones_like(y).float().size())
            loss = (1 - a_lam) * F.cross_entropy(outc, y) + a_lam * (F.binary_cross_entropy_with_logits(sadv, torch.ones_like(sadv).float()))
            grad = torch.autograd.grad(loss, delta)[0].detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta = delta.detach()
    return delta

def min_sep(fs_list):
    minim = None
    for i, fs1 in enumerate(fs_list):
        for j, fs2 in enumerate(fs_list):
            if i == j: 
                continue
            if minim is not None:
                minim += torch.norm(fs1 - fs2).item() #min(minim, torch.norm(fs1 - fs2).item())
            else:
                minim = torch.norm(fs1 - fs2).item()
    return 2* minim/(len(fs_list) * (len(fs_list) - 1))

################## load models ######################
from model_for_cifar.vit import vit_small_patch16_224

base_m = vit_small_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =10,patch_size=args.patch, args=args).cuda()
base_m = nn.DataParallel(base_m)
base_m.eval()
# robust_m = vit_small_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =10,patch_size=args.patch, args=args).cuda()
# robust_m = nn.DataParallel(robust_m)
# robust_m.eval()
# chkpnt = torch.load(r'./finetuned_model/robust_cifar_vit')
# robust_m.load_state_dict(chkpnt['state_dict'])
# chkpnt = torch.load(r'1fullwhite_cifar_vit_small_patch16_224_voting_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_30')
# rprompt = chkpnt['prompts'][0]
# chkpnt = torch.load(r'50lrobustprompt_cifar_vit_small_patch16_224_AT_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')
# rprompt = chkpnt['prompt'][0]
# base_m.load_state_dict(chkpnt['state_dict'])
# chkpnt = torch.load(r'50lnatprompt_cifar_vit_small_patch16_224_natural_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_20')
# cprompt = chkpnt['prompt'][0]
# print(rprompt.size(), cprompt.size(), rpcprompt.size())
# print(ssprompt.size())
chkpnt = torch.load(r'./sepdet_noise/checkpoint_40')
base_m.load_state_dict(chkpnt['state_dict'])
prompt = chkpnt['prompt'][0]
# print(chkpnt.keys())
dprompt = chkpnt['detp'][0]
disc = chkpnt['detp'][1]

epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std
attack_iters = args.attack_iters
restarts = args.eval_restarts
upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)




dacc = 0
acc_c = 0
acc_a = 0
adacc = 0
fp = 0
num_samps = 0
thresh = 0.7
for step, (X, y) in enumerate(test_loader):
    X, y = X.cuda(), y.cuda()
    num_samps += y.size(0)
    noise = attack_pgd(base_m, X, y, epsilon, alpha, 0, restarts, lower_limit, upper_limit, prompt=prompt).detach()
    outc = base_m(X+noise, prompt).detach()
    # p_labels = F.one_hot(outc.max(1)[1], 10).float()
    delta = attack_pgd(base_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=prompt).detach()
    fsa = base_m(X + delta, dprompt, get_fs=True)[1].detach()
    fca = base_m(X + noise, dprompt, get_fs=True)[1].detach()
    sa = disc(fsa[:, dprompt.size(1)]).detach()
    sc = disc(fca[:, dprompt.size(1)]).detach()
    # print(sc.size())
    
    ad = sepdet_atk(base_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, disc=disc, pc=prompt, pd=dprompt,a_lam=0.5).detach()
    fsad = base_m(X + ad, dprompt, get_fs=True)[1].detach()
    sad = disc(fsad[:, dprompt.size(1)]).detach()
    outad = base_m(X+ad, prompt).detach()
    # d = attack_pgd(base_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=rprompt).detach()
    
    dacc += (F.sigmoid(sa).squeeze() > (torch.zeros_like(y) + thresh)).float().sum()
    adacc += (F.sigmoid(sad).squeeze() > (torch.zeros_like(y) + thresh)).float().sum()
    acc_a += (outad.max(1)[1] == y).float().sum()
    acc_c += (outc.max(1)[1] == y).float().sum()
    fp += (F.sigmoid(sc).squeeze() > (torch.zeros_like(y) + thresh)).float().sum()
    # print(fp)

    if (step + 1) % 10 == 0:
        print('##### step {:d} | dacc {:.4f} | adacc {:.4f} | acc_c {:.4f} | acc_a {:.4f} | fp {:.4f} | #####'.format(step, dacc/num_samps, adacc/num_samps,
         acc_c/num_samps, acc_a/num_samps, fp/num_samps))



        
