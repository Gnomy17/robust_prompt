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
from pgd import evaluate_pgd,evaluate_CW,attack_pgd
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

base_m = vit_small_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =11,patch_size=args.patch, args=args).cuda()
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
chkpnt = torch.load(r'./cwdetect_bce0.5_d0.5/checkpoint_10')
base_m.load_state_dict(chkpnt['state_dict'])
prompt = chkpnt['prompt'][0]

epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std
attack_iters = args.attack_iters
restarts = args.eval_restarts
upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)




dacc = 0
acc_c = 0
fp = 0
num_samps = 0
for step, (X, y) in enumerate(train_loader):
    X, y = X.cuda(), y.cuda()
    num_samps += y.size(0)
    outc = base_m(X, prompt).detach()
    # p_labels = F.one_hot(outc.max(1)[1], 10).float()
    delta = 0#attack_pgd(base_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=prompt).detach()
    outa = base_m(X + delta, prompt).detach()
    # d = attack_pgd(base_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=rprompt).detach()
    # out = base_m(X + d, rprompt).detach()

    dacc += (outa[:,-1] > torch.zeros_like(y)).float().sum()
    acc_c += (outc[:, :-1].max(1)[1] == y).float().sum()
    fp += (outc[:,-1] <= torch.zeros_like(y)).float().sum()


    if (step + 1) % 10 == 0:
        print('##### step {:d} | dacc {:.4f} | acc_c {:.4f} fp {:.4f} | #####'.format(step, dacc/num_samps, acc_c/num_samps, fp/num_samps))



        
