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
from pgd import evaluate_pgd,evaluate_CW,evaluate_splits,attack_pgd
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
chkpnt = torch.load(r'50lrobustprompt_cifar_vit_small_patch16_224_AT_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_40')
rprompt = chkpnt['prompt'][0]
base_m.load_state_dict(chkpnt['state_dict'])
# chkpnt = torch.load(r'50lnatprompt_cifar_vit_small_patch16_224_natural_warmup/seed0/weight_decay_0.000100/drop_rate_1.000000/nw_10.000000/checkpoint_20')
# cprompt = chkpnt['prompt'][0]
# print(rprompt.size(), cprompt.size(), rpcprompt.size())
# print(ssprompt.size())

epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std
attack_iters = args.attack_iters
restarts = args.eval_restarts
upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

mean_base = 0
mean_robm = 0
mean_robp = 0
mean_ss = 0
num_samps = 0
fs_rpc = [torch.zeros(768).cuda() for _ in range(10)]
diff_rpc = 0
fs_cpr = [torch.zeros(768).cuda() for _ in range(10)]
diff_cpr = 0 
fs_rm = [torch.zeros(768).cuda() for _ in range(10)]
diff_rm = 0 
fs_rp = [torch.zeros(768).cuda() for _ in range(10)]
diff_rp = 0 

acc = 0
acc_c = 0
acc_ch = 0

for step, (X, y) in enumerate(test_loader):
    X, y = X.cuda(), y.cuda()
    num_samps += y.size(0)
    outc = base_m(X, rprompt).detach()
    # p_labels = F.one_hot(outc.max(1)[1], 10).float()
    delta = attack_pgd(base_m, X, outc.max(1)[1], epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=rprompt).detach()
    outch = base_m(X + delta, rprompt).detach()
    d = attack_pgd(base_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=rprompt).detach()
    out = base_m(X + d, rprompt).detach()

    acc += (out.max(1)[1] == y).float().sum()
    acc_c += (outc.max(1)[1] == y).float().sum()
    acc_ch += (outch.max(1)[1] == outc.max(1)[1]).float().sum()
    # delta = attack_pgd(robust_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=cprompt).detach()
    # delta = attack_pgd(base_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=rprompt).detach()
    # delta = attack_pgd(base_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=rpcprompt).detach()
    # with torch.no_grad():

    #     ### cprompt ####
    #     outcpr, fcpr = base_m(X, cprompt, get_fs=True)
    #     outcpra, fcpra = base_m(X + delta, cprompt, get_fs=True)
    #     for i in range(y.size(0)):
    #         fs_cpr[y[i].item()] += fcpr[i, cprompt.size(1), :].mean()
    #     diff_cpr += torch.norm(fcpra[:, cprompt.size(1), :] - fcpr[:, cprompt.size(1), :], dim=1).sum()

    #     ### rpcprompt  ####
    #     outrpc, frpc = base_m(X, rpcprompt, get_fs=True)
    #     outrpca, frpca = base_m(X + delta, rpcprompt, get_fs=True)
    #     for i in range(y.size(0)):
    #         fs_rpc[y[i].item()] += frpc[i, rpcprompt.size(1), :].mean()
    #     diff_rpc += torch.norm(frpca[:, rpcprompt.size(1), :] - frpc[:, rpcprompt.size(1), :], dim=1).sum()

    #     ### rprompt  ####
    #     outrp, frp = base_m(X, rprompt, get_fs=True)
    #     outrpa, frpa = base_m(X + delta, rprompt, get_fs=True)
    #     for i in range(y.size(0)):
    #         fs_rp[y[i].item()] += frp[i, rprompt.size(1), :].mean()
    #     diff_rp += torch.norm(frpa[:, rprompt.size(1), :] - frp[:, rprompt.size(1), :], dim=1).sum()
        

    #     ### rmodel  ####
    #     outrm, frm = robust_m(X, get_fs=True)
    #     outrm, frma = robust_m(X + delta, get_fs=True)
    #     for i in range(y.size(0)):
    #         fs_rm[y[i].item()] += frm[i, 0, :].mean()
    #     diff_rm += torch.norm(frma[:, 0, :] - frm[:, 0, :], dim=1).sum()

    if (step + 1) % 10 == 0:
        print('##### step {:d} | acc {:.4f} | acc_c {:.4f} acc_ch {:.4f} | #####'.format(step, acc/num_samps, acc_c/num_samps, acc_ch/num_samps))
        # print("cpr sep {:.4f} diff {:.4f}".format(min_sep(fs_cpr)/step, diff_cpr/num_samps))
        # print("rpc sep {:.4f} diff {:.4f}".format(min_sep(fs_rpc)/step, diff_rpc/num_samps))
        # print("rm sep {:.4f} diff {:.4f}".format(min_sep(fs_rm)/step, diff_rm/num_samps))
        # print("rp sep {:.4f} diff {:.4f}".format(min_sep(fs_rp)/step, diff_rp/num_samps))


        
