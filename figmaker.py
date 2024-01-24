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

chkpnt = torch.load(r'./fcorrexp/checkpoint_35')
base_m.load_state_dict(chkpnt['state_dict'])
prompt = chkpnt['prompt'][0]

epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std
attack_iters = args.attack_iters
restarts = args.eval_restarts
upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)




acc_n = 0
acc_c = 0
acc_t = 0
acc_ap = 0
num_samps = 0
mat_c = np.zeros((10,10))
mat_t = np.zeros((10,10))
mat_n = np.zeros((10,10))
mat_ap = np.zeros((10,10))

for step, (X, y) in enumerate(test_loader):
    X, y = X.cuda(), y.cuda()
    num_samps += y.size(0)
    outc = base_m(X, prompt).detach()

    delta = attack_pgd(base_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, prompt=prompt).detach()
    outap = base_m(X + delta, prompt).detach()

    inc = torch.randint(low=0, high=9, size=y.size()).cuda()
    tlabs = (y + inc) % 10# torch.ones_like(y).cuda() * 3#
    dn = attack_pgd(base_m, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit).detach()
    dt = attack_pgd(base_m, X, tlabs, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit).detach()

    outt = base_m(X + dt, prompt).detach()
    outn = base_m(X + dn, prompt).detach()

    acc_c += (outc.max(1)[1] == y).float().sum()
    acc_n += (outn.max(1)[1] == y).float().sum()
    acc_t += (outt.max(1)[1] == tlabs).float().sum()
    acc_ap += (outt.max(1)[1] == y).float().sum()

    for j in range(y.size(0)):
        mat_c[y[j], outc.max(1)[1][j]] += 1
        mat_t[y[j], outt.max(1)[1][j]] += 1
        mat_n[y[j], outn.max(1)[1][j]] += 1
        mat_ap[y[j], outap.max(1)[1][j]] += 1
        

    if (step + 1) % 10 == 0:
        print('##### step {:d} | acc_n {:.4f} | acc_t {:.4f} | acc_c {:.4f} | acc_ap {:.4f} #####'.format(step, acc_n/num_samps, acc_t/num_samps,
            acc_c/num_samps, acc_ap/num_samps))
        fg, axarr = plt.subplots(1,4)

        axarr[0].matshow(mat_c/num_samps)
        axarr[0].axis('off')#.yaxis.tick_left()
        axarr[0].set_title('clean samples')
        axarr[0].set_xlabel('predicted label\n')
        axarr[0].set_ylabel('ground truth label')
        axarr[1].matshow(mat_t/num_samps)
        axarr[1].axis('off')#.yaxis.tick_left()
        axarr[1].set_title('pt samples')
        axarr[1].set_xlabel('predicted label\n')
        axarr[1].set_ylabel('ground truth label')
        axarr[2].matshow(mat_n/num_samps)
        axarr[2].axis('off')#.yaxis.tick_left()
        axarr[2].set_title('pn samples')
        axarr[2].set_xlabel('predicted label\n')
        axarr[2].set_ylabel('ground truth label')
        axarr[3].matshow(mat_ap/num_samps)
        axarr[3].axis('off')#.yaxis.tick_left()
        axarr[3].set_title('ap samples')
        axarr[3].set_xlabel('predicted label\n')
        axarr[3].set_ylabel('ground truth label')
        plt.savefig("./fcorrexp/fcorr_rand.png", dpi=500)
        plt.close() 


        
