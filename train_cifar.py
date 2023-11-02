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
from pgd import evaluate_pgd,evaluate_CW, CW_loss, ACW_loss
from evaluate import evaluate_aa
from auto_LiRPA.utils import logger
import matplotlib.pyplot as plt
from buffer import Buffer
# torch.autograd.set_detect_anomaly(True)
args = get_args()
joint_p = lambda x, y: torch.cat((x, y), dim=1) if y is not None else x 
def make_prompt(length, h_dim, depth=1,init_xavier=True):
    prompt = torch.zeros(1, length, h_dim, depth, requires_grad=True)
    prompt.cuda()
    if init_xavier:
        nn.init.xavier_uniform_(prompt)
    # prompt = nn.Parameter(prompt)
    return prompt

args.out_dir = args.out_dir+"_"+args.dataset+"_"+args.model+"_"+args.method+"_warmup"
args.out_dir = args.out_dir +"/seed"+str(args.seed)
if args.ARD:
    args.out_dir = args.out_dir + "_ARD"
if args.PRM:
    args.out_dir = args.out_dir + "_PRM"
if args.scratch:
    args.out_dir = args.out_dir + "_no_pretrained"
# if args.load:
#     args.out_dir = args.out_dir + "_load"


args.out_dir = args.out_dir + "/weight_decay_{:.6f}/".format(
        args.weight_decay)+ "drop_rate_{:.6f}/".format(args.drop_rate)+"nw_{:.6f}/".format(args.n_w)


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
mean1 = 0
mean2 = 0
mean3 = 0
count = 0

train_loader, test_loader= get_loaders(args)
nclasses = 11 if args.method == 'detect' else 10
if args.model == "vit_base_patch16_224":
    from model_for_cifar.vit import vit_base_patch16_224
    model = vit_base_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    model = nn.DataParallel(model)
elif args.model == "vit_small_robust_cifar":
    from model_for_cifar.vit import vit_small_patch16_224
    model = vit_small_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    model = nn.DataParallel(model)
    chkpnt = torch.load(r'./finetuned_model/robust_cifar_vit')
    model.load_state_dict(chkpnt['state_dict'])
    chkpnt['state_dict'] = 0
elif args.model == "vit_large_patch16_224_in21k":
    from model_for_cifar import vit_large_patch16_224_in21k
    model = vit_large_patch16_224_in21k(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    model = nn.DataParallel(model)

elif args.model == 'vit_finetuned_cifar':
    #### TODO ####
    chkpnt = torch.load(r'./finetuned_model/finetuned_vit')
    from model_for_cifar.vit import vit_base_patch16_224
    model = vit_base_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(chkpnt['state_dict'])
elif args.model == "vit_base_patch16_224_in21k":
    from model_for_cifar.vit import vit_base_patch16_224_in21k
    model = vit_base_patch16_224_in21k(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    model = nn.DataParallel(model)
elif args.model == "vit_small_patch16_224":
    from model_for_cifar.vit import  vit_small_patch16_224
    model = vit_small_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    model = nn.DataParallel(model)
elif args.model == "deit_small_patch16_224":
    from model_for_cifar.deit import  deit_small_patch16_224
    model = deit_small_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses, patch_size=args.patch, args=args).cuda()
    model = nn.DataParallel(model)
elif args.model == "deit_tiny_patch16_224":
    from model_for_cifar.deit import  deit_tiny_patch16_224
    model = deit_tiny_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    model = nn.DataParallel(model)
elif args.model == "convit_base":
    from model_for_cifar.convit import convit_base
    model = convit_base(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    model = nn.DataParallel(model)
elif args.model == "convit_small":
    from model_for_cifar.convit import convit_small
    model = convit_small(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch,args=args).cuda()
    model = nn.DataParallel(model)
elif args.model == "convit_tiny":
    from model_for_cifar.convit import convit_tiny
    model = convit_tiny(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    model = nn.DataParallel(model)
else:
    raise ValueError("Model doesn't exist！")
if args.model_log:
    logger.info('Model{}'.format(model))
model.train()

checkpoint = None
if args.load:
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['state_dict'])


if args.prompted:
    if args.load:
        prompt = (checkpoint['prompt'])[0]
    else:
        prompt = make_prompt(args.prompt_length, model.module.embed_dim, depth=args.prompt_depth)

    prompts = [prompt]
    params = [prompt]

    if not args.freeze_head:
        for p in model.module.head.parameters():
            params.append(p)
    if args.train_patch:
        for p in model.module.patch_embed.parameters():
            params.append(p)
    if args.optim == 'sgd':
        opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay) 
    elif args.optim == 'adam':
        opt = torch.optim.Adam(params, lr=args.lr_max, weight_decay=args.weight_decay)
        
    if args.method == 'sepdet':
        dprompt = make_prompt(args.prompt_length, 768, depth=args.prompt_depth)
        disc = nn.Linear(768, 1).cuda()
        prms = [dprompt]
        for p in disc.parameters():
            prms.append(p)
        # dopt = torch.optim.Adam(prms, lr=args.lr_max, weight_decay=args.weight_decay)
elif args.prefixed:
    if args.load:
        prompt = (checkpoint['prompt'])[0]
    else:
        prompt = make_prompt(args.prompt_length, model.module.embed_dim, depth=model.module.depth)
    
    prompts = [prompt]
    params = [prompt]
    if args.train_patch:
        for p in model.module.patch_embed.parameters():
            params.append(p)
    if not args.freeze_head:
        for p in model.module.head.parameters():
            params.append(p)
    
    if args.optim == 'sgd':
        opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay) 
    
elif args.method in ['voting']:
    prompts = []
    head_params = []
    opts = []
    for p in model.module.head.parameters():
        head_params.append(p)
    for i in range(args.num_prompts):
        if args.load:
            p = (checkpoint['prompts'])[i]
        else:
            p = make_prompt(args.prompt_length, 768)
        prompts.append(p)
        
    # print(len(prompts))
    # print(args.num_prompts)
        o  = torch.optim.SGD([p] + head_params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
        opts.append(o)
else:
    prompt = None
    # model_copy = vit_small_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    # model_copy = nn.DataParallel(model_copy)
    if args.optim == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr = args.lr_max, weight_decay=args.weight_decay)

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


mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu) / std).cuda()
lower_limit = ((0 - mu) / std).cuda()

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


def sepdet_atk(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, disc=None, pd=None, pc=None, a_lam=0.5):
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            outc = model(X + delta, pc)
            _, fsadv = model(X +delta, pd, get_fs=True)
            sadv = disc(fsadv[:, pd.size(1)])
            # print(sadv.size(), torch.ones_like(y).float().size())
            loss = (1 - a_lam) * F.cross_entropy(outc, y) + a_lam * (F.binary_cross_entropy_with_logits(sadv, torch.ones_like(sadv).float()))
            grad = torch.autograd.grad(loss, delta)[0].detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta = delta.detach()
    return delta


def pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate,iters=None, prompt=None, target=None, avoid= None, deep=False):
    model.eval()
    epsilon = epsilon_base.cuda()
    delta = torch.zeros_like(X).cuda()
    if args.delta_init == 'random':
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(args.attack_iters if iters is None else iters):
        if prompt is not None:
            if callable(prompt):
                output = model(X + delta, prompt(X + delta), deep=deep)
            else:
                output = model(X + delta, prompt, deep=deep)
        else:    
            output = model(X + delta)
        if target is None:
            if avoid is None:
                loss = criterion(output, y)
            elif y is None:
                loss = criterion(output, avoid)
            else:
                loss = criterion(output, y) + criterion(output, avoid)
        else:
            loss = -criterion(output, target)
        grad = torch.autograd.grad(loss, delta)[0].detach()
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
    delta = delta.detach()
    model.train()
    if len(handle_list) != 0:
        for handle in handle_list:
            handle.remove()
    return delta



def majority_vote(model, X, y, prompts):
    with torch.no_grad():
        outs = []
        votes = torch.zeros_like(y)
        for p in prompts:
            out = model(X, p)
            outs.append(out.max(1)[1].detach())
            votes += F.one_hot(out.max(1)[1], 10)
        # print(votes)
        return (votes.max(1)[1] == y.max(1)[1]).float().mean().item(), outs


def train_adv(args, model, ds_train, ds_test, logger):
    global prompt, done_prompt, opt

    epsilon_base = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    train_loader, test_loader = ds_train, ds_test

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active :
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.labelsmoothvalue, num_classes=10)


    criterion = nn.CrossEntropyLoss()
    bceloss = nn.BCEWithLogitsLoss()
    mseloss = nn.MSELoss()
    steps_per_epoch = len(train_loader)
    
    
    if args.load:
        logger.info("Resuming at epoch {}".format(checkpoint['epoch'] + 1))

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    lr_steps = args.epochs * steps_per_epoch
    if args.lr_schedule == 'cyclic':
        lr_schedule = lambda t, max_ep: np.interp([t], [0, max_ep // 2, max_ep], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'drops':
        def lr_schedule(t, max_ep):
            if t< max_ep-5:
                return args.lr_max
            elif t< max_ep -2:
                return args.lr_max*0.1
            else:
                return args.lr_max* 0.01
    epoch_s = 0 if not args.load else (checkpoint['epoch'])
    if args.load:
        for k in checkpoint:
            checkpoint[k] = None

    past_p = None
    for epoch in tqdm.tqdm(range(epoch_s + 1, args.epochs + 1)):
        if args.just_eval:
            break
        # evaluate_natural(args, model, test_loader, verbose=False)
        train_loss = 0
        train_acc = 0
        train_clean = 0
        train_prompted = 0
        train_adetect = 0
        train_n = 0
        hist_c = torch.zeros((10)).cuda()
        hist_a = torch.zeros((10)).cuda()
        corr_mats = [torch.zeros((nclasses,nclasses)) for _ in (range(4))]
        fvs = []
        fvs_a = []
        pfvs = []
        pfvs_a = []
        flabels = []

        past_p = prompt.clone().detach()
        ### freeze head weights after warmstart #####
        if epoch == args.ws and args.method == 'sepdet':
            logger.info('Switching to sepdet')
            opt = torch.optim.SGD(prms, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay) 

        def train_step(X, y, t, mixup_fn, hist_a, hist_c, corr_mats):
            global prompt, done_prompt, opt
            model.train()
            # drop_calculation
            def attn_drop_mask_grad(module, grad_in, grad_out, drop_rate):
                new = np.random.rand()
                if new > drop_rate:
                    gamma = 0
                else:
                    gamma = 1
                if len(grad_in) == 1:
                    mask = torch.ones_like(grad_in[0]) * gamma
                    return (mask * grad_in[0][:],)
                else:
                    mask = torch.ones_like(grad_in[0]) * gamma
                    mask_1 = torch.ones_like(grad_in[1]) * gamma
                    return (mask * grad_in[0][:], mask_1 * grad_in[1][:])
            if t < args.n_w:
                drop_rate = t / args.n_w * args.drop_rate
            else:
                drop_rate = args.drop_rate
            drop_hook_func = partial(attn_drop_mask_grad, drop_rate=drop_rate)
            model.eval()
            handle_list = list()
            if args.model in ["vit_base_patch16_224", "vit_base_patch16_224_in21k", "vit_small_patch16_224"]:
                if args.ARD:
                    from model_for_cifar.vit import Block
                    for name, module in model.named_modules():
                        if isinstance(module, Block):
                            handle_list.append(module.drop_path.register_backward_hook(drop_hook_func))
            elif args.model in ["deit_small_patch16_224", "deit_tiny_patch16_224"]:
                if args.ARD:
                    from model_for_cifar.deit import Block
                    for name, module in model.named_modules():
                        if isinstance(module, Block):
                            handle_list.append(module.drop_path.register_backward_hook(drop_hook_func))
            elif args.model in ["convit_base", "convit_small", "convit_tiny"]:
                if args.ARD:
                    from model_for_cifar.convit import Block
                    for name, module in model.named_modules():
                        if isinstance(module, Block):
                            handle_list.append(module.drop_path.register_backward_hook(drop_hook_func))
            model.train()
            if args.method == 'natural' or epoch < args.ws:
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                if args.prompted or args.prefixed:
                    output = model(X, prompt, deep=args.deep_prompt)
                    # print('sag')
                else:
                    output = model(X)
                loss = criterion(output, y)
                loss.backward()
                acc = (output.max(1)[1] == y.max(1)[1]).float().mean().item()

                return loss, acc, y, acc, acc, acc 
            elif args.method == 'AT':
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                
                if args.prompted or args.prefixed:
                    if args.full_white:
                        delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt, deep=args.deep_prompt).detach()
                        if args.past:
                            d2 = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=past_p, deep=args.deep_prompt).detach()
                            out2 = model(X + d2, prompt).detach()
                            acc2 = (out2.max(1)[1] == y.max(1)[1]).float().mean().item()
                    else:
                        delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate).detach() #prompt=prompt, avoid=labs).detach()
                        t = F.one_hot((y.max(1)[1] + 1) % 10, 10).float()
                        d2 =  pgd_attack(model, X, t, epsilon_base, alpha, args, criterion, handle_list, drop_rate).detach()
                        out2 = model(X + d2, prompt).detach()
                        acc2 = (out2.max(1)[1] == t.max(1)[1]).float().mean().item()#(thingy * labs).sum(1).mean().item()#(out2.max(1)[1] == labs.max(1)[1]).float().mean().item()

                    X.detach()
                    out = model(X + delta, prompt, deep=args.deep_prompt)#, get_fs=True)
                    outc = model(X, prompt, deep=args.deep_prompt)
                    # fb = fb.detach()
                    

                    loss = (1 - args.d_lam) * criterion(outc, y) + args.d_lam * criterion(out, y) #+ criterion(outc, y) #+ criterion(outt, y)
                    loss.backward()

                    acc = (out.max(1)[1] == y.max(1)[1]).float().mean().item()
                    acc_c = (outc.max(1)[1] == y.max(1)[1]).float().mean().item()#cosim(fw[:, args.prompt_length, :], fb[:, args.prompt_length, :]).detach().mean().item()
                    # 
                    for j in range(y.size(0)):
                        corr_mats[1][y.max(1)[1][j], out.detach().max(1)[1][j]] += 1
                        corr_mats[0][y.max(1)[1][j], outc.detach().max(1)[1][j]] += 1
                        if not args.full_white or args.past:
                            corr_mats[2][y.max(1)[1][j], out2.detach().max(1)[1][j]] += 1
                        
                    return loss, acc, y, acc2 if (args.past or not args.full_white) else acc_c, acc, acc_c
                else:
                    delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate).detach()
                    X_adv = X + delta
                    output = model(X_adv)
                    loss = criterion(output, y)
                    loss.backward()
                    # labs = (y.max(1)[1] + 1)% 10 #
                    # labs = F.one_hot(labs, 10).float() 
                    # d2 = pgd_attack(model_copy, X, labs, epsilon_base, alpha, args, criterion, handle_list, drop_rate).detach()
                    # out2 = model(X + d2).detach()
                    outc = model(X).detach()
                    # acc2 = (out2.max(1)[1] == labs.max(1)[1]).float().mean().item()#(thingy * labs).sum(1).mean().item()#(out2.max(1)[1] == labs.max(1)[1]).float().mean().item()
                    acc = (output.max(1)[1] == y.max(1)[1]).float().mean().item()
                    acc_c = (outc.max(1)[1] == y.max(1)[1]).float().mean().item()#cosim(fw[:, args.prompt_length, :], fb[:, args.prompt_length, :]).detach().mean().item()
                    for j in range(y.size(0)):
                        corr_mats[1][y.max(1)[1][j], output.detach().max(1)[1][j]] += 1
                        corr_mats[0][y.max(1)[1][j], outc.detach().max(1)[1][j]] += 1
                        # corr_mats[2][y.max(1)[1][j], out2.detach().max(1)[1][j]] += 1
                    return loss, acc, y, acc_c, acc, acc_c
                # output = model(X, prompt)
            elif args.method == 'voting':
                ####### VOTING CAN BE USED FOR ENSMBLING LATER ###########
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                ds = []
                tds = []
                # tar = F.one_hot((y.max(1)[1])%10, 10).float().cuda()
                for i, p in enumerate(prompts):
                
                    d = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=p).detach()
                    # td = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=p, target=y).detach()
                    # tds.append(td)
                    ds.append(d)

                    #### DELTED TRAINING CODE FOR NOW ######
                    

                    accs[i] = acc                
                    opts[i].zero_grad()
                    model.zero_grad()
                    loss.backward()
                    opts[i].step()
                    opts[i].zero_grad()
                    model.zero_grad()
                    losses += loss.item()

                accs_vote = torch.zeros(len(prompts))
                losses /= len(prompts)
                


                delta = 0

                return accs, accs_vote, losses, 0, 0

                      
            elif args.method == 'detect':
                X = X.cuda()
                y = y.cuda()
                bsize = X.size(0)
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                a_label = (nclasses - 1) * torch.ones_like(y.max(1)[1])
                a_label = F.one_hot(a_label, nclasses).float()
                y = torch.cat((y, torch.zeros(y.size(0), 1).cuda()), dim=1)
                # print(y[:,10], a_label[:, 10])
                if args.attack_type == 'pgd':
                    delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt).detach()
                elif args.attack_type == 'cw':
                    delta = cw_attack(model, X, y.max(1)[1], epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt, a_lam=args.a_lam, detection=True).detach()
                elif args.attack_type == 'acw':
                    delta = cw_attack(model, X, y.max(1)[1], epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt, acw=True).detach()
                X_adv = X + delta
                outa = model(X_adv, prompt)
                outc = model(X, prompt)
                ##### TODO : try separating the adv probability from the class predictions, alternatively adding a separate prompt #####
                # loss = (1 - args.d_lam) * criterion(outc[:, :-1], y[:, :-1]) + args.d_lam * (bceloss(outc[:, -1], torch.zeros_like(y.max(1)[1]).float())
                #      + bceloss(outa[:, -1], torch.ones_like(y.max(1)[1]).float()))
                loss = (1 - args.d_lam) * criterion(outc, y) + args.d_lam * criterion(outa, a_label)  #- args.d_lam * torch.minimum(outa[:, -1], torch.tensor(100).cuda()).mean()#*(torch.minimum(outa[:, -1] - torch.max(outa[:, :-1], dim=1)[0].detach(), torch.tensor(10).cuda())).mean() 
                # loss = (1 - args.d_lam) * torch.maximum(outc[:, -1] - outc[np.arange(bsize), y.max(1)[1]], torch.tensor(-10).cuda()
                #  ).mean() + args.d_lam * torch.maximum(outa.max(dim=1)[0] - outa[:, -1], torch.tensor(-10).cuda()).mean()
                model.zero_grad()
                loss.backward()
                
                # if args.attack_type == 'pgd':
                #     d_a = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt, avoid=a_label, a_lam=args.a_lam).detach()
                # elif args.attack_type in ['cw', 'acw']:
                    # print('sag')
                d_a = cw_attack(model, X, y.max(1)[1], epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt, a_lam=args.a_lam, detection=True).detach()
                outad = model(X + d_a, prompt).detach()
                acc_c = (outc.max(1)[1] == y.max(1)[1]).float().mean().item()
                for j in range(y.size(0)):
                    corr_mats[1][y.max(1)[1][j], outa[:, :-1].detach().max(1)[1][j]] += 1
                    corr_mats[0][y.max(1)[1][j], outc[:, :-1].detach().max(1)[1][j]] += 1
                    corr_mats[2][y.max(1)[1][j], outad[:, :-1].detach().max(1)[1][j]] += 1

                acc_a = (outa.max(1)[1] == a_label.max(1)[1]).float().mean().item()#(outa[:, -1] > torch.zeros_like(y.max(1)[1])).float().mean().item() 
                acc = (outad[:, :-1].max(1)[1] == y.max(1)[1]).float().mean().item()#(outc[:, -1] <= torch.zeros_like(y.max(1)[1])).float().mean().item()
                acc_d = (outad.max(1)[1] == a_label.max(1)[1]).float().mean().item()#(outad[:, -1] > torch.zeros_like(y.max(1)[1])).float().mean().item()
                return loss, acc, y, acc_a, acc_d, acc_c
            elif args.method == 'sepdet' and epoch >= args.ws:
                X = X.cuda()
                y = y.cuda()
                bsize = X.size(0)
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)

                y = torch.cat((y, torch.zeros(y.size(0), 1).cuda()), dim=1)
                delta = sepdet_atk(model, X, y.max(1)[1], epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, disc=disc, pd=dprompt, pc=prompt, a_lam=args.a_lam).detach()
                noise = sepdet_atk(model, X, y.max(1)[1], epsilon_base, alpha, 0, 1, lower_limit, upper_limit, disc=disc, pd=dprompt, pc=prompt, a_lam=args.a_lam).detach()
                X_adv = X + delta
                _, fsadv = model(X_adv, dprompt, get_fs=True)
                sadv = disc(fsadv[:, dprompt.size(1)])
                _, fsc = model(X + noise, dprompt, get_fs=True)
                sc = disc(fsc[:, dprompt.size(1)])
                loss = 0.5 * bceloss(sc, torch.zeros_like(sc).float()) + 0.5 * bceloss(sadv, torch.ones_like(sadv).float())
                loss.backward()
                acc_c = (sc < torch.zeros_like(y.max(1)[1])).float().mean().item()
                acc_a = (sadv > torch.zeros_like(y.max(1)[1])).float().mean().item()

                return loss, acc_c, y, acc_a, acc_a, acc_c

            elif args.method == 'TRADES':
                X = X.cuda()
                y = y.cuda()
                epsilon = epsilon_base.cuda()
                beta = args.beta
                batch_size = len(X)
                if args.delta_init == 'random':
                    delta = 0.001 * torch.randn(X.shape).cuda()
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                criterion_kl = nn.KLDivLoss(size_average=False)
                model.eval()

                delta.requires_grad = True
                condition = args.prompted or args.prefixed
                for _ in range(args.attack_iters):
                    loss_kl = criterion_kl(F.log_softmax(model(X + delta, prompt) if condition else model(X + delta), dim=1),
                                           F.softmax(model(X, prompt) if condition else model(X), dim=1))
                    grad = torch.autograd.grad(loss_kl, [delta])[0]
                    # prompt = prompt.detach()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                if len(handle_list) != 0:
                    for handle in handle_list:
                        handle.remove()
                # model.train()
                delta = delta.detach()
                # x_adv = Variable(X+delta, requires_grad=False)
                output = logits = model(X, prompt) if condition else model(X)
                output = output.detach()
                # logits = logits.detach()
                acc_c = (output.max(1)[1] == y).float().mean().item()
                loss_natural = F.cross_entropy(logits, y)
                loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(X+delta, prompt) if condition else model(X+delta), dim=1),
                                                                F.softmax(model(X, prompt) if condition else model(X), dim=1))
                loss = loss_natural + beta * loss_robust
                model.zero_grad()
                loss.backward()

                dpgd = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt if condition else None).detach()
                outa = model(X + dpgd, prompt).detach() if condition else model(X + dpgd).detach()
                acc_a = (outa.detach().max(1)[1] == y).float().mean().item()
                for j in range(y.size(0)):
                    corr_mats[1][y[j], outa.detach().max(1)[1][j]] += 1
                    corr_mats[0][y[j], output.detach().max(1)[1][j]] += 1
                    
                return loss, acc_c, y, acc_a, acc_a, acc_c
            elif args.method == 'ss':
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                

                outc, phic = model(X, prompt, get_fs=True)
                delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt).detach()
                outa, phia = model(X + delta, prompt, get_fs=True)
                outa = outa.detach()
                loss_ce = criterion(outc, y)
                loss_fs = mseloss(phic, phia)
                loss = loss_ce + beta * loss_fs
                loss.backward()

                acc_c = (outc.detach().max(1)[1] == y).float().mean().item()
                acc_a = (outa.detach().max(1)[1] == y).float().mean().item()
                for j in range(y.size(0)):
                    corr_mats[1][y[j], outa.detach().max(1)[1][j]] += 1
                    corr_mats[0][y[j], outc.detach().max(1)[1][j]] += 1
                    
                return loss, acc_c, y, acc_a, acc_a, acc_c
                
            else:
                raise ValueError(args.method)
            opt.zero_grad()
            model.zero_grad()
            (loss / args.accum_steps).backward()
            if args.method == 'AT':
                acc = (output.max(1)[1] == y.max(1)[1]).float().mean()
                if args.prompted:
                    if args.disjoint_prompts:
                        delta = simul_pgd(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate,prompts=[prompt, prompt2]).detach()
                        out = (model(X + delta, prompt2) + model(X+delta, prompt))/2
                        p_acc = (out.max(1)[1] == y.max(1)[1]).float().mean().item()
                    else:
                        delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt).detach()
                        out = model(X+delta, prompt)
                        p_acc = (out.max(1)[1] == y.max(1)[1]).float().mean().item()
                    return loss, acc, y, p_acc, handle_list
            else:
                acc = (output.max(1)[1] == y.max(1)[1]).float().mean()
                if args.disjoint_prompts and args.prompted:
                    delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt).detach()
                    out = model(X+delta, [prompt2, prompt]).detach()
                    p_acc = (out.max(1)[1] == y.max(1)[1]).float().mean().item()
                    return loss, acc, y, p_acc, handle_list
            
            return loss, acc,y, acc.item(), handle_list

        for step, (X, y) in enumerate(train_loader):
            batch_size = args.batch_size // args.accum_steps
            epoch_now = epoch - 1 + (step + 1) / len(train_loader)

            X_ = X[0: batch_size].cuda()  
            y_ = y[0: batch_size].cuda()  
            
            loss, acc,y, p_acc, acc_ad, clean_acc = train_step(X,y,epoch_now,mixup_fn, hist_a, hist_c, corr_mats)
            train_loss += loss.item() * y_.size(0)
            train_acc += acc * y_.size(0)
            
            train_prompted += p_acc * y_.size(0)
            train_adetect += acc_ad * y_.size(0)
            train_clean += clean_acc * y_.size(0)
            train_n += y_.size(0)
            grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) , args.grad_clip)
            opt.step()
            opt.zero_grad()
            model.zero_grad()
            
            if (step + 1 == steps_per_epoch or (step + 1) % 200 == 0) and args.method != 'natural':
                fg, axarr = plt.subplots(1,4)
            
                axarr[0].matshow(corr_mats[0]/train_n)
                axarr[0].yaxis.tick_left()
                axarr[0].set_title('clean samples')
                axarr[0].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_clean/train_n * 100))
                axarr[0].set_ylabel('ground truth label')
                axarr[1].matshow(corr_mats[1]/train_n)
                axarr[1].yaxis.tick_left()
                axarr[1].set_title('perturbed loss(y)')
                axarr[1].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_acc/train_n * 100))
                axarr[2].yaxis.tick_left()
                axarr[2].set_title('perturbed loss(y + 1)')
                axarr[2].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_prompted/train_n * 100))
                axarr[2].matshow(corr_mats[2]/train_n)
                plt.savefig(args.out_dir + "/mat_present_"+str(epoch)+"step_" + str(step) + ".png", dpi=500)
                plt.close()
            if (step + 1) % args.log_interval == 0 or step + 1 == steps_per_epoch:
                logger.info('Training epoch {} step {}/{}, lr {:.4f} loss {:.4f} adaptive acc {:.4f} clean acc {:.4f} detect acc {:.4f} adetect acc {:.4f}'.format(
                    epoch, step + 1, len(train_loader),
                    opt.param_groups[0]['lr'],
                        train_loss / train_n, train_acc / train_n, train_clean/ train_n, train_prompted/ train_n, train_adetect/ train_n
                ))
            lr = lr_schedule(epoch_now, args.ws) if (args.ws and epoch_now < args.ws) else lr_schedule(epoch_now, args.epochs) 
            opt.param_groups[0].update(lr=lr)
        
        path = os.path.join(args.out_dir, 'checkpoint_{}'.format(epoch))

        if epoch == args.epochs or epoch % args.chkpnt_interval == 0:
            if not args.prompted and not args.prefixed:
                prompt = None
            to_save = {'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict(), 'prompt': [prompt]}
            if args.method == 'sepdet':
                to_save['detp'] = [dprompt, disc]
            torch.save(to_save, path)
            logger.info('Checkpoint saved to {}'.format(path))


train_adv(args, model, train_loader, test_loader, logger)


logger.info(args.out_dir)
if not args.prompted and not args.prefixed:
    prompt = None
evaluate_natural(args, model, test_loader, verbose=False, prompt=prompt)


chkpnt = None
args.eval_iters = 10
args.alpha = 2
args.eval_restarts = 1
pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader, prompt=prompt, a_lam=args.a_lam)
logger.info('PGD10 : loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc))


args.eval_iters = 20
args.alpha = 2
cw_loss, cw_acc = evaluate_CW(args, model, test_loader, prompt=prompt, a_lam=0)
logger.info('CW20: loss {:.4f} acc {:.4f}'.format(cw_loss, cw_acc))

args.eval_iters = 50
args.alpha = 2/5
args.eval_restarts = 10
pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader, prompt=prompt, a_lam=args.a_lam)
logger.info('PGD50-10: loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc))


# logger.info('Moving to AA...')
# at_path = os.path.join(args.out_dir, 'result_'+'_autoattack.txt')
# evaluate_aa(args, model,at_path, args.AA_batch, prompt if args.prompted else None)