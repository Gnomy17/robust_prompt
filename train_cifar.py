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
from pgd import evaluate_pgd,evaluate_CW,evaluate_splits, CW_loss
from evaluate import evaluate_aa
from auto_LiRPA.utils import logger
import matplotlib.pyplot as plt
from buffer import Buffer
# torch.autograd.set_detect_anomaly(True)
args = get_args()
joint_p = lambda x, y: torch.cat((x, y), dim=1) if y is not None else x 
def make_prompt(length, h_dim, init_xavier=True):
    prompt = torch.zeros(1, length, h_dim, requires_grad=True)
    prompt.cuda()
    if init_xavier:
        nn.init.xavier_uniform_(prompt)
    # prompt = nn.Parameter(prompt)
    return prompt

args.out_dir = args.out_dir + args.dataset+"_"+args.model+"_"+args.method
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


if args.prompted or args.prompt_too or args.method in ['past_at', 'splits', 'updown']:
    # from model_for_cifar.prompt import Prompt
    # prompt = Prompt(args.prompt_length, 768)
    
    # if args.load:
    #     prompt.load_state_dict(checkpoint['prompt'])
    # params = []
    # prompt.train()
    # prompt.cuda()
    if args.method == 'splits':
        done_prompt = None
    
    if args.load:
        if args.method == 'splits':
            if args.just_eval:
                prompt = joint_p(checkpoint['prompt'][0].detach(), checkpoint['done'][0].detach())
            else:
                prompt = make_prompt(args.prompt_length, 768)#checkpoint['prompt'][0]
                done_prompt = checkpoint['done'][0]
        else:
            prompt = (checkpoint['prompt'])[0]
    else:
        prompt = make_prompt(args.prompt_length, 768)
    if args.method == 'past_at':
        past_prompt = prompt.clone()
        buff = Buffer(args.buffer_size, mode = args.buffer_type,device='cuda')
    prompts = [prompt]
    params = [prompt]
    # params = {'prompt': prompt}
    # prev_p = torch.zeros_like(prompt)
    # prev_p.requires_grad = False
    if args.disjoint_prompts:
        prompt2 = make_prompt(args.prompt_length, 768)
        params.append(prompt2)
    # if args.method == 'PAT_tar':
    #     dim = 768
    #     pred_dim = 512
    #     predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
    #                                 nn.BatchNorm1d(pred_dim),
    #                                 nn.ReLU(inplace=True), # hidden layer
    #                                 nn.Linear(pred_dim, dim))
    #     predictor.cuda()
    #     params += list(predictor.parameters())
    if args.prompt_too:
        for p in model.parameters():
            params.append(p)
    else:
        for p in model.module.head.parameters():
            params.append(p)
        
    if args.optim == 'sgd':
        opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay) 
    elif args.optim == 'adam':
        opt = torch.optim.Adam(params, lr=args.lr_max, weight_decay=args.weight_decay)
    if args.method == 'updown':
        pup = prompt
        optup = opt
        pdown = make_prompt(args.prompt_length, 768)
        dparams = [pdown]
        # for p in model.module.head.parameters():
        #     dparams.append(p)
        optdown = torch.optim.SGD(dparams, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay) 
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
elif args.blocked:
    from model_for_cifar.vit import PatchEmbed, Block
    from model_for_cifar.prompt import PromptBlock
    print(args.block_type)
    if args.block_type == "attention":
        prompt = nn.Sequential(PatchEmbed(img_size=crop_size, patch_size=args.patch, in_chans=3, embed_dim=768), Block(768, 12))
    elif args.block_type == "cnn":
        prompt = PromptBlock(img_size=crop_size, patch_size= args.patch, in_chans=3, middle_dim=768 ,embed_dim= 768, stride=4)
    prompt.cuda()
    prompt.train()
    params = []
    for p in prompt.parameters():
        params.append(p)
    for p in model.module.head.parameters():
        params.append(p) 
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    model_copy = vit_small_patch16_224(pretrained = (not args.scratch),img_size=crop_size,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
    model_copy = nn.DataParallel(model_copy)
    # print(model_copy == model, model == model)
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
                output = model(X, prompt)
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

def cw_attack(model, X, y, epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit, opt=None, prompt=None, a_lam=0):
    # max_loss = torch.zeros(y.shape[0]).cuda()
    # max_delta = torch.zeros_like(X).cuda()
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
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y)
            if a_lam != 0:
                a_label = torch.ones_like(y) * (output.size(1) - 1)
                loss += a_lam * CW_loss(output, a_label)

            grad = torch.autograd.grad(loss, delta)[0].detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta = delta.detach()
        # if prompt is not None:
        #     all_loss = CW_loss(model(X+delta, prompt), y, reduction=False).detach()
        # else:
        #     all_loss = CW_loss(model(X+delta), y, reduction=False).detach()
        # max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        # max_loss = torch.max(max_loss, all_loss)
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
        # patch drop
        add_noise_mask = torch.ones_like(X)
        grid_num_axis = int(args.resize / args.patch)
        max_num_patch = grid_num_axis * grid_num_axis
        ids = [i for i in range(max_num_patch)]
        random.shuffle(ids)
        num_patch = int(max_num_patch * (1 - drop_rate))
        if num_patch !=0:
            ids = np.array(ids[:num_patch])
            rows, cols = ids // grid_num_axis, ids % grid_num_axis
            for r, c in zip(rows, cols):
                add_noise_mask[:, :, r * args.patch:(r + 1) * args.patch,
                c * args.patch:(c + 1) * args.patch] = 0
        if args.PRM:
            delta = delta * add_noise_mask
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



# def CW_loss(out, y_oh, t=0):
#     loss = 0
#     y = y_oh.max(1)[1]
#     for j in range(out.size(1)):
#         inds = torch.Tensor([k for k in range(out.size(1)) if k != j]).long()
#         outs_c = out[y == j]
#         loss += torch.max(torch.max(outs_c[:, inds]) - outs_c[:, j], t)[0].sum()
#     loss /= out.size(0)
#     return loss

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

    # if mixup_active:
    #     criterion = SoftTargetCrossEntropy()
    # else:
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    
    
    if args.load:
        # if args.method == 'voting':
        #     for i, o in enumerate(opts):
        #         o.load_state_dict(checkpoint['opts'][i])
        # else:
        # opt.load_state_dict(checkpoint['opts'])
        logger.info("Resuming at epoch {}".format(checkpoint['epoch'] + 1))
        # del checkpoint
    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    lr_steps = args.epochs * steps_per_epoch
    def lr_schedule(t):
        if t< args.epochs-5:
            return args.lr_max
        elif t< args.epochs-2:
            return args.lr_max*0.1
        else:
            return args.lr_max * 0.01
    epoch_s = 0 if not args.load else (checkpoint['epoch'])
    if args.load:
        for k in checkpoint:
            checkpoint[k] = None
    # prev_prompt = Prompt(args.prompt_length, 768)
    # prev_prompt.set_prompt(prompt)
    # last_ind = args.prompt_length
    # current_ind = int(args.prompt_length* (4/5))
    for epoch in tqdm.tqdm(range(epoch_s + 1, args.epochs + 1)):
        if args.just_eval:
            break
        # evaluate_natural(args, model, test_loader, verbose=False)
        train_loss = 0
        train_acc = 0
        train_clean = 0
        train_prompted = 0
        train_n = 0
        hist_c = torch.zeros((10)).cuda()
        hist_a = torch.zeros((10)).cuda()
        corr_mats = [torch.zeros((nclasses,nclasses)) for _ in (range(4))]
        fvs = []
        fvs_a = []
        pfvs = []
        pfvs_a = []
        flabels = []
        # p10 = prompt.detach().clone()
        # p10.requires_grad = False

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
            if args.method == 'AT' or (args.method == 'ws' and  args.ws <= epoch):
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                
                if args.prompted or args.prompt_too:
                    if args.full_white:
                        delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt).detach()
                        # td = pgd_attack(model, X, None, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt, target=y).detach()
                        # prev_prompt.set_prompt(prompt)
                    elif args.all_classes:
                        i = torch.randint(low=1, high=y.size(1), size=y.max(1)[1].size()).cuda()
                        tars = F.one_hot((y.max(1)[1] + i) % y.size(1), y.size(1))
                        # print(tars.size())
                        delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt, target=tars).detach()
                        output = model(X + delta, prompt)
                        loss = criterion(output, y)
                    else:
                        # labs = torch.ones_like(y.max(1)[1]) * 5 # (y.max(1)[1] + 1)% 10 #
                        # labs = F.one_hot(labs, 10).float()
                        delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate).detach() #prompt=prompt, avoid=labs).detach()
                        t = F.one_hot((y.max(1)[1] + 1) % 10, 10).float()
                        d2 =  pgd_attack(model, X, t, epsilon_base, alpha, args, criterion, handle_list, drop_rate).detach()
                        # d2 = d2[torch.randperm(d2.size()[0])]
                        # out2, fw = model(X + d2, prompt, get_fs=True)
                        # out2 = out2.detach()
                        # fw = fw.detach()
                    X.detach()
                    out = model(X + delta, prompt)#, get_fs=True)
                    outc = model(X, prompt)
                    # fb = fb.detach()
                    out2 = model(X + d2, prompt)
                    # output = (outa + outt).detach()/2
                    # outc, fc = model(X, prompt, get_fs=True)
                    loss = criterion(out, y) #+ criterion(outc, y) #+ criterion(outt, y)
                    loss.backward()
                    
                    # diff = fw[:, args.prompt_length, :] - fc[:, args.prompt_length, :]
                    # diff_n = torch.norm(diff, dim=1).unsqueeze(1).expand(y.size(0), 10)
                    # W = model.module.head.weight.detach()
                    # W_n = torch.norm(W, dim=1).expand(y.size(0), 10)
                    # # print(diff_n.size(), W_n.size())
                    # thingy = (model.module.head(diff)/(W_n * diff_n)).detach()
                    # cosim = nn.CosineSimilarity()
                    acc = (out.max(1)[1] == y.max(1)[1]).float().mean().item()
                    acc_c = (outc.max(1)[1] == y.max(1)[1]).float().mean().item()#cosim(fw[:, args.prompt_length, :], fb[:, args.prompt_length, :]).detach().mean().item()
                    acc2 = (out2.max(1)[1] == t.max(1)[1]).float().mean().item()#(thingy * labs).sum(1).mean().item()#(out2.max(1)[1] == labs.max(1)[1]).float().mean().item()
                    for j in range(y.size(0)):
                        corr_mats[1][y.max(1)[1][j], out.detach().max(1)[1][j]] += 1
                        corr_mats[0][y.max(1)[1][j], outc.detach().max(1)[1][j]] += 1
                        corr_mats[2][y.max(1)[1][j], out2.detach().max(1)[1][j]] += 1
                    return loss, acc, y, acc2, handle_list, acc_c
                    
                # elif args.method == 'detect':
                #     a_label = (nclasses - 1) * torch.ones_like(y.max(1)[1])
                #     a_label = F.one_hot(a_label, nclasses).float()
                #     y = torch.cat((y, torch.zeros_like(y.max(1)[1])), dim=1)
                #     print(y.size(), a_label.size())
                #     delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt, avoid=a_label).detach()
                #     X_adv = X + delta
                #     output = model(X_adv, prompt(X_adv))
                #     loss = criterion(output, y)
                #     if args.mix_lam > 0:
                #         out = model(X, prompt(X))
                #         loss += (args.mix_lam * criterion(out, y))
                #         loss /= (1 + args.mix_lam)
                else:
                    delta = pgd_attack(model_copy, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate).detach()
                    X_adv = X + delta
                    output = model(X_adv)
                    loss = criterion(output, y)
                    loss.backward()
                    labs = (y.max(1)[1] + 1)% 10 #
                    labs = F.one_hot(labs, 10).float() 
                    d2 = pgd_attack(model_copy, X, labs, epsilon_base, alpha, args, criterion, handle_list, drop_rate).detach()
                    out2 = model(X + d2).detach()
                    outc = model(X).detach()
                    acc2 = (out2.max(1)[1] == labs.max(1)[1]).float().mean().item()#(thingy * labs).sum(1).mean().item()#(out2.max(1)[1] == labs.max(1)[1]).float().mean().item()
                    acc = (output.max(1)[1] == y.max(1)[1]).float().mean().item()
                    acc_c = (outc.max(1)[1] == y.max(1)[1]).float().mean().item()#cosim(fw[:, args.prompt_length, :], fb[:, args.prompt_length, :]).detach().mean().item()
                    for j in range(y.size(0)):
                        corr_mats[1][y.max(1)[1][j], output.detach().max(1)[1][j]] += 1
                        corr_mats[0][y.max(1)[1][j], outc.detach().max(1)[1][j]] += 1
                        corr_mats[2][y.max(1)[1][j], out2.detach().max(1)[1][j]] += 1
                    return loss, acc, y, acc2, handle_list, acc_c
                # output = model(X, prompt)
            elif args.method == 'voting':
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
                accs = torch.zeros(len(prompts))
                losses = 0
                for i, p in enumerate(prompts):
                    loss = 0
                    acc = 0
                    out_c = model(X, p)
                    hist_c += F.one_hot(out_c.max(1)[1], 10).sum(dim=0)
                    inds = y.max(1)[1]
                    inds_c = out_c.max(1)[1]
                    for j in range(y.size(0)):
                        corr_mats[i][0, inds[j], inds_c[j]] += 1

                    if args.voting_method == 'all':
                        ind = i + 1 if i%2 == 0 else i - 1
                        next_d = ds[ind]
                        out = model(X + next_d, p)
                        
                        acc += (out.max(1)[1] == y.max(1)[1]).float().mean().item()
                        loss += criterion(out, y) #+ F.mse_loss()
                        for j in range(y.size(0)):
                            corr_mats[i][ind + 1, inds[j], out.max(1)[1][j]] += 1

                        with torch.no_grad():
                            for k, d in enumerate(ds):
                                if k == ind:
                                    continue
                                oute = model(X + d, p).detach()
                                for j in range(y.size(0)):
                                    corr_mats[i][k + 1, inds[j], oute.max(1)[1][j]] += 1
                        
                    elif args.voting_method == 'self':
                        out = model(X + ds[i], p)
                        inds_u = out.max(1)[1]
                        for j in range(y.size(0)):
                            corr_mats[i][i + 1, inds[j], inds_u[j]] += 1
                        hist_a += F.one_hot(out.max(1)[1], 10).sum(dim=0)
                        acc += (y.max(1)[1] == out.max(1)[1]).float().mean().item()
                        
                        # unilabs = torch.ones_like(y)/10
                        # unilabs = F.one_hot((y.max(1)[1] + 1) % 10).float()
                        loss += criterion(out, y) + criterion(out_c, y)
                        with torch.no_grad():
                            for k, d in enumerate(ds):
                                if k == (i):
                                    continue
                                outu = model(X + d, p).detach()
                                for j in range(y.size(0)):
                                    corr_mats[i][k + 1, inds[j], outu.max(1)[1][j]] += 1
                    elif args.voting_method == 'rand':
                        # ds = torch.stack(ds)
                        # inds = torch.randint(low=0, high=len(prompts), size=(ds.size(0),))
                        # rand_d = ds[:, :, :, :, inds]
                        # cosim = nn.CosineSimilarity(2)
                        ind = (i+(1))%len(ds)
                        # ind = (i + torch.randint(low=1, high=len(ds), size=1).item()) % len(ds)
                        # print(ind)
                        next_d = ds[ind]
                        out = model(X + next_d, p)
                        
                        acc += (out.max(1)[1] == y.max(1)[1]).float().mean().item()
                        loss += criterion(out, y) #+ F.mse_loss()

                        outu = model(X + tds[ind], p).detach()
                        for j in range(y.size(0)):
                            corr_mats[i][i + 1, inds[j], outu.max(1)[1][j]] += 1
                        # indu = (ind + 1) % 2
                        # for j in range(y.size(0)):
                        #     corr_mats[i][indu + 1, inds[j],out.max(1)[1][j]] += 1
                        with torch.no_grad():
                            for k, d in enumerate(ds):
                                if k == i:
                                    continue
                                oute = model(X + d, p).detach()
                                for j in range(y.size(0)):
                                    corr_mats[i][k + 1, inds[j], oute.max(1)[1][j]] += 1
                    
                    # loss += criterion(out_c, y)
                    # past_prompts
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
                
                
                #acc_clean, outs = majority_vote(model, X, y, prompts)

                delta = 0#simul_pgd(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompts=prompts).detach()
                #acc_adv, outs = majority_vote(model, X + delta, y, prompts)
                # for i, p in enumerate(prompts):
                    # outc = model(X, p).detach()
                    # labs = F.one_hot(outc.max(1)[1], 10).float()
                    # d = pgd_attack(model, X, labs, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=p).detach()
                    # outu = model(X + d, p).detach()
                    
                    # majority_vote(model, X + d, y, prompts)
                # for k, o in enumerate(outs):
                #     # print(o.size())
                #     accs_vote[k] = (o == y.max(1)[1]).float().mean().item()
                #     for j in range(y.size(0)):
                #         corr_mats[k][len(prompts) + 1, inds[j], o[j]] += 1
                return accs, accs_vote, losses, 0, 0

            elif args.method == 'natural' or (args.method == 'ws' and epoch < args.ws):
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                if args.prompted or args.prompt_too:
                    output, pfv = model(X, prompt, get_fs=True)
                elif args.blocked:
                    output = model(X, prompt(X))
                else:
                    output = model(X)
                # print(output.shape, y.shape)
                loss = criterion(output, y)
                loss.backward()
                acc = (output.max(1)[1] == y.max(1)[1]).float().mean().item()
                # if step >= 770:
                #     ls = y.max(1)[1].detach()
                #     delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt).detach()
                #     output, pfv_a = model(X + delta, prompt, get_fs=True)
                #     delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=None).detach()
                #     _, fv_a = model(X + delta, get_fs=True)
                #     _, fv = model(X, get_fs=True)
                #     fvs_a.append(fv_a[ls % 5 == 0, args.prompt_length, :].detach())
                #     fvs.append(fv[ls % 5 == 0, args.prompt_length, :].detach())
                #     pfvs_a.append(pfv_a[ls % 5 == 0, args.prompt_length, :].detach()) 
                #     pfvs.append(pfv[ls % 5 == 0 , args.prompt_length, :].detach()) 
                    
                #     flabels.append(ls[ls % 5 == 0])
                #     # print('sag')
                return loss, acc, y, acc, handle_list, acc
            elif args.method == 'updown':
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                # outc, fc = model(X, get_fs=True)
                # outc = outc.detach()
                c_y = torch.ones(y.size(0)).long() * 5 
                c_y = F.one_hot(c_y, 10).float().cuda()
                dup = pgd_attack(model, X, c_y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=pup).detach()
                ddown = pgd_attack(model, X, None, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=pdown, target=c_y).detach()
                outup, fdu = model(X + dup, pup, get_fs=True)
                loss = criterion(outup, y)
                optup.zero_grad()
                loss.backward()
                optup.step()
                optup.zero_grad()
                model.zero_grad()
                outdown, fud = model(X + ddown, pdown, get_fs=True)
                loss = criterion(outdown, y)
                optdown.zero_grad()
                loss.backward()
                optdown.step()
                optdown.zero_grad()
                model.zero_grad()
                # out_u, fuu = model(X + dup, pup, get_fs=True)
                out_uc, fuc = model(X, pup, get_fs=True)
                # out_d, fdd = model(X + ddown, pdown, get_fs=True)
                out_dc, fdc = model(X, pdown, get_fs=True)
                # out_dc = out_dc.detach()
                # out_uc = out_uc.detach()
                # out_u = out_u.detach()
                # out_d = out_d.detach()
                cosim = nn.CosineSimilarity()
                w_5 = model.module.head.weight.detach()[5, :]

                # diffu = fdu[:,args.prompt_length, :] - fdc[:,args.prompt_length,:]
                # print(torch.norm(fud[:, args.prompt_length, :], dim=1).size())
                # predu = model.module.head(fuu[:, args.prompt_length, :]/torch.norm(fuu[:, args.prompt_length, :], dim=1).unsqueeze(1) 
                #     + fud[:, args.prompt_length, :]/torch.norm(fud[:, args.prompt_length, :], dim=1).unsqueeze(1)).detach()
                # predd = model.module.head(fdd[:, args.prompt_length, :]/torch.norm(fdd[:, args.prompt_length, :], dim=1).unsqueeze(1)
                #      + fdu[:, args.prompt_length, :]/torch.norm(fdu[:, args.prompt_length, :], dim=1).unsqueeze(1)).detach()
                # print(predu.size())
                
                # print(model.module.head(fuu[:, args.prompt_length, :]).size())
                # print(torch.norm(model.module.head(fuu[:, args.prompt_length, :]).detach() - out_u))
                # print(fc.size())
                # acc_c = (outc.detach().max(1)[1] == y.max(1)[1]).float().mean()
                acc_c = (outup.max(1)[1] == y.max(1)[1]).float().mean().item() +  (outdown.max(1)[1] == y.max(1)[1]).float().mean().item()
                # out_ud = model(X, torch.cat((pup, pdown), dim=1)).detach()
                for j in range(X.size(0)):
                    corr_mats[0][y.max(1)[1][j], outup.detach().max(1)[1][j]] += 1
                    corr_mats[1][y.max(1)[1][j], outdown.detach().max(1)[1][j]] += 1
                    corr_mats[2][y.max(1)[1][j], out_uc.detach().max(1)[1][j]] += 1
                    corr_mats[3][y.max(1)[1][j], out_dc.detach().max(1)[1][j]] += 1
                acc = cosim(w_5, fuc[:, args.prompt_length, :]).detach().mean().item()
                acc2 = cosim(w_5, fdc[:, args.prompt_length, :]).detach().mean().item()
                 # acc_c = (out_ud.max(1)[1] == outc.max(1)[1]).float().mean()
                return loss, acc, y, acc2, handle_list, acc_c
            elif args.method == 'detect':
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                a_label = (nclasses - 1) * torch.ones_like(y.max(1)[1])
                a_label = F.one_hot(a_label, nclasses).float()
                y = torch.cat((y, torch.zeros(y.size(0), 1).cuda()), dim=1)
                # print(y[:,10], a_label[:, 10])
                if args.attack_type == 'pgd':
                    delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt).detach()
                elif args.attack_type == 'cw':
                    delta = cw_attack(model, X, y.max(1)[1], epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt).detach()
                X_adv = X + delta
                outa = model(X_adv, prompt)
                outc = model(X, prompt)
                loss = criterion(outc, y) + .1*criterion(outa, a_label)
                model.zero_grad()
                loss.backward()
                
                if args.attack_type == 'pgd':
                    d_a = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt, avoid=a_label).detach()
                elif args.attack_type == 'cw':
                    d_a = cw_attack(model, X, y.max(1)[1], epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt, a_lam=1).detach()
                outad = model(X + d_a, prompt).detach()
                acc_c = (outc.max(1)[1] == y.max(1)[1]).float().mean().item()
                for j in range(y.size(0)):
                    corr_mats[1][y.max(1)[1][j], outa.detach().max(1)[1][j]] += 1
                    corr_mats[0][y.max(1)[1][j], outc.detach().max(1)[1][j]] += 1
                    corr_mats[2][y.max(1)[1][j], outad.detach().max(1)[1][j]] += 1
                    # corr_mats[2][y.max(1)[1][j], outu.detach().max(1)[1][j]] += 1
                # for j in range(y.size(0), y_adv.size(0)):
                #     corr_mats[2][y_adv.max(1)[1][j], output.detach().max(1)[1][j]]
                acc_a = (outa.max(1)[1] == a_label.max(1)[1]).float().mean().item() 
                acc = (outad.max(1)[1] == y.max(1)[1]).float().mean().item()
                return loss, acc, y, acc_a, handle_list, acc_c
            elif args.method == 'splits':
                
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                # yc = y.detach().clone()
                # X = X[y.max(1)[1] != 9]
                # y = y[y.max(1)[1] != 9]
                outc = model(X, joint_p(prompt, done_prompt))
                delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=joint_p(prompt, done_prompt)).detach()
                losses= 0
                # print(prompt[:,current_ind:last_ind,:].size())
                
                
                deltaa = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=done_prompt).detach()
                outa = model(X + deltaa, joint_p(prompt, done_prompt)).detach()
                acc_a = (outa.max(1)[1] == y.max(1)[1]).float().mean().item() 

                out = model(X + delta, joint_p(prompt, done_prompt))
                loss = criterion(out, y) + criterion(outc, y)
                # if done_prompt is not None:
                #     print(torch.autograd.grad(loss, done_prompt))
                loss.backward()
                # if last_ind < args.prompt_length:
                    # prompt.grad[:,last_ind:,:] *= 0
                losses += loss.item()
                acc = (out.max(1)[1] == y.max(1)[1]).float().mean().item()
                for j in range(y.size(0)):
                    corr_mats[0][0, y.max(1)[1][j], outc.detach().max(1)[1][j]] += 1
                    corr_mats[0][1, y.max(1)[1][j], out.detach().max(1)[1][j]] += 1
                    corr_mats[0][2, y.max(1)[1][j], outa.detach().max(1)[1][j]] += 1
                return loss, acc, y, acc_a, handle_list
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
                for _ in range(args.attack_iters):
                    add_noise_mask = torch.ones_like(X)
                    grid_num_axis = int(args.resize / args.patch)
                    max_num_patch = grid_num_axis * grid_num_axis
                    ids = [i for i in range(max_num_patch)]
                    random.shuffle(ids)
                    num_patch = int(max_num_patch * (1 - drop_rate))
                    if num_patch != 0:
                        ids = np.array(ids[:num_patch])
                        rows, cols = ids // grid_num_axis, ids % grid_num_axis
                        for r, c in zip(rows, cols):
                            add_noise_mask[:, :, r * args.patch:(r + 1) * args.patch,
                            c * args.patch:(c + 1) * args.patch] = 0
                    if args.PRM:
                        delta = delta * add_noise_mask
                    loss_kl = criterion_kl(F.log_softmax(model(X + delta, prompt) if args.prompted else model(X + delta), dim=1),
                                           F.softmax(model(X, prompt) if args.prompted else model(X), dim=1))
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
                output = logits = model(X, prompt) if args.prompted else model(X)
                output = output.detach()
                # logits = logits.detach()
                acc_c = (output.max(1)[1] == y).float().mean().item()
                loss_natural = F.cross_entropy(logits, y)
                loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(X+delta, prompt) if args.prompted else model(X+delta), dim=1),
                                                                F.softmax(model(X, prompt) if args.prompted else model(X), dim=1))
                loss = loss_natural + beta * loss_robust
                model.zero_grad()
                loss.backward()

                dpgd = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt if args.prompted else None).detach()
                outa = model(X + dpgd, prompt).detach() if args.prompted else model(X + dpgd).detach()
                acc_a = (outa.detach().max(1)[1] == y).float().mean().item()
                for j in range(y.size(0)):
                    corr_mats[1][y[j], outa.detach().max(1)[1][j]] += 1
                    corr_mats[0][y[j], output.detach().max(1)[1][j]] += 1
                    
                return loss, acc_c, y, acc_a, handle_list, acc_c
            elif args.method == 'MART':
                X = X.cuda()
                y = y.cuda()
                beta = args.beta
                kl = nn.KLDivLoss(reduction='none')
                model.eval()
                batch_size = len(X)
                epsilon = epsilon_base.cuda()
                delta = torch.zeros_like(X).cuda()
                if args.delta_init == 'random':
                    delta = 0.001 * torch.randn(X.shape).cuda()
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.requires_grad = True
                for _ in range(args.attack_iters):
                    add_noise_mask = torch.ones_like(X)
                    grid_num_axis = int(args.resize / args.patch)
                    max_num_patch = grid_num_axis * grid_num_axis
                    ids = [i for i in range(max_num_patch)]
                    random.shuffle(ids)
                    num_patch = int(max_num_patch * (1 - drop_rate))
                    if num_patch != 0:
                        ids = np.array(ids[:num_patch])
                        rows, cols = ids // grid_num_axis, ids % grid_num_axis
                        for r, c in zip(rows, cols):
                            add_noise_mask[:, :, r * args.patch:(r + 1) * args.patch,
                            c * args.patch:(c + 1) * args.patch] = 0
                    if args.PRM:
                        delta = delta * add_noise_mask
                    output = model(X + delta)
                    loss = F.cross_entropy(output, y)
                    grad = torch.autograd.grad(loss, delta)[0].detach()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta = delta.detach()
                if len(handle_list) != 0:
                    for handle in handle_list:
                        handle.remove()
                model.train()
                x_adv = Variable(X+delta,requires_grad=False)
                logits = model(X)
                logits_adv = model(x_adv)
                adv_probs = F.softmax(logits_adv, dim=1)
                tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
                loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
                nat_probs = F.softmax(logits, dim=1)
                true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
                loss_robust = (1.0 / batch_size) * torch.sum(
                    torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
                loss = loss_adv + float(beta) * loss_robust
            elif args.method == 'ss':
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                delta = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=None).detach()
                d2 = pgd_attack(model, X, y, epsilon_base, alpha, args, criterion, handle_list, drop_rate, prompt=prompt).detach()
                outa, fa = model(X + delta, prompt, get_fs=True)
                outc, fc = model(X, prompt, get_fs=True)
                outad = model(X+d2, prompt).detach()
                diff = fa[:, args.prompt_length, :] - fc[:, args.prompt_length, :].detach()
                loss_c = criterion(outc, y)
                loss_a = (diff**2).sum(1).mean() 
                coeff = .5 #loss_c.item() / loss_a.item() * .5
                loss = (loss_c + coeff * loss_a)
                loss.backward()

                acc = (outa.max(1)[1] == y.max(1)[1]).float().mean().item()
                acc_c = (outc.max(1)[1] == y.max(1)[1]).float().mean().item()
                for j in range(y.size(0)):
                    corr_mats[1][y.max(1)[1][j], outa.detach().max(1)[1][j]] += 1
                    corr_mats[0][y.max(1)[1][j], outc.detach().max(1)[1][j]] += 1
                    corr_mats[2][y.max(1)[1][j], outad.detach().max(1)[1][j]] += 1
                return loss_c, acc, y, loss_a, handle_list, acc_c
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
        if args.method in ['voting']:
            accs_ind = torch.zeros(len(prompts))
            accs_vote = torch.zeros(len(prompts))

        for step, (X, y) in enumerate(train_loader):
            batch_size = args.batch_size // args.accum_steps
            epoch_now = epoch - 1 + (step + 1) / len(train_loader)
            # if step == 2:
            #     break
            X_ = X[0: batch_size].cuda()  # .permute(0, 3, 1, 2)
            y_ = y[0: batch_size].cuda()  # .max(dim=-1).indices
            # X = X[y.detach() != 9].detach()
            # y = y[y.detach()!=9].detach()
            # if X.size(0) % 2 == 1:
            #     X = X[:-1].detach()
            #     y = y[:-1].detach()
            # print(X.size(), y.size())
            if len(X_) == 0:
                break
            # print(y.size())
            if args.method in ['voting']:
                accs, accs_maj, losses, acc_clean, acc_adv = train_step(X, y, epoch_now, mixup_fn, hist_a, hist_c, corr_mats)
                train_n += y_.size(0)
                train_loss += losses * y_.size(0)
                train_clean += acc_clean * y_.size(0)
                accs_ind += accs * y_.size(0)
                accs_vote += accs_maj * y_.size(0)
                train_acc += acc_adv * y_.size(0)
                def floats_to_str(floats):
                    string = "("
                    s = 0
                    n = 0
                    for f in floats:
                        s += f
                        n += 1
                        string += "{:.4f} ".format(f)
                    return string + "{:.4f})".format(s/n)
                # cosim = nn.CosineSimilarity(2)
                # things = lambda x, y: cosim(x,y).detach()
                if (step + 1) % args.log_interval == 0 or step + 1 == steps_per_epoch:
                    logger.info('Training epoch {} step {}/{}, lr {:.4f} loss {:.4f} ind acc {} clean acc {:.4f} vote acc {} adv acc {:.4f}'.format(
                        epoch, step + 1, len(train_loader),
                        opts[0].param_groups[0]['lr'],
                            losses / train_n, floats_to_str(accs_ind / train_n), train_clean/ train_n, floats_to_str([torch.norm(p).item() for p in prompts]), train_acc/train_n
                    ))
                if (step+1) %50 == 0:
                    # fg, axarr = plt.subplots(2,1)
                    # axarr[0].bar(range(10), hist_c.cpu().numpy())
                    # axarr[0].set_title("True labels")
                    # axarr[1].bar(range(10), hist_a.cpu().numpy())
                    # axarr[1].set_title("Adv labels")
                    # path = os.path.join(args.out_dir)
                    # plt.savefig(args.out_dir + "/hist_epoch_"+str(epoch)+"step_"+str(step) + ".png")
                    # plt.clf()
                    fg, axarr = plt.subplots(len(prompts), len(prompts) + 2)
                    if len(prompts) > 1:
                        for i, c in enumerate(corr_mats):
                            for j in range(len(corr_mats) + 2):
                                axarr[i,j].matshow(c[j,:,:]/train_n)
                    else:
                        axarr[0].matshow(corr_mats[0][0,:,:]/train_n)
                        # axarr[0].axis('off')
                        axarr[0].yaxis.tick_left()
                        axarr[0].set_title('clean samples')
                        axarr[0].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_clean/train_n * 100))
                        axarr[0].set_ylabel('ground truth label')
                        axarr[1].matshow(corr_mats[0][1,:,:]/train_n)
                        # axarr[1].axis('off')
                        axarr[1].yaxis.tick_left()
                        axarr[1].set_title('perturbed loss(y)')
                        axarr[1].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_acc/train_n * 100))
                        axarr[2].yaxis.tick_left()
                        axarr[2].set_title('perturbed loss(y + 1)')
                        axarr[2].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_prompted/train_n * 100))
                        axarr[2].matshow(corr_mats[0][2,:,:]/train_n)
                    plt.savefig(args.out_dir + "/mat_present_"+str(epoch)+"step_" + str(step) + ".png")
                    plt.close()
            else:
                # if args.method == 'ws' and args.ws == epoch:
                #     opt = torch.optim.SGD([prompt], lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
                loss, acc,y, p_acc, handle_list, clean_acc = train_step(X,y,epoch_now,mixup_fn, hist_a, hist_c, corr_mats)
                # print(y.max(1)[1].size())
                train_loss += loss.item() * y_.size(0)
                train_acc += acc * y_.size(0)
                
                train_prompted += p_acc * y_.size(0)
                train_clean += clean_acc * y_.size(0)
                train_n += y_.size(0)
                grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) , args.grad_clip)#+ [prompt]
                # cp = prompt.clone()
                opt.step()
                # print(prompt - cp)
                opt.zero_grad()
                model.zero_grad()
                
                if step + 1 == steps_per_epoch and args.method != 'natural':
                    fg, axarr = plt.subplots(1,4)
                    if args.method == 'updown':
                        axarr[0].matshow(corr_mats[0]/train_n)
                        # axarr[0].axis('off')
                        axarr[0].yaxis.tick_left()
                        axarr[0].set_title('clean frozen')
                        # axarr[0].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_clean/train_n * 100))
                        axarr[0].set_ylabel('ground truth label')
                        axarr[1].matshow(corr_mats[1]/train_n)
                        # axarr[1].axis('off')
                        axarr[1].yaxis.tick_left()
                        axarr[1].set_title('clean combined')
                        axarr[1].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_clean/train_n * 100))
                        axarr[2].yaxis.tick_left()
                        axarr[2].set_title('up')
                        axarr[2].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_acc/train_n * 100))
                        axarr[2].matshow(corr_mats[2]/train_n)
                        axarr[3].yaxis.tick_left()
                        axarr[3].set_title('down')
                        axarr[3].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_prompted/train_n * 100))
                        axarr[3].matshow(corr_mats[3]/train_n)
                    elif len(prompts) > 1:
                        for i, c in enumerate(corr_mats):
                            for j in range(len(corr_mats) + 2):
                                axarr[i,j].matshow(c[j,:,:]/train_n)
                    else:
                        axarr[0].matshow(corr_mats[0]/train_n)
                        # axarr[0].axis('off')
                        axarr[0].yaxis.tick_left()
                        axarr[0].set_title('clean samples')
                        axarr[0].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_clean/train_n * 100))
                        axarr[0].set_ylabel('ground truth label')
                        axarr[1].matshow(corr_mats[1]/train_n)
                        # axarr[1].axis('off')
                        axarr[1].yaxis.tick_left()
                        axarr[1].set_title('perturbed loss(y)')
                        axarr[1].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_acc/train_n * 100))
                        axarr[2].yaxis.tick_left()
                        axarr[2].set_title('perturbed loss(y + 1)')
                        axarr[2].set_xlabel('predicted label\n' + "Acc: {:.2f}".format(train_prompted/train_n * 100))
                        axarr[2].matshow(corr_mats[2]/train_n)
                        # axarr[2].axis('off')
                    plt.savefig(args.out_dir + "/mat_present_"+str(epoch)+"step_" + str(step) + ".png", dpi=500)
                    plt.close()
                if (step + 1) % args.log_interval == 0 or step + 1 == steps_per_epoch:
                    logger.info('Training epoch {} step {}/{}, lr {:.4f} loss {:.4f} acc {:.4f} clean acc {:.4f} prompt atk {:.4f}'.format(
                        epoch, step + 1, len(train_loader),
                        opt.param_groups[0]['lr'],
                            train_loss / train_n, train_acc / train_n, train_clean/ train_n, train_prompted/ train_n
                    ))
                # if (step+1) % 3 == 0:
                #     break
            lr = lr_schedule(epoch_now)
            opt.param_groups[0].update(lr=lr)
            # for o in opts:
            #     o.param_groups[0].update(lr=lr)
        if epoch % args.split_interval ==0:
            if args.method == 'splits':
                logger.info("Adding {:d} tokens".format(args.prompt_length))
                done_prompt = prompt if done_prompt is None else torch.cat((prompt.detach(), done_prompt.detach()), dim=1)#.requires_grad_()#joint_p(prompt, done_prompt)#.requires_grad_()
                # done_prompt.requires_grad = True
                prompt = make_prompt(args.prompt_length, 768)
                # prompt.requires_grad = True
                # opt.add_param_group({"params" : [prompt})
                logger.info("Total length is " + str(prompt.size(1) + done_prompt.size(1)))
                # print('sag')
                # print(prompt[:,0,-1])
                # for i in range(done_prompt.size(1)//10):
                #     print(done_prompt[:,i * 10,-1])
                opt = torch.optim.SGD([prompt] + list(model.module.head.parameters()), lr=lr_schedule(epoch_now), momentum=args.momentum, weight_decay=args.weight_decay) 
        
        path = os.path.join(args.out_dir, 'checkpoint_{}'.format(epoch))
        if args.test:
            with open(os.path.join(args.out_dir, 'test_PGD20.txt'),'a') as new:
                args.eval_iters = 20
                args.eval_restarts = 1
                pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader)
                logger.info('test_PGD20 : loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc))
                new.write('{:.4f}   {:.4f}\n'.format(pgd_loss, pgd_acc))
            with open(os.path.join(args.out_dir, 'test_acc.txt'), 'a') as new:
                meter_test = evaluate_natural(args, model, test_loader, verbose=False)
                new.write('{}\n'.format(meter_test))
        if epoch == args.epochs or epoch % args.chkpnt_interval == 0:
            if args.method in ['voting']:
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'opts': [opt], 'prompts': prompts}, path)
            elif args.method in ['splits']:
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'opts': [opt], 'prompt': [prompt], 'done': [done_prompt]}, path)
            else:
                if args.prompted or args.prompt_too:
                    prompt_save = [prompt, prompt2] if args.disjoint_prompts else [prompt]
                elif args.blocked:
                    prompt_save = [prompt.state_dict()]
                else:
                    prompt_save = None
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict(), 'prompt': prompt_save}, path)
            logger.info('Checkpoint saved to {}'.format(path))


train_adv(args, model, train_loader, test_loader, logger)

# args.eval_iters = 10
# args.alpha = 2
logger.info(args.out_dir)
# print(args.out_dir)
evaluate_natural(args, model, test_loader, verbose=False, prompt=prompt)

# # cw_loss, cw_acc = evaluate_CW(args, model, test_loader, prompt=prompt)
# # logger.info('cw20 : loss {:.4f} acc {:.4f}'.format(cw_loss, cw_acc))
# print("sag")
# mats, accs = evaluate_splits(args, logger, model, test_loader, prompt=prompt[:,args.prompt_length:,:], steps=30)
# fg, axarr = plt.subplots(len(mats), len(mats[0]))

# for i in range(len(mats)):
#     for j in range(len(mats[0])):
#         axarr[i,j].matshow(mats[i][j])
#         # axarr[i,j].set_ylabel("{:.2f}".format(accs[i, j]))
#         axarr[i,j].axis('off')
# plt.savefig(args.out_dir + "/mat_of_splits_len{}.png".format(str(prompt.size(1) - args.prompt_length)))
# logger.info('Saved mat to outdir'
chkpnt = None
train_loader = None
args.eval_iters = 10
args.alpha = 2
args.eval_restarts = 1
pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader, prompt=prompt, a_lam=args.a_lam)
logger.info('PGD10 : loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc))


args.eval_iters = 20
args.alpha = 2
cw_loss, cw_acc = evaluate_CW(args, model, test_loader, prompt=prompt, a_lam=args.a_lam, detection=True)
logger.info('CW20: loss {:.4f} acc {:.4f}'.format(cw_loss, cw_acc))

args.eval_iters = 50
args.alpha = 2/5
args.eval_restarts = 10
pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader, prompt=prompt, a_lam=args.a_lam)
logger.info('PGD50-10: loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc))


# logger.info('Moving to AA...')
# at_path = os.path.join(args.out_dir, 'result_'+'_autoattack.txt')
# evaluate_aa(args, model,at_path, args.AA_batch, prompt if args.prompted else None)