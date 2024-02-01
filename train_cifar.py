import numpy as np
import tqdm
import random
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import  SoftTargetCrossEntropy
from timm.data import Mixup
from parser_cifar import get_args
from utils import *
from torch.autograd import Variable
from attacks import attack_pgd, attack_cw
from evaluate import evaluate_aa, evaluate_natural, evaluate_pgd, evaluate_CW
import logging
import matplotlib.pyplot as plt
from buffer import Buffer
import wandb
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
args.name = args.params + "_" + args.dataset+"_"+args.lr_schedule+"_"+args.method + "_" +args.model
args.out_dir = args.out_dir + '_' + args.name
wandb.init(
    project="rpt_cifar",
    name=args.name,
    config=args
)
args.out_dir = args.out_dir +"/seed"+str(args.seed)

print(args.out_dir)
os.makedirs(args.out_dir,exist_ok=True)
logfile = os.path.join(args.out_dir, 'log_{:.4f}.log'.format(args.weight_decay))
from auto_LiRPA.utils import logger
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
    from model_for_cifar.vit import vit_large_patch16_224_in21k
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
    raise ValueError("Model doesn't exist!")
if args.model_log:
    logger.info('Model{}'.format(model))
model.train()

checkpoint = None
if args.load:
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['state_dict'])

if args.params in ['PT','DPT']:
    if args.load:
        prompt = (checkpoint['prompt'])[0]
    else:
        prompt = make_prompt(args.prompt_length, model.module.embed_dim, depth=args.prompt_depth)
        if args.params == 'DPT':
            assert args.prompt_depth == 1
            unexpaned = prompt
            prompt = prompt.expand(1, prompt.size(1), prompt.size(2), model.module.depth)
    prompts = [prompt]
    params = [unexpaned if args.params == 'DPT' else prompt]
        
elif args.params == ['P2T']:
    if args.load:
        prompt = (checkpoint['prompt'])[0]
    else:
        prompt = make_prompt(args.prompt_length, model.module.embed_dim, depth=model.module.depth)
    
    prompts = [prompt]
    params = [prompt]
if args.params == 'FT':  
    prompt = None
    args.prompt_length = 0
    params = model.parameters()
else:
    if args.train_patch:
        for p in model.module.patch_embed.parameters():
            params.append(p)
    if not args.freeze_head:
        for p in model.module.head.parameters():
            params.append(p)
if args.optim == 'sgd':
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay) 
elif args.optim == 'adam':
    opt = torch.optim.Adam(params, lr=args.lr_max, weight_decay=args.weight_decay)      






mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

upper_limit = ((1 - mu) / std).cuda()
lower_limit = ((0 - mu) / std).cuda()



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
        lr_schedule = lambda t, max_ep: np.interp([t], [0, max_ep // 2, max_ep], [args.lr_min, args.lr_max, args.lr_min])[0]
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

        def train_step(X, y, t, mixup_fn, corr_mats):
            global prompt, done_prompt, opt
            model.train()

            model.eval()
            model.train()
            if args.method == 'natural' or epoch < args.ws:
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)
                model(X, prompt)

                loss = criterion(output, y)
                loss.backward()
                acc = (output.max(1)[1] == y.max(1)[1]).float().mean().item()

                if args.eval_nat:
                    dpgd = attack_pgd(model, X, y, epsilon_base, alpha, 1, lower_limit, upper_limit, prompt=prompt if args.full_white else None).detach()
                    out_a = model(X + dpgd, prompt)
                    acc_a = (out_a.max(1)[1] == y.max(1)[1]).float().mean().item()
                    return loss, acc_a, y, acc
                return loss, acc, y, acc 
            elif args.method == 'AT':
                X = X.cuda()
                y = y.cuda()
                if mixup_fn is not None:
                    X, y = mixup_fn(X, y)

                delta = attack_pgd(model, X, y, epsilon_base, alpha, args.attack_iters, 1, lower_limit, upper_limit, prompt=prompt if args.full_white else None)
                X.detach()
                out = model(X + delta, prompt)
                outc = model(X, prompt)
                loss = criterion(out, y)
                loss.backward()

                acc = (out.max(1)[1] == y.max(1)[1]).float().mean().item()
                acc_c = (outc.max(1)[1] == y.max(1)[1]).float().mean().item()
                for j in range(y.size(0)):
                    corr_mats[1][y.max(1)[1][j], out.detach().max(1)[1][j]] += 1
                    corr_mats[0][y.max(1)[1][j], outc.detach().max(1)[1][j]] += 1
                    
                return loss, acc, y, acc_c
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
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)

                delta = delta.detach()

                output = logits = model(X, prompt)
                output = output.detach()
                outa = model(X + delta, prompt)

                acc_c = (output.max(1)[1] == y).float().mean().item()
                loss_natural = F.cross_entropy(logits, y)
                loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(outa, dim=1),
                                                                F.softmax(logits, dim=1))
                loss = loss_natural + beta * loss_robust
                model.zero_grad()
                loss.backward()

                acc_a = (outa.detach().max(1)[1] == y).float().mean().item()
                for j in range(y.size(0)):
                    corr_mats[1][y[j], outa.detach().max(1)[1][j]] += 1
                    corr_mats[0][y[j], output.detach().max(1)[1][j]] += 1
                    
                return loss, acc_a, y, acc_c
            else:
                raise ValueError(args.method)

        for step, (X, y) in enumerate(train_loader):
            batch_size = args.batch_size // args.accum_steps
            epoch_now = epoch - 1 + (step + 1) / len(train_loader)

            X_ = X[0: batch_size].cuda()  
            y_ = y[0: batch_size].cuda()  
            
            loss, acc_a, y, acc_c = train_step(X,y,epoch_now,mixup_fn, corr_mats)
            train_loss += loss.item() * y_.size(0)
            train_acc += acc_a * y_.size(0)
            train_clean += acc_c * y_.size(0)
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
                wandb.config.steps_per_epoch = steps_per_epoch // args.log_interval + 1
                wandb.log(
                    {
                        'loss':train_loss/train_n,
                        'acc':train_acc/train_n,
                        'clean acc':train_clean/train_n,
                        'lr':opt.param_groups[0]['lr']
                    }
                )
                logger.info('Training epoch {} step {}/{}, lr {:.4f} loss {:.4f} adv acc {:.4f} clean acc {:.4f}'.format(
                    epoch, step + 1, len(train_loader),
                    opt.param_groups[0]['lr'],
                        train_loss / train_n, train_acc / train_n, train_clean/ train_n
                ))
            lr = lr_schedule(epoch_now, args.ws) if (args.ws and epoch_now < args.ws) else lr_schedule(epoch_now, args.epochs) 
            opt.param_groups[0].update(lr=lr)
        
        path = os.path.join(args.out_dir, 'checkpoint_{}'.format(epoch))

        if epoch == args.epochs or epoch % args.chkpnt_interval == 0:
            if not args.prompted and not args.prefixed:
                prompt = None
            to_save = {'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict(), 'prompt': [prompt]}
            torch.save(to_save, path)
            logger.info('Checkpoint saved to {}'.format(path))


train_adv(args, model, train_loader, test_loader, logger)


logger.info(args.out_dir)
if not args.prompted and not args.prefixed:
    prompt = None
evaluate_natural(args, model, test_loader, verbose=False, prompt=prompt)

# chkpnt = None
args.eval_iters = 10
args.alpha = 2
args.eval_restarts = 1
pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader, prompt=prompt, a_lam=args.a_lam)
logger.info('PGD10 : loss {:.4f} acc {:.4f}'.format(pgd_loss, pgd_acc))


args.eval_iters = 20
args.alpha = 2
cw_loss, cw_acc = evaluate_CW(args, model, test_loader, prompt=prompt, a_lam=0)
logger.info('CW20: loss {:.4f} acc {:.4f}'.format(cw_loss, cw_acc))



logger.info('Moving to AA...')
at_path = os.path.join(args.out_dir, 'result_'+'_autoattack.txt')
evaluate_aa(args, model, test_loader,at_path, args.AA_batch, prompt=prompt)