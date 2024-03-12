import numpy as np
import tqdm
import random
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import  SoftTargetCrossEntropy
from timm.data import Mixup
from parser import get_args
from utils import *
from torch.autograd import Variable
from attacks import attack_pgd, attack_cw
from losses import *
from evaluate import evaluate_aa, evaluate_natural, evaluate_pgd, evaluate_CW
import logging
import matplotlib.pyplot as plt
from buffer import Buffer
import wandb
from model_for_cifar.model import get_model_prompt
# torch.autograd.set_detect_anomaly(True)
args = get_args()

joint_p = lambda x, y: torch.cat((x, y), dim=1) if y is not None else x 

if 'base' in args.model:
    mname = 'base'
elif 'small' in args.model:
    mname = 'small'
elif 'large' in args.model:
    mname = 'large'
else:
    mname = args.model
args.name = args.params + (str(args.prompt_length) if args.params in ['PT', 'P2T', 'DPT'] else "") + "_" + args.dataset+"_"+args.lr_schedule+"_"+args.method + "_" +mname + ("_deep" if args.deep_p else "") + ("_patch" if args.train_patch else "")
args.out_dir = args.out_dir + args.name
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

train_loader, test_loader= get_loaders(args)



model, prompt, params = get_model_prompt(args)
if args.optim == 'sgd':
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay) 
elif args.optim == 'adam':
    opt = torch.optim.Adam(params, lr=args.lr_max, weight_decay=args.weight_decay)      



def train_adv(args, model, prompt, opt, ds_train, ds_test, logger):


    train_loader, test_loader = ds_train, ds_test

    steps_per_epoch = len(train_loader)
    
    if args.method == 'natural':
        loss_fn = nat_loss
    elif args.method == 'AT':
        loss_fn = AT_loss
    elif args.method == 'TRADES':
        loss_fn = TRADES_loss
    elif args.method == 'PAT':
        loss_fn = PAT_loss
    elif args.method == 'PTRADES':
        loss_fn = PTRADES_loss
    elif args.method == 'PSIM':
        loss_fn = PSIM_loss
    elif args.method == 'PKL':
        loss_fn = PKL_loss
    else:
        raise ValueError(args.method)
    
    if args.load:
        logger.info("Resuming at epoch {}".format(checkpoint['epoch'] + 1))

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
    p_copy = None
    for epoch in range(epoch_s + 1, args.epochs + 1):
        # if p_copy is not None:
        #     print(prompt - p_copy)
        
        if args.just_eval:
            break

        train_loss = 0
        train_acc = 0
        train_clean = 0
        train_prompted = 0
        train_adetect = 0
        train_n = 0

      
        model.train()
        for step, (X, y) in enumerate(train_loader):
            epoch_now = epoch - 1 + (step + 1) / len(train_loader)

            X = X.cuda()
            y=y.cuda()

            
            loss, out_a = loss_fn(model, prompt, X, y, args)
            opt.zero_grad()
            model.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) , args.grad_clip)
            opt.step()
            opt.zero_grad()
            model.zero_grad()

            out_c = model(X, prompt).detach()

            acc_a = (out_a.max(1)[1] == y).float().mean().item()
            acc_c = (out_c.max(1)[1] == y).float().mean().item()
            train_loss += loss.item() * y.size(0)
            train_acc += acc_a * y.size(0)
            train_clean += acc_c * y.size(0)
            train_n += y.size(0)
            
            
            if (step + 1) % args.log_interval == 0 or step + 1 == steps_per_epoch:
                wandb.config.steps_per_epoch = steps_per_epoch // args.log_interval + 1
                wandb.log(
                    {
                        'train adv loss':train_loss/train_n,
                        'train adv acc':train_acc/train_n,
                        'train clean acc':train_clean/train_n,
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
        p_copy = prompt.clone().detach()
        path = os.path.join(args.out_dir, 'checkpoint_{}'.format(epoch))

        if epoch == args.epochs or epoch % args.chkpnt_interval == 0:
            to_save = {'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict(), 'prompt': [prompt]}
            torch.save(to_save, path)
            logger.info('Checkpoint saved to {}'.format(path))
        
        model.eval()
        logger.info('Evaluating epoch...')
        loss_clean, acc_clean = evaluate_natural(args, model, test_loader, logger, prompt=prompt)
        loss_adv, acc_adv = evaluate_pgd(args, model, test_loader, prompt=prompt)
        opt.zero_grad()
        model.zero_grad()
        # loss_cw, acc_cw = evaluate_CW(args, model, test_loader, prompt=prompt)
        # opt.zero_grad()
        # model.zero_grad()
        logger.info('Natural: loss {:.4f} acc {:.4f}'.format(loss_clean, acc_clean))
        logger.info('PGD10 : loss {:.4f} acc {:.4f}'.format(loss_adv, acc_adv))
        wandb.log(
            {
                'test clean loss': loss_clean,
                'test adv loss': loss_adv,
                'test clean acc': acc_clean,
                'test adv acc': acc_adv
            }
        )

def eval_adv(args, model, prompt, test_loader, logger):
    model.eval()
    logger.info('Evaluating natural...')
    loss_clean, acc_clean = evaluate_natural(args, model, test_loader, logger, prompt=prompt)
    logger.info('Nat acc {:.4f}'.format(acc_clean))
    if args.unadapt:
        logger.info('Evaluating adaptive FGSM...')
        args.eval_iters = 1
        args.eval_restarts = 1
        args.alpha = 2*args.epsilon
        loss_fgsm, acc_fgsm = evaluate_pgd(args, model, test_loader, prompt=prompt)
        logger.info('Evaluating adaptive PGD...')
        args.eval_iters = 10
        args.eval_restarts = 1
        args.alpha = 2
        loss_pgd, acc_pgd = evaluate_pgd(args, model, test_loader, prompt=prompt)
        logger.info('Moving to traditional attacks...')
    # logger.info('Evaluating FGSM...')
    # args.eval_iters = 1
    # args.eval_restarts = 1
    # args.alpha = 2*args.epsilon
    #loss_fgsm, acc_fgsm = evaluate_pgd(args, model, test_loader, prompt=prompt, unadapt=args.unadapt)
    #logger.info('Evaluating PGD10...')
    #args.eval_iters = 10
    #args.eval_restarts = 1
    #args.alpha = 2
    #loss_pgd, acc_pgd = evaluate_pgd(args, model, test_loader, prompt=prompt, unadapt=args.unadapt)
    if not args.unadapt:
        logger.info('Evaluating CW...')
        loss_cw, acc_cw = evaluate_CW(args, model, test_loader, prompt=prompt, unadapt=args.unadapt)
        model.zero_grad()
        # aa_path = os.path.join(args.out_dir, 'result_autoattack.txt')
        # _ = evaluate_aa(args, model, test_loader, aa_path, aa_batch=args.AA_batch, prompt=prompt)
    # logger.info({
    #         'final clean loss': loss_clean,
    #         'final clean acc': acc_clean,
    #         'final pgd10 loss': loss_pgd,
    #         'final pgd10 acc': acc_pgd,
    #         'final fgsm loss': loss_fgsm,
    #         'final fgsm acc': acc_fgsm,
    #         'final cw loss': loss_cw,
    #         'final cw acc': acc_cw,
    #     })
    # wandb.log(
    #     {
    #         'final clean loss': loss_clean,
    #         'final clean acc': acc_clean,
    #         'final pgd10 loss': loss_pgd,
    #         'final pgd10 acc': acc_pgd,
    #         'final fgsm loss': loss_fgsm,
    #         'final fgsm acc': acc_fgsm,
    #         'final cw loss': loss_cw,
    #         'final cw acc': acc_cw,
    #     }
    # )



if args.just_eval:
    eval_adv(args, model, prompt, test_loader, logger)
else:
    train_adv(args, model, prompt, opt, train_loader, test_loader, logger)
