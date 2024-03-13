#### Code built upon the repository https://github.com/mo666666/When-Adversarial-Training-Meets-Vision-Transformers

import numpy as np
from parser import get_args
from utils import *
from losses import *
from evaluate import evaluate_aa, evaluate_natural, evaluate_pgd, evaluate_CW
import logging
import wandb
from model import get_model_prompt


#### PARSE ARGS AND SETUP LOGGING #####
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
logging.basicConfig(
    format='%(levelname)-8s %(asctime)-12s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(logfile)
file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
logger.addHandler(file_handler)

logger.info(args)


##### LOAD DATA, MODEL, PROMPTS #####
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

resize_size = args.resize
crop_size = args.crop

train_loader, test_loader= get_loaders(args)



model, prompt, params, epoch_s, opt_dict = get_model_prompt(args)
if args.optim == 'sgd':
    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay) 
elif args.optim == 'adam':
    opt = torch.optim.Adam(params, lr=args.lr_max, weight_decay=args.weight_decay)   
if opt_dict != None:
    opt.load_state_dict(opt_dict)   



##### TRAIN MODEL #####
def train_adv(args, model, prompt, opt, train_loader, test_loader, logger, epoch_s=0):

    steps_per_epoch = len(train_loader)
    #### DETERMINE LOSS FUNCTION ####
    if args.method == 'natural':
        loss_fn = natural
    elif args.method == 'AT':
        loss_fn = AT
    elif args.method == 'TRADES':
        loss_fn = TRADES
    elif args.method == 'ADAPT':
        if args.adapt_loss == 'ce':
            loss_fn = ADAPT_CE
        elif args.adapt_loss == 'kl':
            loss_fn = ADAPT_KL
    else:
        raise ValueError(args.method)
    
    #### IF LOADING RESUME EPOCH ####
    if args.load:
        logger.info("Resuming at epoch {}".format(epoch_s))

    #### LR SCHEDULE ####
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

    #### TRAIN EPOCHS ####
    for epoch in range(epoch_s + 1, args.epochs + 1):
        train_loss = 0
        train_acc = 0
        train_clean = 0
        train_n = 0

      
        model.train()
        #### EPOCH ####
        for step, (X, y) in enumerate(train_loader):
            epoch_now = epoch - 1 + (step + 1) / len(train_loader)

            X = X.cuda()
            y=y.cuda()

            
            loss, out_a = loss_fn(model, prompt, X, y, args)
            opt.zero_grad()
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) , args.grad_clip)
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
            
            #### EVAL DURING EPOCH ####
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
            #### LR SCHEDULE UPDATE ####
            lr = lr_schedule(epoch_now, args.epochs) 
            opt.param_groups[0].update(lr=lr)
        path = os.path.join(args.out_dir, 'checkpoint_{}'.format(epoch))

        ### SAVE CHECKPOINT ####
        if epoch == args.epochs or epoch % args.chkpnt_interval == 0:
            to_save = {'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict(), 'prompt': [prompt]}
            torch.save(to_save, path)
            logger.info('Checkpoint saved to {}'.format(path))
        
        #### EVALUATION EACH EPOCH ####
        model.eval()
        logger.info('Evaluating epoch...')
        loss_clean, acc_clean = evaluate_natural(args, model, test_loader, logger, prompt=prompt)
        loss_adv, acc_adv = evaluate_pgd(args, model, test_loader, prompt=prompt)
        opt.zero_grad()
        model.zero_grad()
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

#### EVALUATE TRAINED MODEL/PROMPT ####
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
    logger.info('Evaluating FGSM...')
    args.eval_iters = 1
    args.eval_restarts = 1
    args.alpha = 2*args.epsilon
    loss_fgsm, acc_fgsm = evaluate_pgd(args, model, test_loader, prompt=prompt, unadapt=args.unadapt)
    logger.info('Evaluating PGD10...')
    args.eval_iters = 10
    args.eval_restarts = 1
    args.alpha = 2
    loss_pgd, acc_pgd = evaluate_pgd(args, model, test_loader, prompt=prompt, unadapt=args.unadapt)
    if not args.unadapt:
        logger.info('Evaluating CW...')
        loss_cw, acc_cw = evaluate_CW(args, model, test_loader, prompt=prompt, unadapt=args.unadapt)
        model.zero_grad()
        aa_path = os.path.join(args.out_dir, 'result_autoattack.txt')
        _ = evaluate_aa(args, model, test_loader, aa_path, aa_batch=args.AA_batch, prompt=prompt)
    logger.info({
            'final clean loss': loss_clean,
            'final clean acc': acc_clean,
            'final pgd10 loss': loss_pgd,
            'final pgd10 acc': acc_pgd,
            'final fgsm loss': loss_fgsm,
            'final fgsm acc': acc_fgsm,
            'final cw loss': loss_cw,
            'final cw acc': acc_cw,
        })
    wandb.log(
        {
            'final clean loss': loss_clean,
            'final clean acc': acc_clean,
            'final pgd10 loss': loss_pgd,
            'final pgd10 acc': acc_pgd,
            'final fgsm loss': loss_fgsm,
            'final fgsm acc': acc_fgsm,
            'final cw loss': loss_cw,
            'final cw acc': acc_cw,
        }
    )



### PERFORM TRAINING/EVALUATION
if args.just_eval:
    eval_adv(args, model, prompt, test_loader, logger)
else:
    train_adv(args, model, prompt, opt, train_loader, test_loader, logger, epoch_s=epoch_s)
