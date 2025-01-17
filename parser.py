import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--method', type=str, default='AT',
                        choices=['AT', 'TRADES', 'MART', 'natural', 'ADAPT', 'NFGSM'])
    parser.add_argument('--adapt-loss', choices=['kl', 'ce'], default='kl')
    parser.add_argument('--params', type=str, default='PT', choices=['FT', 'PT', 'P2T'])
    parser.add_argument('--dataset', type=str, default="cifar10", choices= ['cifar10','cifar100', 'imagenette'])
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--chkpnt_interval', type=int, default=10)
    parser.add_argument('--AA-batch', default=128, type=int,help="Batch size for AA.")
    parser.add_argument('--crop', type=int, default=32)
    parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--load_path', default='', type=str)
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--n_w', type=int, default=10)
    parser.add_argument('--attack-iters', type=int, default=10, help='for pgd training')
    parser.add_argument('--patch', type=int, default=16)
    parser.add_argument('--prompt-depth', type=int, default=1)
    parser.add_argument('--lr-schedule', type=str, default='cyclic', choices=['cyclic', 'drops'])
    parser.add_argument('--unadapt', action='store_true')
    parser.add_argument('--freeze-head', action='store_true')
    parser.add_argument('--prompt_length', type=int, default=25)
    parser.add_argument('--eval-bb', action='store_true')
    parser.add_argument('--eval-en', action='store_true')
    parser.add_argument('--deep-p', action='store_true')
    parser.add_argument('--n_query', type=int, default=10000, help='blackbox attack queries')
    parser.add_argument('--num-eval', type=int, default=10000, help='how many samples to eval')
    parser.add_argument('--train-patch', action='store_true')
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--just-eval", action='store_true')
    parser.add_argument('--eval-restarts', type=int, default=1)
    parser.add_argument('--eval-iters', type=int, default=10)
    parser.add_argument('--data-dir', default='../../datasets/', type=str)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=2, type=float, help='Step size for attacks')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
                        help='Perturbation initialization method')
    parser.add_argument('--out-dir', '--dir', default='./outs/', type=str, help='Output directory')
    parser.add_argument('--model_log', action='store_true')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--name', type=str, default='sample_run')



    ### AdaEA args
    parser.add_argument('--attack_method', type=str, default='AdaEA_DIFGSM')
    parser.add_argument('--fusion_method', type=str, default='add')
    parser.add_argument('--no_norm', action='store_true',
                        help='do not use normalization')
    parser.add_argument('--use_adv_model', action='store_true')
    parser.add_argument('--resize_rate', type=float, default=0.9,
                        help='resize rate')
    parser.add_argument('--diversity_prob', type=float, default=0.5,
                        help='diversity_prob')
    parser.add_argument('--max_value', type=float, default=1.0)
    parser.add_argument('--min_value', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=-0.3)




    args = parser.parse_known_args()[0]
    return args