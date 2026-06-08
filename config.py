import argparse
import math
import os

from data.datasets import DATASET_CONFIG

_DATASETS = ['cifar10', 'tiny-imagenet', 'cifar100', 'stl10', 'caltech256', 'path']
# Final task index used by the linear-probe evaluation (= n_cls/cls_per_task - 1)
_EVAL_TARGET_TASK = {'cifar10': 4, 'tiny-imagenet': 9, 'cifar100': 4, 'stl10': 4, 'caltech256': 15}

# Training
def parse_train_option():
    p = argparse.ArgumentParser('HCDR / CCLIS training')

    p.add_argument('--method', type=str, default='cclis', choices=['cclis', 'hcdr'],
                   help='cclis = baseline; hcdr = baseline + patch NCE + hierarchical PRD')

    # continual-learning schedule
    p.add_argument('--target_task', type=int, default=0)
    p.add_argument('--resume_target_task', type=int, default=None)
    p.add_argument('--end_task', type=int, default=None)
    p.add_argument('--replay_policy', type=str, choices=['random', 'weight'], default='weight')
    p.add_argument('--mem_size', type=int)
    p.add_argument('--cls_per_task', type=int, default=2)

    # distillation
    p.add_argument('--distill_power', type=float, default=0.6)
    p.add_argument('--distill_type', type=str, default='PRD', choices=['PRD', 'IRD'])
    p.add_argument('--IRD_type', type=str, default='all', help='IRD on all data or only past')
    p.add_argument('--current_temp', type=float, default=0.2)
    p.add_argument('--past_temp', type=float, default=0.1)

    # logging / runtime
    p.add_argument('--print_freq', type=int, default=10)
    p.add_argument('--save_freq', type=int, default=1000)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--no_compile', action='store_true', help='disable torch.compile')
    p.add_argument('--legacy_optimizer', action='store_true',
                   help='single param-group SGD (reproduces the pre-refactor behaviour)')

    # epochs / batch (None -> per-dataset default in DATASET_CONFIG)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--start_epoch', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)

    # optimization
    p.add_argument('--learning_rate', type=float, default=0.5)
    p.add_argument('--learning_rate_prototypes', type=float, default=0.01)
    p.add_argument('--lr_decay_epochs', type=str, default='700,800,900')
    p.add_argument('--lr_decay_rate', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--temp', type=float, default=0.5, help='supcon temperature')

    # model / data
    p.add_argument('--model', type=str, default='resnet18')
    p.add_argument('--dataset', type=str, choices=_DATASETS, required=True)
    p.add_argument('--mean', type=str, help='mean tuple as str (path dataset)')
    p.add_argument('--std', type=str, help='std tuple as str (path dataset)')
    p.add_argument('--data_folder', type=str, default='~/data/')
    p.add_argument('--size', type=int, default=32)

    # misc
    p.add_argument('--cosine', action='store_false', help='use cosine annealing (on by default)')
    p.add_argument('--syncBN', action='store_true')
    p.add_argument('--warm', action='store_true')
    p.add_argument('--trial', type=str, default='0')
    p.add_argument('--freeze_prototypes_niters', type=int, default=5)
    p.add_argument('--max_iter', type=int, default=5, help='iterations of score computation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--prefix', type=str, default='')

    # HCDR-only
    p.add_argument('--alpha_patch', type=float, default=0.5, help='weight of patch-level NCE')
    p.add_argument('--lambda_hprd', type=float, default=0.6, help='weight of hierarchical PRD')
    p.add_argument('--top_k_patches', type=int, default=10,
                   help='discriminative patches stored per buffer image')
    p.add_argument('--percentile', type=float, default=90.0,
                   help='percentile for discriminative patch scoring')

    opt = p.parse_args()
    opt.use_hcdr = (opt.method == 'hcdr')
    opt.compile = not opt.no_compile

    _apply_dataset_config(opt)
    _finalize_train_paths(opt)
    return opt


def _apply_dataset_config(opt):
    if opt.dataset == 'path':
        assert opt.data_folder is not None and opt.mean is not None and opt.std is not None
        return
    cfg = DATASET_CONFIG[opt.dataset]
    # n_cls / cls_per_task / size are dataset-intrinsic
    opt.n_cls = cfg['n_cls']
    opt.cls_per_task = cfg['cls_per_task']
    opt.size = cfg['size']
    # epochs / start_epoch / batch_size: CLI value wins, else dataset default
    if opt.epochs is None:
        opt.epochs = cfg['epochs']
    if opt.start_epoch is None:
        opt.start_epoch = cfg['start_epoch']
    if opt.batch_size is None:
        opt.batch_size = cfg['batch_size']


def _finalize_train_paths(opt):
    if opt.data_folder is None:
        opt.data_folder = '~/data/'
    opt.save_freq = opt.epochs // 2

    opt.save_file = './save_{}_{}_{}'.format(opt.replay_policy, opt.mem_size, opt.prefix)
    opt.model_path = opt.save_file + '/{}_models'.format(opt.dataset)
    opt.tb_path = opt.save_file + '/{}_tensorboard'.format(opt.dataset)
    opt.log_path = opt.save_file + '/logs'

    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]

    tag = ''
    if opt.use_hcdr:
        tag = '_hcdr_alpha{}_lamd{}_top{}patches_perc{}'.format(
            opt.alpha_patch, opt.lambda_hprd, opt.top_k_patches, opt.percentile)
    opt.model_name = (
        '{ds}_{sz}_{model}{tag}_lr_{lr}_{lrp}_decay_{wd}_bsz_{bsz}_temp_{temp}_trial_{trial}'
        '_{se}_{ep}_{ct}_{pt}_{dp}_distill_type_{dt}_freeze_prototypes_niters_{fpn}_seed_{seed}'
    ).format(ds=opt.dataset, sz=opt.size, model=opt.model, tag=tag,
             lr=opt.learning_rate, lrp=opt.learning_rate_prototypes, wd=opt.weight_decay,
             bsz=opt.batch_size, temp=opt.temp, trial=opt.trial,
             se=opt.start_epoch if opt.start_epoch is not None else opt.epochs, ep=opt.epochs,
             ct=opt.current_temp, pt=opt.past_temp, dp=opt.distill_power, dt=opt.distill_type,
             fpn=opt.freeze_prototypes_niters, seed=opt.seed)
    if opt.cosine:
        opt.model_name += '_cosine'

    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name += '_warm'
        opt.warmup_from_enc = 0.01
        opt.warmup_from_prot = 0.001
        opt.warm_epochs = 10
        if opt.cosine:
            eta_enc = opt.learning_rate * (opt.lr_decay_rate ** 3)
            eta_prot = opt.learning_rate_prototypes * (opt.lr_decay_rate ** 3)
            cos = (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
            opt.warmup_to_enc = eta_enc + (opt.learning_rate - eta_enc) * cos
            opt.warmup_to_prot = eta_prot + (opt.learning_rate_prototypes - eta_prot) * cos
        else:
            opt.warmup_to_enc = opt.learning_rate
            opt.warmup_to_prot = opt.learning_rate_prototypes

    for sub in (opt.tb_path, opt.model_path, opt.log_path):
        folder = os.path.join(sub, opt.model_name)
        os.makedirs(folder, exist_ok=True)
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    opt.log_folder = os.path.join(opt.log_path, opt.model_name)

# Linear-probe evaluation
def parse_eval_option():
    p = argparse.ArgumentParser('Linear-probe evaluation')

    p.add_argument('--method', type=str, default='cclis', choices=['cclis', 'hcdr'])
    p.add_argument('--replay_policy', type=str, choices=['random', 'weight'], default='weight')
    p.add_argument('--target_task', type=int, default=None,
                   help='task index to evaluate (default: final task of the dataset)')

    p.add_argument('--print_freq', type=int, default=10)
    p.add_argument('--save_freq', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=100)

    p.add_argument('--learning_rate', type=float, default=0.1)
    p.add_argument('--lr_decay_epochs', type=str, default='60,75,90')
    p.add_argument('--lr_decay_rate', type=float, default=0.2)
    p.add_argument('--weight_decay', type=float, default=0)
    p.add_argument('--momentum', type=float, default=0.9)

    p.add_argument('--model', type=str, default='resnet18')
    p.add_argument('--dataset', type=str, choices=_DATASETS[:-1], required=True)
    p.add_argument('--size', type=int, default=32)
    p.add_argument('--data_folder', type=str, default='~/data/')

    p.add_argument('--cosine', action='store_false')
    p.add_argument('--warm', action='store_true')
    p.add_argument('--ckpt', type=str, required=True, help='checkpoint directory')
    p.add_argument('--logpt', type=str, required=True, help='replay-indices log directory')

    opt = p.parse_args()
    opt.use_hcdr = False  # the linear probe only needs the (shared) encoder

    cfg = DATASET_CONFIG[opt.dataset]
    opt.n_cls = cfg['n_cls']
    opt.cls_per_task = cfg['cls_per_task']
    opt.size = cfg['size']
    if opt.target_task is None:
        opt.target_task = _EVAL_TARGET_TASK[opt.dataset]

    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]
    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.format(
        opt.dataset, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size)
    if opt.cosine:
        opt.model_name += '_cosine'
    if opt.warm:
        opt.model_name += '_warm'
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.origin_ckpt = opt.ckpt
    opt.ckpt = os.path.join(opt.ckpt, 'last_{}_{}.pth'.format(opt.replay_policy, opt.target_task))
    opt.logpt = os.path.join(opt.logpt, 'replay_indices_{}_{}.npy'.format(
        opt.replay_policy, opt.target_task))
    return opt