"""HCDR component ablation harness.

Verifies that each component HCDR adds on top of the CCLIS baseline actually
contributes to final accuracy, by training/evaluating a small grid of
configurations and reporting the deltas between them.

HCDR's objective is

    L = L_global_NCE  +  alpha_patch * L_patch_NCE  +  lambda_hprd * L_H_PRD

so a component is ablated simply by zeroing its weight (``--alpha_patch 0`` or
``--lambda_hprd 0``).  This isolates the *loss term's* contribution while
holding the architecture fixed (the patch heads still exist in every ``hcdr``
run).  The separate ``cclis`` config additionally removes the patch
architecture entirely, so the grid also exposes any pure-architecture effect.

Default grid (each trained + linear-probe evaluated):

    name              method  alpha_patch  lambda_hprd   isolates
    ----------------  ------  -----------  -----------   --------------------------
    cclis             cclis   --           --            baseline, no patch heads
    hcdr_none         hcdr    0            0             patch arch only, both off
    hcdr_patch_only   hcdr    A            0             + patch-NCE
    hcdr_hprd_only    hcdr    0            L             + hierarchical-PRD
    hcdr_full         hcdr    A            L             full method

From these you can read off:
    patch-NCE effect = hcdr_patch_only - hcdr_none   and   hcdr_full - hcdr_hprd_only
    H-PRD effect     = hcdr_hprd_only  - hcdr_none   and   hcdr_full - hcdr_patch_only
    architecture     = hcdr_none       - cclis
    full gain        = hcdr_full       - cclis

Run several seeds (``--seeds 0,1,2``) to get mean +/- std per cell; ablation
claims should be backed by more than one seed.

This script only orchestrates the existing ``train.py`` / ``evaluate.py``
entrypoints via subprocess; it does not change any training behaviour.

Examples
--------
    # quick smoke test (cheap settings, one seed), just print the commands:
    python ablation.py --dataset cifar10 --mem-size 200 \
        --start-epoch 5 --epochs 3 --seeds 0 --dry-run

    # real run on tiny-imagenet, three seeds, on GPU 2:
    python ablation.py --dataset tiny-imagenet --mem-size 2000 \
        --seeds 0,1,2 --gpu 2 --extra-train "--no_compile"

    # add a top-k patch sensitivity sweep at full alpha/lambda:
    python ablation.py --dataset cifar100 --mem-size 2000 \
        --seeds 0,1 --topk-sweep 5,10,20
"""
import argparse
import csv
import glob
import os
import re
import statistics
import subprocess
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

_CLASS_IL_RE = re.compile(r'Average accuracy for Class-IL:\s*([-+0-9.eE]+)')
_TASK_IL_RE = re.compile(r'Average accuracy for Task-IL:\s*([-+0-9.eE]+)')


# --------------------------------------------------------------------------- #
# Configuration grid                                                          #
# --------------------------------------------------------------------------- #
def build_grid(opt):
    """Return the list of ablation configs given the CLI options."""
    A, L = opt.alpha, opt.lam
    grid = [
        dict(name='cclis', method='cclis'),
        dict(name='hcdr_none', method='hcdr', alpha=0.0, lam=0.0),
        dict(name='hcdr_patch_only', method='hcdr', alpha=A, lam=0.0),
        dict(name='hcdr_hprd_only', method='hcdr', alpha=0.0, lam=L),
        dict(name='hcdr_full', method='hcdr', alpha=A, lam=L),
    ]
    if opt.configs:
        wanted = set(opt.configs.split(','))
        grid = [c for c in grid if c['name'] in wanted]
        missing = wanted - {c['name'] for c in grid}
        if missing:
            sys.exit('Unknown config name(s): {}'.format(', '.join(sorted(missing))))

    if opt.topk_sweep:
        for k in [int(x) for x in opt.topk_sweep.split(',')]:
            grid.append(dict(name='hcdr_full_top{}'.format(k), method='hcdr',
                             alpha=A, lam=L, top_k=k))
    return grid


def train_args_for(cfg, seed, opt):
    """CLI args (after ``train.py``) for one config + seed."""
    args = ['--method', cfg['method'],
            '--dataset', opt.dataset,
            '--mem_size', str(opt.mem_size),
            '--seed', str(seed),
            '--replay_policy', opt.policy]
    if opt.start_epoch is not None:
        args += ['--start_epoch', str(opt.start_epoch)]
    if opt.epochs is not None:
        args += ['--epochs', str(opt.epochs)]
    if cfg['method'] == 'hcdr':
        args += ['--alpha_patch', str(cfg.get('alpha', opt.alpha)),
                 '--lambda_hprd', str(cfg.get('lam', opt.lam))]
        if 'top_k' in cfg:
            args += ['--top_k_patches', str(cfg['top_k'])]
    if opt.extra_train:
        args += opt.extra_train.split()
    return args


# --------------------------------------------------------------------------- #
# Subprocess helpers                                                          #
# --------------------------------------------------------------------------- #
def resolve_folders(train_args):
    """Ask config.py for the exact save/log folders these train args produce.

    parse_train_option() is fully deterministic in its arguments, so this gives
    the same directories train.py will write to -- no fragile globbing.
    """
    snippet = (
        'import sys, config\n'
        'sys.argv = ["train.py"] + {!r}\n'
        'opt = config.parse_train_option()\n'
        'print(opt.save_folder)\n'
        'print(opt.log_folder)\n'
    ).format(train_args)
    out = subprocess.run([sys.executable, '-c', snippet], cwd=PKG_DIR,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError('Could not resolve folders:\n' + out.stderr)
    save_folder, log_folder = out.stdout.strip().splitlines()[-2:]
    return save_folder, log_folder


def env_with_gpu(opt):
    env = os.environ.copy()
    if opt.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    return env


def run(cmd, env, log_path):
    """Run a command, streaming stdout to console and a log file. Returns text."""
    print('  $ {}'.format(' '.join(cmd)))
    lines = []
    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(cmd, cwd=PKG_DIR, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            lines.append(line)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError('Command failed (exit {}): {}'.format(proc.returncode, ' '.join(cmd)))
    return ''.join(lines)


def parse_acc(text):
    c = _CLASS_IL_RE.search(text)
    t = _TASK_IL_RE.search(text)
    return (float(c.group(1)) if c else None,
            float(t.group(1)) if t else None)


def checkpoint_exists(save_folder, policy):
    return bool(glob.glob(os.path.join(save_folder, 'last_{}_*.pth'.format(policy))))


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return (float('nan'), float('nan'))
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (m, s)


def fmt(m, s):
    return '{:6.2f} +/- {:4.2f}'.format(m, s)


def print_table(results, grid_names):
    print('\n' + '=' * 72)
    print('ABLATION SUMMARY  (mean +/- std over seeds)')
    print('=' * 72)
    print('{:<20}  {:>15}  {:>15}'.format('config', 'Class-IL', 'Task-IL'))
    print('-' * 72)
    summ = {}
    for name in grid_names:
        cil = agg([r['class_il'] for r in results if r['config'] == name])
        til = agg([r['task_il'] for r in results if r['config'] == name])
        summ[name] = {'class_il': cil, 'task_il': til}
        print('{:<20}  {:>15}  {:>15}'.format(name, fmt(*cil), fmt(*til)))
    print('-' * 72)

    def delta(a, b, metric):
        if a in summ and b in summ:
            return summ[a][metric][0] - summ[b][metric][0]
        return None

    print('\nComponent contributions (Class-IL percentage points):')
    interp = [
        ('patch-NCE  (full - hprd_only)', 'hcdr_full', 'hcdr_hprd_only'),
        ('patch-NCE  (patch_only - none)', 'hcdr_patch_only', 'hcdr_none'),
        ('H-PRD      (full - patch_only)', 'hcdr_full', 'hcdr_patch_only'),
        ('H-PRD      (hprd_only - none)', 'hcdr_hprd_only', 'hcdr_none'),
        ('patch arch (none - cclis)', 'hcdr_none', 'cclis'),
        ('full gain  (full - cclis)', 'hcdr_full', 'cclis'),
    ]
    for label, a, b in interp:
        d = delta(a, b, 'class_il')
        if d is not None:
            print('  {:<34} {:+6.2f}'.format(label, d))
    print('=' * 72)
    return summ


def write_csv(path, per_run, summ):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'seed', 'class_il', 'task_il'])
        for r in per_run:
            w.writerow([r['config'], r['seed'], r['class_il'], r['task_il']])
        w.writerow([])
        w.writerow(['config', 'class_il_mean', 'class_il_std',
                    'task_il_mean', 'task_il_std'])
        for name, s in summ.items():
            w.writerow([name, '{:.4f}'.format(s['class_il'][0]),
                        '{:.4f}'.format(s['class_il'][1]),
                        '{:.4f}'.format(s['task_il'][0]),
                        '{:.4f}'.format(s['task_il'][1])])
    print('\nWrote results to {}'.format(path))


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser('HCDR ablation harness',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', required=True)
    p.add_argument('--mem-size', type=int, required=True)
    p.add_argument('--seeds', default='0', help='comma-separated seeds')
    p.add_argument('--policy', default='weight', choices=['weight', 'random'])
    p.add_argument('--start-epoch', type=int, default=None,
                   help='override (defaults to the dataset config)')
    p.add_argument('--epochs', type=int, default=None,
                   help='override (defaults to the dataset config)')
    p.add_argument('--alpha', type=float, default=0.5, help='alpha_patch when enabled')
    p.add_argument('--lam', type=float, default=0.6, help='lambda_hprd when enabled')
    p.add_argument('--configs', default=None,
                   help='comma-separated subset of config names to run')
    p.add_argument('--topk-sweep', default=None,
                   help='comma-separated top_k_patches values to add at full alpha/lambda')
    p.add_argument('--gpu', default=None, help='value for CUDA_VISIBLE_DEVICES')
    p.add_argument('--extra-train', default=None,
                   help='extra args forwarded verbatim to train.py, e.g. "--no_compile"')
    p.add_argument('--extra-eval', default=None,
                   help='extra args forwarded verbatim to evaluate.py')
    p.add_argument('--reuse-checkpoints', action='store_true',
                   help='skip training a config/seed if its checkpoint already exists')
    p.add_argument('--out', default='ablation_results.csv')
    p.add_argument('--logdir', default='ablation_logs')
    p.add_argument('--dry-run', action='store_true',
                   help='print the train/eval commands without running them')
    opt = p.parse_args()

    seeds = [int(s) for s in opt.seeds.split(',')]
    grid = build_grid(opt)
    grid_names = [c['name'] for c in grid]
    env = env_with_gpu(opt)
    os.makedirs(os.path.join(PKG_DIR, opt.logdir), exist_ok=True)

    print('Ablation plan: dataset={} mem_size={} policy={} seeds={}'.format(
        opt.dataset, opt.mem_size, opt.policy, seeds))
    print('Configs: {}'.format(', '.join(grid_names)))

    per_run = []
    for cfg in grid:
        for seed in seeds:
            tag = '{}_seed{}'.format(cfg['name'], seed)
            print('\n' + '-' * 72)
            print('CONFIG {} | seed {}'.format(cfg['name'], seed))
            print('-' * 72)

            t_args = train_args_for(cfg, seed, opt)
            train_cmd = [sys.executable, 'train.py'] + t_args

            if opt.dry_run:
                save_folder = log_folder = '<resolved-at-runtime>'
                print('  [dry-run] train: {}'.format(' '.join(train_cmd)))
            else:
                save_folder, log_folder = resolve_folders(t_args)

            eval_cmd = [sys.executable, 'evaluate.py',
                        '--method', cfg['method'],
                        '--dataset', opt.dataset,
                        '--replay_policy', opt.policy,
                        '--ckpt', save_folder,
                        '--logpt', log_folder]
            if opt.extra_eval:
                eval_cmd += opt.extra_eval.split()

            if opt.dry_run:
                print('  [dry-run] eval:  {}'.format(' '.join(eval_cmd)))
                per_run.append(dict(config=cfg['name'], seed=seed,
                                    class_il=None, task_il=None))
                continue

            if opt.reuse_checkpoints and checkpoint_exists(save_folder, opt.policy):
                print('  reusing existing checkpoint in {}'.format(save_folder))
            else:
                run(train_cmd, env, os.path.join(PKG_DIR, opt.logdir, tag + '_train.log'))

            eval_text = run(eval_cmd, env,
                            os.path.join(PKG_DIR, opt.logdir, tag + '_eval.log'))
            class_il, task_il = parse_acc(eval_text)
            if class_il is None:
                print('  WARNING: could not parse accuracy for {}'.format(tag))
            else:
                print('  -> Class-IL {:.2f} | Task-IL {:.2f}'.format(
                    class_il, task_il if task_il is not None else float('nan')))
            per_run.append(dict(config=cfg['name'], seed=seed,
                                class_il=class_il, task_il=task_il))

    if opt.dry_run:
        print('\n[dry-run] no experiments executed.')
        return

    summ = print_table(per_run, grid_names)
    write_csv(os.path.join(PKG_DIR, opt.out), per_run, summ)


if __name__ == '__main__':
    main()