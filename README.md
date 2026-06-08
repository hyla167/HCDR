# HCDR — Hierarchical Contrastive Distillation Replay

A modular PyTorch implementation of **HCDR**, a rehearsal-based continual
learning method for image classification, together with its predecessor
baseline **CCLIS** (Contrastive Continual Learning with Importance Sampling).

Both methods share the same training scaffold — importance-sampled replay
selection, an importance-sampled supervised-contrastive global objective, and
prototype distillation across tasks. HCDR adds two terms on top:

```
L = L_global_NCE  +  alpha_patch · L_patch_NCE  +  lambda_hprd · L_H_PRD
```

* **L_patch_NCE** — patch-level InfoNCE that contrasts spatial patch tokens
  against per-class *patch prototypes*.
* **L_H_PRD** — *hierarchical* prototype-relational distillation: for every
  patch token, the softmax distribution over the global class prototypes is
  distilled from the frozen past model into the current one, so each local
  region's relationship to the global class concept stays stable across tasks.

A `PatchBuffer` tracks, for each replay image, its frozen global embedding and
the positions of its top-K discriminative patches (selected by a cosine
boundary score gated at the 90th within-class percentile).

## Package layout

```
HCDR/
├── train.py            # unified training entrypoint (--method {cclis,hcdr})
├── evaluate.py         # unified linear-probe evaluation
├── config.py           # argument parsing + per-dataset config (train & eval)
├── run.sh              # full train→evaluate pipeline over mem sizes
├── requirements.txt
├── networks/
│   └── resnet.py       # ResNet-18/34/50/101, SupConResNet (+ HCDR patch heads),
│                       #   SupCEResNet, LinearClassifier
├── losses/
│   ├── supcon.py       # ISSupConLoss (importance-sampled SupCon + scoring)
│   └── hcdr.py         # HCDRLoss, PatchBuffer, discriminative-patch scoring
├── data/
│   ├── datasets.py     # TinyImagenet, STL10/Caltech256 wrappers, transforms,
│   │                   #   per-dataset config & channel statistics
│   └── loaders.py      # replay sampling, train/eval DataLoader construction
└── utils/
    └── util.py         # meters, LR schedules, optimizer, checkpoint I/O, sampler
```

The four original monolithic scripts (`cclis.py`, `hcdr.py`, `eval_cclis.py`,
`eval_hcdr.py`) are collapsed into this package. CCLIS is simply HCDR with the
patch terms disabled, so a single `train.py` covers both via `--method`; and
since the two eval scripts were identical, there is one `evaluate.py`.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2.1 is required (the code uses `torch.amp`, `torch.compile`, and the
fused SGD path on CUDA).

## Data

* **cifar10 / cifar100 / stl10** — downloaded automatically by torchvision on
  first run.
* **tiny-imagenet** — downloaded automatically from Google Drive (the processed
  archive) into `--data_folder`.
* **caltech256** — torchvision cannot auto-download it reliably; obtain the
  `256_ObjectCategories` archive yourself and point `--data_folder` at it. A
  reproducible 80/20 per-class split (seed 42, clutter class excluded) is built
  on load.

Set the data root with `--data_folder` (default `~/data/`).

## Usage

Run commands from inside the `hcdr_cl/` directory (imports are package-relative).

Train:

```bash
python train.py --method hcdr  --dataset tiny-imagenet --mem_size 2000 \
                --start_epoch 500 --epochs 50
python train.py --method cclis --dataset tiny-imagenet --mem_size 2000 \
                --start_epoch 500 --epochs 50
```

Evaluate (linear probe) — point `--ckpt`/`--logpt` at the model and log
directories produced by training:

```bash
python evaluate.py --method hcdr --dataset tiny-imagenet \
                   --ckpt  <run>/tiny-imagenet_models/<model_name> \
                   --logpt <run>/logs/<model_name>
```

Full sweep (training + evaluation over several memory sizes):

```bash
bash run.sh
```

`run.sh` runs each configuration, then `ls -td | grep` selects the matching
run directory — HCDR runs carry an `hcdr` tag in their `model_name`, so the
two methods are still distinguishable under a shared save root.

### Key HCDR arguments

| flag | default | meaning |
|------|---------|---------|
| `--alpha_patch`   | 0.5 | weight of the patch-level NCE term |
| `--lambda_hprd`   | 0.6 | weight of the hierarchical-PRD term |
| `--top_k_patches` | 10  | discriminative patches stored per buffer image |

`--method cclis` ignores all three (the patch heads are never built).

## Behavioural notes / fixes vs. the original scripts

The refactor preserves the algorithms exactly, with a few deliberate
corrections that are applied identically to both methods (so any CCLIS-vs-HCDR
comparison stays fair):

1. **Optimizer built before `torch.compile`.** The originals called the
   optimizer setup *after* compiling; the resulting `_orig_mod.` prefix on the
   state-dict keys broke the exact-match check that creates the per-group
   prototype learning rate, silently collapsing everything to one param group
   and one LR. The optimizer is now constructed on the uncompiled model so the
   prototype LR (and the HCDR patch-head/patch-prototype groups) take effect.
   Pass `--legacy_optimizer` to reproduce the old single-group behaviour.
2. **`torch.compile` is optional** via `--no_compile`.
3. **Checkpoint loading in evaluation uses `strict=False`** and strips both
   `_orig_mod.` (compile) and `module.` (DataParallel) prefixes. HCDR
   checkpoints carry patch-head / patch-prototype tensors the probe doesn't
   use; these are reported and ignored, while the encoder/head/prototypes load
   normally.
4. **AMP precision is chosen at runtime:** bf16 autocast when the GPU supports
   it (no `GradScaler` needed), otherwise fp16 + a single shared `GradScaler`.
5. **Contrastive logits use float32** consistently (`LOGITS_DTYPE` in
   `losses/supcon.py`). The original CCLIS used float64 and HCDR float32.
6. **`path` dataset std fix:** the original computed the normalization std from
   `opt.mean`; it now correctly uses `opt.std`.
7. **`channels_last` memory format** is applied to the model and inputs;
   unused imports (e.g. `scipy`) were removed.

## Ablation study

`ablation.py` orchestrates `train.py` + `evaluate.py` over a small grid to
verify that each HCDR component actually contributes to accuracy. Because the
objective is `L = L_global_NCE + alpha_patch·L_patch_NCE + lambda_hprd·L_H_PRD`,
a term is ablated simply by zeroing its weight, which isolates that loss term
while keeping the architecture fixed:

| config | method | `alpha_patch` | `lambda_hprd` | isolates |
|--------|--------|---------------|---------------|----------|
| `cclis`           | cclis | — | — | baseline, no patch heads |
| `hcdr_none`       | hcdr  | 0 | 0 | patch architecture only, both terms off |
| `hcdr_patch_only` | hcdr  | A | 0 | + patch-NCE |
| `hcdr_hprd_only`  | hcdr  | 0 | L | + hierarchical-PRD |
| `hcdr_full`       | hcdr  | A | L | full method |

Each config gets a unique `model_name`, so runs never collide; the script asks
`config.py` for the exact save/log directories rather than globbing.

```bash
# quick smoke test (cheap settings) — just print the commands first
python ablation.py --dataset cifar10 --mem-size 200 \
    --start-epoch 5 --epochs 3 --seeds 0 --dry-run

# real run, three seeds, on GPU 2
python ablation.py --dataset tiny-imagenet --mem-size 2000 --seeds 0,1,2 --gpu 2

# optional top_k_patches sensitivity sweep at full alpha/lambda
python ablation.py --dataset cifar100 --mem-size 2000 --seeds 0,1 --topk-sweep 5,10,20

# optional percentile sweep
python ablation.py --dataset tiny-imagenet --mem-size 2000 --seeds 0 --percentile-sweep 80,85,90
```

It prints a mean ± std table per config (over seeds) and the per-component
contributions in percentage points — e.g. patch-NCE as `hcdr_full − hcdr_hprd_only`
and H-PRD as `hcdr_full − hcdr_patch_only` — and writes per-run and summary rows
to `ablation_results.csv` (per-run stdout logs go to `ablation_logs/`). Useful
flags: `--configs` to run a subset, `--reuse-checkpoints` to skip configs whose
checkpoint already exists, and `--extra-train`/`--extra-eval` to forward args
(e.g. `--extra-train "--no_compile"`). Run more than one seed before drawing
conclusions, since single-seed differences can be noise.

## Checkpoints & logs

Training writes, per task `t`, under the run's model/log folders:

* `last_{policy}_{t}.pth` — model + optimizer state (final task only)
* `replay_indices_{policy}_{t}.npy`, `importance_weight_{policy}_{t}.npy`,
  `subset_indices_{policy}_{t}.npy`, `score_{policy}_{t}.npy`

Evaluation reads the final-task checkpoint and replay indices, writes
`acc_buffer_{t}.txt` (best Class-IL accuracy plus per-class accuracies) and a
`linear_{t}.pth` classifier checkpoint, and prints mean ± std Class-IL and
Task-IL accuracy.