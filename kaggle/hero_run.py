# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hero run on free Kaggle GPU (T4/P100, 16 GB)
#
# _Maintainer note: this .py file is the source of truth. Never edit the
# .ipynb directly — regenerate it with `jupytext --to ipynb kaggle/hero_run.py`._
#
# Same code + configs as the local pilot, scaled to the full 91-patient cohort.
# Nothing here requires a paid GPU — the trade is wall-clock time, not feasibility.
#
# **Session plan** (free tier: ~30 h GPU/week, ~9 h max per session):
#
# 1. This notebook, GPU enabled: install → clone → wire paths → train.
# 2. First session: set `EPOCHS = 30`, run the training cell. Outputs persist to `/kaggle/working`.
# 3. Later sessions: attach the previous session's output as `prev-checkpoints`, run the **resume** cell with a higher `EPOCHS`.
# 4. After every session, paste the `val epoch` lines into `info/pilot.md` (Run 4+) back home — results tie to commits, per repo convention 4.

# %% [markdown]
# ## Step 0 — one-time: create the input dataset (on kaggle.com, 5 min)
#
# Kaggle → Datasets → New Dataset → upload the folder `world-model-inputs/` with this layout:
#
# ```
# world-model-inputs/
#   lumiere_preprocessed/<Patient>/<visit>/{CT1,T1,T2,FLAIR}.nii.gz  # full cohort, built by scripts/preprocess.py
#   lumiere_meta/*.csv                                              # rano/demographics/mri_params/completeness
#   BrainIAC.ckpt                                                   # official weights ONLY (never community ports — repo D8)
# ```
#
# Keep it **private**. Then in this notebook: Add input → `world-model-inputs` (lands at `/kaggle/input/world-model-inputs`).
# For follow-up sessions, also attach the prior session's output as `prev-checkpoints` (see resume cell).

# %%
# Train-time deps only. NO hd-bet here: it would drag transformers>=5 +
# torchvision into the env and break the peft import (repo D16).
# Preprocessing stays on the dev machine; Kaggle only trains.
# !pip install --quiet monai "nibabel>=5.2" "SimpleITK>=2.4" "peft>=0.8" "transformers<5" "einops>=0.7" "scikit-learn>=1.3" "pyyaml>=6.0" "safetensors>=0.4" "tqdm>=4.65"
# Kaggle images ship torchao 0.10, which recent peft treats as fatal
# (found-but-too-old raises; absence is handled gracefully). We never
# use torchao quantization, so remove it outright.
# !pip uninstall --quiet -y torchao
# Leave the image's torch+torchvision alone: they are version-matched, and
# nothing in our path needs torchvision. Only `pip uninstall -y torchvision`
# if a peft/transformers import error tells you to (repo D16).

# %%
import torch
print('cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), f'{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')
import transformers, peft, monai
print('transformers', transformers.__version__, '| peft', peft.__version__, '| monai', monai.__version__)
assert transformers.__version__.startswith('4'), 'need transformers<5 for peft (repo D16)'

# %%
# If the repo is private this fails — then attach it as a Kaggle code dataset
# or use a token URL instead.
# !rm -rf world-model && git clone https://github.com/nairadithya/mri-world-model.git world-model
# %cd world-model
# !git rev-parse --short HEAD  # RECORD this hash with your results (repo convention 4)

# %%
# Wire the read-only Kaggle inputs + writable checkpoint dir into a run config.
# raw_root=None: full preprocess covers every imaged visit; imageless visits
# are dropped by design (repo D13). Processed-only is intentional, not a gap.
import os, yaml

IN = '/kaggle/input/datasets/nairadithya/preprocessed-mri-data'
# (dataset root holds lumiere_preprocessed/, lumiere_meta/, BrainIAC.ckpt directly;
# adjust if your mount path differs — the asserts below will tell you)
for p in ['lumiere_preprocessed', 'lumiere_meta', 'BrainIAC.ckpt']:
    assert os.path.exists(os.path.join(IN, p)), f'missing {p} — check Step 0 dataset layout'
print('input patients:', len(os.listdir(os.path.join(IN, 'lumiere_preprocessed'))))

cfg = yaml.safe_load(open('config/default.yaml'))
cfg['data']['root'] = f'{IN}/lumiere_preprocessed'
cfg['data']['raw_root'] = None
cfg['data']['meta_dir'] = f'{IN}/lumiere_meta'
cfg['model']['brainiac']['checkpoint'] = f'{IN}/BrainIAC.ckpt'
cfg['training']['checkpoint_dir'] = '/kaggle/working/checkpoints'
yaml.safe_dump(cfg, open('kaggle.yaml', 'w'))
print('wrote kaggle.yaml; checkpoint_dir -> /kaggle/working/checkpoints')

# %%
# FIRST session (and any fresh start). Edit EPOCHS to fit the time you have;
# ~9 h session cap — stop early freely, checkpoints save every epoch.
# Batch 1 fits 16 GB (chunked encoding, repo D15); batch 2+ needs 24 GB VRAM.
# %env EPOCHS=30
# %env BATCH=1
!(echo "CLI: --epochs $EPOCHS --batch-size $BATCH"; python scripts/run_train.py --config kaggle.yaml --epochs $EPOCHS --batch-size $BATCH --no-wandb) 2>&1 | tee /kaggle/working/train_ep$EPOCHS.log

# If $EPOCHS ever expands empty (argparse: 'expected one argument'),
# the variable handoff broke. Fallback: hardcode the number, e.g.
# # !python scripts/run_train.py --config kaggle.yaml --epochs 30 --batch-size 1 --no-wandb 2>&1 | tee /kaggle/working/train_ep30.log

# %%
# FOLLOW-UP sessions: attach the previous session's output as input dataset
# `prev-checkpoints` (must contain best.pt), then resume. Resume from BEST,
# not last: last.pt may hold overfit tail-epoch weights (Run 4), while
# best.pt is the lowest-val checkpoint and only ever improves. Fresh
# optimizer + LR schedule per leg (documented `--resume-from` limitation);
# loss may blip for an epoch. --epochs here means ADDITIONAL epochs
# (the counter restarts each leg); best.pt tracking makes over-long legs
# safe — the optimum is kept regardless.
import glob, shutil, os
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
# Recursive: Kaggle mounts datasets at varying depths
# (/kaggle/input/<slug>/ vs /kaggle/input/datasets/<user>/<slug>/).
prev = sorted(glob.glob('/kaggle/input/**/prev-checkpoints/*.pt', recursive=True))
assert prev, 'attach previous session output as prev-checkpoints first'
for p in prev:
    shutil.copy(p, '/kaggle/working/checkpoints/')
    print('staged', os.path.basename(p))

# %env EPOCHS=30  # additional epochs this leg (counter restarts; best.pt is safe)
# %env BATCH=1
# %env LR=0.00002  # leg 2+: 5x below the 1e-4 that found the ep-8 optimum (Run 4
# overfit past it). Exploit with small steps; omit --lr for config default.
!(echo "CLI: --epochs $EPOCHS --batch-size $BATCH --lr $LR --resume-from best.pt"; python scripts/run_train.py --config kaggle.yaml --epochs $EPOCHS --batch-size $BATCH --lr $LR --no-wandb --resume-from /kaggle/working/checkpoints/best.pt) 2>&1 | tee /kaggle/working/train_ep$EPOCHS.log

# %% [markdown]
# ## After each session
#
# 1. `/kaggle/working/checkpoints/best.pt` + `last.pt` are your outputs — download them, or save the version to attach as `prev-checkpoints` next time.
# 2. Run the notes cell, then copy `/kaggle/working/run_notes.md` home into `info/pilot.md` (new Run section, with this notebook's commit hash).
# 3. Gate (repo A7): JEPA mean vs persistence 0.0043 — plus the train/val generalization gap. Run the eval cell below for the per-patient table.

# %%
# OPTIONAL: per-patient JEPA vs persistence on the current best.pt
# (CPU-friendly on GPU; mirrors repo A6/A7 method). Needs no extra deps.
import sys, glob, torch, yaml
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from src.data.collate import make_collate
from src.data.dataset import LUMIEREDataset
from src.model.jepa_model import JEPAWorldModel
from src.train.baselines import PersistenceBaseline

cfg = yaml.safe_load(open('kaggle.yaml'))
device = torch.device('cuda')
size = tuple(cfg['preprocessing'].get('target_size', [96, 96, 96]))
common = dict(meta_dir=cfg['data']['meta_dir'], processed_root=cfg['data']['root'],
                raw_root=None, modalities=tuple(cfg['data'].get('modalities', ['CT1', 'T1', 'T2', 'FLAIR'])),
                min_visits=cfg['data'].get('min_visits', 2))
collate = make_collate(size)
model = JEPAWorldModel(cfg)
model.load_state_dict(torch.load('/kaggle/working/checkpoints/best.pt', map_location='cpu')['model'])
model.eval().to(device)
pers = PersistenceBaseline(model.projector)
pats = sorted(os.listdir(cfg['data']['root']))
J = P = n = 0
for p in pats:
    ds = LUMIEREDataset(patients=[p], **common)
    if len(ds) == 0:
        continue
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    jl = pl = m = 0
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            jl += model(b)['loss'].item()
            pl += pers(b, model.encode_visits)['loss'].item()
            m += 1
    print(f'{p}: JEPA={jl / m:.4f} persist={pl / m:.4f}')
    J += jl / m; P += pl / m; n += 1
print(f'MEAN over {n} patients: JEPA={J / n:.4f} persist={P / n:.4f}')

# %% [markdown]
# ## Collect the notes file (run after training and/or eval)
#
# Parses every `train_ep*.log` into one paste-ready file:
# `/kaggle/working/run_notes.md`. It persists as session output — download it
# or copy its printed contents straight into `info/pilot.md` back home.

# %%
import datetime, glob, os, re, subprocess, yaml

commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                            capture_output=True, text=True).stdout.strip()
cfg = yaml.safe_load(open('kaggle.yaml'))
tr, data = cfg['training'], cfg['data']

legs = []
for log in sorted(glob.glob('/kaggle/working/train_ep*.log')):
    txt = open(log).read()
    vals = re.findall(r'^val epoch (\d+): loss=([0-9.]+) std=([0-9.]+) rank=([0-9.]+)', txt, re.M)
    done = re.findall(r'^done\. best val loss: ([0-9.]+)', txt, re.M)
    cli = re.findall(r'^CLI: (.*)', txt, re.M)
    legs.append((log, vals, done[-1] if done else 'n/a', cli[-1] if cli else 'pre-CLI-echo logs: see cell source'))

L = []
L.append(f'# Hero-run notes — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}')
L.append(f'- commit: `{commit}` | config: `kaggle.yaml` (from `config/default.yaml`)')
L.append(f"- data: {len(os.listdir(data['root']))} patients | weights: official `BrainIAC.ckpt`")
L.append(f"- batch {tr.get('batch_size')} | lr {tr.get('lr')} | warmup {tr.get('warmup_epochs')} | seed {data.get('seed')} | AMP {tr.get('use_amp')}")
for log, vals, best, cli in legs:
    L.append(f'\n## {os.path.basename(log)} — best val {best} — CLI: {cli}')
    L += [f'- epoch {e}: loss={l} std={s} rank={r}' for e, l, s, r in vals]
L.append('\n## Eval (JEPA vs persistence per patient)')
L.append('(paste the eval-cell table here, or: not run)')
L.append('\n## Gate')
L.append('JEPA mean vs persistence 0.0043 (A6); watch the train/val generalization gap (A7).')
notes = '\n'.join(L) + '\n'
open('/kaggle/working/run_notes.md', 'w').write(notes)
print(notes)
