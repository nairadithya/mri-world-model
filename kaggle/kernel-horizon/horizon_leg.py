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
# # Horizon leg — multi-horizon JEPA (pushable run notebook)
#
# _Maintainer note: this .py file is the source of truth. Never edit the
# .ipynb directly — regenerate it with `jupytext --to ipynb kaggle/kernel-horizon/horizon_leg.py`.
# NOTE: unlike `kaggle/hero_run.py`, the shell commands here are LIVE (no `#`
# comments) — `kaggle kernels push` executes the notebook as-is. Do not
# py_compile this file; it is notebook source, not a script._
#
# Plan: resume the 0.0081 champion with `--horizon` (gap-conditioned head,
# 1/n-weighted loss), 30 epochs at LR 2e-5, batch 1, T4 only.
# Gate: val loss AND per-horizon eval below vs the frozen-probe numbers
# (n=1: 0.0050; n=5: 0.0065 vs persistence 0.0134).

# %%
# Fail fast on the wrong GPU: P100 (sm_60) has no kernels in this image's
# torch build — a P100 session burns quota while failing every CUDA op.
import torch
assert torch.cuda.is_available(), 'no GPU allocated — aborting'
name = torch.cuda.get_device_name(0)
print('device:', name, f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
assert 'T4' in name, f'wrong GPU ({name}) — this run needs T4; aborting to save quota'

# %%
# Train-time deps only. NO hd-bet (would drag transformers>=5 + torchvision
# into the env and break the peft import). Preprocessing stays local.
!pip install --quiet monai "nibabel>=5.2" "SimpleITK>=2.4" "peft>=0.8" "transformers<5" "einops>=0.7" "scikit-learn>=1.3" "pyyaml>=6.0" "safetensors>=0.4" "tqdm>=4.65"
!pip uninstall --quiet -y torchao
import transformers, peft, monai
print('transformers', transformers.__version__, '| peft', peft.__version__, '| monai', monai.__version__)
assert transformers.__version__.startswith('4'), 'need transformers<5 for peft'

# %%
# Pin the exact code the hero run was gated on.
!rm -rf world-model && git clone https://github.com/nairadithya/mri-world-model.git world-model
%cd world-model
!git checkout e0bd393
!git rev-parse --short HEAD  # RECORD this hash with your results

# %%
# Stage the champion and wire paths. Datasets mount at /kaggle/input/<slug>/
# (depth varies — hence the recursive search, not a hardcoded path).
import glob, shutil, os, yaml

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
prev = sorted(glob.glob('/kaggle/input/**/prev-checkpoints/*.pt', recursive=True))
assert prev, 'Prev Checkpoints dataset missing or has no .pt files'
for p in prev:
    shutil.copy(p, '/kaggle/working/checkpoints/')
    print('staged', os.path.basename(p))

found = [d for d in glob.glob('/kaggle/input/**/lumiere_preprocessed', recursive=True)
         if os.path.isdir(d)]
assert found, 'Preprocessed MRI Data dataset missing'
IN = os.path.dirname(found[0])
print('inputs at', IN)
for p in ['lumiere_preprocessed', 'lumiere_meta', 'BrainIAC.ckpt']:
    assert os.path.exists(os.path.join(IN, p)), f'missing {p}'
print('input patients:', len(os.listdir(os.path.join(IN, 'lumiere_preprocessed'))))

cfg = yaml.safe_load(open('config/default.yaml'))
cfg['data']['root'] = f'{IN}/lumiere_preprocessed'
cfg['data']['raw_root'] = None
cfg['data']['meta_dir'] = f'{IN}/lumiere_meta'
cfg['model']['brainiac']['checkpoint'] = f'{IN}/BrainIAC.ckpt'
cfg['training']['checkpoint_dir'] = '/kaggle/working/checkpoints'
# The --horizon CLI flag enables the head in the trainer's in-memory config,
# but eval cells re-read this file — persist the flag so they agree.
cfg['model']['predictor']['horizon']['enabled'] = True
yaml.safe_dump(cfg, open('kaggle.yaml', 'w'))
print('wrote kaggle.yaml (horizon enabled)')

# %%
# HORIZON leg. Resume line must list the 14 horizon_* keys as randomly
# initialized (champion predates the head) — anything else missing means the
# wrong checkpoint got staged. Checkpoint every epoch; ~9h session cap.
!python scripts/run_train.py --config kaggle.yaml --epochs 30 --batch-size 1 --lr 0.00002 --horizon --no-wandb --resume-from /kaggle/working/checkpoints/best.pt 2>&1 | tee /kaggle/working/train_hz.log

# %%
# Per-horizon JEPA vs persistence on the new best.pt (same-pair comparison).
import sys
import torch.nn.functional as F
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from src.data.collate import make_collate
from src.data.dataset import LUMIEREDataset
from src.model.jepa_model import JEPAWorldModel

cfg = yaml.safe_load(open('kaggle.yaml'))
assert cfg['model']['predictor'].get('horizon', {}).get('enabled'), \
    'kaggle.yaml lost the horizon flag — check the wiring cell'
device = torch.device('cuda')
size = tuple(cfg['preprocessing'].get('target_size', [96, 96, 96]))
common = dict(meta_dir=cfg['data']['meta_dir'], processed_root=cfg['data']['root'],
              raw_root=None, modalities=tuple(cfg['data'].get('modalities', ['CT1', 'T1', 'T2', 'FLAIR'])),
              min_visits=cfg['data'].get('min_visits', 2))
collate = make_collate(size)
model = JEPAWorldModel(cfg)
model.load_state_dict(torch.load('/kaggle/working/checkpoints/best.pt', map_location='cpu')['model'], strict=False)
model.eval().to(device)
by_n = {}
pats = sorted(os.listdir(cfg['data']['root']))
with torch.no_grad():
    for p in pats:
        ds = LUMIEREDataset(patients=[p], **common)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
        for b in loader:
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            hz = model(b)['horizon']
            if hz is None:
                continue
            pe = 1 - (F.normalize(hz['zt'], dim=-1) * F.normalize(hz['zu'], dim=-1)).sum(dim=-1)
            for n_, je, pe_ in zip(hz['n'].tolist(), hz['err'].tolist(), pe.tolist()):
                by_n.setdefault(int(n_), []).append((je, pe_))
print(f"{'n':>4} {'pairs':>7} {'jepa':>8} {'persist':>8}")
for n_ in sorted(by_n):
    je = sum(r[0] for r in by_n[n_] ) / len(by_n[n_])
    pe = sum(r[1] for r in by_n[n_]) / len(by_n[n_])
    print(f"{n_:>4} {len(by_n[n_]):>7} {je:>8.4f} {pe:>8.4f}")

# %%
# Collect the notes file for the repo record.
import datetime, re, subprocess

commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                        capture_output=True, text=True).stdout.strip()
txt = open('/kaggle/working/train_hz.log').read()
vals = re.findall(r'^val epoch (\d+): loss=([0-9.]+) std=([0-9.]+) rank=([0-9.]+)', txt, re.M)
done = re.findall(r'^done\. best val loss: ([0-9.]+)', txt, re.M)
L = [f'# Horizon-leg notes — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}',
     f'- commit: `{commit}` | CLI: --epochs 30 --batch-size 1 --lr 0.00002 --horizon --resume-from champion(best.pt)',
     f'- best val: {done[-1] if done else "n/a"}']
L += [f'- epoch {e}: loss={l} std={s} rank={r}' for e, l, s, r in vals]
L += ['\n## Per-horizon eval (JEPA vs persistence)', '(paste the eval-cell table here)']
notes = '\n'.join(L) + '\n'
open('/kaggle/working/run_notes.md', 'w').write(notes)
print(notes)
