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
# # LoRA leg — supervised vision finetune (pushable run notebook)
#
# _Maintainer note: this .py file is the source of truth. Never edit the
# .ipynb directly — regenerate it with
# `jupytext --to ipynb kaggle/kernel-lora/lora_leg.py`.
# NOTE: unlike `kaggle/hero_run.py`, the shell commands here are LIVE (no `#`
# comments) — `kaggle kernels push` executes the notebook as-is. Do not
# py_compile this file; it is notebook source, not a script._
#
# Plan: `scripts/finetune_lora.py`, the one untried SOTA recipe — optimize the
# image representation for RANO. Trainable = LoRA + projector + fusion + head,
# temporal frozen; class-weighted CE on state_t -> RANO_{t+1}, JEPA
# distillation regularizer, augmentation, early stop on the 13 `dev`.
#
# Gate (locked protocol, `info/eval_protocol.md`): beat the frozen readout on
# the reserved `final` 13 with a paired patient-cluster CI excluding 0
# (frozen reference: honest ~0.39, seed-42 0.448). If it does not, the encoder
# is at its data-limited ceiling and only external data remains.

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
# Pin the exact commit that contains scripts/finetune_lora.py + the
# augmentation collate. RECORD the printed hash with the results.
!rm -rf world-model && git clone https://github.com/nairadithya/mri-world-model.git world-model
%cd world-model
!git checkout e4a1b5d
!git rev-parse --short HEAD

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
# The champion is the prev-checkpoints best.pt (val 0.0081); select by
# definition if several files land (D22: never trust a filename).
import torch as _t
champ = '/kaggle/working/checkpoints/best.pt'
assert os.path.exists(champ), 'expected best.pt in prev-checkpoints'
print('champion val:', _t.load(champ, map_location='cpu', weights_only=False).get('val_loss'))

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
yaml.safe_dump(cfg, open('kaggle.yaml', 'w'))
print('wrote kaggle.yaml')

# %%
# LORA leg. 20 epochs, LR 2e-5, accumulation 8 (batch-1 memory), augmentation,
# JEPA distillation lambda 0.1 to keep dynamics from collapsing. Early stop on
# dev macro-F1; final is touched only by the locked eval below.
!python -u scripts/finetune_lora.py --config kaggle.yaml --champion /kaggle/working/checkpoints/best.pt --epochs 20 --lr 0.00002 --accum-steps 8 --jepa-lambda 0.1 --augment --patience 5 --checkpoint-dir /kaggle/working/lora 2>&1 | tee /kaggle/working/train_lora.log

# %%
# Locked-protocol eval. Encode the finetuned champion's states into a fresh
# cache, then score with the frozen readout on the same folds as A25/A26.
!python -u scripts/probe_rano.py --config kaggle.yaml --champion /kaggle/working/lora/best.pt --cache /kaggle/working/lora_cache.pt --encode --cv-unseen --feat states_forecast --hidden 256 --train-pool unseen --cohort unseen --boot 10000 2>&1 | tee /kaggle/working/eval_lora_unseen.log

# %%
# Reserved final 13 (transfer framing), reusing the cache from the cell above.
!python -u scripts/probe_rano.py --cache /kaggle/working/lora_cache.pt --cv-unseen --feat states_forecast --hidden 256 --train-pool train --cohort final --boot 10000 2>&1 | tee /kaggle/working/eval_lora_final.log

# %%
# Collect the notes file for the repo record.
import datetime, re, subprocess

commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                        capture_output=True, text=True).stdout.strip()
txt = open('/kaggle/working/train_lora.log').read()
eps = re.findall(r'^epoch (\d+): loss=([0-9.]+) ce=([0-9.]+) jepa=([0-9.]+) dev_macroF1=([0-9.]+)', txt, re.M)
best = re.findall(r'^best dev macro-F1 ([0-9.]+) @ epoch (\d+)', txt, re.M)
unseen = open('/kaggle/working/eval_lora_unseen.log').read().strip().splitlines()
final = open('/kaggle/working/eval_lora_final.log').read().strip().splitlines()
L = [f'# LoRA-leg notes — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}',
     f'- commit: `{commit}` | finetune_lora.py --epochs 20 --lr 2e-5 --accum-steps 8 --jepa-lambda 0.1 --augment',
     f'- best dev macro-F1: {best[-1] if best else "n/a"}']
L += [f'- epoch {e}: loss={l} ce={c} jepa={j} dev_f1={f}' for e, l, c, j, f in eps]
L += ['\n## Locked eval (within-unseen CV, 26)', '```'] + unseen + ['```']
L += ['\n## Locked eval (transfer -> reserved final 13)', '```'] + final + ['```']
notes = '\n'.join(L) + '\n'
open('/kaggle/working/run_notes.md', 'w').write(notes)
print(notes)
