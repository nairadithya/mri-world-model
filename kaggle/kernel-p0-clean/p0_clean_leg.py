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
# # P0 clean JEPA retrain — survival-free prospective input contract
#
# _Maintainer note: this .py is the source of truth; regenerate the notebook
# with `jupytext --to ipynb kaggle/kernel-p0-clean/p0_clean_leg.py`. Shell
# commands are LIVE._
#
# This is a clean from-scratch JEPA training leg using official BRAINIAC
# initialization and `include_survival: false`. The base BRAINIAC ViT remains
# frozen; LoRA adapters and JEPA heads train. Outputs feed the local cache
# rebuild.

# %%
import torch
assert torch.cuda.is_available(), 'no GPU allocated — aborting'
name = torch.cuda.get_device_name(0)
print('device:', name, f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
assert 'T4' in name, f'wrong GPU ({name}) — aborting'

# %%
!pip install --quiet monai "nibabel>=5.2" "SimpleITK>=2.4" "peft>=0.8" "transformers<5" "einops>=0.7" "scikit-learn>=1.3" "pyyaml>=6.0" "safetensors>=0.4" "tqdm>=4.65"
!pip uninstall --quiet -y torchao
import transformers, peft, monai
print('transformers', transformers.__version__, '| peft', peft.__version__, '| monai', monai.__version__)
assert transformers.__version__.startswith('4')

# %%
!rm -rf world-model && git clone --branch exp/p0-clean https://github.com/nairadithya/mri-world-model.git world-model
%cd world-model
!git rev-parse --short HEAD

# %%
import glob, os, yaml

found = [d for d in glob.glob('/kaggle/input/**/lumiere_preprocessed', recursive=True)
         if os.path.isdir(d)]
assert found, 'Preprocessed MRI Data dataset missing'
IN = os.path.dirname(found[0])
for p in ['lumiere_preprocessed', 'lumiere_meta', 'BrainIAC.ckpt']:
    assert os.path.exists(os.path.join(IN, p)), f'missing {p}'
print('input patients:', len(os.listdir(os.path.join(IN, 'lumiere_preprocessed'))))

cfg = yaml.safe_load(open('config/default.yaml'))
cfg['data']['root'] = f'{IN}/lumiere_preprocessed'
cfg['data']['raw_root'] = None
cfg['data']['meta_dir'] = f'{IN}/lumiere_meta'
cfg['data']['include_survival'] = False
cfg['model']['brainiac']['checkpoint'] = f'{IN}/BrainIAC.ckpt'
cfg['training']['checkpoint_dir'] = '/kaggle/working/p0_clean'
yaml.safe_dump(cfg, open('kaggle.yaml', 'w'))
print('wrote kaggle.yaml (survival-free clinical input)')

# %%
# From scratch, 30 epochs, batch 1, accumulation 8, lr 1e-4, warmup 5.
!python -u scripts/harness.py train jepa --config kaggle.yaml --epochs 30 --batch-size 1 --accum-steps 8 --lr 0.0001 --warmup-epochs 5 --no-wandb --checkpoint-dir /kaggle/working/p0_clean 2>&1 | tee /kaggle/working/train_p0_clean.log

# %%
# Preserve both checkpoints for the local cache rebuild.
!cp /kaggle/working/p0_clean/best.pt /kaggle/working/p0_clean_best.pt
!cp /kaggle/working/p0_clean/last.pt /kaggle/working/p0_clean_last.pt

# %%
import datetime, re, subprocess
commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                        capture_output=True, text=True).stdout.strip()
txt = open('/kaggle/working/train_p0_clean.log').read()
vals = re.findall(r'^val epoch (\d+): loss=([0-9.]+) std=([0-9.]+) rank=([0-9.]+)', txt, re.M)
best = re.findall(r'^done\. best val loss: ([0-9.]+)', txt, re.M)
L = [f'# P0 clean retrain notes — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}',
     f'- commit: `{commit}` | survival-free, official BRAINIAC, 30ep, batch1, accum8',
     f'- best val loss: {best[-1] if best else "n/a"}']
L += [f'- val epoch {e}: loss={l} std={s} rank={r}' for e, l, s, r in vals]
notes = '\n'.join(L) + '\n'
open('/kaggle/working/run_notes.md', 'w').write(notes)
print(notes)
