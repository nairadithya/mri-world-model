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
# # CNN leg — supervised 3D ResNet-18 comparator (pushable run notebook)
#
# _Maintainer note: this .py is the source of truth; regenerate the .ipynb with
# `jupytext --to ipynb kaggle/kernel-cnn/cnn_leg.py`. Shell commands are LIVE._
#
# Purpose: train the supervised `pure CNN` comparator on LUMIERE (pair framing,
# locked protocol) and encode LUMIERE features for the cross-site adaptability
# comparison against the JEPA encoder. SAILOR is controlled-access and is NOT on
# Kaggle — its features are encoded locally with this same checkpoint.

# %%
import torch
assert torch.cuda.is_available(), 'no GPU allocated — aborting'
name = torch.cuda.get_device_name(0)
print('device:', name, f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
assert 'T4' in name, f'wrong GPU ({name}) — this run needs T4; aborting to save quota'

# %%
!pip install --quiet monai "nibabel>=5.2" "SimpleITK>=2.4" "peft>=0.8" "transformers<5" "einops>=0.7" "scikit-learn>=1.3" "pyyaml>=6.0" "safetensors>=0.4" "tqdm>=4.65"
!pip uninstall --quiet -y torchao
import transformers, monai
print('transformers', transformers.__version__, '| monai', monai.__version__)
assert transformers.__version__.startswith('4')

# %%
# Pin the commit that contains the CNN comparator + adaptation harness.
!rm -rf world-model && git clone https://github.com/nairadithya/mri-world-model.git world-model
%cd world-model
!git checkout bdd64c7
!git rev-parse --short HEAD

# %%
import glob, os, yaml

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
yaml.safe_dump(cfg, open('kaggle.yaml', 'w'))
print('wrote kaggle.yaml')

# %%
# Train the pure CNN (pair framing), report the reserved final, then encode
# LUMIERE pair features (SAILOR is skipped — not mounted on Kaggle).
!python -u scripts/train_supervised_cnn.py --config kaggle.yaml --train --eval-final --encode --encode-scope lum --augment --epochs 20 --lr 0.0001 --pair-batch 4 --patience 8 --min-epochs 5 --checkpoint-dir /kaggle/working/cnn --ckpt /kaggle/working/cnn/best.pt --feature-cache /kaggle/working/cnn_features.pt 2>&1 | tee /kaggle/working/train_cnn.log

# %%
import datetime, re, subprocess

commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                        capture_output=True, text=True).stdout.strip()
txt = open('/kaggle/working/train_cnn.log').read()
eps = re.findall(r'^epoch (\d+): loss=([0-9.]+) dev_macroF1=([0-9.]+)', txt, re.M)
best = re.findall(r'^best dev macro-F1 ([0-9.]+) @ epoch (\d+)', txt, re.M)
final = re.findall(r'^FINAL \(reserved\): \d+ patients macro-F1 ([0-9.]+)', txt, re.M)
L = [f'# CNN-leg notes — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}',
     f'- commit: `{commit}` | train_supervised_cnn.py pair framing, ResNet-18 3D',
     f'- best dev macro-F1: {best[-1] if best else "n/a"}',
     f'- reserved final macro-F1: {final[-1] if final else "n/a"}']
L += [f'- epoch {e}: loss={l} dev_f1={f}' for e, l, f in eps]
notes = '\n'.join(L) + '\n'
open('/kaggle/working/run_notes.md', 'w').write(notes)
print(notes)
