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
# # CNN-2D leg — competitive supervised comparator (pushable run notebook)
#
# _Maintainer note: this .py is the source of truth; regenerate the .ipynb with
# `jupytext --to ipynb kaggle/kernel-cnn2d/cnn2d_leg.py`. Shell commands LIVE._
#
# Trains the 2D axial-slice ResNet-18 comparator to convergence on LUMIERE
# (locked protocol) and encodes LUMIERE pair features for the cross-site
# adaptability comparison against JEPA. SAILOR stays local (controlled access).

# %%
import torch
assert torch.cuda.is_available(), 'no GPU allocated — aborting'
name = torch.cuda.get_device_name(0)
print('device:', name, f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
assert 'T4' in name, f'wrong GPU ({name}) — aborting'

# %%
!pip install --quiet monai "nibabel>=5.2" "SimpleITK>=2.4" "peft>=0.8" "transformers<5" "einops>=0.7" "scikit-learn>=1.3" "pyyaml>=6.0" "safetensors>=0.4" "tqdm>=4.65"
!pip uninstall --quiet -y torchao
import transformers, monai
print('transformers', transformers.__version__, '| monai', monai.__version__)
assert transformers.__version__.startswith('4')

# %%
!rm -rf world-model && git clone https://github.com/nairadithya/mri-world-model.git world-model
%cd world-model
!git checkout f403516
!git rev-parse --short HEAD

# %%
# Stage the DeepBraTumIA atlas seg masks (ROI cropping) from the private
# lumiere-autoseg-masks dataset. Kaggle auto-extracts the upload and gunzips
# .nii.gz -> .nii in place, so we link the Imaging tree (absolute links only).
import glob, os
imgs = [p for p in glob.glob('/kaggle/input/**/Imaging', recursive=True)
        if os.path.isdir(p) and glob.glob(os.path.join(p, 'Patient-*'))]
assert imgs, 'lumiere-autoseg-masks dataset not mounted'
os.makedirs('data/autoseg/extracted', exist_ok=True)
link = os.path.abspath('data/autoseg/extracted/Imaging')
if not os.path.exists(link):
    os.symlink(imgs[0], link)
n = len(glob.glob('data/autoseg/extracted/Imaging/*/week-*/DeepBraTumIA-segmentation/'
                  'atlas/segmentation/seg_mask.nii*'))
print('staged seg masks:', n, 'from', imgs[0])
assert n > 500, 'too few seg masks staged'

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
# 2D ROI-crop training (tumor-centred 64^3 crops, tumor-bearing slices),
# reserved-final eval, LUMIERE
# feature encode (SAILOR not mounted).
!python -u scripts/train_supervised_cnn2d.py --config kaggle.yaml --train --eval-final --encode --encode-scope lum --augment --roi --crop 64 --epochs 30 --lr 0.0003 --slice-batch 32 --patience 8 --min-epochs 5 --checkpoint-dir /kaggle/working/cnn2d --ckpt /kaggle/working/cnn2d/best.pt --feature-cache /kaggle/working/cnn2d_features.pt 2>&1 | tee /kaggle/working/train_cnn2d.log

# %%
import datetime, re, subprocess

commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                        capture_output=True, text=True).stdout.strip()
txt = open('/kaggle/working/train_cnn2d.log').read()
eps = re.findall(r'^epoch (\d+): loss=([0-9.]+) steps=(\d+) dev_macroF1=([0-9.]+)', txt, re.M)
best = re.findall(r'^best dev macro-F1 ([0-9.]+) @ epoch (\d+)', txt, re.M)
final = re.findall(r'^FINAL \(reserved\): \d+ patients macro-F1 ([0-9.]+)', txt, re.M)
L = [f'# CNN-2D-leg notes — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}',
     f'- commit: `{commit}` | train_supervised_cnn2d.py (2D axial slices)',
     f'- best dev macro-F1: {best[-1] if best else "n/a"}',
     f'- reserved final macro-F1: {final[-1] if final else "n/a"}']
L += [f'- epoch {e}: loss={l} steps={s} dev_f1={f}' for e, l, s, f in eps]
notes = '\n'.join(L) + '\n'
open('/kaggle/working/run_notes.md', 'w').write(notes)
print(notes)
