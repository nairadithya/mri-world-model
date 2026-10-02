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
# # Clean-data rescue: augmentation
#
# Source of truth is this .py; regenerate with jupytext. Commands are LIVE.
# R33 protocol: info/clean_rescues.md. SAILOR is evaluated locally only.

# %%
import torch
assert torch.cuda.is_available(), 'No GPU allocated'
print('device:', torch.cuda.get_device_name(0))
assert 'T4' in torch.cuda.get_device_name(0), 'Wrong GPU; abort before training'

# %%
!pip install --quiet monai "nibabel>=5.2" "SimpleITK>=2.4" "peft>=0.8" "transformers<5" "einops>=0.7" "scikit-learn>=1.3" "pyyaml>=6.0" "safetensors>=0.4" "tqdm>=4.65"
!pip uninstall --quiet -y torchao
import transformers, peft, monai
assert transformers.__version__.startswith('4')

# %%
import subprocess
subprocess.run(['git', 'clone', '--branch', 'exp/clean-data-rescues',
                'https://github.com/nairadithya/mri-world-model.git',
                '/kaggle/working/world-model'], check=True)
%cd /kaggle/working/world-model
subprocess.run(['git', 'checkout', '36b7078d3e328d95f85af7aed74512626d12f396'], check=True)
subprocess.run(['pip', 'freeze'], stdout=open('/kaggle/working/environment.txt', 'w'), check=True)

# %%
import glob, os, zipfile, yaml, json
from pathlib import Path
from src.harness.analysis.clean_rescues import sha256
from src.harness.data.anatomy_manifest import build_manifest

candidates = sorted(glob.glob('/kaggle/input/**/p0_clean_best.pt', recursive=True))
assert len(candidates) == 1, f'Clean checkpoint mount ambiguous: {candidates}'
champion = candidates[0]
assert sha256(champion) == '7186a6267d7aa42720381e6c2780f9c86fe20e71067074f8eacdf63d48199d0a', 'Wrong control checkpoint'
checkpoint = torch.load(champion, map_location='cpu', weights_only=False)
assert checkpoint.get('opt'), 'Optimizer stripped; full basin matrix is invalid'
assert checkpoint['config']['data']['include_survival'] is False
print('clean control:', checkpoint['epoch'], checkpoint['val_loss'])
del checkpoint

roots = sorted(glob.glob('/kaggle/input/**/lumiere_preprocessed', recursive=True))
assert len(roots) == 1, f'MRI mount ambiguous: {roots}'
input_root = Path(roots[0]).parent
masks = sorted(glob.glob('/kaggle/input/**/Patient-*/week-*/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii*', recursive=True))
if not masks:
    archives = sorted(glob.glob('/kaggle/input/**/Imaging.zip', recursive=True))
    assert len(archives) == 1, f'Mask archive missing/ambiguous: {archives}'
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall('/kaggle/working/lesion-supervision')
    masks = sorted(glob.glob('/kaggle/working/lesion-supervision/**/Patient-*/week-*/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii*', recursive=True))
assert masks, 'Native-affine masks missing'
mask_root = masks[0].split('/Patient-', 1)[0]
manifest = build_manifest(lumiere_root=str(input_root / 'lumiere_preprocessed'),
                          lumiere_volumes='/kaggle/working/absent-volumes',
                          lumiere_masks=mask_root, sailor_root='/kaggle/working/absent-sailor')
assert sum(p['usable'] for p in manifest['pairs']) == 496, 'Audited anatomy pair set changed'
Path('/kaggle/working/anatomy_manifest.json').write_text(json.dumps(manifest, indent=2))
cfg = yaml.safe_load(Path('config/default.yaml').read_text())
cfg['data'].update(root=str(input_root / 'lumiere_preprocessed'), raw_root=None,
                   meta_dir=str(input_root / 'lumiere_meta'), include_survival=False)
cfg['model']['brainiac']['checkpoint'] = str(input_root / 'BrainIAC.ckpt')
Path('/kaggle/working/rescue_base.yaml').write_text(yaml.safe_dump(cfg))

# %%
# All dependent commands checked; a failed training arm cannot become a score.
subprocess.run(['python', '-u', '-m', 'src.harness.train.rescues',
                '--config', '/kaggle/working/rescue_base.yaml',
                '--champion', champion, '--experiment', 'augmentation',
                '--manifest', '/kaggle/working/anatomy_manifest.json',
                '--out-dir', '/kaggle/working/clean_rescues'], check=True)

# %%
status = json.loads(Path('/kaggle/working/clean_rescues/status.json').read_text())
print(json.dumps(status, indent=2))
# Scores determine scientific verdicts; parsing/optional files never assert
# that a run passed. Persisted checkpoints retain optimizer states.
for path in sorted(Path('/kaggle/working/clean_rescues').glob('**/*eval.json')):
    result = json.loads(path.read_text())
    print(path.parent.name, {k: v['persistence_relative_mae'] for k, v in result['anatomy'].items()})
