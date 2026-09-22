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
# # P2.4 learned lesion-transition run
#
# Source of truth: this percent-format file. The run anatomy-supervises LoRA,
# caches adapted lesion tokens, then trains the elapsed-time transition GRU.

# %%
import torch
assert torch.cuda.is_available(), 'no GPU allocated — aborting'
name = torch.cuda.get_device_name(0)
print('device:', name, f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
assert 'T4' in name, f'wrong GPU ({name}); this run requires T4'

# %%
!pip install --quiet monai "nibabel>=5.2" "SimpleITK>=2.4" "peft>=0.8" "transformers<5" "einops>=0.7" "scikit-learn>=1.3" "pyyaml>=6.0" "safetensors>=0.4" "tqdm>=4.65"
!pip uninstall --quiet -y torchao
import transformers, peft, monai
print('transformers', transformers.__version__, 'peft', peft.__version__, 'monai', monai.__version__)
assert transformers.__version__.startswith('4')

# %%
!rm -rf world-model
!git clone https://github.com/nairadithya/mri-world-model.git world-model
%cd world-model
!git checkout 667b2dc
!git rev-parse --short HEAD

# %%
import glob, os, shutil, yaml, torch as _torch

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
previous = sorted(glob.glob('/kaggle/input/**/prev-checkpoints/*.pt', recursive=True))
assert previous, 'previous checkpoint dataset missing'
for path in previous:
    shutil.copy(path, '/kaggle/working/checkpoints/')
champion = '/kaggle/working/checkpoints/best.pt'
assert os.path.exists(champion), 'best.pt champion missing'
print('champion val', _torch.load(champion, map_location='cpu', weights_only=False).get('val_loss'))

roots = [path for path in glob.glob('/kaggle/input/**/lumiere_preprocessed', recursive=True)
         if os.path.isdir(path)]
assert roots, 'preprocessed LUMIERE missing'
input_root = os.path.dirname(roots[0])
mask_roots = [path for path in glob.glob('/kaggle/input/**/Imaging', recursive=True)
              if os.path.isdir(path) and glob.glob(path + '/Patient-*/week-*/DeepBraTumIA-segmentation')]
assert mask_roots, 'lesion supervision masks missing'
mask_root = mask_roots[0]
print('images', input_root, 'masks', mask_root)

cfg = yaml.safe_load(open('config/default.yaml'))
cfg['data']['root'] = input_root + '/lumiere_preprocessed'
cfg['data']['raw_root'] = None
cfg['data']['meta_dir'] = input_root + '/lumiere_meta'
cfg['model']['brainiac']['checkpoint'] = input_root + '/BrainIAC.ckpt'
yaml.safe_dump(cfg, open('kaggle.yaml', 'w'))

# %%
!python -u scripts/harness.py train lesion --config kaggle.yaml --champion /kaggle/working/checkpoints/best.pt --mask-root "$mask_root" --anatomy-epochs 8 --forecast-epochs 200 --lr 0.0002 --accum 8 --out /kaggle/working/lesion_learned.pt 2>&1 | tee /kaggle/working/train_lesion.log

# %%
import json
result = json.load(open('/kaggle/working/lesion_learned.pt.json'))
print(json.dumps(result, indent=2))
assert result['locked_unseen_relative'] == result['locked_unseen_relative'], 'non-finite result'
verdict = 'PASS' if result['locked_unseen_relative'] < 1.0 else 'FAIL'
print('LOCKED PERSISTENCE GATE:', verdict, result['locked_unseen_relative'])

# %%
import datetime, subprocess
commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                        capture_output=True, text=True).stdout.strip()
notes = [f'# P2.4 lesion-transition run — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}',
         f'- commit: `{commit}`', '- GPU: Tesla T4',
         '- anatomy LoRA: 8 epochs, lr 2e-4, accumulation 8',
         '- transition GRU: up to 200 epochs, patience 20',
         f'- locked relative MAE: {result["locked_unseen_relative"]:.6f}',
         f'- gate: {verdict}']
open('/kaggle/working/run_notes.md', 'w').write('\n'.join(notes) + '\n')
print('\n'.join(notes))
