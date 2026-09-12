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
# # A32 leg — JEPA-side objective changes (pushable run notebook)
#
# _Maintainer note: this .py is the source of truth; regenerate with
# `jupytext --to ipynb kaggle/kernel-a32/a32_leg.py`. Shell commands LIVE._
#
# Resumes the champion for a short accum-8 leg with the three A32 changes:
# surgery-window pair exclusion (R7), SSL augmentation with a clean EMA-target
# view, and transition-balanced JEPA loss (R13). Gate: same-space persistence
# (split_gate) and the locked RANO probe (A25) vs the frozen champion
# (0.309 within-unseen CV / 0.448 final seed-42).

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
!rm -rf world-model && git clone https://github.com/nairadithya/mri-world-model.git world-model
%cd world-model
!git checkout HARNESS_COMMIT  # TODO(harness): pin the commit containing scripts/harness.py (was 7320184)
!git rev-parse --short HEAD

# %%
import glob, shutil, os, yaml

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
prev = sorted(glob.glob('/kaggle/input/**/prev-checkpoints/*.pt', recursive=True))
assert prev, 'Prev Checkpoints dataset missing'
for p in prev:
    shutil.copy(p, '/kaggle/working/checkpoints/')
champ = '/kaggle/working/checkpoints/best.pt'
assert os.path.exists(champ)
print('champion val:', torch.load(champ, map_location='cpu', weights_only=False).get('val_loss'))

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
cfg['data']['augment_train'] = True
cfg['model']['brainiac']['checkpoint'] = f'{IN}/BrainIAC.ckpt'
cfg['training']['checkpoint_dir'] = '/kaggle/working/a32'
cfg['training']['surgery_window'] = True
cfg['training']['transition_weighting'] = True
yaml.safe_dump(cfg, open('kaggle.yaml', 'w'))
print('wrote kaggle.yaml (augment + surgery_window + transition_weighting)')

# %%
# Short accum-8 leg from the champion. Val prints the JEPA loss; the gate is
# the separate eval cells below.
!python -u scripts/harness.py train jepa --config kaggle.yaml --epochs 6 --batch-size 1 --accum-steps 8 --lr 0.00002 --augment --surgery-window --transition-weighting --no-wandb --resume-from /kaggle/working/checkpoints/best.pt --checkpoint-dir /kaggle/working/a32 2>&1 | tee /kaggle/working/train_a32.log

# %%
# Locked-protocol RANO probe on the new representation (within-unseen CV).
!python -u scripts/harness.py encode --config kaggle.yaml --champion /kaggle/working/a32/best.pt --cache /kaggle/working/a32_cache.pt --views vision fused states clinical 2>&1 | tee /kaggle/working/encode_a32.log
!python -u scripts/harness.py eval --cache /kaggle/working/a32_cache.pt --task rano4_forecast --view states_forecast --readout mlp --hidden 256 --train-pool unseen --cohort unseen --boot 10000 2>&1 | tee /kaggle/working/eval_a32_unseen.log

# %%
# Reserved final 13 (transfer framing), reusing the cache above.
!python -u scripts/harness.py eval --cache /kaggle/working/a32_cache.pt --task rano4_forecast --view states_forecast --readout mlp --hidden 256 --train-pool train --cohort final --boot 10000 2>&1 | tee /kaggle/working/eval_a32_final.log

# %%
import datetime, re, subprocess
commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                        capture_output=True, text=True).stdout.strip()
txt = open('/kaggle/working/train_a32.log').read()
vals = re.findall(r'^val epoch (\d+): loss=([0-9.]+) std=([0-9.]+) rank=([0-9.]+)', txt, re.M)
best = re.findall(r'^done\. best val loss: ([0-9.]+)', txt, re.M)
unseen = open('/kaggle/working/eval_a32_unseen.log').read().strip().splitlines()
final = open('/kaggle/working/eval_a32_final.log').read().strip().splitlines()
L = [f'# A32-leg notes — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}',
     f'- commit: `{commit}` | resume champion, 6ep accum-8 lr 2e-5 + A32 flags',
     f'- best val loss: {best[-1] if best else "n/a"}',
     '- (compare frozen champion val loss 0.0081 / test 0.0074)']
L += [f'- val epoch {e}: loss={l} std={s} rank={r}' for e, l, s, r in vals]
L += ['\n## Locked eval (within-unseen CV, 26)', '```'] + unseen + ['```']
L += ['\n## Locked eval (transfer -> reserved final 13)', '```'] + final + ['```']
notes = '\n'.join(L) + '\n'
open('/kaggle/working/run_notes.md', 'w').write(notes)
print(notes)
