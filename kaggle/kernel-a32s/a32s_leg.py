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
# # A32-scratch leg — from-scratch JEPA with the A32 objective changes (pushable run notebook)
#
# _Maintainer note: this .py is the source of truth; regenerate with
# `jupytext --to ipynb kaggle/kernel-a32s/a32s_leg.py`. Shell commands LIVE._
#
# The proper A32 test (A33): the champion's own leg was from scratch, 30
# epochs, batch 1, lr 1e-4, warmup 5. Repeat that exact schedule with the
# three A32 flags (surgery-window, augmentation + clean EMA target,
# transition weighting) and compare the locked RANO probe and the same-space
# persistence gate against the frozen champion (0.309 CV / 0.448 final).

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
!git checkout 7320184
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
cfg['data']['augment_train'] = True
cfg['model']['brainiac']['checkpoint'] = f'{IN}/BrainIAC.ckpt'
cfg['training']['checkpoint_dir'] = '/kaggle/working/a32s'
cfg['training']['surgery_window'] = True
cfg['training']['transition_weighting'] = True
yaml.safe_dump(cfg, open('kaggle.yaml', 'w'))
print('wrote kaggle.yaml (augment + surgery_window + transition_weighting)')

# %%
# From scratch, 30 epochs, batch 1, lr 1e-4, warmup 5 — the champion's schedule.
!python -u scripts/run_train.py --config kaggle.yaml --epochs 30 --batch-size 1 --lr 0.0001 --warmup-epochs 5 --augment --surgery-window --transition-weighting --no-wandb --checkpoint-dir /kaggle/working/a32s 2>&1 | tee /kaggle/working/train_a32s.log

# %%
# Same-space persistence gate (needs the model's own states/targets cache).
!python -u scripts/horizon_probe.py --config kaggle.yaml --champion /kaggle/working/a32s/best.pt --cache /kaggle/working/a32s_hcache.pt --encode 2>&1 | tee /kaggle/working/encode_a32s.log
!python -u scripts/split_gate.py --champion /kaggle/working/a32s/best.pt --cache /kaggle/working/a32s_hcache.pt --boot 10000 2>&1 | tee /kaggle/working/gate_a32s.log

# %%
# Locked RANO probe (within-unseen CV + reserved final), reusing one cache.
!python -u scripts/probe_rano.py --config kaggle.yaml --champion /kaggle/working/a32s/best.pt --cache /kaggle/working/a32s_cache.pt --encode --cv-unseen --feat states_forecast --hidden 256 --train-pool unseen --cohort unseen --boot 10000 2>&1 | tee /kaggle/working/eval_a32s_unseen.log
!python -u scripts/probe_rano.py --cache /kaggle/working/a32s_cache.pt --cv-unseen --feat states_forecast --hidden 256 --train-pool train --cohort final --boot 10000 2>&1 | tee /kaggle/working/eval_a32s_final.log

# %%
import datetime, re, subprocess
commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                        capture_output=True, text=True).stdout.strip()
txt = open('/kaggle/working/train_a32s.log').read()
vals = re.findall(r'^val epoch (\d+): loss=([0-9.]+) std=([0-9.]+) rank=([0-9.]+)', txt, re.M)
best = re.findall(r'^done\. best val loss: ([0-9.]+)', txt, re.M)
def tail(p):
    try:
        return open(p).read().strip().splitlines()
    except OSError:
        return ['(missing)']
L = [f'# A32-scratch notes — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}',
     f'- commit: `{commit}` | from scratch, 30ep batch1 lr 1e-4 warmup5 + A32 flags',
     f'- best val loss: {best[-1] if best else "n/a"} (not comparable to the '
     f'champion 0.0081 — surgery-window removes the hard pairs, weighting changes the objective)']
L += [f'- val epoch {e}: loss={l} std={s} rank={r}' for e, l, s, r in vals]
L += ['\n## Persistence gate (split_gate)', '```'] + tail('/kaggle/working/gate_a32s.log') + ['```']
L += ['\n## Locked eval (within-unseen CV, 26)', '```'] + tail('/kaggle/working/eval_a32s_unseen.log') + ['```']
L += ['\n## Locked eval (transfer -> reserved final 13)', '```'] + tail('/kaggle/working/eval_a32s_final.log') + ['```']
notes = '\n'.join(L) + '\n'
open('/kaggle/working/run_notes.md', 'w').write(notes)
print(notes)
