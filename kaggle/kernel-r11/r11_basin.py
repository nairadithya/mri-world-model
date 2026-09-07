# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # R11 — basin-hold matrix (does continued training eject, or the resume?)
#
# _Maintainer note: this .py file is the source of truth. Never edit the
# .ipynb directly — regenerate it with `jupytext --to ipynb kaggle/kernel-r11/r11_basin.py`.
# Shell commands here are LIVE — `kaggle kernels push` executes the notebook
# as-is. Do not py_compile this file; it is notebook source, not a script._
#
# Plan: resume the 0.0081 champion, 4 legs × 5 epochs, flat LR 2e-5, batch 1,
# warmup 1 (identical schedules — the ONLY differences are optimizer state and
# accumulation). D22's hypothesis (fresh momentum ejects the narrow basin)
# predicts: fresh-opt legs drift from epoch 1, loaded-opt legs hold.
# - A: accum 1, fresh opt (Run-5 replication — expect drift)
# - B: accum 1, loaded opt (--resume-opt)
# - C: accum 8, fresh opt
# - D: accum 8, loaded opt
# Gate: epoch-1 val per leg (drift-from-ep1 is the D22 signature) + best val.
# Verdict rule: B holds while A drifts → momentum was the ejector; C/D hold
# while A/B drift → batch-1 noise was; all drift → basin truly unholdable
# (D22 stands, no more training ever).

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
# Pin the exact code the R11 gate was designed on (accumulation + --resume-opt
# + --checkpoint-dir all in this tree).
!rm -rf world-model && git clone https://github.com/nairadithya/mri-world-model.git world-model
%cd world-model
!git checkout acf7503
!git rev-parse --short HEAD  # RECORD this hash with your results

# %%
# Stage the champion by DEFINITION (lowest val_loss in prev-checkpoints),
# not by filename — a resume leg's best.pt once overwrote the staged copy
# (D22), so names are untrustworthy. Abort if nothing is champion-grade.
import glob, shutil, os, yaml

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
cands = sorted(glob.glob('/kaggle/input/**/prev-checkpoints/*.pt', recursive=True))
assert cands, 'Prev Checkpoints dataset missing or has no .pt files'
rows = []
for p in cands:
    ck = None
    try:
        ck = torch.load(p, map_location='cpu')
        rows.append((p, ck.get('epoch', '?'), ck.get('val_loss', float('inf'))))
    except Exception as e:
        print('unreadable', os.path.basename(p), e)
    finally:
        del ck
for p, e, v in rows:
    print(f'{os.path.basename(p):30s} epoch={e} val={v} size={os.path.getsize(p)//10**6}MB')
ok = [(p, e, v) for p, e, v in rows if isinstance(v, float) and v < 0.0085]
assert len(ok) == 1, f'champion ambiguous/missing (need exactly one file with val<0.0085): {ok}'
champ = ok[0][0]
print('CHAMPION:', os.path.basename(champ), 'val=', ok[0][2])
shutil.copy(champ, '/kaggle/working/checkpoints/champion.pt')
open('/kaggle/working/CHAMPION', 'w').write('/kaggle/working/checkpoints/champion.pt')

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
print('wrote kaggle.yaml (plain 1-step JEPA; per-leg flags live in the train cells below)')

# %%
# Leg A: accum 1, FRESH opt (Run-5 replication — expect drift from epoch 1).
!python scripts/run_train.py --config kaggle.yaml --epochs 5 --batch-size 1 --lr 0.00002 --warmup-epochs 1 --no-wandb --accum-steps 1 --resume-from $(cat /kaggle/working/CHAMPION) --checkpoint-dir /kaggle/working/checkpoints/legA 2>&1 | tee /kaggle/working/train_A.log

# %%
# Leg B: accum 1, LOADED opt. Resume line must say "loaded optimizer".
!python scripts/run_train.py --config kaggle.yaml --epochs 5 --batch-size 1 --lr 0.00002 --warmup-epochs 1 --no-wandb --accum-steps 1 --resume-from $(cat /kaggle/working/CHAMPION) --resume-opt --checkpoint-dir /kaggle/working/checkpoints/legB 2>&1 | tee /kaggle/working/train_B.log

# %%
# Leg C: accum 8, fresh opt.
!python scripts/run_train.py --config kaggle.yaml --epochs 5 --batch-size 1 --lr 0.00002 --warmup-epochs 1 --no-wandb --accum-steps 8 --resume-from $(cat /kaggle/working/CHAMPION) --checkpoint-dir /kaggle/working/checkpoints/legC 2>&1 | tee /kaggle/working/train_C.log

# %%
# Leg D: accum 8, loaded opt.
!python scripts/run_train.py --config kaggle.yaml --epochs 5 --batch-size 1 --lr 0.00002 --warmup-epochs 1 --no-wandb --accum-steps 8 --resume-from $(cat /kaggle/working/CHAMPION) --resume-opt --checkpoint-dir /kaggle/working/checkpoints/legD 2>&1 | tee /kaggle/working/train_D.log

# %%
# Verdict table: val trajectory per leg straight from the logs.
import datetime, re, subprocess

commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                        capture_output=True, text=True).stdout.strip()
print('code:', commit)
print(f"{'leg':>5} {'setup':>22} {'ep1-val':>8} {'best-val':>9} {'ep5-val':>8}")
verdict = {}
for leg, setup in [('A', 'accum1/fresh'), ('B', 'accum1/loaded'),
                   ('C', 'accum8/fresh'), ('D', 'accum8/loaded')]:
    try:
        txt = open(f'/kaggle/working/train_{leg}.log').read()
    except FileNotFoundError:
        print(f"{leg:>5} {setup:>22} NO LOG — leg never ran")
        verdict[leg] = (float('nan'), float('nan'), float('nan'))
        continue
    vals = re.findall(r'^val epoch (\d+): loss=([0-9.]+) std=([0-9.]+) rank=([0-9.]+)', txt, re.M)
    res = re.findall(r'resumed weights from .* epoch ([^,]+), (\w+) optimizer', txt)
    assert res and res[0][1] == ('loaded' if 'loaded' in setup else 'fresh'), \
        f'leg {leg}: optimizer mode wrong in log — {res} vs {setup}; DO NOT TRUST THIS LEG'
    ep1 = float(vals[0][1]) if vals else float('nan')
    best = min(float(v[1]) for v in vals) if vals else float('nan')
    last = float(vals[-1][1]) if vals else float('nan')
    verdict[leg] = (ep1, best, last)
    print(f"{leg:>5} {setup:>22} {ep1:>8.4f} {best:>9.4f} {last:>8.4f}")
champ_v = 0.0081
L = [f'# R11 basin-hold notes — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}',
     f'- commit: `{commit}` | champion val 0.0081 | flat LR 2e-5, warmup 1, batch 1, 5 epochs/leg',
     f'- A accum1/fresh: ep1 {verdict["A"][0]:.4f}, best {verdict["A"][1]:.4f}',
     f'- B accum1/loaded: ep1 {verdict["B"][0]:.4f}, best {verdict["B"][1]:.4f}',
     f'- C accum8/fresh: ep1 {verdict["C"][0]:.4f}, best {verdict["C"][1]:.4f}',
     f'- D accum8/loaded: ep1 {verdict["D"][0]:.4f}, best {verdict["D"][1]:.4f}',
     '- reading: B holds while A drifts → momentum was the ejector; '
     'C/D hold while A/B drift → batch-1 noise was; all drift → D22 stands.']
open('/kaggle/working/run_notes.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
