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
# # R11 repair — durability + dose-response (v1 partial verdict in hand)
#
# _Maintainer note: this .py file is the source of truth. Never edit the
# .ipynb directly — regenerate it with `jupytext --to ipynb kaggle/kernel-r11/r11_basin.py`.
# Shell commands here are LIVE — `kaggle kernels push` executes the notebook
# as-is. Do not py_compile this file; it is notebook source, not a script._
#
# v1 outcome (A17): leg A (accum-1/fresh) ejected 0.0086→0.0139, replicating
# Run 5; leg C (accum-8/fresh) held flat 0.0078→0.0080 — and leg-C best
# (0.0078) sits BELOW champion (0.0081): accumulation may improve, not just
# hold. Legs B/D (loaded-opt) crashed correctly: ferried champions are
# opt-stripped, so momentum data does not exist anywhere — dropped, not
# retried. v1's verdict cell died on an untested log-regex (space-vs-paren);
# every parser below is tested against v1's real logs before push.
#
# Plan (flat LR 2e-5, batch 1, warmup 1, identical schedules):
# - C2: accum-8 fresh, 20 epochs from champion (durability: does the hold last?)
# - E: accum-4 fresh, 10 epochs from champion (dose-response midpoint)
# Gate: C2 holds 20ep → training continues (R13 unblocked); C2 drifts after
# epN → hold is transient (report N); E between A-ref and C2 → noise mechanism
# confirmed by dose. A-ref (v1): 0.0086→0.0092→0.0108→0.0121→0.0139.

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
# Pin the exact code v1 was gated on (accumulation + bucketing + G2/G3 all in
# this tree; repair changes notebook only).
!rm -rf world-model && git clone https://github.com/nairadithya/mri-world-model.git world-model
%cd world-model
!git checkout 3241596  # pinned: includes scripts/harness.py (D33-D35)
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
# Leg C2: accum-8 fresh, 20 epochs from champion (durability).
# -u: unbuffered stdout so the tee'd log is complete by construction.
!python -u scripts/harness.py train jepa --config kaggle.yaml --epochs 20 --batch-size 1 --lr 0.00002 --warmup-epochs 1 --no-wandb --accum-steps 8 --resume-from $(cat /kaggle/working/CHAMPION) --checkpoint-dir /kaggle/working/checkpoints/legC2 2>&1 | tee /kaggle/working/train_C2.log

# %%
# Leg E: accum-4 fresh, 10 epochs from champion (dose-response midpoint).
!python -u scripts/harness.py train jepa --config kaggle.yaml --epochs 10 --batch-size 1 --lr 0.00002 --warmup-epochs 1 --no-wandb --accum-steps 4 --resume-from $(cat /kaggle/working/CHAMPION) --checkpoint-dir /kaggle/working/checkpoints/legE 2>&1 | tee /kaggle/working/train_E.log

# %%
# Verdict table. NEVER assert-fails the session: every leg reports a status
# (OK / CRASH / NO LOG / UNKNOWN) and the table is written regardless.
# Parser patterns below were tested against v1's real logs (train_A/B/C/D)
# before push — the v1 failure was an untested `" epoch"`-vs-`"(epoch"`
# mismatch, never again.
import datetime, re, subprocess

commit = subprocess.run('git rev-parse --short HEAD', shell=True,
                        capture_output=True, text=True).stdout.strip()
print('code:', commit)

VAL_RE = r'^val epoch (\d+): loss=([0-9.]+)'
RESUME_RE = r'epoch ([^,]+), (\w+) optimizer'
CRASH_RES = [r'(ValueError: [^\n]{0,150})', r'(RuntimeError: [^\n]{0,150})',
             r'(CUDA out of memory[^\n]{0,80})']


def parse_leg(leg):
    try:
        txt = open(f'/kaggle/working/train_{leg}.log').read()
    except FileNotFoundError:
        return {'status': 'NO LOG', 'vals': [], 'resume': '?', 'crash': ''}
    vals = re.findall(VAL_RE, txt, re.M)
    res = re.findall(RESUME_RE, txt)
    crash = ''
    for pat in CRASH_RES:
        m = re.findall(pat, txt)
        if m:
            crash = m[0][:150]
            break
    return {'status': 'CRASH' if crash and not vals else 'OK',
            'vals': [(int(e), float(v)) for e, v in vals],
            'resume': res[0][1] if res else 'UNKNOWN (warn: resume line unparsed)',
            'crash': crash}


print(f"{'leg':>5} {'setup':>14} {'status':>8} {'ep1':>8} {'ep5':>8} "
      f"{'ep10':>8} {'ep20':>8} {'best':>8} {'resume':>8}")
verdict = {}
for leg, setup in [('C2', 'accum8/fresh'), ('E', 'accum4/fresh')]:
    r = parse_leg(leg)
    by_ep = dict(r['vals'])
    get = lambda e: f"{by_ep[e]:.4f}" if e in by_ep else 'n/a'
    best = min((v for _, v in r['vals']), default=float('nan'))
    verdict[leg] = (by_ep.get(1, float('nan')), best, r['status'])
    best_s = f"{best:.4f}" if best == best else "n/a"
    print(f"{leg:>5} {setup:>14} {r['status']:>8} {get(1):>8} {get(5):>8} "
          f"{get(10):>8} {get(20):>8} {best_s:>8} {r['resume']:>8}")
    if r['crash']:
        print(f'      crash: {r["crash"]}')
    if 'UNKNOWN' in r['resume']:
        print(f'      warn: resume line unparsed — leg identity unverified')
print('A-ref (v1, accum1/fresh): 0.0086 → 0.0092 → 0.0108 → 0.0121 → 0.0139 (ejects)')
champ_v = 0.0081
L = [f'# R11-repair notes — {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}',
     f'- commit: `{commit}` | champion val 0.0081 | flat LR 2e-5, warmup 1, batch 1',
     f'- C2 accum8/fresh 20ep: ep1 {verdict["C2"][0]:.4f}, best {verdict["C2"][1]:.4f}, {verdict["C2"][2]}',
     f'- E accum4/fresh 10ep: ep1 {verdict["E"][0]:.4f}, best {verdict["E"][1]:.4f}, {verdict["E"][2]}',
     '- reading: C2 holds 20ep → durability proven, training continues (R13 unblocked); '
     'C2 drifts after epN → report N, hold is transient; '
     'E between A-ref and C2 → dose-response confirms noise mechanism.']
open('/kaggle/working/run_notes.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
