"""Plot the recorded contributions to the MU 24-week Dice difference."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/opencode/world-model-report-matplotlib')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
name = 'patient_forecast_interpretation_metrics.json'
data = (ROOT / 'metrics' / name).read_bytes()
provenance = json.loads((ROOT / 'metrics' / 'provenance.json').read_text())
if hashlib.sha256(data).hexdigest() != provenance[name]['sha256']:
    raise ValueError(f'Aggregate hash mismatch: {name}')

result = json.loads(data)['results']['MU:24:current']
compartments = result['compartments']
strata = compartments['2']['strata']
empty = strata['future_empty']['contribution_to_overall_gain']
nonempty = strata['future_nonempty']['contribution_to_overall_gain']
other = sum(compartments[k]['contribution_to_overall_gain'] for k in ('1', '3'))
overall = result['overall_gain']
if abs(empty + nonempty + other - overall) > 1e-10:
    raise ValueError('Contributions do not sum to the overall difference')

fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    'font.family': 'IBM Plex Sans' if 'IBM Plex Sans' in fonts else 'DejaVu Sans',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
})
labels = ['Future-empty core', 'Future-nonempty core', 'Other compartments', 'Overall difference']
values = [100 * x for x in (empty, nonempty, other, overall)]
fig, ax = plt.subplots(figsize=(5.7, 2.8), layout='constrained')
ax.axvline(0, color='#767676', linewidth=0.8, linestyle='--')
for i, (value, marker, color) in enumerate(zip(
    values, ['o', 'o', 'o', 's'], ['#0F4D92', '#B64342', '#767676', '#0F4D92']
)):
    ax.scatter(value, i, marker=marker, color=color, s=28, zorder=3)
    ax.annotate(f'{value:+.2f}', (value, i), xytext=(6, 0),
                textcoords='offset points', va='center', fontsize=8)
ax.set_yticks(range(len(labels)), labels)
ax.set_ylim(3.6, -0.6)
ax.set_xlim(-2, 12.5)
ax.set_xlabel('Dice difference, percentage points (current-only minus persistence)')
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
for extension in ('pdf', 'png'):
    fig.savefig(ROOT / 'figures' / f'updated_forecast_attribution.{extension}', dpi=300)
plt.close(fig)
