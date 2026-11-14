"""Regenerate the report's four new figures from bundled aggregate JSON."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/world-model-report-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
BLUE, RED, GRAY = '#0F4D92', '#B64342', '#767676'
fonts = {f.name for f in font_manager.fontManager.ttflist}
font = 'IBM Plex Sans' if 'IBM Plex Sans' in fonts else 'DejaVu Sans'
plt.rcParams.update({'font.family': font, 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})

def read(name):
    return json.loads((ROOT/'metrics'/f'{name}.json').read_text())

def save(fig, name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf')
    fig.savefig(ROOT/'figures'/f'{name}.png', dpi=300)
    plt.close(fig)

def interval(ax, rows, xlabel, title, colors=None):
    for i, (label, mean, ci) in enumerate(rows):
        c = (colors or [BLUE]*len(rows))[i]
        ax.errorbar(mean, i, xerr=[[mean-ci[0]], [ci[1]-mean]],
                    fmt='o', color=c, capsize=3, markersize=4, linewidth=1.3)
    ax.axvline(0, color=GRAY, linewidth=.8, linestyle='--')
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows)-.5, -.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc='left', fontweight='bold', pad=12)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

for name, meta in json.loads((ROOT/'metrics/provenance.json').read_text()).items():
    assert hashlib.sha256((ROOT/'metrics'/name).read_bytes()).hexdigest() == meta['sha256']

r65, r69 = read('r65_lumiere_metrics'), read('r69_spatial_access_metrics')
fig, axes = plt.subplots(2, 1, figsize=(5.7, 4.8), layout='constrained')
rows=[]
for label, key in [('Add T1','B-A_tumor_mae'), ('Add T2','C-B_tumor_mae')]:
    d=r65['primary_contrasts'][key]; rows.append((label,d['mean'],d['ci98_75']))
interval(axes[0], rows, 'Lesion MAE difference (standardized intensity)',
         'A  Sequence additions (98.75% intervals)')
rows=[]
for label,key in [('Lesion','lesion_mae'),('Nonlesion','nonlesion_mae')]:
    d=r69['enabled_minus_zero'][key]['seed_average']; rows.append((label,d['mean'],d['interval95']))
interval(axes[1], rows, 'MAE difference (spatial access minus zeroed)',
         'B  Higher-resolution features (95% intervals)')
save(fig,'updated_completion')

c=read('patient_cavity_observation_metrics')['datasets']
fig,axes=plt.subplots(2,1,figsize=(5.7,5.1),layout='constrained')
rows=[]
for label,cohort,key in [('MU: original three','MU','native_available_label_dice'),
 ('LUMIERE: conditional','LUMIERE','scored_compartment_dice'),
 ('LUMIERE: native','LUMIERE','native_available_label_dice')]:
    d=c[cohort][key];rows.append((label,d['new_minus_baseline'],d['paired_patient_ci95']))
interval(axes[0],rows,'Dice difference (cavity model minus baseline)',
         'A  Anatomy Dice by cohort and readout', [BLUE,BLUE,RED])
rows=[]
for label,key in [('Core volume error','class_2_volume_error_ml'),
                  ('Core inside cavity','core_in_reference_cavity_ml')]:
    d=c['MU'][key];rows.append((label,d['new_minus_baseline'],d['paired_patient_ci95']))
interval(axes[1],rows,'Volume difference (mL; cavity model minus baseline)',
         'B  MU: core-volume and cavity errors')
save(fig,'updated_cavity')

h=read('patient_structured_forecasting_metrics')['horizons']
fig,axes=plt.subplots(2,1,figsize=(5.7,5.4),layout='constrained')
keys=list(h)
for off,method,label,marker,color in [(-.18,'persistence','Persistence','s',GRAY),
 (0,'current','Current-only','o',BLUE),(.18,'history','History','^',RED)]:
    axes[0].scatter([h[k]['methods'][method]['dice'] for k in keys],
                    np.arange(4)+off,label=label,marker=marker,color=color,s=24)
labels=[f'{k.replace(":", " / ")} weeks' for k in keys]
axes[0].set_yticks(range(4),labels);axes[0].set_ylim(3.6,-.6)
axes[0].set_xlim(0,.5);axes[0].set_xlabel('Three-compartment Dice')
axes[0].set_title('A  Point scores by cohort and requested horizon',loc='left',fontweight='bold',pad=12)
axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=3,fontsize=8,frameon=False)
rows=[]
for label,k in zip(labels,keys):
    d=h[k]['history_minus_control']['current']['dice']
    rows.append((label,d['difference'],d['patient_bootstrap95']))
interval(axes[1],rows,'Dice difference (history minus current-only)',
         'B  Added history: paired 95% intervals')
save(fig,'updated_forecast')

a=read('patient_acquisition_jepa_metrics')
fig,ax=plt.subplots(figsize=(5.7,2.4),layout='constrained')
rows=[]
for label,key in [('T1','1'),('T1 + CT1','01'),('T1 + T2','12'),('T1 + CT1 + T2','012')]:
    d=a['fixed_panel_retention'][key]['lesion_deficit'];rows.append((label,d['difference'],d['ci95']))
interval(ax,rows,'Lesion FLAIR MAE difference (new minus reference)',
         'Completion from the observed sequences')
save(fig,'updated_acquisition')
print('Generated four PDF/PNG figures from verified aggregate snapshots.')
