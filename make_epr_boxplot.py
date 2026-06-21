#!/usr/bin/env python3
"""
Publication-standard box-and-whisker plot of shoreline change rate (EPR) by island,
grouped by settlement regime (inhabited vs uninhabited), 1990-2024.

Reads per-transect EPR from data/dsas_output/transects_<island>.shp for the four
study islands and renders a journal-quality figure to:
    final_maps/png_300dpi/09_epr_boxplot.{png,pdf}

Scaling note: the y-axis is driven by the box/whisker extent (not by the single most
extreme transect), so Mousuni's long erosional tail no longer squashes the four boxes.
Transects that fall beyond the plotted range are counted under each box, and a SCALE
toggle ('linear' | 'symlog') is provided in case the near-stable islands need more room.

    conda activate sundarbans
    python ~/sundarbans_paper/scripts/make_epr_boxplot.py
"""
from pathlib import Path
import numpy as np, geopandas as gpd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCALE = 'linear'   # 'linear' (default) or 'symlog' (compress the deep tail, enlarge near-zero boxes)

# ---- APA: use a sans-serif face *within* the figure (Arial/Helvetica), sans-serif mathtext ----
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Arial','Helvetica','Liberation Sans','DejaVu Sans']
plt.rcParams['mathtext.fontset']='dejavusans'
plt.rcParams['axes.unicode_minus']=True

# project root — auto-detect (WSL dir moved to H:\wsl_archive\sundarbans_paper;
# the old home location is kept as a fallback so the script runs either way)
def _resolve_root():
    for p in (Path('/mnt/h/wsl_archive/sundarbans_paper'),
              Path.home()/'sundarbans_paper',
              Path('/mnt/d/sundarbans_paper')):
        if p.exists(): return p
    return Path('/mnt/h/wsl_archive/sundarbans_paper')
PROJ=_resolve_root()
DSAS=PROJ/'data'/'dsas_output'
OUTP=PROJ/'final_maps'/'png_300dpi'; OUTP.mkdir(parents=True, exist_ok=True)
print(f'project root: {PROJ}')

# regime-ordered: two inhabited, then two uninhabited
ISLANDS=['Kumirmari','Mousuni','Hamilton','Lothian']
AREA={'Kumirmari':19.09,'Mousuni':26.19,'Hamilton':84.94,'Lothian':36.30}
REGIME={'Kumirmari':'Inhabited','Mousuni':'Inhabited','Hamilton':'Uninhabited','Lothian':'Uninhabited'}
POS={'Kumirmari':1.0,'Mousuni':2.0,'Hamilton':3.6,'Lothian':4.6}
FILL={'Inhabited':'#bb5a41','Uninhabited':'#3f6f99'}        # muted terracotta / steel blue

def load_epr(island):
    p=DSAS/f'transects_{island.lower()}.shp'
    if not p.exists(): print(f'  [skip] {p.name} not found'); return None
    g=gpd.read_file(p)
    col=next((c for c in ['EPR','epr','EPR_obs','EPR_m_yr'] if c in g.columns), None)
    if col is None: print(f'  [skip] no EPR column in {p.name}'); return None
    v=g[col].astype(float).values
    return v[np.isfinite(v)]

data={i:load_epr(i) for i in ISLANDS}
data={i:v for i,v in data.items() if v is not None and len(v)}
if not data:
    raise SystemExit('No transect EPR data found \u2014 run on the machine with data/dsas_output/.')

order=[i for i in ISLANDS if i in data]
vals=[data[i] for i in order]
positions=[POS[i] for i in order]

# ---- robust statistics for scaling (box + 1.5xIQR whiskers, NOT the extreme fliers) ----
q1=np.array([np.percentile(v,25) for v in vals])
q3=np.array([np.percentile(v,75) for v in vals])
iqr=q3-q1
wlo=np.array([v[v>=q1[k]-1.5*iqr[k]].min() for k,v in enumerate(vals)])
whi=np.array([v[v<=q3[k]+1.5*iqr[k]].max() for k,v in enumerate(vals)])
box_lo=float(q1.min()); box_hi=float(q3.max())          # lowest / highest box edge
core=max(box_hi-box_lo, 1.0)                            # full box-block span
# Focus the axis on the band where the four BOXES live so the near-stable islands stay
# legible. The window ALWAYS contains every box; a long whisker/tail of the dominant
# island (Mousuni) is clipped — in EITHER direction — to ~a third of a core beyond the
# box block, and flagged with an arrow + the extreme value.
m_lo=0.32*core; m_hi=0.28*core
y_data_lo=max(float(wlo.min()), box_lo-m_lo)
y_data_hi=min(float(whi.max()), box_hi+m_hi)
y_data_lo=min(y_data_lo,-0.02*core); y_data_hi=max(y_data_hi,0.02*core)   # keep the 0 line in view
# islands whose data runs past the clipped window, with direction + extreme value
clip_lo={isl:(float(v.min()),int((v<y_data_lo).sum())) for isl,v in zip(order,vals) if v.min()<y_data_lo-1e-9}
clip_hi={isl:(float(v.max()),int((v>y_data_hi).sum())) for isl,v in zip(order,vals) if v.max()>y_data_hi+1e-9}

fig,ax=plt.subplots(figsize=(8.4,6.4))
ax.axhline(0,color='#555555',lw=1.0,ls=(0,(5,4)),zorder=1)            # stable reference
ax.text(POS['Lothian']+0.55,0,'stable  (0 m yr$^{-1}$)',va='center',ha='left',
        fontsize=8.5,color='#555555',style='italic',
        bbox=dict(fc='white',ec='none',alpha=0.9,pad=0.6))

bp=ax.boxplot(vals,positions=positions,widths=0.62,patch_artist=True,
              whis=1.5,showmeans=True,manage_ticks=False,
              medianprops=dict(color='black',lw=1.6),
              whiskerprops=dict(color='black',lw=1.1),
              capprops=dict(color='black',lw=1.1),
              flierprops=dict(marker='o',markersize=3,markerfacecolor='none',
                              markeredgecolor='#7a7a7a',alpha=0.6,markeredgewidth=0.6),
              meanprops=dict(marker='D',markerfacecolor='white',markeredgecolor='black',
                             markersize=6,markeredgewidth=1.1))
for patch,isl in zip(bp['boxes'],order):
    patch.set_facecolor(FILL[REGIME[isl]]); patch.set_alpha(0.55)
    patch.set_edgecolor('black'); patch.set_linewidth(1.2)

if SCALE=='symlog':
    lt=max(round(float(np.median(iqr)),1),2.0)
    ax.set_yscale('symlog', linthresh=lt, linscale=1.0)

# ---- axis limits: headroom for brackets (+ any up-arrow), footroom for the down-arrow ----
top_pad=(0.42 if clip_hi else 0.26)*core; bot_pad=0.18*core
y_hi=y_data_hi+top_pad; y_lo=y_data_lo-bot_pad
ax.set_xlim(0.3, POS['Lothian']+1.4); ax.set_ylim(y_lo, y_hi)

# keep all box-plot artists inside the focus window; whatever lies beyond it is conveyed
# by the arrows + footnote only (so the picture and the "beyond range" count agree)
from matplotlib.patches import Rectangle as _Rect
_clip=_Rect((0.3,y_data_lo),(POS['Lothian']+1.4)-0.3,y_data_hi-y_data_lo,transform=ax.transData)
for _k in ('whiskers','caps','fliers','boxes','medians','means'):
    for _a in bp.get(_k,[]): _a.set_clip_path(_clip)

# ---- clipped tails: dark-red down-arrow for an erosional tail, blue up-arrow for accretion ----
for isl,(vmin,n_) in clip_lo.items():
    x=POS[isl]
    ax.annotate('', xy=(x,y_lo+0.02*core), xytext=(x,y_data_lo+0.02*core),
                arrowprops=dict(arrowstyle='-|>',color='#7a1f12',lw=1.6,mutation_scale=13,shrinkA=0,shrinkB=0))
    ax.text(x+0.12,y_lo+0.06*core,f'to {vmin:+.1f}',ha='left',va='bottom',
            fontsize=8.0,color='#7a1f12',style='italic')
for isl,(vmax,n_) in clip_hi.items():
    x=POS[isl]
    ax.annotate('', xy=(x,y_data_hi+0.085*core), xytext=(x,y_data_hi+0.01*core),
                arrowprops=dict(arrowstyle='-|>',color='#1f4e79',lw=1.6,mutation_scale=13,shrinkA=0,shrinkB=0))
    ax.text(x+0.12,y_data_hi+0.03*core,f'to {vmax:+.1f}',ha='left',va='bottom',
            fontsize=8.0,color='#1f4e79',style='italic')

# ---- regime brackets above any up-arrow ----
def bracket(x0,x1,y,label,color):
    ax.plot([x0,x0,x1,x1],[y,y+0.045*core,y+0.045*core,y],color=color,lw=1.3,clip_on=False)
    ax.text((x0+x1)/2,y+0.075*core,label,ha='center',va='bottom',fontsize=10.2,
            fontweight='bold',color=color)
ybr=y_data_hi+(0.18*core if clip_hi else 0.10*core)
bracket(POS['Kumirmari'],POS['Mousuni'],ybr,'Inhabited (settled, embanked)',FILL['Inhabited'])
bracket(POS['Hamilton'],POS['Lothian'],ybr,'Uninhabited (protected)',FILL['Uninhabited'])

ax.set_xticks([POS[i] for i in order])
ax.set_xticklabels([rf'{i}'+'\n'+rf'({AREA[i]:.2f} km$^2$)'+'\n'+rf'$\bar{{x}}$={data[i].mean():+.2f},  n={len(data[i])}' for i in order], fontsize=9.8)
ax.set_ylabel('End-Point Rate, EPR (m yr$^{-1}$)', fontsize=12)
ax.tick_params(axis='y', labelsize=10)
ax.grid(axis='y', color='#cfcfcf', lw=0.5, ls=':', alpha=0.8, zorder=0)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.spines['left'].set_linewidth(1.0); ax.spines['bottom'].set_linewidth(1.0)

# APA: no title inside the image. The figure number (bold) and the title (italic, Title
# Case) belong ABOVE the figure as document text; this script prints both at the end so
# you can paste them in. The image itself stays clean (plot + legend + arrows only).

# legend / methods footnote
leg=[Line2D([0],[0],marker='s',color='none',markerfacecolor=FILL['Inhabited'],alpha=0.55,
            markeredgecolor='black',markersize=11,label='Inhabited'),
     Line2D([0],[0],marker='s',color='none',markerfacecolor=FILL['Uninhabited'],alpha=0.55,
            markeredgecolor='black',markersize=11,label='Uninhabited'),
     Line2D([0],[0],marker='D',color='none',markerfacecolor='white',markeredgecolor='black',
            markersize=7,label='mean'),
     Line2D([0],[0],color='black',lw=1.6,label='median')]
ax.legend(handles=leg,loc='lower left',frameon=True,framealpha=0.95,edgecolor='#999999',
          fontsize=9,ncol=2,handlelength=1.4,columnspacing=1.1,borderpad=0.6)
beyond=[]
for isl in order:
    v=data[isl]; k=int((v<y_data_lo).sum())+int((v>y_data_hi).sum())
    if k: beyond.append(f'{isl} ({k})')
beyond_txt=(' Transects beyond the plotted range: '+', '.join(beyond)+'.') if beyond else ''
# APA: the "Box = IQR ..." explanation is NOT burned into the image; it becomes the
# document Note. (printed below). The image stays clean (plot + legend + arrows only).

fig.tight_layout()
png=OUTP/'09_epr_boxplot.png'; pdf=OUTP/'09_epr_boxplot.pdf'
fig.savefig(png,dpi=300,bbox_inches='tight'); fig.savefig(pdf,bbox_inches='tight'); plt.close(fig)
print('Saved:',png); print('Saved:',pdf)
print(f'  y-axis: [{y_lo:.1f}, {y_hi:.1f}]  (data whiskers [{y_data_lo:.1f}, {y_data_hi:.1f}], scale={SCALE})')
for i in order:
    v=data[i]; print(f'  {i:10s} n={len(v):5d}  mean={v.mean():+6.2f}  median={np.median(v):+6.2f}  %eroding={100*(v<0).mean():4.1f}')

# ---- APA caption + note text (the document text that goes around the image) ----
title_txt='Distribution of Per-Transect End-Point Rate (EPR) by Island, 1990\u20132024'
note_txt=('EPR = end-point rate (m yr\u207b\u00b9); negative values denote erosion, positive values '
          'accretion. Boxes show the inter-quartile range and median (line); white diamonds show the '
          'mean; whiskers extend to 1.5 \u00d7 IQR; open circles are outliers within range. Arrows mark '
          'distributions whose tails extend beyond the plotted range, labelled with the extreme value.'
          +beyond_txt)
print('\n================ APA figure text ================')
print('ABOVE the image (two lines):')
print('  Figure 8                       [bold]')
print(f'  {title_txt}   [italic, Title Case]')
print('BELOW the image:')
print(f'  Note. {note_txt}')
print('=================================================')
