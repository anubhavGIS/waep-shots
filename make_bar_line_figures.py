#!/usr/bin/env python3
"""
make_bar_line_figures.py — two summary figures for the Sundarbans chapter, in the paper's
serif style, built from the corrected post-tidal-fix numbers (edit the DATA block to match
your latest summary.json if needed):
  14_change_rate_barchart.png  — grouped bar: mean EPR vs LRR by island
  15_kalman_projection_lines.png — line: Kalman net change 2024 -> 2034 -> 2044 by island
Saves PNG (300 dpi) + PDF into final_maps/png_300dpi and final_maps/pdf.

    conda activate sundarbans
    python ~/sundarbans_paper/scripts/make_bar_line_figures.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

PROJECT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1]/'final_maps').exists() else Path.home()/'sundarbans_paper'
PNG_DIR = PROJECT/'final_maps'/'png_300dpi'; PDF_DIR = PROJECT/'final_maps'/'pdf'
PNG_DIR.mkdir(parents=True, exist_ok=True); PDF_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif','Nimbus Roman'],
    'mathtext.fontset':'dejavuserif','axes.edgecolor':'#444444','axes.linewidth':0.9,
    'xtick.color':'#222','ytick.color':'#222','text.color':'#111',
})
# ---- DATA (corrected, post tidal-fix) ----
ISL=['Kumirmari','Hamilton','Mousuni','Lothian']; REG=['inhabited','uninhabited','inhabited','uninhabited']
EPR=[-0.62,-0.89,-4.54,0.54]; LRR=[-0.85,-0.95,-4.69,1.44]
KAL={'Kumirmari':[0,-8,-16],'Hamilton':[0,-9,-19],'Mousuni':[0,-44,-93],'Lothian':[0,5,12]}; YEARS=[2024,2034,2044]
C_INH='#B2182B'; C_UNI='#2166AC'; col=[C_INH if r=='inhabited' else C_UNI for r in REG]; MARK={'inhabited':'o','uninhabited':'^'}

def save(fig,name):
    fig.tight_layout()
    fig.savefig(PNG_DIR/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(PDF_DIR/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    print('saved',name)

# FIG 1
fig,ax=plt.subplots(figsize=(8.6,5.1),dpi=300); ax.set_axisbelow(True); ax.yaxis.grid(True,color='#e7e7e7',lw=0.8)
x=np.arange(4); w=0.38
b1=ax.bar(x-w/2,EPR,w,color=col,edgecolor='#1a1a1a',lw=0.7,zorder=3)
b2=ax.bar(x+w/2,LRR,w,color=col,edgecolor='#1a1a1a',lw=0.7,alpha=0.45,hatch='////',zorder=3)
ax.axhline(0,color='#333',lw=1.1,zorder=4)
for bars,vals in [(b1,EPR),(b2,LRR)]:
    for rect,v in zip(bars,vals):
        ax.annotate(f'{v:+.2f}',(rect.get_x()+rect.get_width()/2,v),ha='center',va='bottom' if v>=0 else 'top',
                    fontsize=8.2,xytext=(0,3 if v>=0 else -3),textcoords='offset points')
ax.set_xticks(x); ax.set_xticklabels(ISL,fontsize=10.5); ax.set_ylim(-5.4,1.9); ax.set_xlim(-0.55,4.25)
ax.set_ylabel('Shoreline-change rate (m yr$^{-1}$)',fontsize=11)
ax.set_title('Mean shoreline-change rates by island, 1990\u20132024',fontsize=12.5,fontweight='bold',pad=12)
ax.axvline(1.5,color='#c9c9c9',ls=(0,(2,2)),lw=1.0,zorder=1)
ax.text(0.5,1.55,'Interior pair (sheltered)',ha='center',fontsize=9.2,style='italic',color='#555')
ax.text(2.5,1.55,'Coastal pair (exposed)',ha='center',fontsize=9.2,style='italic',color='#555')
ax.annotate('',(3.62,1.0),(3.62,-1.0),arrowprops=dict(arrowstyle='<->',color='#9a9a9a',lw=0.9))
ax.text(3.74,0.55,'accretion',fontsize=8,color=C_UNI,va='center'); ax.text(3.74,-0.55,'erosion',fontsize=8,color=C_INH,va='center')
for s in ('top','right'): ax.spines[s].set_visible(False)
l1=ax.legend(handles=[Patch(fc=C_INH,ec='#1a1a1a',label='Inhabited'),Patch(fc=C_UNI,ec='#1a1a1a',label='Uninhabited')],
             loc='lower left',fontsize=8.6,frameon=False,title='Settlement',title_fontsize=8.8,bbox_to_anchor=(0.005,0.02)); ax.add_artist(l1)
ax.legend(handles=[Patch(fc='#888',ec='#1a1a1a',label='EPR (end-point rate)'),
                   Patch(fc='#888',ec='#1a1a1a',alpha=0.45,hatch='////',label='LRR (linear regression rate)')],
          loc='lower left',fontsize=8.6,frameon=False,bbox_to_anchor=(0.175,0.02))
save(fig,'14_change_rate_barchart')

# FIG 2
fig,ax=plt.subplots(figsize=(8.2,5.3),dpi=300); ax.set_axisbelow(True); ax.yaxis.grid(True,color='#e7e7e7',lw=0.8)
for isl,reg in zip(ISL,REG):
    c=C_INH if reg=='inhabited' else C_UNI
    ax.plot(YEARS,KAL[isl],marker=MARK[reg],color=c,lw=2.1,markersize=7.5,markeredgecolor='#111',markeredgewidth=0.7,zorder=3)
    dy=2 if isl=='Kumirmari' else (-2 if isl=='Hamilton' else 0)
    ax.annotate(f'  {isl}  ({KAL[isl][-1]:+d} m)',(2044,KAL[isl][-1]+dy),fontsize=8.8,va='center',color=c)
ax.axhline(0,color='#333',lw=1.1,zorder=2)
ax.set_xticks(YEARS); ax.set_xticklabels(YEARS,fontsize=10.5); ax.set_xlim(2022,2050.5)
ax.set_ylabel('Projected net shoreline change from 2024 (m)',fontsize=11)
ax.set_title('Kalman-filter shoreline projections to 2034 and 2044',fontsize=12.5,fontweight='bold',pad=12)
ax.text(2032.5,-30,'negative = landward retreat (erosion)',fontsize=8.4,style='italic',color='#777')
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.legend(handles=[Line2D([0],[0],color=C_INH,marker='o',lw=2.1,mec='#111',label='Inhabited'),
                   Line2D([0],[0],color=C_UNI,marker='^',lw=2.1,mec='#111',label='Uninhabited')],
          loc='lower left',fontsize=9,frameon=False,title='Settlement',title_fontsize=9)
save(fig,'15_kalman_projection_lines')
print('done')
