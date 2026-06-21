"""Figure 2: DSAS Linear Regression Rate (LRR) heatmap of four Sundarbans islands,
restyled to match Figure 1 (neatline, no-box heading, Esri Dark Gray basemap,
degree-minute coordinates, white ArcGIS 'Scale Line 1' bottom-right, fixed-size
north arrows, shared horizontal colorbar, single-line sources credit)."""
import warnings, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec, colormaps
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from xyzservices import TileProvider

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family':'serif',
                     'font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
                     'font.size':9,'axes.linewidth':1.0,'axes.edgecolor':'black'})

EsriDarkGray = TileProvider(
    name='Esri.WorldDarkGrayBase',
    url='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    attribution='Esri')

HOME      = Path.home()
PROJECT   = HOME / 'sundarbans_paper'
ANCILLARY = PROJECT / 'data' / 'ancillary'
DSAS_OUT  = PROJECT / 'data' / 'dsas_output'
OUT_PNG   = PROJECT / 'final_maps' / 'png_300dpi' / '02_dsas_lrr_heatmap.png'
OUT_PDF   = PROJECT / 'final_maps' / 'pdf' / '02_dsas_lrr_heatmap.pdf'
for d in (OUT_PNG.parent, OUT_PDF.parent): d.mkdir(parents=True, exist_ok=True)

ISLANDS  = ['Kumirmari','Hamilton','Mousuni','Lothian']
SUBTITLE = {'Kumirmari':'INTERIOR \u00b7 INHABITED','Hamilton':'INTERIOR \u00b7 UNINHABITED',
            'Mousuni':'COASTAL \u00b7 INHABITED','Lothian':'COASTAL \u00b7 UNINHABITED'}
UTM = 'EPSG:32645'; WGS = 'EPSG:4326'

# ---------- data ----------
islands_utm = gpd.read_file(ANCILLARY / 'four_islands.shp').to_crs(UTM)
name_col = next((c for c in ['island','Island','NAME','name'] if c in islands_utm.columns), None)
with open(DSAS_OUT / 'summary.json') as f:
    summary = json.load(f)

# ---------- helpers (identical style to Figure 1 v18) ----------
def dd_to_dm(dd):
    s='-' if dd<0 else ''; dd=abs(dd); d=int(dd); m=int(round((dd-d)*60))
    if m==60: d+=1; m=0
    return f"{s}{d}\u00b0{m:02d}'"
def x_fmt(x,_): return "0\u00b0" if x==0 else f"{dd_to_dm(abs(x))}{'E' if x>=0 else 'W'}"
def y_fmt(y,_): return "0\u00b0" if y==0 else f"{dd_to_dm(abs(y))}{'N' if y>=0 else 'S'}"

def fit_box(ax, fig, zoom=None):
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim(); dx=x1-x0; dy=y1-y0
    cos_lat=np.cos(np.radians((y0+y1)/2))
    bb=ax.get_position(); aw=bb.width*fig.get_figwidth(); ah=bb.height*fig.get_figheight()
    box_r=ah/aw; dpr=dy/(dx*cos_lat)
    if dpr<box_r:
        nd=dx*cos_lat*box_r; cy=(y0+y1)/2; ax.set_ylim(cy-nd/2,cy+nd/2)
    elif dpr>box_r:
        nd=dy/(cos_lat*box_r); cx=(x0+x1)/2; ax.set_xlim(cx-nd/2,cx+nd/2)
    try:
        kw=dict(crs=4326, source=EsriDarkGray, attribution=False)
        if zoom is not None: kw['zoom']=zoom
        ctx.add_basemap(ax, **kw)
    except Exception:
        try:
            ctx.add_basemap(ax, crs=4326, source=ctx.providers.CartoDB.DarkMatter, attribution=False)
        except Exception:
            ax.set_facecolor('#3A3A3A')

def fmt_panel(ax, xbins=3, ybins=3):
    ax.grid(True, color='white', linestyle='--', linewidth=0.4, alpha=0.40)
    ax.xaxis.set_major_formatter(FuncFormatter(x_fmt)); ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, prune='both'))
    ax.tick_params(axis='x', labelsize=6, direction='out', length=2.5, pad=2)
    ax.tick_params(axis='y', labelsize=6, direction='out', length=2.5, pad=2, labelleft=True, labelright=False)
    plt.setp(ax.get_yticklabels(), rotation=90, ha='center', va='center')
    ax2=ax.secondary_yaxis('right'); ax2.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=ybins, prune='both'))
    ax2.tick_params(axis='y', labelsize=6, direction='out', length=2.5, pad=2)
    plt.setp(ax2.get_yticklabels(), rotation=-90, ha='center', va='center')
    for sp in ax.spines.values(): sp.set_linewidth(1.2); sp.set_color('black')

def north_arrow_box(ax, fig, box_w_in, box_h_in, topleft_axes=(0.86, 0.975)):
    bb=ax.get_position(); aw_in=bb.width*fig.get_figwidth(); ah_in=bb.height*fig.get_figheight()
    bw=box_w_in/aw_in; bh=box_h_in/ah_in
    lx,ty=topleft_axes; x0=lx; y0=ty-bh; cx=x0+bw/2
    ax.add_patch(Rectangle((x0,y0),bw,bh,transform=ax.transAxes,facecolor='white',
                           edgecolor='black',linewidth=0.7,alpha=0.96,zorder=15))
    ax.text(cx,y0+bh*0.90,'N',transform=ax.transAxes,ha='center',va='top',
            fontsize=9.5,fontweight='bold',family='serif',zorder=17)
    tri_top=y0+bh*0.55; tri_bot=y0+bh*0.14; tri_hw=bw*0.28
    tri=np.array([[cx,tri_top],[cx-tri_hw,tri_bot],[cx+tri_hw,tri_bot]])
    ax.add_patch(Polygon(tri,transform=ax.transAxes,facecolor='black',edgecolor='black',
                         linewidth=0.7,zorder=16))

def scale_line_1(ax, total_km, n_div=4, label_every=1):
    halo=[pe.withStroke(linewidth=1.5, foreground='#2A2A2A')]
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    kpd=111.32*np.cos(np.radians((y0+y1)/2))
    total_deg=total_km/kpd; div_deg=total_deg/n_div; div_km=total_km/n_div
    km_w=(x1-x0)*0.05; rmarg=(x1-x0)*0.04
    px=x1-rmarg-km_w-total_deg; py=y0+(y1-y0)*0.045; tick_up=(y1-y0)*0.015
    ax.plot([px,px+total_deg],[py,py],color='white',linewidth=1.1,zorder=11,
            solid_capstyle='butt',path_effects=halo)
    for i in range(n_div+1):
        xt=px+i*div_deg
        ax.plot([xt,xt],[py,py+tick_up],color='white',linewidth=1.0,zorder=11,
                solid_capstyle='butt',path_effects=halo)
        if i % label_every == 0:
            ax.text(xt,py-(y1-y0)*0.008,f'{i*div_km:g}',ha='center',va='top',fontsize=4.8,
                    color='white',family='serif',fontweight='bold',zorder=12,path_effects=halo)
    ax.text(px+total_deg+(x1-x0)*0.012,py,'km',ha='left',va='center',fontsize=5.2,
            color='white',family='serif',fontweight='bold',zorder=12,path_effects=halo)

def nice_scale_km(span_km):
    target=span_km*0.22
    if target<=0: return 1,4
    base=10**np.floor(np.log10(target)); r=target/base
    return (base if r<1.5 else (2*base if r<3.5 else 5*base)), 4

def panel_title(fig, ax, text, fontsize=9.5):
    bb=ax.get_position()
    fig.text((bb.x0+bb.x1)/2, bb.y1+0.004, text, ha='center', va='bottom',
             fontsize=fontsize, fontweight='bold', family='serif',
             bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.0))

# ============ FIGURE ============
fig=plt.figure(figsize=(12.5,14.2),facecolor='white')
fig.patches.append(Rectangle((0.008,0.008),0.984,0.984,transform=fig.transFigure,
                             fill=False,edgecolor='black',linewidth=2.5,zorder=100))
fig.text(0.5,0.965,
         'LINEAR REGRESSION RATE (LRR) OF SHORELINE CHANGE, 1990\u20132024\n'
         'FOUR ESTUARINE ISLANDS, INDIAN SUNDARBANS, WEST BENGAL, INDIA',
         ha='center',va='center',fontsize=12,fontweight='bold',family='serif',
         linespacing=1.35,zorder=101)

gs=gridspec.GridSpec(3,2,figure=fig,height_ratios=[1,1,0.045],hspace=0.20,wspace=0.20,
                     top=0.930,bottom=0.060,left=0.065,right=0.945)
axes={isl:fig.add_subplot(gs[i//2,i%2]) for i,isl in enumerate(ISLANDS)}
cax=fig.add_subplot(gs[2,:])

vmin,vmax=-10,10
norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)
cmap=colormaps.get_cmap('RdBu')

for isl in ISLANDS:
    ax=axes[isl]
    poly_utm = islands_utm[islands_utm[name_col]==isl].geometry.iloc[0]
    poly_ll  = gpd.GeoSeries([poly_utm], crs=UTM).to_crs(WGS).iloc[0]
    tr_utm   = gpd.read_file(DSAS_OUT / f'transects_{isl.lower()}.shp').to_crs(UTM)

    # clip each transect to a coastal band (shoreline + 440 m) in METRES, then reproject
    band = poly_utm.buffer(440.0)
    segs=[]; lrrs=[]
    for _,row in tr_utm.iterrows():
        s=row.geometry.intersection(band)
        if s.is_empty: continue
        if s.geom_type=='MultiLineString':
            o=Point(row.geometry.coords[0]); s=min(s.geoms, key=lambda g:g.distance(o))
        if s.geom_type!='LineString': continue
        segs.append(s); lrrs.append(float(row['LRR']))
    seg_ll = gpd.GeoSeries(segs, crs=UTM).to_crs(WGS)

    b=poly_ll.bounds; padx=(b[2]-b[0])*0.10; pady=(b[3]-b[1])*0.10
    ax.set_xlim(b[0]-padx,b[2]+padx); ax.set_ylim(b[1]-pady,b[3]+pady)
    fit_box(ax, fig)   # auto zoom for each island's extent

    geoms=[poly_ll] if poly_ll.geom_type=='Polygon' else list(poly_ll.geoms)
    for g in geoms:
        xs,ys=g.exterior.xy
        ax.fill(xs,ys,facecolor='#d9d9d9',alpha=0.42,edgecolor='white',linewidth=0.7,zorder=2)
    lrr_c=np.clip(np.array(lrrs),vmin,vmax)
    for s,c in zip(seg_ll.values, lrr_c):
        x,y=s.xy
        ax.plot(x,y,color=cmap(norm(c)),linewidth=0.9,alpha=0.95,zorder=3)

    fmt_panel(ax)
    s=summary[isl]
    txt=(f"EPR  {s['EPR_mean']:+.2f} \u00b1 {s['EPR_std']:.2f} m/yr\n"
         f"LRR  {s['LRR_mean']:+.2f} \u00b1 {s['LRR_std']:.2f} m/yr\n"
         f"E {s['pct_erosion']:.0f}% / S {s['pct_stable']:.0f}% / A {s['pct_accretion']:.0f}%\n"
         f"n = {s['n_transects']} transects")
    ax.text(0.035,0.965,txt,transform=ax.transAxes,va='top',ha='left',fontsize=6.6,
            family='monospace',zorder=18,
            bbox=dict(boxstyle='round,pad=0.4',facecolor='white',alpha=0.92,edgecolor='#888',linewidth=0.8))

fig.canvas.draw()
_kb=axes['Kumirmari'].get_position()
REF_W_IN=0.102*_kb.width*fig.get_figwidth()
REF_H_IN=0.15725*_kb.height*fig.get_figheight()
for isl in ISLANDS:
    north_arrow_box(axes[isl], fig, REF_W_IN, REF_H_IN, topleft_axes=(0.86,0.975))
    ax=axes[isl]; x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    span_km=(x1-x0)*111.32*np.cos(np.radians((y0+y1)/2))
    km,ndiv=nice_scale_km(span_km)
    scale_line_1(ax, total_km=km, n_div=ndiv, label_every=1)

sm=ScalarMappable(cmap=cmap,norm=norm); sm.set_array([])
cbar=fig.colorbar(sm,cax=cax,orientation='horizontal')
cbar.set_ticks([-10,-5,0,5,10]); cbar.ax.tick_params(labelsize=8)
for t in cbar.ax.get_xticklabels(): t.set_family('serif')
cbar.set_label('Linear Regression Rate, LRR (m yr$^{-1}$)        \u2190 Erosion          Accretion \u2192',
               fontsize=9.5,family='serif',labelpad=5)
cbar.outline.set_linewidth(1.0)

fig.canvas.draw()
for isl in ISLANDS:
    panel_title(fig, axes[isl], f'{isl.upper()} \u2014 {SUBTITLE[isl]}', fontsize=9.5)

# figure source credit removed — see figure_sources.md

fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
fig.savefig(OUT_PDF, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close(fig)
print(f'Saved: {OUT_PNG}')
print(f'Saved: {OUT_PDF}')
