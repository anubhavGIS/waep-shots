"""Kalman-filter shoreline forecast maps (polished). Produces BOTH:
  - a combined 2x2 figure (day5_kalman_forecasts)  -> 13_kalman_forecast.{png,pdf}
  - four separate per-island figures               -> 13_kalman_forecast_<island>.{png,pdf}
Each map shows per-transect forecast shoreline positions 2024 -> 2034 -> 2044 over an
ocean basemap, in the shared cartographic style (neatline; no-box serif heading with
the setting inline; degree-minute coordinates; inset north arrow that does not touch
the frame; scale bar inside the frame; per-island stats box BELOW the map; legend;
sources kept within the neatline). Forecast geometry is computed in UTM 45N and
reprojected to WGS 84; each map is fitted to its island so none is distorted.

Basemap: tries Esri 'Ocean' first, then automatically falls through to other ocean/
water bases (Esri World Imagery, CARTO Voyager, OpenStreetMap) if a provider returns
the 'Map data not available' placeholder or is unreachable."""
import warnings, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.patches import Rectangle, Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D
import geopandas as gpd
from pyproj import Transformer

try:
    import contextily as cx
    HAVE_CX = True
except Exception:
    HAVE_CX = False
OCEAN_FALLBACK = '#cfe6f2'

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family':'serif',
                     'font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
                     'font.size':9,'axes.linewidth':1.0,'axes.edgecolor':'black'})

HOME      = Path.home()
PROJECT   = HOME / 'sundarbans_paper'
ANCILLARY = PROJECT / 'data' / 'ancillary'
KALMAN    = PROJECT / 'data' / 'kalman_output'
PNG_DIR   = PROJECT / 'final_maps' / 'png_300dpi'
PDF_DIR   = PROJECT / 'final_maps' / 'pdf'
for d in (PNG_DIR, PDF_DIR): d.mkdir(parents=True, exist_ok=True)

UTM='EPSG:32645'; WGS='EPSG:4326'
ISLANDS=['Kumirmari','Hamilton','Mousuni','Lothian']
POS={'Kumirmari':(0,0),'Hamilton':(0,1),'Mousuni':(1,0),'Lothian':(1,1)}
SETTING={'Kumirmari':'Interior','Hamilton':'Interior','Mousuni':'Coastal','Lothian':'Coastal'}
ISLE_EDGE='#37474F'
YR_COLOR={'2024':'#000000','2034':'#ff7f0e','2044':'#d62728'}

# Basemap candidates, tried in order; first one that returns real tiles wins.
# (Esri Ocean is reliably cached only to ~z10; others are global to high zoom.)
CANDIDATES=[]
BASEMAP_CREDIT={
    'Esri Ocean Basemap':'Esri Ocean Basemap (Esri, GEBCO, NOAA, Nat. Geographic, DeLorme, et al.)',
    'Esri World Imagery':'Esri World Imagery (Esri, Maxar, Earthstar Geographics, et al.)',
    'CARTO Voyager':'CARTO Voyager (\u00a9 OpenStreetMap contributors, \u00a9 CARTO)',
    'OpenStreetMap':'\u00a9 OpenStreetMap contributors',
    'ocean tint':'ocean tint (basemap tiles unavailable on this machine)'}
if HAVE_CX:
    P=cx.providers
    CANDIDATES=[('Esri World Topo Map', P.Esri.WorldTopoMap, 13),
                ('Esri World Topo Map (z14)', P.Esri.WorldTopoMap, 14),
                ('CARTO Positron (no labels)', P.CartoDB.PositronNoLabels, 13),
                ('OpenStreetMap',      P.OpenStreetMap.Mapnik, 12)]
STATE={'basemap':'ocean tint'}

islands_utm = gpd.read_file(ANCILLARY / 'four_islands.shp').to_crs(UTM)
name_col = next((c for c in ['island','Island','NAME','name'] if c in islands_utm.columns), None)
T_U2W = Transformer.from_crs(UTM, WGS, always_xy=True)

import json
try:
    with open(KALMAN / 'forecast_summary.json') as f: SUMMARY = json.load(f)
except Exception:
    SUMMARY = {}

# ---------- helpers ----------
def dd_to_dm(dd):
    s='-' if dd<0 else ''; dd=abs(dd); d=int(dd); m=int(round((dd-d)*60))
    if m==60: d+=1; m=0
    return f"{s}{d}\u00b0{m:02d}'"
def x_fmt(x,_): return "0\u00b0" if x==0 else f"{dd_to_dm(abs(x))}{'E' if x>=0 else 'W'}"
def y_fmt(y,_): return "0\u00b0" if y==0 else f"{dd_to_dm(abs(y))}{'N' if y>=0 else 'S'}"

def fmt_panel(ax, lab=7.5):
    ax.grid(True, color='white', linestyle=':', linewidth=0.5, alpha=0.5, zorder=1)
    ax.xaxis.set_major_formatter(FuncFormatter(x_fmt)); ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    ax.tick_params(axis='x', labelsize=lab, direction='out', length=2.6, pad=2)
    ax.tick_params(axis='y', labelsize=lab, direction='out', length=2.6, pad=2)
    plt.setp(ax.get_yticklabels(), rotation=90, ha='center', va='center')
    for sp in ax.spines.values(): sp.set_linewidth(1.2); sp.set_color('black')

NA_W_IN, NA_H_IN = 0.32, 0.44
def north_arrow_box(ax, fig, gap=0.05):
    bb=ax.get_position(); aw_in=bb.width*fig.get_figwidth(); ah_in=bb.height*fig.get_figheight()
    bw=NA_W_IN/aw_in; bh=NA_H_IN/ah_in
    x0=1.0-gap-bw; y0=1.0-gap-bh; cx_=x0+bw/2
    ax.add_patch(Rectangle((x0,y0),bw,bh,transform=ax.transAxes,facecolor='white',
                           edgecolor='black',linewidth=0.7,alpha=0.97,zorder=15))
    ax.text(cx_,y0+bh*0.90,'N',transform=ax.transAxes,ha='center',va='top',
            fontsize=8.5,fontweight='bold',family='serif',zorder=17)
    tri_top=y0+bh*0.56; tri_bot=y0+bh*0.16; tri_hw=bw*0.27
    tri=np.array([[cx_,tri_top],[cx_-tri_hw,tri_bot],[cx_+tri_hw,tri_bot]])
    ax.add_patch(Polygon(tri,transform=ax.transAxes,facecolor='black',edgecolor='black',linewidth=0.7,zorder=16))

def scale_line(ax, total_km, n_div=4, fs=6.0):
    halo=[pe.withStroke(linewidth=1.8, foreground='white')]
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    kpd=111.32*np.cos(np.radians((y0+y1)/2))
    total_deg=total_km/kpd; div_deg=total_deg/n_div; div_km=total_km/n_div
    px=x0+(x1-x0)*0.06; py=y0+(y1-y0)*0.06; tick_up=(y1-y0)*0.018
    ax.plot([px,px+total_deg],[py,py],color='black',linewidth=1.3,zorder=12,solid_capstyle='butt',path_effects=halo)
    for i in range(n_div+1):
        xt=px+i*div_deg
        ax.plot([xt,xt],[py,py+tick_up],color='black',linewidth=1.0,zorder=12,solid_capstyle='butt',path_effects=halo)
        ax.text(xt,py-(y1-y0)*0.011,f'{i*div_km:g}',ha='center',va='top',fontsize=fs,
                color='black',family='serif',zorder=12,path_effects=halo)
    ax.text(px+total_deg+(x1-x0)*0.012,py,'km',ha='left',va='center',fontsize=fs+0.4,
            color='black',family='serif',zorder=12,path_effects=halo)

def nice_scale_km(span_km):
    target=span_km*0.35
    if target<=0: return 1,4
    base=10**np.floor(np.log10(target)); r=target/base
    return (base if r<1.5 else (2*base if r<3.5 else 5*base)), 4

def reproj_xy(xs, ys):
    lon, lat = T_U2W.transform(np.asarray(xs,float), np.asarray(ys,float)); return np.asarray(lon), np.asarray(lat)

def stats_text(island):
    s = SUMMARY.get(island)
    if not s: return None
    return (f"\u0394 2034   {s['change_2034_mean']:+.0f} m         \u0394 2044   {s['change_2044_mean']:+.0f} \u00b1 {s['change_2044_std']:.0f} m\n"
            f"{s['pct_eroding_2044']:.0f}% of transects eroding by 2044         n = {s['n_transects']} transects")

def _looks_blank(im):
    """True if the basemap image is the near-uniform light-grey 'Map data not available' placeholder."""
    a=np.asarray(im.get_array(),dtype=float)
    if a.size==0: return True
    if a.max()>1.5: a=a/255.0
    rgb=a[...,:3].reshape(-1,3)
    mean=rgb.mean(axis=0); std=float(rgb.std(axis=0).mean())
    light=mean.mean()>0.78
    blue_excess=mean[2]-mean[0]                       # ocean tiles are bluish; placeholder is grey
    grayish=(blue_excess<0.025) and (abs(mean[1]-mean[0])<0.04)
    return bool(light and grayish and std<0.08)

def add_ocean(ax, ex):
    ex0,ex1,ey0,ey1=ex
    if HAVE_CX:
        for name,prov,z in CANDIDATES:
            n0=len(ax.images)
            try:
                cx.add_basemap(ax, crs=WGS, source=prov, zoom=z, attribution=False, zorder=0)
            except Exception:
                while len(ax.images)>n0: ax.images[-1].remove()
                continue
            if len(ax.images)>n0 and _looks_blank(ax.images[-1]):
                while len(ax.images)>n0: ax.images[-1].remove()
                continue
            if len(ax.images)>n0:
                STATE['basemap']=name
                ax.set_xlim(ex0,ex1); ax.set_ylim(ey0,ey1); return
    ax.set_facecolor(OCEAN_FALLBACK)
    ax.set_xlim(ex0,ex1); ax.set_ylim(ey0,ey1)

def island_extent(poly_ll, pts):
    bx=[poly_ll.bounds[0],poly_ll.bounds[2]]; by=[poly_ll.bounds[1],poly_ll.bounds[3]]
    for lon,lat in pts.values():
        if len(lon): bx+=[float(np.nanmin(lon)),float(np.nanmax(lon))]; by+=[float(np.nanmin(lat)),float(np.nanmax(lat))]
    minx,maxx,miny,maxy=min(bx),max(bx),min(by),max(by)
    mx=(maxx-minx)*0.10 or 0.01; my=(maxy-miny)*0.10 or 0.01
    return (minx-mx,maxx+mx,miny-my,maxy+my)

def load_island(island):
    poly_ll = gpd.GeoSeries([islands_utm[islands_utm[name_col]==island].geometry.iloc[0]],
                            crs=UTM).to_crs(WGS).iloc[0]
    csv=KALMAN / f'forecast_{island.lower()}.csv'; pts={}
    if csv.exists():
        df=pd.read_csv(csv)
        # --- drop runaway transects whose 2034/2044 forecast lands implausibly far offshore ---
        cx0=df['cur_x'].to_numpy(float); cy0=df['cur_y'].to_numpy(float)
        d34=np.hypot(df['x_2034'].to_numpy(float)-cx0, df['y_2034'].to_numpy(float)-cy0)
        d44=np.hypot(df['x_2044'].to_numpy(float)-cx0, df['y_2044'].to_numpy(float)-cy0)
        dmax=np.fmax(d34,d44); fin=np.isfinite(dmax)
        if fin.any():
            med=np.nanmedian(dmax[fin]); mad=np.nanmedian(np.abs(dmax[fin]-med))
            cap=max(800.0, med + 6.0*1.4826*mad)          # robust cap (>= 800 m of 20-yr displacement)
        else:
            cap=np.inf
        keep=fin & (dmax<=cap); ndrop=int((~keep).sum())
        if ndrop: print(f'  [{island}] dropped {ndrop} runaway transect(s): 20-yr forecast displacement > {cap:.0f} m')
        df=df[keep].reset_index(drop=True)
        pts={'2024':reproj_xy(df['cur_x'],df['cur_y']),
             '2034':reproj_xy(df['x_2034'],df['y_2034']),
             '2044':reproj_xy(df['x_2044'],df['y_2044'])}
    return poly_ll, pts

def setup_panel(ax, island, scale_fs=6.0, lab=7.5, title=True):
    poly_ll, pts = load_island(island)
    ex=island_extent(poly_ll, pts); ex0,ex1,ey0,ey1=ex; mean_lat=(ey0+ey1)/2
    ax.set_xlim(ex0,ex1); ax.set_ylim(ey0,ey1)
    add_ocean(ax, ex)
    ax.set_xlim(ex0,ex1); ax.set_ylim(ey0,ey1)
    ax.set_aspect(1.0/np.cos(np.radians(mean_lat)))
    for gg in ([poly_ll] if poly_ll.geom_type=='Polygon' else list(poly_ll.geoms)):
        xs,ys=gg.exterior.xy
        ax.fill(xs,ys,facecolor='white',edgecolor=ISLE_EDGE,linewidth=1.4,alpha=0.42,zorder=2,
                path_effects=[pe.withStroke(linewidth=2.1, foreground='white')])
    for yr in ['2024','2034','2044']:
        if yr in pts and len(pts[yr][0]):
            lon,lat=pts[yr]
            ax.scatter(lon,lat,s=6,c=YR_COLOR[yr],alpha=0.82,linewidths=0,zorder={'2024':4,'2034':5,'2044':6}[yr])
    if not pts:
        ax.text(0.5,0.5,'(forecast CSV unavailable)',transform=ax.transAxes,ha='center',va='center',
                fontsize=9,family='serif',color='#444',zorder=8)
    fmt_panel(ax, lab=lab)
    if title:
        ax.set_title(f'{island} \u00b7 {SETTING[island]}', fontsize=11, fontweight='bold', family='serif', pad=4)
    span_km=(ex1-ex0)*111.32*np.cos(np.radians(mean_lat)); km,ndiv=nice_scale_km(span_km)
    scale_line(ax, total_km=km, n_div=ndiv, fs=scale_fs)

def neatline(fig):
    fig.patches.append(Rectangle((0.008,0.008),0.984,0.984,transform=fig.transFigure,
                                 fill=False,edgecolor='black',linewidth=2.5,zorder=100))

def sources_block(fig, y1, y2, fs):
    cred=BASEMAP_CREDIT.get(STATE['basemap'], STATE['basemap'])
    fig.text(0.5,y1,
             'Sources:  Forecast positions \u2014 2-D Kalman filter (constant-velocity state model) on DSAS transect '
             'time-series (1990\u20132024);   Tidal correction \u2014 FES2022;   Shorelines \u2014 Landsat MNDWI + Otsu;   '
             'Island polygons \u2014 GEE MNDWI',
             ha='center',va='center',fontsize=fs,family='serif',style='italic',zorder=101)
    fig.text(0.5,y2,
             f'Basemap \u2014 {cred};   CRS \u2014 WGS 84 (EPSG:4326), forecast geometry in UTM 45N (EPSG:32645);   '
             'Cartography \u2014 A. Ghosh (2026)'.replace('{cred}',cred),
             ha='center',va='center',fontsize=fs,family='serif',style='italic',zorder=101)

def year_legend(fig, y, fs):
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=YR_COLOR['2024'],markeredgecolor='none',markersize=8,label='2024 (Kalman state)'),
             Line2D([0],[0],marker='o',color='none',markerfacecolor=YR_COLOR['2034'],markeredgecolor='none',markersize=8,label='2034 forecast'),
             Line2D([0],[0],marker='o',color='none',markerfacecolor=YR_COLOR['2044'],markeredgecolor='none',markersize=8,label='2044 forecast')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5,y),
               ncol=3, frameon=False, fontsize=fs+3.5, prop={'family':'serif'}, handlelength=1.8, columnspacing=2.6)

# ---------- combined 2x2 (day5_kalman_forecasts) ----------
def build_combined():
    fig=plt.figure(figsize=(13.2,14.6), facecolor='white'); neatline(fig)
    fig.text(0.5,0.962,'KALMAN-FILTER SHORELINE FORECASTS \u2014 PER-TRANSECT POSITIONS 2024 \u2192 2034 \u2192 2044\n'
             'FOUR ESTUARINE ISLANDS, INDIAN SUNDARBANS, WEST BENGAL, INDIA',
             ha='center',va='center',fontsize=11.0,fontweight='bold',family='serif',linespacing=1.35,zorder=101)
    gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.52,wspace=0.18,top=0.885,bottom=0.165,left=0.065,right=0.95)
    axd={}
    for island in ISLANDS:
        r,c=POS[island]; ax=fig.add_subplot(gs[r,c]); setup_panel(ax, island, scale_fs=5.2, lab=6.5); axd[island]=ax
    fig.canvas.draw()
    for island,ax in axd.items():
        north_arrow_box(ax, fig, gap=0.05)
        st=stats_text(island)
        if st:
            bb=ax.get_position()
            fig.text((bb.x0+bb.x1)/2, bb.y0-0.028, st, ha='center', va='top', fontsize=6.2,
                     family='serif', linespacing=1.4, zorder=101,
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#999', linewidth=0.6))
    year_legend(fig, 0.066, 5.0)
    # figure source credit removed — see figure_sources.md
    png=PNG_DIR/'13_kalman_forecast.png'; pdf=PDF_DIR/'13_kalman_forecast.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    fig.savefig(pdf, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close(fig); print(f'Saved: {png}')

# ---------- one file per island ----------
def build_island(island):
    fig=plt.figure(figsize=(10.0,11.2), facecolor='white'); neatline(fig)
    fig.text(0.5,0.953,
             f'KALMAN-FILTER SHORELINE FORECAST \u2014 {island.upper()} \u00b7 {SETTING[island].upper()}\n'
             'PER-TRANSECT POSITIONS 2024 \u2192 2034 \u2192 2044 \u00b7 INDIAN SUNDARBANS, WEST BENGAL, INDIA',
             ha='center',va='center',fontsize=11.0,fontweight='bold',family='serif',linespacing=1.4,zorder=101)
    ax=fig.add_axes([0.09, 0.175, 0.84, 0.700]); setup_panel(ax, island, scale_fs=6.0, lab=7.5, title=False)
    fig.canvas.draw(); north_arrow_box(ax, fig, gap=0.05)
    st=stats_text(island)
    if st:
        bb=ax.get_position()
        fig.text((bb.x0+bb.x1)/2, bb.y0-0.030, st, ha='center', va='top', fontsize=7.5,
                 family='serif', linespacing=1.45, zorder=101,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#999', linewidth=0.7))
    year_legend(fig, 0.072, 5.4)
    # figure source credit removed — see figure_sources.md
    base=f'13_kalman_forecast_{island.lower()}'
    png=PNG_DIR/f'{base}.png'; pdf=PDF_DIR/f'{base}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    fig.savefig(pdf, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close(fig); print(f'Saved: {png}')

build_combined()
for island in ISLANDS:
    build_island(island)
print(f'Basemap used: {STATE["basemap"]}' + ('' if HAVE_CX else '  (contextily not installed)'))
