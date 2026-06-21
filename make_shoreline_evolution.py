"""Shoreline evolution maps (polished). Produces BOTH:
  - a combined 2x2 figure          -> 04_shoreline_evolution.{png,pdf}
  - four separate per-island files -> 04_shoreline_evolution_<island>.{png,pdf}
Each map shows three tidally-corrected shoreline epochs (1990 purple, 2010 teal,
2024 yellow) over an ocean basemap, in the shared cartographic style (neatline;
no-box serif heading with the setting inline; degree-minute coordinates; inset
north arrow that does not touch the frame; scale bar inside the frame; epoch-line
legend; sources kept within the neatline). Shoreline geometry is read from
shorelines_corrected.gpkg (UTM 45N) and reprojected to WGS 84; each map is fitted
to its island so none is distorted.

Basemap: tries Esri 'Ocean' first, then automatically falls through to other ocean/
water bases (Esri World Imagery, CARTO Voyager, OpenStreetMap) if a provider returns
the 'Map data not available' placeholder or is unreachable."""
import warnings, math
from pathlib import Path
import numpy as np
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
SHORE_GPKG= PROJECT / 'data' / 'processed' / 'shorelines' / 'shorelines_corrected.gpkg'
PNG_DIR   = PROJECT / 'final_maps' / 'png_300dpi'
PDF_DIR   = PROJECT / 'final_maps' / 'pdf'
for d in (PNG_DIR, PDF_DIR): d.mkdir(parents=True, exist_ok=True)

UTM='EPSG:32645'; WGS='EPSG:4326'
ISLANDS=['Kumirmari','Hamilton','Mousuni','Lothian']
POS={'Kumirmari':(0,0),'Hamilton':(0,1),'Mousuni':(1,0),'Lothian':(1,1)}
SETTING={'Kumirmari':'Interior','Hamilton':'Interior','Mousuni':'Coastal','Lothian':'Coastal'}

EPOCHS=['1990','2010','2024']
EPOCH_COLOR={'1990':'#440154','2010':'#21918c','2024':'#fde725'}   # viridis purple -> teal -> yellow
EPOCH_Z={'1990':3,'2010':4,'2024':5}

# Basemap candidates, tried in order; first one that returns real tiles wins.
CANDIDATES=[]
BASEMAP_CREDIT={
    'CARTO Positron (OSM, no roads/labels)':'CARTO Positron (no labels) (\u00a9 OpenStreetMap contributors, \u00a9 CARTO)',
    'CARTO Voyager (OSM, label-free)':'CARTO Voyager (no labels) (\u00a9 OpenStreetMap contributors, \u00a9 CARTO)',
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
                ('OpenStreetMap',                         P.OpenStreetMap.Mapnik,     12)]
STATE={'basemap':'ocean tint'}

islands_utm = gpd.read_file(ANCILLARY / 'four_islands.shp').to_crs(UTM)
name_col = next((c for c in ['island','Island','NAME','name'] if c in islands_utm.columns), None)
T_U2W = Transformer.from_crs(UTM, WGS, always_xy=True)

# ---- load shorelines once, reproject to WGS84, filter to landsat ----
def _load_shorelines():
    try:
        sl = gpd.read_file(SHORE_GPKG, layer='shorelines_corrected')
    except Exception:
        try:
            sl = gpd.read_file(SHORE_GPKG)
        except Exception:
            return None, None
    if sl.crs is None:
        sl = sl.set_crs(UTM)
    sl = sl.to_crs(WGS)
    if 'sensor' in sl.columns:
        sl = sl[sl['sensor'].astype(str).str.lower() == 'landsat']
    ncol = next((c for c in ['island','Island','NAME','name'] if c in sl.columns), None)
    return sl, ncol
SL, SL_NAME = _load_shorelines()

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

def _looks_blank(im):
    a=np.asarray(im.get_array(),dtype=float)
    if a.size==0: return True
    if a.max()>1.5: a=a/255.0
    rgb=a[...,:3].reshape(-1,3)
    mean=rgb.mean(axis=0); std=float(rgb.std(axis=0).mean())
    light=mean.mean()>0.78
    blue_excess=mean[2]-mean[0]
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

def island_shorelines(island):
    """Return {epoch: [(lon,lat), ...parts]} for the three epochs (WGS84)."""
    out={}
    if SL is None or SL_NAME is None: return out
    sub=SL[SL[SL_NAME]==island]
    for epoch in EPOCHS:
        match=sub[sub['epoch'].astype(str)==epoch]
        parts=[]
        for geom in match.geometry:
            if geom is None: continue
            for g in (geom.geoms if geom.geom_type=='MultiLineString' else [geom]):
                try:
                    xs,ys=g.xy
                except Exception:
                    continue
                if len(xs): parts.append((np.asarray(xs),np.asarray(ys)))
        if parts: out[epoch]=parts
    return out

def island_extent(poly_ll, shore):
    bx=[poly_ll.bounds[0],poly_ll.bounds[2]]; by=[poly_ll.bounds[1],poly_ll.bounds[3]]
    for parts in shore.values():
        for lon,lat in parts:
            if len(lon):
                bx+=[float(np.nanmin(lon)),float(np.nanmax(lon))]
                by+=[float(np.nanmin(lat)),float(np.nanmax(lat))]
    minx,maxx,miny,maxy=min(bx),max(bx),min(by),max(by)
    mx=(maxx-minx)*0.10 or 0.01; my=(maxy-miny)*0.10 or 0.01
    return (minx-mx,maxx+mx,miny-my,maxy+my)

def setup_panel(ax, island, scale_fs=6.0, lab=7.5, title=True):
    poly_ll = gpd.GeoSeries([islands_utm[islands_utm[name_col]==island].geometry.iloc[0]],
                            crs=UTM).to_crs(WGS).iloc[0]
    shore = island_shorelines(island)
    ex=island_extent(poly_ll, shore); ex0,ex1,ey0,ey1=ex; mean_lat=(ey0+ey1)/2
    ax.set_xlim(ex0,ex1); ax.set_ylim(ey0,ey1)
    add_ocean(ax, ex)
    ax.set_xlim(ex0,ex1); ax.set_ylim(ey0,ey1)
    ax.set_aspect(1.0/np.cos(np.radians(mean_lat)))
    # light land fill (no edge: the epoch lines provide the boundaries)
    for gg in ([poly_ll] if poly_ll.geom_type=='Polygon' else list(poly_ll.geoms)):
        xs,ys=gg.exterior.xy
        ax.fill(xs,ys,facecolor='white',edgecolor='none',alpha=0.28,zorder=2)
    # three shoreline epochs, each with a thin white halo so they read on any basemap
    for epoch in EPOCHS:
        if epoch in shore:
            first=True
            for lon,lat in shore[epoch]:
                ax.plot(lon,lat,color=EPOCH_COLOR[epoch],linewidth=1.9,alpha=0.95,
                        solid_capstyle='round',solid_joinstyle='round',zorder=EPOCH_Z[epoch],
                        path_effects=[pe.Stroke(linewidth=3.1,foreground='white',alpha=0.9),pe.Normal()])
                first=False
    if not shore:
        ax.text(0.5,0.5,'(shoreline gpkg unavailable)',transform=ax.transAxes,ha='center',va='center',
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
             'Sources:  Shorelines \u2014 Landsat 5/7/8/9 MNDWI (Xu 2006) + Otsu thresholding, three epochs '
             '(1990, 2010, 2024);   Tidal correction \u2014 FES2022 (\u0394x = h / tan \u03b2, \u03b2 = 0.0075);   '
             'Island polygons \u2014 GEE MNDWI',
             ha='center',va='center',fontsize=fs,family='serif',style='italic',zorder=101)
    fig.text(0.5,y2,
             f'Basemap \u2014 {cred};   CRS \u2014 WGS 84 (EPSG:4326), shoreline geometry in UTM 45N (EPSG:32645);   '
             'Cartography \u2014 A. Ghosh (2026)',
             ha='center',va='center',fontsize=fs,family='serif',style='italic',zorder=101)

def epoch_legend(fig, y, fs):
    handles=[Line2D([0],[0],color=EPOCH_COLOR['1990'],lw=2.8,label='1990',
                    path_effects=[pe.Stroke(linewidth=4.0,foreground='white',alpha=0.9),pe.Normal()]),
             Line2D([0],[0],color=EPOCH_COLOR['2010'],lw=2.8,label='2010',
                    path_effects=[pe.Stroke(linewidth=4.0,foreground='white',alpha=0.9),pe.Normal()]),
             Line2D([0],[0],color=EPOCH_COLOR['2024'],lw=2.8,label='2024',
                    path_effects=[pe.Stroke(linewidth=4.0,foreground='white',alpha=0.9),pe.Normal()])]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5,y),
               ncol=3, frameon=False, fontsize=fs+3.5, prop={'family':'serif'},
               handlelength=2.6, columnspacing=2.8, title='Tidally-corrected shoreline (year)',
               title_fontsize=fs+2.5)

# ---------- combined 2x2 ----------
def build_combined():
    fig=plt.figure(figsize=(13.2,13.8), facecolor='white'); neatline(fig)
    fig.text(0.5,0.958,'SHORELINE EVOLUTION  1990 \u2192 2010 \u2192 2024  (TIDALLY CORRECTED)\n'
             'FOUR ESTUARINE ISLANDS, INDIAN SUNDARBANS, WEST BENGAL, INDIA',
             ha='center',va='center',fontsize=11.0,fontweight='bold',family='serif',linespacing=1.35,zorder=101)
    gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.30,wspace=0.17,top=0.890,bottom=0.115,left=0.065,right=0.95)
    axd={}
    for island in ISLANDS:
        r,c=POS[island]; ax=fig.add_subplot(gs[r,c]); setup_panel(ax, island, scale_fs=5.2, lab=6.5); axd[island]=ax
    fig.canvas.draw()
    for island,ax in axd.items():
        north_arrow_box(ax, fig, gap=0.05)
    epoch_legend(fig, 0.062, 5.2)
    # figure source credit removed — see figure_sources.md
    png=PNG_DIR/'04_shoreline_evolution.png'; pdf=PDF_DIR/'04_shoreline_evolution.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    fig.savefig(pdf, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close(fig); print(f'Saved: {png}')

# ---------- one file per island ----------
def build_island(island):
    fig=plt.figure(figsize=(10.0,10.8), facecolor='white'); neatline(fig)
    fig.text(0.5,0.953,
             f'SHORELINE EVOLUTION 1990 \u2192 2010 \u2192 2024 \u2014 {island.upper()} \u00b7 {SETTING[island].upper()}\n'
             'TIDALLY CORRECTED \u00b7 INDIAN SUNDARBANS, WEST BENGAL, INDIA',
             ha='center',va='center',fontsize=11.0,fontweight='bold',family='serif',linespacing=1.4,zorder=101)
    ax=fig.add_axes([0.09, 0.145, 0.84, 0.745]); setup_panel(ax, island, scale_fs=6.0, lab=7.5, title=False)
    fig.canvas.draw(); north_arrow_box(ax, fig, gap=0.05)
    epoch_legend(fig, 0.066, 5.4)
    # figure source credit removed — see figure_sources.md
    base=f'04_shoreline_evolution_{island.lower()}'
    png=PNG_DIR/f'{base}.png'; pdf=PDF_DIR/f'{base}.pdf'
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    fig.savefig(pdf, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close(fig); print(f'Saved: {png}')

build_combined()
for island in ISLANDS:
    build_island(island)
print(f'Basemap used: {STATE["basemap"]}' + ('' if HAVE_CX else '  (contextily not installed)'))
