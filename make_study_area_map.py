"""Figure 1 v17: heading without box, single-line sources fitting inside neatline,
ArcGIS Pro Scale Line 1 in WHITE (no box) at bottom-right, compact; globe beside legend."""
import urllib.request, warnings, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec, colormaps
from matplotlib.patches import Rectangle, Patch, Polygon, Ellipse, Circle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import geopandas as gpd
from shapely.geometry import Polygon as SPolygon, MultiPolygon, Point
import contextily as ctx
import numpy as np
import pyproj
from xyzservices import TileProvider

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 9, 'axes.linewidth': 1.0, 'axes.edgecolor': 'black',
})

EsriDarkGray = TileProvider(
    name='Esri.WorldDarkGrayBase',
    url='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    attribution='Esri'
)

HOME = Path.home()
PROJECT = HOME / 'sundarbans_paper'
ANCILLARY = PROJECT / 'data' / 'ancillary'
SOI_DIR = PROJECT / 'data' / 'india_boundary'
WBD_DIR = PROJECT / 'data' / 'wb_districts'
NE_DIR  = PROJECT / 'data' / 'natural_earth'
OUT_PNG = PROJECT / 'final_maps' / 'png_300dpi' / '01_study_area_location.png'
OUT_PDF = PROJECT / 'final_maps' / 'pdf' / '01_study_area_location.pdf'
for d in (SOI_DIR, WBD_DIR, NE_DIR, OUT_PNG.parent, OUT_PDF.parent):
    d.mkdir(parents=True, exist_ok=True)

SOI_SHP = SOI_DIR / 'STATE_BOUNDARY.shp'
if not SOI_SHP.exists():
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx']:
        m = list(SOI_DIR.glob(f'*STATE_BOUNDARY{ext}'))
        if m and not (SOI_DIR / f'STATE_BOUNDARY{ext}').exists():
            shutil.copy(m[0], SOI_DIR / f'STATE_BOUNDARY{ext}')
WBD_SHP = WBD_DIR / 'West_Bengal_District.shp'
if not WBD_SHP.exists():
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx']:
        m = list(WBD_DIR.glob(f'*West_Bengal_District{ext}'))
        if m and not (WBD_DIR / f'West_Bengal_District{ext}').exists():
            shutil.copy(m[0], WBD_DIR / f'West_Bengal_District{ext}')

def download_ne():
    base = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/'
    for fn in ['ne_110m_land.geojson', 'ne_110m_admin_0_countries.geojson']:
        fp = NE_DIR / fn
        if not fp.exists():
            urllib.request.urlretrieve(base + fn, fp)
download_ne()

india_soi = gpd.read_file(SOI_SHP).to_crs(epsg=4326)
india_soi_clean = india_soi[~india_soi['STATE'].str.contains('DISPUTED', na=False)].copy()
wb_state = india_soi_clean[india_soi_clean['STATE'] == 'WEST BENGAL'].copy()
wb_districts = gpd.read_file(WBD_SHP).to_crs(epsg=4326)
s24p = wb_districts[wb_districts['DISTRICT'].str.contains('SOUTH 24', case=False, na=False)]
four_islands = gpd.read_file(ANCILLARY / 'four_islands.shp').to_crs(epsg=4326)
name_col = next((c for c in ['island','Island','NAME','name'] if c in four_islands.columns), None)
world_land = gpd.read_file(NE_DIR / 'ne_110m_land.geojson')
world_countries = gpd.read_file(NE_DIR / 'ne_110m_admin_0_countries.geojson')
india_world = world_countries[world_countries['NAME'].str.contains('India', case=False, na=False)]


def dd_to_dm(dd):
    s = '-' if dd < 0 else ''
    dd = abs(dd); d = int(dd); mf = (dd - d) * 60; m = int(round(mf))
    if m == 60: d += 1; m = 0
    return f"{s}{d}°{m:02d}'"
def x_fmt(x, _): return "0°" if x == 0 else f"{dd_to_dm(abs(x))}{'E' if x>=0 else 'W'}"
def y_fmt(y, _): return "0°" if y == 0 else f"{dd_to_dm(abs(y))}{'N' if y>=0 else 'S'}"

def fit_box(ax, fig, zoom=None):
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    dx=x1-x0; dy=y1-y0
    cos_lat=np.cos(np.radians((y0+y1)/2))
    bb=ax.get_position()
    aw=bb.width*fig.get_figwidth(); ah=bb.height*fig.get_figheight()
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
            kw=dict(crs=4326, source=ctx.providers.CartoDB.DarkMatter, attribution=False)
            if zoom is not None: kw['zoom']=zoom
            ctx.add_basemap(ax, **kw)
        except Exception as e:
            print(f'basemap: {e}'); ax.set_facecolor('#3A3A3A')

def north_arrow_box(ax, fig, box_w_in, box_h_in, topleft_axes=(0.025, 0.975)):
    """N arrow + box of FIXED physical size (inches) so it's identical in every panel."""
    bb = ax.get_position()
    aw_in = bb.width * fig.get_figwidth()
    ah_in = bb.height * fig.get_figheight()
    bw = box_w_in / aw_in
    bh = box_h_in / ah_in
    lx, ty = topleft_axes
    x0 = lx; y0 = ty - bh; cx = x0 + bw/2
    ax.add_patch(Rectangle((x0, y0), bw, bh, transform=ax.transAxes,
                           facecolor='white', edgecolor='black', linewidth=0.7,
                           alpha=0.96, zorder=15))
    ax.text(cx, y0 + bh*0.90, 'N', transform=ax.transAxes, ha='center', va='top',
            fontsize=9.5, fontweight='bold', family='serif', zorder=17)
    tri_top = y0 + bh*0.55; tri_bot = y0 + bh*0.14; tri_hw = bw*0.28
    tri = np.array([[cx, tri_top], [cx - tri_hw, tri_bot], [cx + tri_hw, tri_bot]])
    ax.add_patch(Polygon(tri, transform=ax.transAxes, facecolor='black',
                         edgecolor='black', linewidth=0.7, zorder=16))

def scale_line_1(ax, total_km, n_div=5, label_every=1):
    """ArcGIS Pro 'Scale Line 1' — WHITE, no box, bottom-RIGHT, compact.
    label_every>1 labels only every Nth tick (alternate values)."""
    halo = [pe.withStroke(linewidth=1.5, foreground='#2A2A2A')]
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim()
    kpd=111.32*np.cos(np.radians((y0+y1)/2))
    total_deg=total_km/kpd
    div_deg=total_deg/n_div
    div_km=total_km/n_div
    km_w = (x1-x0)*0.045
    rmarg = (x1-x0)*0.040
    px = x1 - rmarg - km_w - total_deg
    py = y0 + (y1-y0)*0.040    # moved a bit lower
    tick_up = (y1-y0)*0.012
    ax.plot([px, px+total_deg], [py, py], color='white', linewidth=1.1, zorder=11,
            solid_capstyle='butt', path_effects=halo)
    for i in range(n_div+1):
        xt = px + i*div_deg
        ax.plot([xt, xt], [py, py+tick_up], color='white', linewidth=1.0, zorder=11,
                solid_capstyle='butt', path_effects=halo)
        if i % label_every == 0:
            ax.text(xt, py-(y1-y0)*0.006, f'{int(round(i*div_km))}',
                    ha='center', va='top', fontsize=4.8, color='white', family='serif',
                    fontweight='bold', zorder=12, path_effects=halo)
    ax.text(px+total_deg+(x1-x0)*0.010, py, 'km',
            ha='left', va='center', fontsize=5.2, color='white', family='serif',
            fontweight='bold', zorder=12, path_effects=halo)

def fmt_panel(ax, xbins=2, ybins=3):
    ax.grid(True, color='white', linestyle='--', linewidth=0.4, alpha=0.45)
    ax.xaxis.set_major_formatter(FuncFormatter(x_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, prune='both'))
    ax.tick_params(axis='x', labelsize=6, direction='out', length=2.5, pad=2)
    ax.tick_params(axis='y', labelsize=6, direction='out', length=2.5, pad=2,
                   labelleft=True, labelright=False)
    plt.setp(ax.get_yticklabels(), rotation=90, ha='center', va='center')
    ax2 = ax.secondary_yaxis('right')
    ax2.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=ybins, prune='both'))
    ax2.tick_params(axis='y', labelsize=6, direction='out', length=2.5, pad=2)
    plt.setp(ax2.get_yticklabels(), rotation=-90, ha='center', va='center')
    for sp in ax.spines.values():
        sp.set_linewidth(1.2); sp.set_color('black')

def fmt_panel_circular(ax, xbins=3, ybins=3):
    ax.grid(True, color='white', linestyle='--', linewidth=0.4, alpha=0.55, zorder=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins, prune='both'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, prune='both'))
    ax.tick_params(axis='both', labelleft=False, labelright=False,
                   labeltop=False, labelbottom=False,
                   left=False, right=False, top=False, bottom=False)

def make_circle_fixed_radius(ax, fig, radius_inches=0.75):
    fig.canvas.draw()
    bb = ax.get_position()
    aw_in = bb.width * fig.get_figwidth()
    ah_in = bb.height * fig.get_figheight()
    rx_axes = radius_inches / aw_in
    ry_axes = radius_inches / ah_in
    bg = Ellipse((0.5, 0.5), rx_axes*2, ry_axes*2, transform=ax.transAxes,
                 facecolor='#3A3A3A', edgecolor='none', zorder=0)
    ax.add_patch(bg)
    clip = Ellipse((0.5, 0.5), rx_axes*2, ry_axes*2, transform=ax.transAxes,
                   facecolor='none', edgecolor='black', linewidth=1.5, zorder=15)
    ax.add_patch(clip)
    for img in ax.images: img.set_clip_path(clip)
    for coll in ax.collections: coll.set_clip_path(clip)
    for gl in ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines():
        gl.set_clip_path(clip)
    for sp in ax.spines.values(): sp.set_visible(False)
    return clip

def draw_globe(ax, land_gdf, india_gdf, lon_0=82, lat_0=20):
    R = 6371000
    ortho = pyproj.CRS.from_proj4(f'+proj=ortho +lat_0={lat_0} +lon_0={lon_0} +ellps=sphere +units=m')
    tf = pyproj.Transformer.from_crs('EPSG:4326', ortho, always_xy=True)
    def gc(lon, lat):
        la,lo=np.radians(lat),np.radians(lon); la0,lo0=np.radians(lat_0),np.radians(lon_0)
        return np.degrees(np.arccos(np.clip(np.sin(la)*np.sin(la0)+np.cos(la)*np.cos(la0)*np.cos(lo-lo0),-1,1)))
    def proj_vis(geom):
        def ring(coords):
            lon=np.array([c[0] for c in coords]); lat=np.array([c[1] for c in coords])
            d=gc(lon,lat)
            if np.all(d>90): return None
            x,y=tf.transform(lon,lat)
            good=np.isfinite(x)&np.isfinite(y)&(d<=90)
            if good.sum()<3: return None
            return list(zip(x[good],y[good]))
        polys=geom.geoms if geom.geom_type=='MultiPolygon' else [geom]
        out=[]
        for poly in polys:
            ext=ring(list(poly.exterior.coords))
            if ext and len(ext)>=3:
                try:
                    p=SPolygon(ext)
                    if not p.is_valid: p=p.buffer(0)
                    if p.is_valid and p.area>0: out.append(p)
                except Exception: pass
        if not out: return None
        return MultiPolygon(out) if len(out)>1 else out[0]
    disk=Point(0,0).buffer(R)
    land_g=[g.intersection(disk) for g in (proj_vis(geom) for geom in land_gdf.geometry) if g is not None]
    land_g=[g for g in land_g if not g.is_empty]
    india_g=[g for g in (proj_vis(geom) for geom in india_gdf.geometry) if g is not None]
    ax.add_patch(Circle((0,0), R, facecolor='#9DC3E6', edgecolor='#2A2A2A', linewidth=1.3, zorder=0))
    gpd.GeoSeries(land_g).plot(ax=ax, facecolor='#C2D6A4', edgecolor='#7A8B66', linewidth=0.3, zorder=1)
    if india_g:
        parts=[]
        for g in india_g:
            parts += [g] if g.geom_type=='Polygon' else list(g.geoms)
        ig=MultiPolygon(parts)
        ib=ig.bounds; pad=R*0.05
        ax.add_patch(Rectangle((ib[0]-pad, ib[1]-pad), (ib[2]-ib[0])+2*pad, (ib[3]-ib[1])+2*pad,
                               fill=False, edgecolor='red', linewidth=2.0, zorder=5))
    ax.set_xlim(-R*1.06, R*1.06); ax.set_ylim(-R*1.06, R*1.06)
    ax.set_aspect('equal'); ax.axis('off')

def panel_title(fig, ax, text, fontsize=10):
    bb = ax.get_position()
    fig.text((bb.x0 + bb.x1)/2, bb.y1 + 0.003, text,
             ha='center', va='bottom', fontsize=fontsize,
             fontweight='bold', family='serif',
             bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                       edgecolor='black', linewidth=1.0))


# ============ FIGURE ============
fig = plt.figure(figsize=(11, 14.5), facecolor='white')
fig.patches.append(Rectangle((0.008,0.008), 0.984, 0.984,
                              transform=fig.transFigure, fill=False,
                              edgecolor='black', linewidth=2.5, zorder=100))

# Heading: NO box, just title text inside the neatline
fig.text(0.5, 0.971,
         'STUDY AREA MAP OF FOUR ESTUARINE ISLANDS '
         '(KUMIRMARI, HAMILTON, MOUSUNI, LOTHIAN)\n'
         'INDIAN SUNDARBANS, WEST BENGAL, INDIA',
         ha='center', va='center', fontsize=12, fontweight='bold',
         family='serif', linespacing=1.35, zorder=101)

SRC_LINE_Y = 0.034
gs = gridspec.GridSpec(3, 6, figure=fig,
                       width_ratios=[1,1,1,1,1,1],
                       height_ratios=[1.2, 1.5, 0.55],
                       wspace=0.22, hspace=0.15,
                       left=0.04, right=0.97, top=0.935, bottom=0.050)
ax_india = fig.add_subplot(gs[0, 0:3])
ax_wb    = fig.add_subplot(gs[0, 3:6])
ax_reg   = fig.add_subplot(gs[1, 0:6])
ax_m     = fig.add_subplot(gs[2, 0])
ax_l     = fig.add_subplot(gs[2, 1])
ax_h     = fig.add_subplot(gs[2, 2])
ax_k     = fig.add_subplot(gs[2, 3])
ax_leg   = fig.add_subplot(gs[2, 4])
ax_globe = fig.add_subplot(gs[2, 5])

cmap = colormaps.get_cmap('Pastel1')

ax_india.set_xlim(67, 98); ax_india.set_ylim(6, 38)
india_soi_clean.plot(ax=ax_india,
    color=[cmap((i*17)%9/9) for i in range(len(india_soi_clean))],
    edgecolor='black', linewidth=0.4, alpha=0.65)
wb_state.plot(ax=ax_india, facecolor='red', edgecolor='black', linewidth=0.8,
              hatch='///', alpha=0.7)
fit_box(ax_india, fig, zoom=4)
fmt_panel(ax_india)
scale_line_1(ax_india, total_km=500, n_div=4, label_every=2)

b = wb_state.total_bounds; pad=0.3
ax_wb.set_xlim(b[0]-pad, b[2]+pad); ax_wb.set_ylim(b[1]-pad, b[3]+pad)
wb_districts.plot(ax=ax_wb,
    color=[cmap((i*13)%9/9) for i in range(len(wb_districts))],
    edgecolor='black', linewidth=0.5, alpha=0.7)
s24p.plot(ax=ax_wb, facecolor='magenta', edgecolor='black',
          linewidth=0.8, hatch='///', alpha=0.8)
fit_box(ax_wb, fig, zoom=6)
fmt_panel(ax_wb)
scale_line_1(ax_wb, total_km=100, n_div=5)

b = four_islands.total_bounds; mx,my=0.15,0.12
ax_reg.set_xlim(b[0]-mx, b[2]+mx); ax_reg.set_ylim(b[1]-my, b[3]+my)
s24p.plot(ax=ax_reg, facecolor='pink', edgecolor='black', linewidth=0.4, alpha=0.30)
ic = {'Kumirmari':'#C8A2C8','Hamilton':'#90EE90','Mousuni':'#FFA07A','Lothian':'#87CEEB'}
for _, row in four_islands.iterrows():
    name = row[name_col]
    gpd.GeoSeries([row.geometry], crs=four_islands.crs).plot(
        ax=ax_reg, facecolor=ic.get(name,'#ccc'),
        edgecolor='black', linewidth=1.0, alpha=0.95)
fit_box(ax_reg, fig, zoom=10)
fmt_panel(ax_reg, xbins=5, ybins=3)
centroids = {row[name_col]: row.geometry.centroid for _, row in four_islands.iterrows()}
offset = 0.04
label_positions = {
    'Mousuni':   (centroids['Mousuni'].x - offset*1.4,   centroids['Mousuni'].y),
    'Lothian':   (centroids['Lothian'].x + offset*1.4,   centroids['Lothian'].y),
    'Hamilton':  (centroids['Hamilton'].x - offset*1.4,  centroids['Hamilton'].y - 0.03),
    'Kumirmari': (centroids['Kumirmari'].x - offset*1.4, centroids['Kumirmari'].y + 0.03),
}
for name, (lx, ly) in label_positions.items():
    cen = centroids[name]
    ax_reg.annotate(name, xy=(cen.x, cen.y), xytext=(lx, ly),
        ha='center', va='center', fontsize=9, fontweight='bold',
        family='serif', zorder=20,
        bbox=dict(boxstyle='square,pad=0.22', facecolor='white',
                  edgecolor='black', linewidth=0.7),
        arrowprops=dict(arrowstyle='->', color='black', linewidth=1.3,
                        connectionstyle='arc3,rad=0.0', shrinkA=1, shrinkB=3))
scale_line_1(ax_reg, total_km=10, n_div=5)

fig.canvas.draw()
# Fixed-size north arrows on the 3 main maps — all matching INDIA's physical size
_ib = ax_india.get_position()
REF_W_IN = 0.102   * _ib.width  * fig.get_figwidth()
REF_H_IN = 0.15725 * _ib.height * fig.get_figheight()
north_arrow_box(ax_india, fig, REF_W_IN, REF_H_IN, topleft_axes=(0.025, 0.975))
north_arrow_box(ax_wb,    fig, REF_W_IN, REF_H_IN, topleft_axes=(0.025, 0.975))
north_arrow_box(ax_reg,   fig, REF_W_IN, REF_H_IN, topleft_axes=(0.020, 0.975))
panels = {'Mousuni':ax_m, 'Lothian':ax_l, 'Hamilton':ax_h, 'Kumirmari':ax_k}
min_dim_in = min(min(ax.get_position().width * fig.get_figwidth(),
                      ax.get_position().height * fig.get_figheight())
                  for ax in panels.values())
common_radius_in = min_dim_in * 0.42
for name, ax in panels.items():
    row = four_islands[four_islands[name_col]==name]
    if len(row)==0: continue
    b = row.total_bounds
    p = max(b[2]-b[0], b[3]-b[1])*0.35
    ax.set_xlim(b[0]-p, b[2]+p); ax.set_ylim(b[1]-p, b[3]+p)
    row.plot(ax=ax, facecolor=ic[name], edgecolor='black', linewidth=1.2, alpha=0.95)
    fit_box(ax, fig, zoom=12)
    fmt_panel_circular(ax)
    make_circle_fixed_radius(ax, fig, radius_inches=common_radius_in)

ax_leg.axis('off'); ax_leg.set_xlim(0,1); ax_leg.set_ylim(0,1)
lh = [
    Patch(facecolor='red', edgecolor='black', hatch='///', alpha=0.7, label='West Bengal'),
    Patch(facecolor='magenta', edgecolor='black', hatch='///', alpha=0.8, label='South 24 Parganas'),
    Patch(facecolor='#FFA07A', edgecolor='black', label='Mousuni\n(Coastal–Inhabited)'),
    Patch(facecolor='#87CEEB', edgecolor='black', label='Lothian\n(Coastal–Uninhabited)'),
    Patch(facecolor='#90EE90', edgecolor='black', label='Hamilton\n(Interior–Uninhabited)'),
    Patch(facecolor='#C8A2C8', edgecolor='black', label='Kumirmari\n(Interior–Inhabited)'),
]
leg = ax_leg.legend(handles=lh, loc='center', bbox_to_anchor=(0.5, 0.5),
                    frameon=True, fontsize=6, title='LEGEND',
                    title_fontsize=9, prop={'family':'serif'},
                    handleheight=1.8, labelspacing=0.55)
leg.get_frame().set_edgecolor('black'); leg.get_frame().set_linewidth(1.3)
leg.get_title().set_fontweight('bold'); leg.get_title().set_family('serif')

draw_globe(ax_globe, world_land, india_world, lon_0=82, lat_0=20)

sources_line = ('Sources:  India boundary — Survey of India (2024);   '
                'West Bengal districts — Govt. of West Bengal / NIC;   '
                'Basemap — Esri Dark Gray Canvas;   '
                'Study islands — Landsat 9 OLI/TIRS (USGS, 2024);   '
                'CRS — WGS 84 (EPSG:4326);   Cartography — A. Ghosh (2026)')
# figure source credit removed — see figure_sources.md

fig.canvas.draw()
panel_title(fig, ax_india, 'INDIA', fontsize=11)
panel_title(fig, ax_wb,    'WEST BENGAL', fontsize=11)
panel_title(fig, ax_reg,   'INDIAN SUNDARBANS — FOUR STUDY ISLANDS', fontsize=11)
panel_title(fig, ax_m,     'MOUSUNI', fontsize=10)
panel_title(fig, ax_l,     'LOTHIAN', fontsize=10)
panel_title(fig, ax_h,     'HAMILTON', fontsize=10)
panel_title(fig, ax_k,     'KUMIRMARI', fontsize=10)
panel_title(fig, ax_globe, 'GLOBAL LOCATION', fontsize=9)

fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
fig.savefig(OUT_PDF, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.1)
plt.close(fig)
print(f'Saved: {OUT_PNG}')
print(f'Saved: {OUT_PDF}')
