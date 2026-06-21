"""Day 4 Step 3 (v2): Cleaner layout with proper colorbar + rotated x-labels + refined validation."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
import geopandas as gpd

BASE_DIR  = Path.home() / 'sundarbans_paper'
DSAS_OUT  = BASE_DIR / 'data' / 'dsas_output'
ANCILLARY = BASE_DIR / 'data' / 'ancillary'
FIGURES   = BASE_DIR / 'figures'
FIGURES.mkdir(exist_ok=True)

with open(DSAS_OUT / 'summary.json') as f:
    summary = json.load(f)
islands_gdf = gpd.read_file(ANCILLARY / 'four_islands.shp').to_crs('EPSG:32645')

ISLANDS = ['Kumirmari', 'Hamilton', 'Mousuni', 'Lothian']
SETTINGS = {'Kumirmari': 'Interior • Inhabited',
            'Hamilton':  'Interior • Uninhabited',
            'Mousuni':   'Coastal • Inhabited',
            'Lothian':   'Coastal • Uninhabited'}

# === FIGURE 1: 4-panel LRR heatmap with dedicated colorbar axes ===
fig = plt.figure(figsize=(17, 17))
gs = gridspec.GridSpec(3, 2, figure=fig,
                       height_ratios=[1, 1, 0.035],
                       hspace=0.32, wspace=0.22,
                       top=0.95, bottom=0.045, left=0.06, right=0.97)

axes = {island: fig.add_subplot(gs[i // 2, i % 2])
        for i, island in enumerate(ISLANDS)}
cax = fig.add_subplot(gs[2, :])

vmin, vmax = -10, 10
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu

for island in ISLANDS:
    ax = axes[island]
    poly = islands_gdf[islands_gdf['island'] == island].iloc[0].geometry
    geoms = [poly] if poly.geom_type == 'Polygon' else poly.geoms
    for g in geoms:
        x, y = g.exterior.xy
        ax.fill(x, y, color='#d8d8d8', alpha=0.55,
                edgecolor='#666', linewidth=0.8, zorder=1)
    tr = gpd.read_file(DSAS_OUT / f'transects_{island.lower()}.shp')
    lrr_c = np.clip(tr['LRR'].values, vmin, vmax)
    for i, row in tr.iterrows():
        x, y = row.geometry.xy
        ax.plot(x, y, color=cmap(norm(lrr_c[i])), linewidth=0.7, alpha=0.85, zorder=2)
    s = summary[island]
    txt = (f'EPR: {s["EPR_mean"]:+.2f} ± {s["EPR_std"]:.2f} m/yr\n'
           f'LRR: {s["LRR_mean"]:+.2f} ± {s["LRR_std"]:.2f} m/yr\n'
           f'E {s["pct_erosion"]:.0f}% / S {s["pct_stable"]:.0f}% / A {s["pct_accretion"]:.0f}%\n'
           f'n = {s["n_transects"]} transects')
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     alpha=0.92, edgecolor='#aaa'))
    ax.set_title(f'{island} — {SETTINGS[island]}', fontsize=13, fontweight='bold', pad=8)
    ax.set_aspect('equal')
    ax.set_xlabel('Easting (m, UTM 45N)', fontsize=10)
    ax.set_ylabel('Northing (m, UTM 45N)', fontsize=10)
    # Rotate x-labels and limit ticks
    ax.tick_params(axis='x', rotation=35, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=6)
    ax.grid(True, alpha=0.25)

# Dedicated colorbar axes
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label('Linear Regression Rate, LRR (m/yr)   |   ← Erosion          Accretion →',
               fontsize=12, labelpad=6)
cbar.ax.tick_params(labelsize=10)

fig.suptitle('Shoreline Change Rates 1990–2024: 2×2 Paired-Site Design',
             fontsize=15, fontweight='bold', y=0.985)
plt.savefig(FIGURES / 'day4_transects_2x2.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {FIGURES / "day4_transects_2x2.png"}')

# === REFINED MOUSUNI VALIDATION (worst-section + correct azimuth) ===
print('\n=== Refined Mousuni validation ===')
mousuni = gpd.read_file(DSAS_OUT / 'transects_mousuni.shp')

print('\nMean EPR by erosion percentile (most-eroded sections):')
for pct in [5, 10, 15, 20, 25, 30, 50]:
    thr = mousuni['EPR'].quantile(pct/100)
    sub = mousuni[mousuni['EPR'] <= thr]
    print(f'  Worst {pct:2d}% (n={len(sub):3d}): mean EPR = {sub["EPR"].mean():+7.2f} m/yr')

print('\nMean EPR by azimuth quadrant (corrected: 0=N, 90=E, 180=S, 270=W):')
for az_min, az_max, name in [(0,90,'NE→sea side'), (90,180,'SE→sea/channel'),
                              (180,270,'SW→open Bay'), (270,360,'NW→Muriganga')]:
    sub = mousuni[(mousuni['azimuth'] >= az_min) & (mousuni['azimuth'] < az_max)]
    print(f'  {name:20s} ({az_min:3d}-{az_max:3d}°): n={len(sub):3d}, '
          f'mean EPR = {sub["EPR"].mean():+7.2f} m/yr,  '
          f'min = {sub["EPR"].min():+7.2f} m/yr')

# Most-eroded transect location
worst = mousuni.loc[mousuni['EPR'].idxmin()]
print(f'\nMost-eroded transect:')
print(f'  Azimuth: {worst["azimuth"]:.1f}° (W=270, NW=315)')
print(f'  Location UTM: ({worst["origin_x"]:.0f}, {worst["origin_y"]:.0f})')
print(f'  EPR: {worst["EPR"]:+.2f} m/yr  |  Published worst-section: -23.55 m/yr')

# Sea-facing combined (SW+NW, where the worst erosion is concentrated)
sea_facing = mousuni[(mousuni['azimuth'] >= 180) & (mousuni['azimuth'] < 360)]
print(f'\nSea/channel-facing (SW+NW, az 180-360°): n={len(sea_facing)}')
print(f'  Mean EPR: {sea_facing["EPR"].mean():+.2f} m/yr')
print(f'  Worst 25%: {sea_facing[sea_facing["EPR"] <= sea_facing["EPR"].quantile(0.25)]["EPR"].mean():+.2f} m/yr')

# Copy to desktop
desktop = Path('/mnt/c/Users/anubg/Desktop')
if desktop.exists():
    shutil.copy(FIGURES / 'day4_transects_2x2.png', desktop / 'day4_transects_2x2.png')
    print(f'\nCopied to desktop: day4_transects_2x2.png')
