"""Day 4 Step 3: Visualize DSAS transect rates + validate against published."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
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

# === FIGURE 1: 4-panel LRR heatmap on transects ===
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
positions = {'Kumirmari': (0,0), 'Hamilton': (0,1),
             'Mousuni':   (1,0), 'Lothian':  (1,1)}
vmin, vmax = -10, 10
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.cm.RdBu

for island in ISLANDS:
    ax = axes[positions[island]]
    poly = islands_gdf[islands_gdf['island'] == island].iloc[0].geometry
    geoms = [poly] if poly.geom_type == 'Polygon' else poly.geoms
    for g in geoms:
        x, y = g.exterior.xy
        ax.fill(x, y, color='#d8d8d8', alpha=0.55, edgecolor='#666', linewidth=0.8, zorder=1)

    tr = gpd.read_file(DSAS_OUT / f'transects_{island.lower()}.shp')
    lrr_c = np.clip(tr['LRR'].values, vmin, vmax)
    for i, row in tr.iterrows():
        x, y = row.geometry.xy
        ax.plot(x, y, color=cmap(norm(lrr_c[i])), linewidth=0.7, alpha=0.85, zorder=2)

    s = summary[island]
    txt = (f'EPR: {s["EPR_mean"]:+.2f} ± {s["EPR_std"]:.2f} m/yr\n'
           f'LRR: {s["LRR_mean"]:+.2f} ± {s["LRR_std"]:.2f} m/yr\n'
           f'Erosion {s["pct_erosion"]:.0f}% / Stable {s["pct_stable"]:.0f}% / Accretion {s["pct_accretion"]:.0f}%\n'
           f'n = {s["n_transects"]} transects')
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top',
            fontsize=9.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#aaa'))

    ax.set_title(f'{island} — {SETTINGS[island]}', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlabel('Easting (m, UTM 45N)', fontsize=9)
    ax.set_ylabel('Northing (m, UTM 45N)', fontsize=9)
    ax.grid(True, alpha=0.25)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                    fraction=0.04, pad=0.06, shrink=0.7)
cbar.set_label('Linear Regression Rate, LRR (m/yr)\n← Erosion       Accretion →',
               fontsize=11)

fig.suptitle('Shoreline Change Rates 1990–2024: 2×2 Paired-Site Design',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.savefig(FIGURES / 'day4_transects_2x2.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {FIGURES / "day4_transects_2x2.png"}')

# === FIGURE 2: EPR distribution boxplots, 2x2 layout ===
fig, ax = plt.subplots(figsize=(10, 6))
data = []; positions_x = [1, 2, 4, 5]; colors = []
labels = []
for i, island in enumerate(ISLANDS):
    tr = gpd.read_file(DSAS_OUT / f'transects_{island.lower()}.shp')
    data.append(tr['EPR'].values)
    labels.append(f'{island}\n({SETTINGS[island].split(" • ")[1]})')
    colors.append('#4a7cb8' if island in ('Kumirmari', 'Mousuni') else '#7fa55c')

bp = ax.boxplot(data, positions=positions_x, widths=0.7, patch_artist=True,
                showfliers=False, medianprops=dict(color='black', linewidth=1.5))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)

ax.axhline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.5, label='Stable (EPR = 0)')
ax.set_xticks(positions_x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('End Point Rate, EPR (m/yr)', fontsize=11)
ax.set_title('Distribution of EPR per Island (1990–2024)', fontsize=12, fontweight='bold')

# Group labels
ax.text(1.5, ax.get_ylim()[1]*0.92, 'INTERIOR', ha='center', fontsize=11, fontweight='bold', color='#444')
ax.text(4.5, ax.get_ylim()[1]*0.92, 'COASTAL',  ha='center', fontsize=11, fontweight='bold', color='#444')
ax.axvline(3, color='gray', linestyle=':', alpha=0.5)
ax.grid(True, alpha=0.25, axis='y'); ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(FIGURES / 'day4_boxplot_epr.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {FIGURES / "day4_boxplot_epr.png"}')

# === MOUSUNI EAST-COAST VALIDATION (tighter check vs published -23.55) ===
print('\n=== Mousuni east-coast validation ===')
mousuni = gpd.read_file(DSAS_OUT / 'transects_mousuni.shp')
# East-facing = azimuth 45-135° (transects pointing east-ish from island interior)
east = mousuni[(mousuni['azimuth'] > 45) & (mousuni['azimuth'] < 135)]
south = mousuni[(mousuni['azimuth'] > 135) & (mousuni['azimuth'] < 225)]
print(f'  East-facing transects: n={len(east)} of {len(mousuni)}')
print(f'    Mean EPR:   {east["EPR"].mean():+.2f} m/yr')
print(f'    Median EPR: {east["EPR"].median():+.2f} m/yr')
print(f'    Min EPR:    {east["EPR"].min():+.2f} m/yr')
print(f'    % eroding:  {100*(east["EPR"] < -0.5).mean():.0f}%')
print(f'  South-facing transects: n={len(south)}')
print(f'    Mean EPR:   {south["EPR"].mean():+.2f} m/yr')
print(f'  Published Mousuni east-coast erosion: -23.55 m/yr')
print(f'  Validation match if east-facing mean is between -15 and -25 m/yr')

# === 2×2 INTERACTION ANALYSIS (preliminary, before formal stats on Day 6) ===
print('\n=== 2×2 interaction analysis (preliminary) ===')
print('\nMain effect — ANTHROPOGENIC (Inhabited − Uninhabited):')
int_an = summary['Kumirmari']['EPR_mean'] - summary['Hamilton']['EPR_mean']
cst_an = summary['Mousuni']['EPR_mean']   - summary['Lothian']['EPR_mean']
print(f'  Interior:  {int_an:+.2f} m/yr  (Kumirmari − Hamilton)')
print(f'  Coastal:   {cst_an:+.2f} m/yr  (Mousuni − Lothian)')

print('\nMain effect — GEOGRAPHIC (Interior − Coastal):')
inh_geo = summary['Kumirmari']['EPR_mean'] - summary['Mousuni']['EPR_mean']
uni_geo = summary['Hamilton']['EPR_mean']  - summary['Lothian']['EPR_mean']
print(f'  Inhabited:    {inh_geo:+.2f} m/yr  (Kumirmari − Mousuni)')
print(f'  Uninhabited:  {uni_geo:+.2f} m/yr  (Hamilton − Lothian)')

inter = int_an - cst_an
print(f'\nINTERACTION (anthropogenic effect difference between settings):')
print(f'  Δ = {inter:+.2f} m/yr')
print(f'  Sign of anthropogenic effect FLIPS between settings:')
print(f'    Interior: human presence → +{int_an:.2f} m/yr (protective)')
print(f'    Coastal:  human presence → {cst_an:.2f} m/yr (associated with erosion)')
print(f'  Formal Mann-Whitney U + KS tests on Day 6.')

# Copy to Windows
desktop = Path('/mnt/c/Users/anubg/Desktop')
if desktop.exists():
    for f in FIGURES.glob('day4_*.png'):
        shutil.copy(f, desktop / f.name)
        print(f'  Copied: {f.name}')
