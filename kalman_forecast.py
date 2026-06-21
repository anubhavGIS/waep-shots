"""Day 5: Kalman filter shoreline forecasts for 2034 and 2044."""
import json, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import geopandas as gpd
from shapely.geometry import Point

BASE = Path.home() / 'sundarbans_paper'
DSAS_IN, DSAS_OUT = BASE/'data'/'dsas_input', BASE/'data'/'dsas_output'
KOUT = BASE/'data'/'kalman_output'; KOUT.mkdir(exist_ok=True)
ANC, FIGS = BASE/'data'/'ancillary', BASE/'figures'
TARGETS = [2034, 2044]
ISLANDS = ['Kumirmari', 'Hamilton', 'Mousuni', 'Lothian']
SETTINGS = {'Kumirmari': 'Interior . Inhabited', 'Hamilton': 'Interior . Uninhabited',
            'Mousuni': 'Coastal . Inhabited', 'Lothian': 'Coastal . Uninhabited'}

def to_dec_year(dt):
    return dt.year + (dt - datetime(dt.year, 1, 1)).days / (datetime(dt.year+1, 1, 1) - datetime(dt.year, 1, 1)).days

def tr_dist(t, sl):
    if not t.intersects(sl): return None
    i = t.intersection(sl); o = Point(t.coords[0])
    if i.geom_type == 'Point': return float(o.distance(i))
    if i.geom_type == 'MultiPoint': return float(min(o.distance(p) for p in i.geoms))
    if i.geom_type == 'LineString': return float(o.distance(Point(i.coords[0])))
    if i.geom_type == 'GeometryCollection':
        ds = [float(o.distance(g)) for g in i.geoms if g.geom_type == 'Point']
        return min(ds) if ds else None
    return None

def kalman(yrs, dists, uncs, targets):
    """2D Kalman filter, constant-velocity state model. State x = [position, velocity]."""
    n = len(dists)
    if n < 3: return None
    y = np.asarray(yrs, float); d = np.asarray(dists, float); u = np.asarray(uncs, float)
    x = np.array([d[0], 0.0])
    P = np.array([[u[0]**2, 0.0], [0.0, 100.0]])
    Q = np.array([[1.0, 0.0], [0.0, 0.25]])
    H = np.array([[1.0, 0.0]])
    for i in range(1, n):
        dt = y[i] - y[i-1]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        x = F @ x; P = F @ P @ F.T + Q
        innov = d[i] - (H @ x)[0]
        S = (H @ P @ H.T)[0, 0] + u[i]**2
        K = (P @ H.T).flatten() / S
        x = x + K * innov
        P = (np.eye(2) - np.outer(K, H[0])) @ P
    result = {'cur': {'year': float(y[-1]), 'pos': float(x[0]), 'vel': float(x[1])}}
    for t in targets:
        dt = t - y[-1]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        x_f = F @ x; P_f = F @ P @ F.T + Q * abs(dt)
        sig = float(np.sqrt(P_f[0, 0]))
        result[t] = {'pos': float(x_f[0]), 'sigma': sig, 'change': float(x_f[0] - x[0])}
    return result

all_results = {}; all_dfs = {}

for island in ISLANDS:
    print('\n=== ' + island + ' ===')
    sl = gpd.read_file(DSAS_IN / ('shorelines_' + island.lower() + '.shp'))
    tr = gpd.read_file(DSAS_OUT / ('transects_' + island.lower() + '.shp'))
    sl_years = [to_dec_year(datetime.strptime(d, '%m/%d/%Y')) for d in sl['DATE_']]
    sl_uncs = sl['UNCY'].values
    shorelines = sl.geometry.values

    rows = []
    for _, t in tr.iterrows():
        ds, ys, us = [], [], []
        for i, ln in enumerate(shorelines):
            d = tr_dist(t.geometry, ln)
            if d is not None:
                ds.append(d); ys.append(sl_years[i]); us.append(sl_uncs[i])
        if len(ds) < 6: continue
        o = np.argsort(ys)
        ys, ds, us = [ys[k] for k in o], [ds[k] for k in o], [us[k] for k in o]
        fc = kalman(ys, ds, us, TARGETS)
        if fc is None: continue
        az = np.radians(t['azimuth']); nx, ny = np.sin(az), np.cos(az)
        rows.append({
            'transect_i': t['transect_i'], 'origin_x': t['origin_x'], 'origin_y': t['origin_y'],
            'azimuth': t['azimuth'], 'EPR_obs': t['EPR'], 'LRR_obs': t['LRR'],
            'velocity_k': fc['cur']['vel'],
            'cur_x': t['origin_x'] + fc['cur']['pos']*nx, 'cur_y': t['origin_y'] + fc['cur']['pos']*ny,
            'x_2034': t['origin_x'] + fc[2034]['pos']*nx, 'y_2034': t['origin_y'] + fc[2034]['pos']*ny,
            'sigma_2034': fc[2034]['sigma'], 'change_2034': fc[2034]['change'],
            'x_2044': t['origin_x'] + fc[2044]['pos']*nx, 'y_2044': t['origin_y'] + fc[2044]['pos']*ny,
            'sigma_2044': fc[2044]['sigma'], 'change_2044': fc[2044]['change'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(KOUT / ('forecast_' + island.lower() + '.csv'), index=False)
    all_dfs[island] = df

    c34, c44 = df['change_2034'].values, df['change_2044'].values
    v_k, lrr = df['velocity_k'].values, df['LRR_obs'].values
    print(f'  Forecasts: {len(df)} / {len(tr)} transects')
    print(f'  Velocity check: Kalman={v_k.mean():+.2f}, LRR={lrr.mean():+.2f} m/yr')
    print(f'  D 2024->2034: {c34.mean():+.1f} +/- {c34.std():.1f} m (median {np.median(c34):+.1f})')
    print(f'  D 2024->2044: {c44.mean():+.1f} +/- {c44.std():.1f} m (median {np.median(c44):+.1f})')

    all_results[island] = {
        'n_transects': int(len(df)),
        'velocity_kalman_mean': float(v_k.mean()), 'velocity_kalman_std': float(v_k.std()),
        'change_2034_mean': float(c34.mean()), 'change_2034_std': float(c34.std()),
        'change_2044_mean': float(c44.mean()), 'change_2044_std': float(c44.std()),
        'pct_eroding_2034': float(100 * (c34 < -1).mean()),
        'pct_eroding_2044': float(100 * (c44 < -1).mean()),
    }

with open(KOUT / 'forecast_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print('\n' + '='*82)
print('KALMAN FORECAST SUMMARY (per-island mean +/- std)')
print('='*82)
print(f'{"Island":12} {"Kalman v":>16} {"D 2024->2034":>20} {"D 2024->2044":>20} {"% eroded":>10}')
print('-'*82)
for island in ISLANDS:
    r = all_results[island]
    print(f'{island:12} {r["velocity_kalman_mean"]:+6.2f} +/- {r["velocity_kalman_std"]:4.2f}   '
          f'{r["change_2034_mean"]:+7.1f} +/- {r["change_2034_std"]:5.1f}   '
          f'{r["change_2044_mean"]:+7.1f} +/- {r["change_2044_std"]:5.1f}   '
          f'{r["pct_eroding_2044"]:6.0f}%')
print('='*82)

# === VISUALIZATION ===
islands_gdf = gpd.read_file(ANC / 'four_islands.shp').to_crs('EPSG:32645')
fig = plt.figure(figsize=(17, 17))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22,
                       top=0.94, bottom=0.05, left=0.06, right=0.97)
axes = {island: fig.add_subplot(gs[i // 2, i % 2]) for i, island in enumerate(ISLANDS)}

for island in ISLANDS:
    ax = axes[island]
    poly = islands_gdf[islands_gdf['island'] == island].iloc[0].geometry
    for g in ([poly] if poly.geom_type == 'Polygon' else poly.geoms):
        x, y = g.exterior.xy
        ax.fill(x, y, color='#d8d8d8', alpha=0.5, edgecolor='#444', linewidth=0.8, zorder=1)
    df = all_dfs[island].sort_values('transect_i').reset_index(drop=True)
    ax.scatter(df['cur_x'],  df['cur_y'],  s=4, color='black',   alpha=0.7, label='2024 (Kalman)', zorder=3)
    ax.scatter(df['x_2034'], df['y_2034'], s=4, color='#ff7f0e', alpha=0.7, label='2034 forecast', zorder=4)
    ax.scatter(df['x_2044'], df['y_2044'], s=4, color='#d62728', alpha=0.7, label='2044 forecast', zorder=5)
    r = all_results[island]
    txt = (f'D 2034: {r["change_2034_mean"]:+5.1f} +/- {r["change_2034_std"]:.1f} m\n'
           f'D 2044: {r["change_2044_mean"]:+5.1f} +/- {r["change_2044_std"]:.1f} m\n'
           f'Eroded by 2044: {r["pct_eroding_2044"]:.0f}% transects')
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#aaa'))
    ax.set_title(island + ' - ' + SETTINGS[island], fontsize=13, fontweight='bold', pad=8)
    ax.set_aspect('equal')
    ax.set_xlabel('Easting (m, UTM 45N)', fontsize=10)
    ax.set_ylabel('Northing (m, UTM 45N)', fontsize=10)
    ax.tick_params(axis='x', rotation=35, labelsize=9); ax.tick_params(axis='y', labelsize=9)
    ax.locator_params(axis='x', nbins=5); ax.locator_params(axis='y', nbins=6)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower right', fontsize=9, markerscale=2)

fig.suptitle('Kalman Filter Shoreline Forecasts: 2024 -> 2034 -> 2044',
             fontsize=15, fontweight='bold', y=0.98)
plt.savefig(FIGS / 'day5_kalman_forecasts.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved: ' + str(FIGS / 'day5_kalman_forecasts.png'))

dt = Path('/mnt/c/Users/anubg/Desktop')
if dt.exists():
    shutil.copy(FIGS / 'day5_kalman_forecasts.png', dt / 'day5_kalman_forecasts.png')
    print('Copied to desktop')
