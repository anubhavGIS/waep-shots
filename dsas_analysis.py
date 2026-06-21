"""Day 4 Step 2: DSAS-equivalent transect analysis (Himmelstoss et al. 2018) in Python."""
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from scipy import stats

BASE_DIR    = Path.home() / 'sundarbans_paper'
DSAS_IN     = BASE_DIR / 'data' / 'dsas_input'
DSAS_OUT    = BASE_DIR / 'data' / 'dsas_output'
DSAS_OUT.mkdir(exist_ok=True)
ANCILLARY   = BASE_DIR / 'data' / 'ancillary'

TRANSECT_SPACING_M = 50    # 50 m along baseline
TRANSECT_LENGTH_M  = 3000  # 3 km outward (well past historical shoreline range)
MIN_INTERSECTIONS  = 6     # require ≥6 of 12 epochs for stable LRR

islands_gdf = gpd.read_file(ANCILLARY / 'four_islands.shp').to_crs('EPSG:32645')
ISLAND_CENTROIDS = {row['island']: row.geometry.centroid for _, row in islands_gdf.iterrows()}

def generate_transects(baseline, spacing_m, length_m, centroid):
    """Generate transects perpendicular to baseline, pointing outward (away from centroid)."""
    transects = []
    total = baseline.length
    n = int(total / spacing_m) + 1
    delta = 5.0
    for i in range(n):
        d = min(i * spacing_m, total - 1)
        pt = baseline.interpolate(d)
        pt_b = baseline.interpolate(max(0, d - delta))
        pt_a = baseline.interpolate(min(total, d + delta))
        tx, ty = pt_a.x - pt_b.x, pt_a.y - pt_b.y
        norm = np.hypot(tx, ty)
        if norm < 1e-6: continue
        tx, ty = tx/norm, ty/norm
        # Two perpendicular options
        nx1, ny1 = -ty,  tx
        nx2, ny2 =  ty, -tx
        # Pick the one pointing AWAY from centroid (outward)
        dcx, dcy = pt.x - centroid.x, pt.y - centroid.y
        dot1 = nx1*dcx + ny1*dcy
        dot2 = nx2*dcx + ny2*dcy
        nx, ny = (nx1, ny1) if dot1 > dot2 else (nx2, ny2)
        end = Point(pt.x + nx*length_m, pt.y + ny*length_m)
        transect = LineString([pt, end])
        azimuth = (np.degrees(np.arctan2(nx, ny)) + 360) % 360
        transects.append({
            'transect_id': i, 'origin_x': pt.x, 'origin_y': pt.y,
            'azimuth': azimuth, 'geometry': transect,
        })
    return transects

def transect_distance(transect, shoreline):
    """First intersection distance from transect origin (closest crossing seaward)."""
    if not transect.intersects(shoreline): return None
    isec = transect.intersection(shoreline)
    origin = Point(transect.coords[0])
    if isec.geom_type == 'Point':
        return float(origin.distance(isec))
    elif isec.geom_type == 'MultiPoint':
        return float(min(origin.distance(p) for p in isec.geoms))
    elif isec.geom_type == 'LineString':
        return float(origin.distance(Point(isec.coords[0])))
    elif isec.geom_type == 'GeometryCollection':
        # Pick the closest Point
        dists = []
        for g in isec.geoms:
            if g.geom_type == 'Point':
                dists.append(float(origin.distance(g)))
        return min(dists) if dists else None
    return None

def compute_rates(distances, dates, uncertainties):
    """NSM, SCE, EPR, LRR, LR², LSE, LCI95, WLR, WCI95."""
    order = np.argsort([d.toordinal() for d in dates])
    dates = [dates[i] for i in order]
    d = np.array([distances[i] for i in order])
    u = np.array([uncertainties[i] for i in order])
    yr = np.array([(dt - dates[0]).days / 365.25 for dt in dates])
    n = len(d)
    if n < 2: return None

    nsm = float(d[-1] - d[0])
    sce = float(d.max() - d.min())
    epr = nsm / yr[-1] if yr[-1] > 0 else 0.0

    if n >= 3:
        # OLS
        slope, intercept = np.polyfit(yr, d, 1)
        pred = slope*yr + intercept
        rss = float(np.sum((d - pred)**2))
        ss_tot = float(np.sum((d - d.mean())**2))
        lr2 = 1.0 - rss/ss_tot if ss_tot > 1e-6 else 1.0
        x_var = float(np.sum((yr - yr.mean())**2))
        if n > 2 and x_var > 1e-6:
            lse = float(np.sqrt(rss/(n-2)/x_var))
            lci = float(stats.t.ppf(0.975, n-2) * lse)
        else:
            lse = lci = float('nan')
        lrr = float(slope)
        # WLS (weights = 1/UNCY²)
        try:
            w = 1.0 / u**2
            W = np.diag(w)
            X = np.vstack([yr, np.ones(n)]).T
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ d)
            wlr = float(beta[0])
            cov = np.linalg.inv(X.T @ W @ X)
            wse = float(np.sqrt(cov[0,0]))
            wci = float(stats.t.ppf(0.975, n-2) * wse)
        except np.linalg.LinAlgError:
            wlr = wse = wci = float('nan')
    else:
        lrr = epr; lr2 = lse = lci = wlr = wse = wci = float('nan')

    return {
        'n_obs': n, 'time_span': float(yr[-1]),
        'NSM': nsm, 'SCE': sce, 'EPR': epr,
        'LRR': lrr, 'LR2': lr2, 'LSE': lse, 'LCI95': lci,
        'WLR': wlr, 'WSE': wse, 'WCI95': wci,
    }

summary = {}

for island in ['Kumirmari', 'Hamilton', 'Mousuni', 'Lothian']:
    print(f'\n=== {island} ===')
    sl_gdf = gpd.read_file(DSAS_IN / f'shorelines_{island.lower()}.shp')
    bl_gdf = gpd.read_file(DSAS_IN / f'baseline_{island.lower()}.shp')
    baseline = bl_gdf.iloc[0].geometry
    dates = [datetime.strptime(d, '%m/%d/%Y') for d in sl_gdf['DATE_']]
    uncertainties = sl_gdf['UNCY'].values
    shorelines = sl_gdf.geometry.values
    centroid = ISLAND_CENTROIDS[island]
    print(f'  Shorelines: {len(sl_gdf)}, baseline: {baseline.length/1000:.2f} km')
    transects = generate_transects(baseline, TRANSECT_SPACING_M, TRANSECT_LENGTH_M, centroid)
    print(f'  Generated {len(transects)} transects')

    rows = []
    for t in transects:
        dists = []; dts = []; uncs = []
        for i, sl in enumerate(shorelines):
            d = transect_distance(t['geometry'], sl)
            if d is not None:
                dists.append(d); dts.append(dates[i]); uncs.append(uncertainties[i])
        if len(dists) >= MIN_INTERSECTIONS:
            rates = compute_rates(dists, dts, uncs)
            if rates:
                row = {**t, **rates, 'n_intersections': len(dists)}
                rows.append(row)
    print(f'  Valid transects: {len(rows)} / {len(transects)} (≥{MIN_INTERSECTIONS} epochs)')

    rates_gdf = gpd.GeoDataFrame(rows, crs='EPSG:32645', geometry='geometry')
    out_shp = DSAS_OUT / f'transects_{island.lower()}.shp'
    rates_gdf.to_file(out_shp, driver='ESRI Shapefile')

    epr = rates_gdf['EPR'].values
    lrr = rates_gdf['LRR'].values
    nsm = rates_gdf['NSM'].values
    wlr = rates_gdf['WLR'].dropna().values

    summary[island] = {
        'n_transects': int(len(rates_gdf)),
        'baseline_length_km': float(baseline.length/1000),
        'NSM_mean': float(nsm.mean()), 'NSM_std': float(nsm.std()),
        'NSM_median': float(np.median(nsm)),
        'EPR_mean': float(epr.mean()), 'EPR_std': float(epr.std()),
        'EPR_median': float(np.median(epr)),
        'EPR_min': float(epr.min()), 'EPR_max': float(epr.max()),
        'EPR_p05': float(np.percentile(epr, 5)), 'EPR_p95': float(np.percentile(epr, 95)),
        'LRR_mean': float(lrr.mean()), 'LRR_std': float(lrr.std()),
        'WLR_mean': float(wlr.mean()) if len(wlr) else None,
        'WLR_std':  float(wlr.std())  if len(wlr) else None,
        'pct_erosion':   float(100 * (epr < -0.5).sum() / len(epr)),
        'pct_stable':    float(100 * (np.abs(epr) <= 0.5).sum() / len(epr)),
        'pct_accretion': float(100 * (epr > 0.5).sum() / len(epr)),
    }
    s = summary[island]
    print(f'  EPR: {s["EPR_mean"]:+6.2f} ± {s["EPR_std"]:.2f} m/yr  '
          f'(range [{s["EPR_min"]:+.2f}, {s["EPR_max"]:+.2f}], median {s["EPR_median"]:+.2f})')
    print(f'  LRR: {s["LRR_mean"]:+6.2f} ± {s["LRR_std"]:.2f} m/yr')
    print(f'  WLR: {s["WLR_mean"]:+6.2f} ± {s["WLR_std"]:.2f} m/yr')
    print(f'  Classification: {s["pct_erosion"]:.0f}% eroding / {s["pct_stable"]:.0f}% stable / {s["pct_accretion"]:.0f}% accreting')

with open(DSAS_OUT / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSaved: {DSAS_OUT}/summary.json + 4 transect shapefiles')

# === 2×2 RESULTS TABLE ===
print(f'\n{"="*72}')
print('2×2 FACTORIAL: SHORELINE CHANGE RATES (m/yr, mean ± std EPR)')
print(f'{"="*72}')
print(f'{"":18}{"Inhabited":>26}{"Uninhabited":>26}')
i = summary
print(f'{"INTERIOR":18}{i["Kumirmari"]["EPR_mean"]:+8.2f} ± {i["Kumirmari"]["EPR_std"]:5.2f} (Kumirmari)'
      f'  {i["Hamilton"]["EPR_mean"]:+8.2f} ± {i["Hamilton"]["EPR_std"]:5.2f} (Hamilton)')
print(f'{"COASTAL":18}{i["Mousuni"]["EPR_mean"]:+8.2f} ± {i["Mousuni"]["EPR_std"]:5.2f} (Mousuni) '
      f'  {i["Lothian"]["EPR_mean"]:+8.2f} ± {i["Lothian"]["EPR_std"]:5.2f} (Lothian)')
print(f'{"="*72}')

# Published validation
print('\nPublished erosion rate at Mousuni east coast (literature): -23.55 m/yr')
print(f'Our analysis at Mousuni (all transects):   EPR mean = {summary["Mousuni"]["EPR_mean"]:+.2f} m/yr,  '
      f'min = {summary["Mousuni"]["EPR_min"]:+.2f} m/yr')
print('Match expected if our minimum (most-eroded transect) is close to -23.55')
