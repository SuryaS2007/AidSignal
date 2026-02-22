"""
build_data.py — Processes all CSVs in ./data/ and outputs ./data/countries.json
Run once before opening index.html: python build_data.py
"""

import pandas as pd
import numpy as np
import json
import math
import os
import re

DATA_DIR = 'data'
OUTPUT_FILE = f'{DATA_DIR}/countries.json'


def normalize_series(s):
    """Min-max normalize a pandas Series to [0, 1]. Returns 0 if constant."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def to_python(v):
    """Convert numpy scalars / NaN to JSON-safe Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    return v


def get_bucket(score):
    """Map a mismatch score [0-1] to a display bucket [1-5], or None."""
    if score is None:
        return None
    if score < 0.2:
        return 1
    if score < 0.4:
        return 2
    if score < 0.6:
        return 3
    if score < 0.8:
        return 4
    return 5


# ─────────────────────────────────────────────────────────────────────────────
# 1. GOVERNANCE RISK SCORE  (from Risk_Factor.ipynb logic)
#    Inputs: WGICSV.csv
#    Output: iso3 -> risk_index [0-1] (higher = worse governance / higher risk)
# ─────────────────────────────────────────────────────────────────────────────
print(">> Loading WGI governance risk data ...")
wgi_raw = pd.read_csv(f'{DATA_DIR}/WGICSV.csv')

TARGET_INDS = ['CC.EST', 'RL.EST', 'GE.EST', 'PV.EST']
WEIGHTS     = {'CC.EST': 0.4, 'RL.EST': 0.2, 'GE.EST': 0.2, 'PV.EST': 0.2}

wgi_filt = wgi_raw[wgi_raw['Indicator Code'].isin(TARGET_INDS)][
    ['Country Code', 'Country Name', 'Indicator Code', '2023']
].rename(columns={'2023': 'value'}).copy()

wgi_filt['value'] = pd.to_numeric(wgi_filt['value'], errors='coerce')

wgi_pivot = (
    wgi_filt
    .pivot_table(index=['Country Code', 'Country Name'],
                 columns='Indicator Code', values='value')
    .reset_index()
)
wgi_pivot.columns.name = None

for ind in TARGET_INDS:
    if ind in wgi_pivot.columns:
        mn = wgi_pivot[ind].min()
        mx = wgi_pivot[ind].max()
        wgi_pivot[f'{ind}_quality'] = (wgi_pivot[ind] - mn) / (mx - mn)
        wgi_pivot[f'{ind}_risk'] = 1 - wgi_pivot[f'{ind}_quality']
    else:
        wgi_pivot[f'{ind}_risk'] = 0.5  # neutral if missing

wgi_pivot['risk_index'] = sum(
    WEIGHTS[ind] * wgi_pivot[f'{ind}_risk'].fillna(0.5)
    for ind in TARGET_INDS
)

risk_df = wgi_pivot[['Country Code', 'Country Name', 'risk_index']].rename(
    columns={'Country Code': 'iso3', 'Country Name': 'name'}
)
print(f"  -> {len(risk_df)} countries/entities in WGI data")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEVERITY SCORE  (from hpc_hno_2026.csv)
#    Severity = population_in_need / total_population, normalised to [0-1]
# ─────────────────────────────────────────────────────────────────────────────
print(">> Loading HNO 2026 severity data ...")
hno = pd.read_csv(f'{DATA_DIR}/hpc_hno_2026.csv')

plan_rows = hno[hno['Cluster'] == 'ALL'][
    ['Country ISO3', 'Population', 'In Need']
].rename(columns={
    'Country ISO3': 'iso3',
    'Population': 'population',
    'In Need': 'population_in_need'
}).copy()

plan_rows['population'] = pd.to_numeric(plan_rows['population'], errors='coerce')
plan_rows['population_in_need'] = pd.to_numeric(plan_rows['population_in_need'], errors='coerce')
plan_rows = plan_rows.dropna(subset=['population_in_need'])

plan_rows['severity_raw'] = (
    plan_rows['population_in_need'] /
    plan_rows['population'].replace(0, np.nan)
)
plan_rows['severity_score'] = normalize_series(plan_rows['severity_raw'].fillna(
    plan_rows['severity_raw'].median()
))

severity_df = plan_rows[['iso3', 'population', 'population_in_need', 'severity_score']].copy()
print(f"  -> {len(severity_df)} countries with HNO severity data")

# Load supplemental population_in_need for countries not in HNO
print(">> Loading supplemental population_in_need data ...")
supp_path = f'{DATA_DIR}/population_supplement.csv'
if os.path.exists(supp_path):
    supp_df = pd.read_csv(supp_path)
    supp_df['population_in_need'] = pd.to_numeric(supp_df['population_in_need'], errors='coerce')
    # Only use supplement for iso3 codes NOT already in severity_df
    existing_iso3 = set(severity_df['iso3'])
    supp_new = supp_df[~supp_df['iso3'].isin(existing_iso3)].copy()
    supp_new['population'] = None
    supp_new['severity_score'] = None
    severity_df = pd.concat([severity_df, supp_new[['iso3', 'population', 'population_in_need', 'severity_score']]], ignore_index=True)
    print(f"  -> added {len(supp_new)} countries from supplement")
else:
    print("  -> no supplement file found, skipping")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FTS REQUIREMENTS / FUNDING  (mismatch components)
#    fts_requirements_funding_global.csv has a metadata row 2 -> skiprows=[1]
# ─────────────────────────────────────────────────────────────────────────────
print(">> Loading FTS requirements/funding data ...")
fts_raw = pd.read_csv(
    f'{DATA_DIR}/fts_requirements_funding_global.csv',
    skiprows=[1],        # skip the metadata header row
    dtype=str
)

# Drop any residual metadata rows (lines starting with '#')
fts_raw = fts_raw[~fts_raw['countryCode'].astype(str).str.startswith('#')].copy()

for col in ['year', 'requirements', 'funding']:
    fts_raw[col] = pd.to_numeric(fts_raw[col], errors='coerce')

fts_valid = fts_raw.dropna(subset=['year', 'requirements', 'funding'])
fts_valid = fts_valid[(fts_valid['requirements'] > 0) & (fts_valid['year'] <= 2026)]

# Aggregate per country × year (sum plans if multiple exist)
fts_agg = (
    fts_valid
    .groupby(['countryCode', 'year'], as_index=False)
    .agg(requirements=('requirements', 'sum'), funding=('funding', 'sum'))
)
fts_agg['percent_funded'] = (
    fts_agg['funding'] / fts_agg['requirements'] * 100
).round(1)

# Keep most recent year per country
fts_latest = (
    fts_agg
    .sort_values('year', ascending=False)
    .groupby('countryCode', as_index=False)
    .first()
    .rename(columns={'countryCode': 'iso3'})
)

fts_latest['gap']     = fts_latest['requirements'] - fts_latest['funding']
fts_latest['inv_fpp'] = (1 - fts_latest['funding'] / fts_latest['requirements']).clip(0, 1)

print(f"  -> {len(fts_latest)} countries with FTS requirements data")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CBPF ALLOCATIONS  (partial: 5 countries only)
# ─────────────────────────────────────────────────────────────────────────────
print(">> Loading CBPF allocation data ...")
alloc_raw = pd.read_csv(f'{DATA_DIR}/Allocations__20260221_042047_UTC.csv')

FUND_TO_ISO3 = {
    'DRC':               'COD',
    'Ethiopia':          'ETH',
    'Mozambique (RhPF)': 'MOZ',
    'Sudan':             'SDN',
    'Fiji (AP-Rhpf)':   'FJI',
}
alloc_raw['iso3'] = alloc_raw['PooledFund'].map(FUND_TO_ISO3)
alloc_clean = alloc_raw.dropna(subset=['iso3'])
alloc_df = (
    alloc_clean
    .groupby('iso3')['Budget']
    .sum()
    .reset_index()
    .rename(columns={'Budget': 'cbpf_allocation'})
)
print(f"  -> {len(alloc_df)} countries with CBPF allocations")


# ─────────────────────────────────────────────────────────────────────────────
# 4.5. MARKET VOLATILITY SCORE  (from IMF CPI data)
#    Formula: 0.6·Level_norm + 0.4·Vol_norm  (2024 monthly MoM % change)
#    ISO3 extracted from SERIES_CODE prefix
# ─────────────────────────────────────────────────────────────────────────────
print(">> Loading IMF CPI market volatility data ...")
imf_candidates = [f for f in os.listdir(DATA_DIR) if f.startswith('dataset_') and 'IMF' in f and f.endswith('.csv')]
if imf_candidates:
    imf_path = f'{DATA_DIR}/{imf_candidates[0]}'
    imf_raw = pd.read_csv(imf_path, low_memory=False)
    imf_mom = imf_raw[
        imf_raw['FREQUENCY'].astype(str).str.lower().eq('monthly') &
        imf_raw['TYPE_OF_TRANSFORMATION'].astype(str).str.contains(
            'period-over-period percent change', case=False, na=False)
    ].copy()
    imf_mom['iso3'] = imf_mom['SERIES_CODE'].str.split('.').str[0]
    month_cols_2024 = [c for c in imf_raw.columns if re.match(r'^2024-M\d{2}$', str(c))]
    imf_long = imf_mom.melt(
        id_vars=['iso3'], value_vars=month_cols_2024,
        var_name='period', value_name='mom_inflation'
    )
    imf_long['mom_inflation'] = pd.to_numeric(imf_long['mom_inflation'], errors='coerce')
    imf_long = imf_long.dropna(subset=['mom_inflation'])
    imf_agg = imf_long.groupby('iso3', as_index=False).agg(
        inflation_level=('mom_inflation', lambda x: float(np.mean(np.abs(x)))),
        inflation_vol=('mom_inflation', lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
        months=('mom_inflation', 'count')
    )
    imf_agg.loc[imf_agg['months'] < 6, 'inflation_vol'] = np.nan
    imf_agg['level_norm'] = normalize_series(pd.to_numeric(imf_agg['inflation_level'], errors='coerce'))
    imf_agg['vol_norm']   = normalize_series(pd.to_numeric(imf_agg['inflation_vol'],   errors='coerce').fillna(
        pd.to_numeric(imf_agg['inflation_vol'], errors='coerce').median()
    ))
    imf_agg['market_volatility_score'] = (0.6 * imf_agg['level_norm'] + 0.4 * imf_agg['vol_norm']).round(4)
    mv_df = imf_agg[['iso3', 'market_volatility_score']].dropna(subset=['market_volatility_score'])
    print(f"  -> market volatility scores for {len(mv_df)} countries")
else:
    mv_df = pd.DataFrame(columns=['iso3', 'market_volatility_score'])
    print("  -> IMF CPI file not found, skipping market volatility")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MISMATCH SCORE  (from Mismatch Index.ipynb logic, adapted for local use)
#    Formula: 0.4·Severity + 0.3·Gap + 0.2·InvFPP + 0.1·InvPFPP
#    All components normalised to [0-1] before weighting
# ─────────────────────────────────────────────────────────────────────────────
print(">> Computing mismatch scores ...")
mis = (
    fts_latest
    .merge(severity_df[['iso3', 'severity_score', 'population', 'population_in_need']],
           on='iso3', how='left')
    .merge(alloc_df, on='iso3', how='left')
)

mis['cbpf_allocation'] = mis['cbpf_allocation'].fillna(0)
mis['severity_score']  = mis['severity_score'].fillna(0)

# InvPFPP: how much of funding came from pooled funds (0 if no allocation)
safe_funding = mis['funding'].replace(0, np.nan)
mis['pfpp']     = (mis['cbpf_allocation'] / safe_funding).fillna(0)
mis['inv_pfpp'] = (1 - mis['pfpp']).clip(0, 1)

# Normalise all components
mis['sev_norm']      = normalize_series(mis['severity_score'])
mis['gap_norm']      = normalize_series(mis['gap'].clip(lower=0))
mis['inv_fpp_norm']  = normalize_series(mis['inv_fpp'])
mis['inv_pfpp_norm'] = normalize_series(mis['inv_pfpp'])

mis['mismatch_score'] = (
    0.4 * mis['sev_norm'] +
    0.3 * mis['gap_norm'] +
    0.2 * mis['inv_fpp_norm'] +
    0.1 * mis['inv_pfpp_norm']
)

print(f"  -> mismatch scores computed for {len(mis)} countries")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ACTIVE CRISES / CLUSTERS  (from fts_requirements_funding_cluster_global.csv)
#    Distinct cluster names per country, most recent year only
# ─────────────────────────────────────────────────────────────────────────────
print(">> Loading FTS cluster / crisis data ...")
cl_raw = pd.read_csv(
    f'{DATA_DIR}/fts_requirements_funding_cluster_global.csv',
    skiprows=[1],
    dtype=str
)

cl_raw = cl_raw[~cl_raw['countryCode'].astype(str).str.startswith('#')].copy()
cl_raw['year'] = pd.to_numeric(cl_raw['year'], errors='coerce')
cl_raw = cl_raw.dropna(subset=['year', 'cluster'])
cl_raw = cl_raw[
    (~cl_raw['cluster'].isin(['Not specified', ''])) &
    (cl_raw['year'] <= 2026)
]

# Keep clusters from the most recent year per country
max_year_per_country = (
    cl_raw.groupby('countryCode')['year']
    .max()
    .reset_index()
    .rename(columns={'year': 'max_year'})
)
cl_filtered = cl_raw.merge(max_year_per_country, on='countryCode')
cl_filtered = cl_filtered[cl_filtered['year'] == cl_filtered['max_year']]

clusters_df = (
    cl_filtered
    .groupby('countryCode')['cluster']
    .apply(lambda x: sorted(list(x.dropna().unique())))
    .reset_index()
    .rename(columns={'countryCode': 'iso3', 'cluster': 'clusters'})
)
print(f"  -> cluster data for {len(clusters_df)} countries")


# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD FINAL COUNTRY DATASET
#    Base: all WGI countries (risk scores) -> merge mismatch + clusters
# ─────────────────────────────────────────────────────────────────────────────
print(">> Assembling final country dataset ...")

final = risk_df.merge(
    mis[['iso3', 'mismatch_score', 'requirements', 'funding',
         'percent_funded', 'population', 'population_in_need', 'gap']],
    on='iso3', how='left'
).merge(
    clusters_df, on='iso3', how='left'
).merge(
    mv_df, on='iso3', how='left'
)

records = []
for _, row in final.iterrows():
    ms_raw   = to_python(row.get('mismatch_score'))
    risk_raw = to_python(row.get('risk_index'))

    ms_val   = round(ms_raw, 4) if ms_raw is not None else None
    risk_val = round(risk_raw * 100, 1) if risk_raw is not None else None

    req = row.get('requirements')
    fund = row.get('funding')
    pin  = row.get('population_in_need')

    clusters_cell = row.get('clusters')
    clusters_val  = clusters_cell if isinstance(clusters_cell, list) else []

    mv_raw = row.get('market_volatility_score')
    mv_val = round(float(mv_raw), 4) if pd.notna(mv_raw) else None

    record = {
        'iso3':                    str(row['iso3']),
        'name':                    str(row['name']),
        'mismatch_score':          ms_val,
        'mismatch_bucket':         get_bucket(ms_val),
        'risk_score':              risk_val,
        'requirements':            int(req)  if pd.notna(req)  else None,
        'funding':                 int(fund) if pd.notna(fund) else None,
        'percent_funded':          to_python(row.get('percent_funded')),
        'population_in_need':      int(pin)  if pd.notna(pin)  else None,
        'clusters':                clusters_val,
        'market_volatility_score': mv_val,
    }
    records.append(record)

os.makedirs(DATA_DIR, exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, separators=(',', ':'))

# ─── Summary ─────────────────────────────────────────────────────────────────
total          = len(records)
has_mismatch   = sum(1 for r in records if r['mismatch_score'] is not None)
has_clusters   = sum(1 for r in records if r['clusters'])
bucket_counts  = {i: sum(1 for r in records if r['mismatch_bucket'] == i) for i in range(1, 6)}

print(f"\nOK  Generated {OUTPUT_FILE}")
print(f"   {total} total countries / entities")
print(f"   {has_mismatch} with mismatch scores (will be coloured on map)")
print(f"   {total - has_mismatch} without mismatch data (will be grey on map)")
print(f"   {has_clusters} with active crises / clusters")
print(f"   Bucket distribution: {bucket_counts}")
