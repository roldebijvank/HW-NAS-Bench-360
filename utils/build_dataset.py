"""Merge raw device CSVs, expand to all 15,625 archs via iso-map, and write results/dataset.parquet.

  python utils/build_dataset.py [--out results/dataset.parquet]
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


TARGET_COLS = [
    'arch_idx', 'device', 'framework', 'task',
    'lat_ms', 'lat_ms_var', 'energy_mj', 'accuracy'
]


def load_pi(path):
    df = pd.read_csv(path)
    df = df[df['status'] == 'ok'].copy()
    acc_cols = [c for c in df.columns if c.startswith('acc_')]
    df = df.drop(columns=acc_cols + ['status', 'error'], errors='ignore')
    df = df.rename(columns={'runtime': 'framework', 'lat_ms_median': 'lat_ms'})
    return df[['arch_idx', 'device', 'framework', 'task', 'lat_ms', 'lat_ms_var', 'energy_mj']]


def load_jetson(path):
    df = pd.read_csv(path)
    df = df[df['status'] == 'ok'].copy()
    df = df.rename(columns={
        'runtime': 'framework',
        'lat_ms_median': 'lat_ms',
        'energy_mj_net': 'energy_mj'
    })
    df = df.drop(
        columns=['energy_mj_raw', 'idle_power_mw', 'n_passes', 'status', 'error'],
        errors='ignore'
    )
    return df[['arch_idx', 'device', 'framework', 'task', 'lat_ms', 'lat_ms_var', 'energy_mj']]


def load_pixel(path):
    df = pd.read_csv(path)
    df = df[df['status'] == 'ok'].copy()
    df = df.rename(columns={'runtime': 'framework', 'lat_ms_median': 'lat_ms'})
    df['energy_mj'] = np.nan
    df = df.drop(columns=['status', 'error'], errors='ignore')
    return df[['arch_idx', 'device', 'framework', 'task', 'lat_ms', 'lat_ms_var', 'energy_mj']]


def build_iso_map(hw_pickle_path, measured_indices):
    # Arch strings come from the 10MB HW-NAS-Bench pickle (same 0..15624 NB201
    # indexing as the accuracy file), not the 5GB NAS-Bench-201 .pth API.
    import pickle
    import warnings
    from xautodl.models.cell_searchs.genotypes import Structure

    with open(hw_pickle_path, 'rb') as f, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d = pickle.load(f)
    configs = d['nasbench201']['cifar10']['config']
    items = configs.items() if isinstance(configs, dict) else enumerate(configs)

    measured = set(measured_indices)
    # canonical (isomorphism-invariant) str -> a *measured* representative idx
    canon_to_rep = {}
    arch_canon = {}
    for idx, cfg in items:
        idx = int(idx)
        canon = Structure.str2structure(cfg['arch_str']).to_unique_str(consider_zero=True)
        arch_canon[idx] = canon
        if idx in measured:
            canon_to_rep.setdefault(canon, idx)

    # every full idx -> measured rep of its iso class (None if class unmeasured)
    return {idx: canon_to_rep.get(canon) for idx, canon in arch_canon.items()}

def expand_isomorphic(df, iso_map):
    mapping = pd.DataFrame(
        {'full_idx': list(iso_map.keys()), 'arch_idx': list(iso_map.values())}
    ).dropna().astype(int)

    expanded = mapping.merge(df, on='arch_idx', how='left')
    expanded = expanded.drop(columns=['arch_idx']).rename(columns={'full_idx': 'arch_idx'})
    return expanded


def join_accuracy(df, acc_path):
    path = Path(acc_path)
    acc = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)

    # Accept either long (arch_idx, task, accuracy) or wide (arch_idx, acc_<task>).
    if 'accuracy' not in acc.columns:
        acc_cols = [c for c in acc.columns if c.startswith('acc_')]
        acc = acc.melt(
            id_vars='arch_idx', value_vars=acc_cols,
            var_name='task', value_name='accuracy'
        )
        acc['task'] = acc['task'].str.removeprefix('acc_')

    acc = acc[['arch_idx', 'task', 'accuracy']]
    # darcy: source stores eval_acc1es = 1 - rel-L2; convert back to raw
    # relative L2 error (lower is better) so the dataset carries the real metric.
    darcy = acc['task'] == 'darcy'
    acc.loc[darcy, 'accuracy'] = 1.0 - acc.loc[darcy, 'accuracy']
    # accuracy file is keyed by full arch_idx, so merge directly (no iso routing).
    return df.merge(acc, on=['arch_idx', 'task'], how='left')


def main():
    parser = argparse.ArgumentParser(description='Build HW-NAS-Bench-360 Parquet dataset.')
    parser.add_argument('--pi', required=True, help='Path to pi.csv')
    parser.add_argument('--jetson', required=True, help='Path to jetson.csv')
    parser.add_argument('--pixel', required=True, help='Path to pixel.csv')
    parser.add_argument('--accuracy', required=True,
                        help='CSV or Parquet with columns arch_idx, task, accuracy')
    parser.add_argument('--hw-pickle', default=None,
                        help='Path to HW-NAS-Bench pickle (.pickle); required if --iso-map not given')
    parser.add_argument('--iso-map', default=None,
                        help='Path to precomputed iso_map JSON (optional)')
    parser.add_argument('--out', default='dataset.parquet', help='Output Parquet path')
    args = parser.parse_args()

    if not args.iso_map and not args.hw_pickle:
        parser.error('Either --iso-map or --hw-pickle must be provided.')

    pi = load_pi(args.pi)
    jetson = load_jetson(args.jetson)
    pixel = load_pixel(args.pixel)
    combined = pd.concat([pi, jetson, pixel], ignore_index=True)

    measured_indices = set(combined['arch_idx'].unique())

    if args.iso_map:
        with open(args.iso_map) as f:
            raw = json.load(f)
        iso_map = {int(k): (int(v) if v is not None else None) for k, v in raw.items()}
    else:
        iso_map = build_iso_map(args.hw_pickle, measured_indices)

    expanded = expand_isomorphic(combined, iso_map)
    expanded = join_accuracy(expanded, args.accuracy)

    expanded = (
        expanded[TARGET_COLS]
        .sort_values(['arch_idx', 'device', 'framework', 'task'])
        .reset_index(drop=True)
    )

    expanded.to_parquet(args.out, index=False)
    n_archs = expanded['arch_idx'].nunique()
    print(f"Wrote {len(expanded)} rows ({n_archs} unique arch indices) to {args.out}")


if __name__ == '__main__':
    main()