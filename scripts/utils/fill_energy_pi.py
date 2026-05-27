#!/usr/bin/env python3
"""Fill energy_mj column in pi.csv from energy_pi.csv (J -> mJ)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENERGY = ROOT / "results" / "energy_pi.csv"
PI = ROOT / "results" / "pi.csv"
OUT = PI

energy_map = {}
with ENERGY.open() as f:
    for row in csv.DictReader(f):
        key = (row["arch_index"], row["task"], row["framework"])
        energy_map[key] = float(row["energy_per_inference_J"]) * 1000.0

with PI.open() as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

filled = 0
for row in rows:
    key = (row["arch_idx"], row["task"], row["runtime"])
    if key in energy_map:
        row["energy_mj"] = f"{energy_map[key]:.6f}"
        filled += 1

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"filled {filled}/{len(rows)} rows in {OUT}")
