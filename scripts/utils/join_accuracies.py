"""Read results/latency_rpi.csv from Pi, join with per-task accuracies,
write results/hwnas_bench_360_v1.csv with acc column added.
"""
from pathlib import Path
import csv, argparse

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.utils.task_specs import TASKS, load_accuracies


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--lat",  default=str(ROOT / "results" / "latency_rpi.csv"))
  ap.add_argument("--out",  default=str(ROOT / "results" / "hwnas_bench_360_v1.csv"))
  args = ap.parse_args()

  print("loading accuracies...", flush=True)
  accs = {}
  for t in TASKS:
    a = load_accuracies(t)
    accs[t] = a or {}
    print(f"  {t}: {len(accs[t])} entries", flush=True)

  with open(args.lat, newline="") as f:
    rows = list(csv.DictReader(f))
  print(f"loaded {len(rows)} latency rows from {args.lat}", flush=True)

  out_cols = list(rows[0].keys()) + ["acc"]
  with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_cols)
    w.writeheader()
    for r in rows:
      arch_idx = int(r["arch_idx"])
      r["acc"] = accs.get(r["task"], {}).get(arch_idx, "")
      w.writerow(r)
  print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
  main()
