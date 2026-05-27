"""Check pi.csv and latency_pixel.csv have all non-iso arch x framework x task combos,
and that latency (and energy on pi) values are present.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from scripts.utils.arch_iter import non_iso_indices

PI_CSV = ROOT / "results" / "pi.csv"
PX_CSV = ROOT / "results" / "latency_pixel.csv"

PI_FRAMEWORKS = ["litert", "onnx", "torchmobile"]
PI_TASKS = ["cifar100", "ninapro", "darcy"]

PX_FRAMEWORKS = ["litert-npu"]
PX_TASKS = ["cifar100", "ninapro", "darcy"]


def check(df, archs, frameworks, tasks, check_energy):
  idx = df.set_index(["arch_idx", "runtime", "task"])
  missing_rows = []
  missing_lat = []
  missing_energy = []
  for a in archs:
    for fw in frameworks:
      for tk in tasks:
        key = (a, fw, tk)
        if key not in idx.index:
          missing_rows.append(key)
          continue
        row = idx.loc[key]
        if isinstance(row, pd.DataFrame):
          row = row.iloc[0]
        if pd.isna(row.get("lat_ms_median")):
          missing_lat.append(key)
        if check_energy and pd.isna(row.get("energy_mj")):
          missing_energy.append(key)
  return missing_rows, missing_lat, missing_energy


def fmt(label, items, limit=20):
  print(f"\n{label}: {len(items)}")
  for k in items[:limit]:
    print(f"  {k}")
  if len(items) > limit:
    print(f"  ... +{len(items)-limit} more")


def main():
  archs = non_iso_indices()
  print(f"non-iso archs: {len(archs)}")

  pi = pd.read_csv(PI_CSV)
  px = pd.read_csv(PX_CSV)
  print(f"pi rows: {len(pi)}  pixel rows: {len(px)}")

  expected_pi = len(archs) * len(PI_FRAMEWORKS) * len(PI_TASKS)
  expected_px = len(archs) * len(PX_FRAMEWORKS) * len(PX_TASKS)
  print(f"expected pi combos: {expected_pi}  pixel combos: {expected_px}")

  print("\n=== PI ===")
  mr, ml, me = check(pi, archs, PI_FRAMEWORKS, PI_TASKS, check_energy=True)
  fmt("missing rows", mr)
  fmt("missing latency", ml)
  fmt("missing energy", me)

  print("\n=== PIXEL ===")
  mr, ml, _ = check(px, archs, PX_FRAMEWORKS, PX_TASKS, check_energy=False)
  fmt("missing rows", mr)
  fmt("missing latency", ml)

  total = len(mr) + len(ml)
  print("\nall good" if total == 0 and not any([]) else "\ndone")


if __name__ == "__main__":
  main()
