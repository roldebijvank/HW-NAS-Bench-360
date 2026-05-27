"""Check latency_pixel.csv for missing non-iso archs and errors.

Run as module so imports work:
  uv run python -m scripts.pixel.check_latency_csv
"""
import argparse
import csv
from pathlib import Path

from scripts.utils.arch_iter import non_iso_indices
from scripts.utils.task_specs import TASKS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "results" / "latency_pixel.csv"
DEFAULT_OUT_DIR = ROOT / "results"


def _has_latency(val):
  if val is None:
    return False
  s = str(val).strip()
  if not s:
    return False
  try:
    float(s)
  except ValueError:
    return False
  return True


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv", default=str(DEFAULT_CSV),
                  help="CSV path (default: results/latency_pixel.csv)")
  ap.add_argument("--device", default="pixel",
                  help="Device value in CSV (default: pixel)")
  ap.add_argument("--tasks", nargs="*", default=list(TASKS.keys()),
                  choices=list(TASKS.keys()))
  ap.add_argument("--runtimes", nargs="*", default=None,
                  help="Runtime labels to expect (default: from CSV)")
  ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                  help="Output directory for reports.")
  ap.add_argument("--max-list", type=int, default=10)
  ap.add_argument("--show-all", action="store_true")
  args = ap.parse_args()

  csv_path = Path(args.csv)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  if not csv_path.exists():
    raise SystemExit(f"missing csv: {csv_path}")

  rows = []
  runtimes_in_data = set()
  with open(csv_path, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
      if row.get("device") != args.device:
        continue
      rows.append(row)
      rt = (row.get("runtime") or "").strip()
      if rt:
        runtimes_in_data.add(rt)

  tasks = list(args.tasks)
  if args.runtimes:
    runtimes = list(args.runtimes)
  else:
    runtimes = sorted(runtimes_in_data)

  if not runtimes:
    raise SystemExit("no runtimes found in CSV; pass --runtimes")

  expected_archs = non_iso_indices()
  expected_set = set(expected_archs)

  rows_by_key = {}
  error_rows = []
  for row in rows:
    try:
      arch_idx = int(row.get("arch_idx"))
    except (TypeError, ValueError):
      continue
    if arch_idx not in expected_set:
      continue
    task = row.get("task")
    runtime = row.get("runtime")
    if task not in tasks or runtime not in runtimes:
      continue
    key = (arch_idx, task, runtime)
    rows_by_key.setdefault(key, []).append(row)

    status = (row.get("status") or "").strip()
    lat_ok = _has_latency(row.get("lat_ms_median"))
    if status != "ok" or not lat_ok:
      error_rows.append(row)

  missing = []
  errors = []
  ok = 0
  for arch_idx in expected_archs:
    for task in tasks:
      for runtime in runtimes:
        key = (arch_idx, task, runtime)
        rows_for_key = rows_by_key.get(key, [])
        if not rows_for_key:
          missing.append((arch_idx, task, runtime, "missing row"))
          continue
        ok_row = False
        for row in rows_for_key:
          status = (row.get("status") or "").strip()
          lat_ok = _has_latency(row.get("lat_ms_median"))
          if status == "ok" and lat_ok:
            ok_row = True
            break
        if ok_row:
          ok += 1
        else:
          errors.append((arch_idx, task, runtime, "no ok latency"))

  missing_out = out_dir / "pixel_latency_missing.tsv"
  errors_out = out_dir / "pixel_latency_errors.tsv"

  with open(missing_out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["arch_idx", "task", "runtime", "reason"])
    for rec in missing:
      w.writerow(rec)

  with open(errors_out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["arch_idx", "task", "runtime", "reason"])
    for rec in errors:
      w.writerow(rec)

  def show_list(label, items):
    if not items:
      return
    show = items if args.show_all or len(items) <= args.max_list else items[:args.max_list]
    suffix = "" if len(show) == len(items) else f" (+{len(items) - len(show)} more)"
    print(f"{label}: {show}{suffix}")

  print(f"csv={csv_path} device={args.device}")
  print(f"non_iso_archs={len(expected_archs)} tasks={tasks} runtimes={runtimes}")
  print(f"ok={ok} missing={len(missing)} errors={len(errors)}")
  show_list("missing sample", missing)
  show_list("errors sample", errors)
  print(f"missing report: {missing_out}")
  print(f"errors report: {errors_out}")


if __name__ == "__main__":
  main()
