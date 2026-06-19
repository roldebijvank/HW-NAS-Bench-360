"""Bench arch dirs on Pixel 6a (all frameworks, CPU only, no energy) inside proot-distro Ubuntu.

  python3 ~/HW-NAS-Bench-360/device/pixel/bench.py [--task cifar100] [--framework litert]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.pipeline_config import TASKS, FRAMEWORKS, DEVICES
from utils.measure import gate_temp, time_loop
from utils.runners import MAKERS
from utils.runner_utils import read_completed, parse_arch_list
from utils.bench_loop import list_arch_dirs, run_bench_loop

DATA_ROOT = Path.home() / "HW-NAS-Bench-360"
ARCH_ROOT = DATA_ROOT / "archs"
RESULTS_DIR = DATA_ROOT / "results"

DEVICE = "pixel"
TIMED = 40
MIN_WINDOW_S = 0.5

BIG_CORES = {4, 5, 6, 7}
THERMAL_ZONE = "/sys/class/thermal/thermal_zone0/temp"
TEMP_CEILING_C = 45.0
TEMP_RESUME_C = 40.0

TASK_SHAPES = {t: TASKS[t]["input_shape"] for t in TASKS}
RUNTIME_EXT = {n: FRAMEWORKS[n]["ext"] for n in DEVICES[DEVICE]["frameworks"]}

CSV_COLS = ["device", "arch_idx", "task", "runtime",
            "lat_ms_median", "lat_ms_var",
            "energy_mj", "status", "error"]


def pixel_temp():
  try:
    with open(THERMAL_ZONE) as f:
      return float(f.read().strip()) / 1000.0
  except (OSError, ValueError):
    return None


def bench_arch(arch_root, arch_idx, task_fw_pairs):
  rows = []
  d = arch_root / f"arch_{arch_idx}"
  inputs = {}
  for task, runtime in task_fw_pairs:
    if task not in inputs:
      inputs[task] = np.random.randn(1, *TASK_SHAPES[task]).astype(np.float32)
    x_np = inputs[task]
    ext = RUNTIME_EXT[runtime]
    art = d / f"{task}_{runtime}.{ext}"
    row = {"device": DEVICE, "arch_idx": arch_idx,
           "task": task, "runtime": runtime}
    if not art.exists():
      row["status"] = "missing"; row["error"] = "artifact not found"
      rows.append(row); continue
    gate_temp(pixel_temp, TEMP_CEILING_C, TEMP_RESUME_C,
              label=f"arch {arch_idx} {task}/{runtime}: ")
    try:
      step = MAKERS[runtime](art, x_np)
      med, var, _, n = time_loop(step, None, timed=TIMED,
                                  min_window_s=MIN_WINDOW_S)
      row["lat_ms_median"] = med
      row["lat_ms_var"] = var
      row["status"] = "ok"
    except Exception as e:
      row["status"] = "error"; row["error"] = str(e)[:200]
      print(f"  arch {arch_idx} {task}/{runtime}: ERROR {e}",
            file=sys.stderr, flush=True)
    rows.append(row)
  return rows


def main():
  os.sched_setaffinity(0, BIG_CORES)

  ap = argparse.ArgumentParser()
  ap.add_argument("--task", choices=list(TASK_SHAPES), action="append",
                  default=[], help="repeat to select subset; default all")
  ap.add_argument("--framework", choices=list(RUNTIME_EXT), action="append",
                  default=[], help="repeat to select subset; default all")
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--arch-root", type=Path, default=ARCH_ROOT)
  ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
  args = ap.parse_args()

  tasks = args.task or list(TASK_SHAPES.keys())
  frameworks = args.framework or list(RUNTIME_EXT.keys())
  results_dir = args.results_dir.expanduser()
  csv_path = results_dir / "latency_pixel.csv"

  arch_root = args.arch_root.expanduser()
  on_device = list_arch_dirs(arch_root)
  if not on_device:
    print(f"no arch_*/ under {arch_root}", file=sys.stderr)
    sys.exit(2)

  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  else:
    indices = on_device
  indices = indices[args.start:]
  if args.limit:
    indices = indices[:args.limit]

  done_per_key = {
    (t, fw): read_completed(results_dir / f"completed_pixel_{t}_{fw}.txt")
    for t in tasks for fw in frameworks
  }

  def bench_fn(arch_idx, pending_pairs):
    return bench_arch(arch_root, arch_idx, pending_pairs)

  run_bench_loop(
    indices=indices,
    on_device_set=set(on_device),
    csv_path=csv_path,
    csv_cols=CSV_COLS,
    pending_fn=lambda arch_idx: [(t, fw) for t in tasks for fw in frameworks
                                 if arch_idx not in done_per_key[(t, fw)]],
    bench_fn=bench_fn,
    done_per_key=done_per_key,
    completion_path_fn=lambda key: results_dir / f"completed_pixel_{key[0]}_{key[1]}.txt",
    mark_done_fn=lambda rows, pending: {(r["task"], r["runtime"]) for r in rows
                                        if r.get("status") == "ok"},
  )


if __name__ == "__main__":
  main()
