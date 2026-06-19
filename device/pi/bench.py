"""Bench arch dirs on Pi5 across all frameworks measuring latency and PMIC energy.

  taskset -c 3 python3 ~/HW-NAS-Bench-360/device/pi/bench.py [--task cifar100] [--arch N]
"""
import argparse
import os
import re
import subprocess
import sys
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
CSV_PATH = RESULTS_DIR / "latency_pi.csv"

DEVICE = "pi5"
TIMED  = 40

TEMP_CEILING_C = 60.0
TEMP_RESUME_C  = 55.0

TASK_SHAPES = {t: TASKS[t]["input_shape"] for t in TASKS}
RUNTIME_EXT = {n: FRAMEWORKS[n]["ext"] for n in DEVICES[DEVICE]["frameworks"]}
ENERGY_READER_CLS = DEVICES[DEVICE]["energy_reader"]
TEMP_READER = DEVICES[DEVICE]["temp_reader"]

CSV_COLS = ["device", "arch_idx", "task", "runtime",
            "lat_ms_median", "lat_ms_var",
            "energy_mj_median", "status", "error"]


def read_throttled():
  try:
    out = subprocess.run(["vcgencmd", "get_throttled"],
                         capture_output=True, text=True, timeout=2).stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return None
  m = re.search(r"throttled=(0x[0-9a-fA-F]+)", out)
  return m.group(1) if m else None


def bench_arch(arch_root, arch_idx, tasks, energy_enabled,
               periodic_bin, vcgencmd_bin):
  rows = []
  d = arch_root / f"arch_{arch_idx}"
  for task in tasks:
    shape = TASK_SHAPES[task]
    x_np = np.random.randn(1, *shape).astype(np.float32)
    for runtime, ext in RUNTIME_EXT.items():
      art = d / f"{task}_{runtime}.{ext}"
      row = {"device": DEVICE, "arch_idx": arch_idx,
             "task": task, "runtime": runtime}
      if not art.exists():
        row["status"] = "missing"; row["error"] = "artifact not found"
        rows.append(row); continue
      gate_temp(TEMP_READER, TEMP_CEILING_C, TEMP_RESUME_C,
                label=f"arch {arch_idx} {task}/{runtime}: ")
      try:
        step = MAKERS[runtime](art, x_np)
        reader = (ENERGY_READER_CLS(periodic_bin=periodic_bin,
                                    vcgencmd_bin=vcgencmd_bin)
                  if energy_enabled else None)
        med, var, energy_mj, n = time_loop(step, reader, timed=TIMED)
        row["lat_ms_median"] = med
        row["lat_ms_var"] = var
        if energy_mj is not None:
          row["energy_mj_median"] = energy_mj / n
        row["status"] = "ok"
      except Exception as e:
        row["status"] = "error"; row["error"] = str(e)[:200]
      rows.append(row)
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", choices=list(TASK_SHAPES), action="append",
                  default=[], help="repeat to select subset; default all")
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--arch-root", type=Path, default=ARCH_ROOT,
                  help="Directory containing arch_* folders")
  ap.add_argument("--energy", action="store_true",
                  help="Measure energy via periodic+vcgencmd pmic_read_adc")
  ap.add_argument("--periodic-bin", type=Path, default=DATA_ROOT / "periodic",
                  help="Path to periodic binary")
  ap.add_argument("--vcgencmd-bin", default="vcgencmd",
                  help="vcgencmd binary (default: vcgencmd)")
  args = ap.parse_args()

  tasks = args.task or list(TASK_SHAPES.keys())

  arch_root = args.arch_root.expanduser()
  on_pi = list_arch_dirs(arch_root)
  if not on_pi:
    print(f"no arch_*/ under {arch_root}; run host/convert.py first",
          file=sys.stderr); sys.exit(2)

  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  else:
    indices = on_pi
  indices = indices[args.start:]
  if args.limit: indices = indices[:args.limit]

  done_per_key = {t: read_completed(RESULTS_DIR / f"completed_pi_{t}.txt")
                  for t in tasks}

  energy_enabled = args.energy
  if energy_enabled:
    pb = args.periodic_bin.expanduser()
    if not (pb.exists() and pb.is_file() and os.access(pb, os.X_OK)):
      print(f"periodic not executable: {pb}; disabling energy",
            file=sys.stderr)
      energy_enabled = False

  periodic_bin = args.periodic_bin.expanduser()
  vcgencmd_bin = args.vcgencmd_bin

  def bench_fn(arch_idx, pending_tasks):
    while True:
      read_throttled()
      rows = bench_arch(arch_root, arch_idx, pending_tasks,
                        energy_enabled, periodic_bin, vcgencmd_bin)
      raw = read_throttled()
      throttled = bool(raw and int(raw, 16) & 0x7) if raw else False
      if throttled:
        print(f"arch {arch_idx}: THROTTLED {raw}, retry", flush=True)
        gate_temp(TEMP_READER, TEMP_RESUME_C, TEMP_RESUME_C,
                  label=f"arch {arch_idx}: ")
        continue
      return rows

  run_bench_loop(
    indices=indices,
    on_device_set=set(on_pi),
    csv_path=CSV_PATH,
    csv_cols=CSV_COLS,
    pending_fn=lambda arch_idx: [t for t in tasks
                                 if arch_idx not in done_per_key[t]],
    bench_fn=bench_fn,
    done_per_key=done_per_key,
    completion_path_fn=lambda t: RESULTS_DIR / f"completed_pi_{t}.txt",
    mark_done_fn=lambda rows, pending: set(pending),
  )


if __name__ == "__main__":
  main()
