"""Phase 2 (Jetson): bench arch dirs under ~/bep/archs/ (override via --arch-root).
Runs ON the Jetson (self-contained, no scripts.utils dependency). ONNX Runtime
only. For each arch and each task (filter via --task), times the onnx artifact
(10 warmup + 40 timed). Pre-gates 60C via /sys/class/thermal and retries archs
that stay hot. Appends rows to ~/bep/results/latency.csv and
~/bep/results/completed_jetson_<task>.txt.

Energy: measured via the on-board INA3221 monitor on the POM_5V_IN rail (total
board input power). A background thread polls the sysfs power node at ~550 Hz
and integrates power over time by trapezoidal summation (E = sum P_i*dt_i) to
millijoules. 10 warmup passes precede measurement; the monitor is started
immediately before the 40 timed passes and stopped directly after, bounding the
energy window to the latency window. Reported value includes board idle draw.

  taskset -c 0 python3 ~/bep/scripts/jetson/bench.py --energy [--task cifar100] [--arch N]
"""
import argparse
import csv
import glob
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

DATA_ROOT = Path.home() / "bep-rik"
ARCH_ROOT = DATA_ROOT / "archs"
RESULTS_DIR = DATA_ROOT / "results"
CSV_PATH = RESULTS_DIR / "latency.csv"

DEVICE = "jetson"
WARMUP = 10
TIMED  = 40

# INA3221 / POM_5V_IN total board input rail (mW, JetPack 4.x iio layout).
POWER_NODE = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input"
SAMPLE_HZ  = 550.0

TEMP_CEILING_C = 90.0
TEMP_RESUME_C  = 80.0
TEMP_POLL_S    = 5.0
THERMAL_GLOB   = "/sys/class/thermal/thermal_zone*/temp"

TASK_SHAPES = {
  "cifar100": (3, 32, 32),
  "ninapro":  (1, 52, 16),
  "darcy":    (1, 88, 88),
}

# ONNX Runtime only on Jetson.
RUNTIME_EXT = {"onnx": "onnx"}

CSV_COLS = ["device", "arch_idx", "task", "runtime",
            "lat_ms_median", "lat_ms_var",
            "energy_mj", "status", "error"]


def ensure_csv(path):
  path.parent.mkdir(parents=True, exist_ok=True)
  if not path.exists():
    with open(path, "w", newline="") as f:
      csv.writer(f).writerow(CSV_COLS)


def append_row(path, row):
  with open(path, "a", newline="") as f:
    csv.writer(f).writerow([row.get(k, "") for k in CSV_COLS])


def read_completed(path):
  if not path.exists(): return set()
  with open(path) as f:
    return {int(s.strip()) for s in f if s.strip()}


def append_completed(path, arch_idx):
  with open(path, "a") as f:
    f.write(f"{arch_idx}\n")


def parse_arch_list(path):
  out = []
  with open(path) as f:
    for line in f:
      s = line.strip()
      if s and not s.startswith("#"): out.append(int(s))
  return out


def read_temp_c():
  temps = []
  for p in glob.glob(THERMAL_GLOB):
    try:
      with open(p) as f:
        temps.append(float(f.read().strip()) / 1000.0)
    except (OSError, ValueError):
      continue
  return max(temps) if temps else None


def cooldown(label=""):
  t = read_temp_c()
  if t is None or t < TEMP_RESUME_C: return
  print(f"  {label}cooling {t:.1f}C -> <{TEMP_RESUME_C}C", flush=True)
  last = time.time()
  while True:
    time.sleep(TEMP_POLL_S)
    t = read_temp_c()
    if t is None or t < TEMP_RESUME_C:
      print(f"  {label}cool", flush=True)
      return
    if time.time() - last >= 30:
      print(f"  {label}still {t:.1f}C", flush=True)
      last = time.time()


def gate_temp(label=""):
  t = read_temp_c()
  if t is not None and t >= TEMP_CEILING_C: cooldown(label)


def integrate_energy_mj(samples):
  """Trapezoidal integral of power (mW) over time (s) -> mJ."""
  if len(samples) < 2: return None
  energy_mj = 0.0
  for i in range(1, len(samples)):
    t0, p0 = samples[i - 1]
    t1, p1 = samples[i]
    dt = t1 - t0
    if dt <= 0: continue
    energy_mj += (p0 + p1) * 0.5 * dt
  return energy_mj


class EnergySampler:
  """Background INA3221 poller. samples = [(t_perf_s, power_mw), ...]."""

  def __init__(self, power_node, sample_hz=SAMPLE_HZ):
    self.power_node = Path(power_node)
    self.interval = 1.0 / sample_hz
    self.samples = []
    self._stop = threading.Event()
    self._thread = None

  def _read_power_mw(self):
    with open(self.power_node) as f:
      return float(f.read().strip())

  def start(self):
    self.samples = []
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def _run(self):
    deadline = time.perf_counter()
    while not self._stop.is_set():
      try:
        p = self._read_power_mw()
        self.samples.append((time.perf_counter(), p))
      except (OSError, ValueError):
        pass
      deadline += self.interval
      remaining = deadline - time.perf_counter()
      if remaining > 0:
        time.sleep(remaining)
      else:
        deadline = time.perf_counter()

  def stop(self):
    self._stop.set()
    if self._thread:
      self._thread.join(timeout=2)
    return integrate_energy_mj(self.samples)


def time_loop(fn, energy_sampler=None):
  for _ in range(WARMUP): fn()
  samples = []
  energy_mj = None
  if energy_sampler: energy_sampler.start()
  try:
    for _ in range(TIMED):
      t0 = time.perf_counter_ns()
      fn()
      samples.append((time.perf_counter_ns() - t0) / 1e6)
  finally:
    if energy_sampler:
      energy_mj = energy_sampler.stop()
  return statistics.median(samples), statistics.variance(samples), energy_mj


def make_step_onnx(path, x_np, use_gpu=True):
  import onnxruntime as ort
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
  providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
               if use_gpu else ["CPUExecutionProvider"])
  sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
  name = sess.get_inputs()[0].name
  def step(): sess.run(None, {name: x_np})
  return step


def list_arch_dirs(arch_root):
  if not arch_root.exists(): return []
  out = []
  for p in arch_root.iterdir():
    if not (p.is_dir() and p.name.startswith("arch_")): continue
    try: out.append(int(p.name.split("_", 1)[1]))
    except ValueError: continue
  return sorted(out)


def bench_arch(arch_root, arch_idx, tasks, energy_enabled, power_node, use_gpu):
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
      gate_temp(label=f"arch {arch_idx} {task}/{runtime}: ")
      try:
        step = make_step_onnx(art, x_np, use_gpu=use_gpu)
        sampler = EnergySampler(power_node) if energy_enabled else None
        med, var, energy_mj = time_loop(step, sampler)
        row["lat_ms_median"] = med
        row["lat_ms_var"] = var
        if energy_mj is not None:
          row["energy_mj_median"] = energy_mj / TIMED
        row["status"] = "ok"
      except Exception as e:
        row["status"] = "error"; row["error"] = str(e)[:200]
        print(f"  arch {arch_idx} {task}/{runtime}: ERROR {e}",
              file=sys.stderr, flush=True)
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
                  help="Measure energy via INA3221 POM_5V_IN rail")
  ap.add_argument("--power-node", default=POWER_NODE,
                  help="sysfs power node (mW) for the POM_5V_IN rail")
  ap.add_argument("--cpu", action="store_true",
                  help="Use CPUExecutionProvider (default: CUDA GPU)")
  args = ap.parse_args()

  tasks = args.task or list(TASK_SHAPES.keys())
  ensure_csv(CSV_PATH)

  arch_root = args.arch_root.expanduser()
  on_dev = list_arch_dirs(arch_root)
  if not on_dev:
    print(f"no arch_*/ under {arch_root}; convert/copy archs first",
          file=sys.stderr); sys.exit(2)

  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  else:
    indices = on_dev
  indices = indices[args.start:]
  if args.limit: indices = indices[:args.limit]

  on_dev_set = set(on_dev)
  done_per_task = {t: read_completed(RESULTS_DIR / f"completed_jetson_{t}.txt")
                   for t in tasks}

  energy_enabled = args.energy
  if energy_enabled and not os.access(args.power_node, os.R_OK):
    print(f"power node not readable: {args.power_node}; disabling energy",
          file=sys.stderr)
    energy_enabled = False

  total = len(indices)
  for i, arch_idx in enumerate(indices, 1):
    if arch_idx not in on_dev_set:
      print(f"[{i}/{total}] arch {arch_idx}: not on device", flush=True); continue
    remaining = [t for t in tasks if arch_idx not in done_per_task[t]]
    if not remaining:
      print(f"[{i}/{total}] arch {arch_idx}: done", flush=True); continue

    t0 = time.time()
    rows = bench_arch(arch_root, arch_idx, remaining, energy_enabled,
                      args.power_node, use_gpu=not args.cpu)

    ok = 0
    for r in rows:
      append_row(CSV_PATH, r)
      if r.get("status") == "ok": ok += 1
    ok_tasks = {r["task"] for r in rows if r.get("status") == "ok"}
    for t in ok_tasks:
        append_completed(RESULTS_DIR / f"completed_jetson_{t}.txt", arch_idx)
        done_per_task[t].add(arch_idx)
    dt = time.time() - t0
    print(f"[{i}/{total}] arch {arch_idx}: {ok}/{len(rows)} ok ({dt:.1f}s)",
          flush=True)


if __name__ == "__main__":
  main()
