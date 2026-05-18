"""Phase 2 (Pi): bench arch dirs under ~/bep/archs/ (override via --arch-root). Runs ON the Pi (self-
contained, no scripts.utils dependency). For each arch and each task (filter
via --task), times every runtime artifact (10 warmup + 40 timed). Pre-gates
60C via vcgencmd and retries archs that throttle. Appends rows to
~/bep/results/latency.csv and ~/bep/results/completed_pi_<task>.txt.

  taskset -c 3 python3 ~/bep/scripts/pi/bench.py [--task cifar100] [--arch N]
"""
import argparse
import csv
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DATA_ROOT = Path.home() / "bep"
ARCH_ROOT = DATA_ROOT / "archs"
RESULTS_DIR = DATA_ROOT / "results"
CSV_PATH = RESULTS_DIR / "latency.csv"

DEVICE = "pi5"
WARMUP = 10
TIMED  = 40
PIN_CORE = 3

TEMP_CEILING_C = 60.0
TEMP_RESUME_C  = 55.0
TEMP_POLL_S    = 5.0

TASK_SHAPES = {
  "cifar100": (3, 32, 32),
  "ninapro":  (1, 52, 16),
  "darcy":    (1, 88, 88),
}

RUNTIME_EXT = {
  "litert":      "tflite",
  "onnx":        "onnx",
  "torchmobile": "ptl",
}

CSV_COLS = ["device", "arch_idx", "task", "runtime",
            "lat_ms_median", "lat_ms_var",
            "energy_mj_median", "status", "error"]


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
  try:
    out = subprocess.run(["vcgencmd", "measure_temp"],
                         capture_output=True, text=True, timeout=2).stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return None
  m = re.search(r"temp=([\d.]+)", out)
  return float(m.group(1)) if m else None


def read_throttled():
  try:
    out = subprocess.run(["vcgencmd", "get_throttled"],
                         capture_output=True, text=True, timeout=2).stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return None
  m = re.search(r"throttled=(0x[0-9a-fA-F]+)", out)
  return m.group(1) if m else None


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


def time_loop(fn):
  for _ in range(WARMUP): fn()
  samples = []
  for _ in range(TIMED):
    t0 = time.perf_counter_ns()
    fn()
    samples.append((time.perf_counter_ns() - t0) / 1e6)
  return statistics.median(samples), statistics.variance(samples)


def make_step_litert(path, x_np):
  try:
    from tflite_runtime.interpreter import Interpreter
  except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
  interp = Interpreter(model_path=str(path), num_threads=1)
  interp.allocate_tensors()
  inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
  def step():
    interp.set_tensor(inp["index"], x_np)
    interp.invoke()
    interp.get_tensor(out["index"])
  return step


def make_step_onnx(path, x_np):
  import onnxruntime as ort
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
  sess = ort.InferenceSession(str(path), sess_options=so,
                              providers=["CPUExecutionProvider"])
  name = sess.get_inputs()[0].name
  def step(): sess.run(None, {name: x_np})
  return step


def make_step_torchmobile(path, x_np):
  import torch
  torch.set_num_threads(1)
  m = torch.jit.load(str(path)); m.eval()
  x = torch.from_numpy(x_np)
  def step():
    with torch.no_grad(): m(x)
  return step


MAKERS = {
  "litert":      make_step_litert,
  "onnx":        make_step_onnx,
  "torchmobile": make_step_torchmobile,
}


def list_arch_dirs(arch_root):
  if not arch_root.exists(): return []
  out = []
  for p in arch_root.iterdir():
    if not (p.is_dir() and p.name.startswith("arch_")): continue
    try: out.append(int(p.name.split("_", 1)[1]))
    except ValueError: continue
  return sorted(out)


def bench_arch(arch_root, arch_idx, tasks):
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
        step = MAKERS[runtime](art, x_np)
        med, var = time_loop(step)
        row["lat_ms_median"] = med
        row["lat_ms_var"] = var
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
  args = ap.parse_args()

  tasks = args.task or list(TASK_SHAPES.keys())
  ensure_csv(CSV_PATH)

  arch_root = args.arch_root.expanduser()
  on_pi = list_arch_dirs(arch_root)
  if not on_pi:
    print(f"no arch_*/ under {arch_root}; run pi/convert.py first",
          file=sys.stderr); sys.exit(2)

  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  else:
    indices = on_pi
  indices = indices[args.start:]
  if args.limit: indices = indices[:args.limit]

  on_pi_set = set(on_pi)
  done_per_task = {t: read_completed(RESULTS_DIR / f"completed_pi_{t}.txt")
                   for t in tasks}

  total = len(indices)
  for i, arch_idx in enumerate(indices, 1):
    if arch_idx not in on_pi_set:
      print(f"[{i}/{total}] arch {arch_idx}: not on Pi", flush=True); continue
    remaining = [t for t in tasks if arch_idx not in done_per_task[t]]
    if not remaining:
      print(f"[{i}/{total}] arch {arch_idx}: done", flush=True); continue

    while True:
      read_throttled()
      t0 = time.time()
      rows = bench_arch(arch_root, arch_idx, remaining)
      raw = read_throttled()
      throttled = bool(raw and int(raw, 16) & 0x7) if raw else False
      if throttled:
        print(f"[{i}/{total}] arch {arch_idx}: THROTTLED {raw}, retry",
              flush=True)
        cooldown(label=f"arch {arch_idx}: ")
        continue
      break

    ok = 0
    for r in rows:
      append_row(CSV_PATH, r)
      if r.get("status") == "ok": ok += 1
    for t in remaining:
      append_completed(RESULTS_DIR / f"completed_pi_{t}.txt", arch_idx)
      done_per_task[t].add(arch_idx)
    dt = time.time() - t0
    print(f"[{i}/{total}] arch {arch_idx}: {ok}/{len(rows)} ok ({dt:.1f}s)",
          flush=True)


if __name__ == "__main__":
  main()
