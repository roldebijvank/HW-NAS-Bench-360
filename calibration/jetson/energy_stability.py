"""Find how many INA3221 samples per measurement give stable energy/inference
on the Jetson Nano (ONNX Runtime).

Mirrors scripts/utils/energy_stability.py but uses the jetson/bench.py
methodology: total board power on the POM_5V_IN rail (includes idle draw),
polled at ~550Hz, integrated by trapezoidal summation to millijoules, divided
by the inference count in the window. No idle subtraction (same as bench.py).

Self-contained (no scripts.* imports) so it can be copied to the Jetson
alongside bench.py.

Procedure:
  1. Warm up the model (10 passes).
  2. Run it in a tight loop for `--total-s` seconds while a background thread
     polls in_power0_input at ~550Hz, recording (t, power_mW, cumulative infs).
  3. For each candidate sample-count N in `--sizes`, slice the stream into
     non-overlapping N-sample windows; each window yields one synthetic
     measurement = trapezoidal energy / inferences in the window.
  4. Report mean/median/std/CV/IQR across windows, and the smallest N whose
     chunk-to-chunk CV <= `--tol-pct`.

  python3 ~/HW-NAS-Bench-360/calibration/jetson/energy_stability.py \
    --arch-index 0 --task cifar100 \
    --model-path /mnt/usb/archs/arch_0/cifar100_onnx.onnx \
    --total-s 60
"""
import argparse
import csv
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np

# Matches jetson/bench.py.
POWER_NODE = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input"
SAMPLE_HZ = 550.0
SAMPLE_INTERVAL_S = 1.0 / SAMPLE_HZ
WARMUP = 10
# Pin the sampler thread here so it doesn't contend with inference (core 0).
SAMPLER_CORE = 1


def pin_thread_to_core(core):
  """Pin the calling thread to one CPU core (Linux only; pid 0 = self)."""
  if core is None or not hasattr(os, "sched_setaffinity"):
    return
  try:
    os.sched_setaffinity(0, {core})
  except OSError:
    pass

TASK_SHAPES = {
  "cifar100": (3, 32, 32),
  "ninapro":  (1, 52, 16),
  "darcy":    (1, 88, 88),
}

DEFAULT_OUTPUT = Path.home() / "HW-NAS-Bench-360" / "results" / "energy_stability_jetson.csv"
# ~0.1, 0.25, 0.5, 1, 2, 4, 8 s at 550Hz.
DEFAULT_SIZES = [55, 138, 275, 550, 1100, 2200, 4400]

CSV_HEADER = [
  "arch_index", "task", "framework",
  "n_samples", "measure_s", "n_chunks",
  "mean_mJ", "median_mJ", "std_mJ", "cv_pct", "iqr_pct",
  "min_mJ", "max_mJ",
]


def make_step_onnx(path, x_np, use_gpu=True):
  import onnxruntime as ort
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
  providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
               if use_gpu else ["CPUExecutionProvider"])
  sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
  active = sess.get_providers()
  on_gpu = "CUDAExecutionProvider" in active
  print(f"ONNX providers: {active}  GPU: {'YES' if on_gpu else 'NO'}",
        flush=True)
  name = sess.get_inputs()[0].name
  def step(): sess.run(None, {name: x_np})
  return step


class _Sampler:
  """Polls in_power0_input (mW) at ~550Hz, tagging each sample with the
  cumulative inference count at sample time."""

  def __init__(self, power_node, stop_event, inf_counter, core=SAMPLER_CORE):
    self.power_node = Path(power_node)
    self.stop_event = stop_event
    self.inf_counter = inf_counter
    self.core = core
    self.t = []
    self.power_mw = []
    self.inf_at_sample = []
    self.missed = 0
    self.read_errors = 0
    self.last_error = None
    self.thread = None

  def _read_power_mw(self):
    with open(self.power_node) as f:
      return float(f.read().strip())

  def start(self):
    self.thread = threading.Thread(target=self._run, daemon=True)
    self.thread.start()

  def _run(self):
    pin_thread_to_core(self.core)
    deadline = time.perf_counter()
    while not self.stop_event.is_set():
      try:
        p = self._read_power_mw()
        now = time.perf_counter()
        self.power_mw.append(p)
        self.t.append(now)
        self.inf_at_sample.append(self.inf_counter[0])
      except (OSError, ValueError) as e:
        self.read_errors += 1
        self.last_error = e
      deadline += SAMPLE_INTERVAL_S
      remaining = deadline - time.perf_counter()
      if remaining > 0:
        time.sleep(remaining)
      else:
        self.missed += 1
        deadline = time.perf_counter()

  def stop(self):
    if self.thread:
      self.thread.join(timeout=2)


def _collect(step, power_node, total_s, sampler_core):
  inf_counter = [0]
  stop_event = threading.Event()
  sampler = _Sampler(power_node, stop_event, inf_counter, core=sampler_core)
  sampler.start()
  t_end = time.perf_counter() + total_s
  while time.perf_counter() < t_end:
    step()
    inf_counter[0] += 1
  stop_event.set()
  sampler.stop()
  return sampler


def _chunk_estimates(t, power_mw, inf_at_sample, n):
  """Energy/inference (mJ) per non-overlapping window of n samples,
  trapezoidal, no idle subtraction (matches bench.py)."""
  total = len(power_mw)
  n_chunks = total // n
  out = []
  for c in range(n_chunks):
    a = c * n
    b = a + n
    e_mj = 0.0
    for i in range(a + 1, b):
      dt = t[i] - t[i - 1]
      if dt <= 0: continue
      e_mj += (power_mw[i - 1] + power_mw[i]) * 0.5 * dt
    inf_before = inf_at_sample[a - 1] if a > 0 else 0
    di = inf_at_sample[b - 1] - inf_before
    if di > 0:
      out.append(e_mj / di)
  return out


def _stats(values):
  if len(values) < 2:
    return None
  mean = sum(values) / len(values)
  median = statistics.median(values)
  std = statistics.stdev(values)
  cv = (std / abs(mean) * 100.0) if mean else float("inf")
  q1, q3 = np.percentile(values, [25, 75])
  iqr_pct = ((q3 - q1) / abs(median) * 100.0) if median else float("inf")
  return {
    "mean": mean, "median": median, "std": std,
    "cv_pct": cv, "iqr_pct": iqr_pct,
    "min": min(values), "max": max(values),
  }


def run(arch_index, task, framework, model_path, power_node,
        total_s, sizes, tol_pct, use_gpu, output_path, sampler_core):
  if task not in TASK_SHAPES:
    raise SystemExit(f"unknown task: {task}")
  model_path = Path(model_path)
  if not model_path.exists():
    raise SystemExit(f"model not found: {model_path}")

  # Probe the power node up front so a bad path / permission fails loudly.
  try:
    with open(power_node) as f:
      probe_mw = float(f.read().strip())
    print(f"[power] {power_node} -> {probe_mw:.1f} mW", flush=True)
  except (OSError, ValueError) as e:
    raise SystemExit(
      f"cannot read power node {power_node}: {e}\n"
      f"check the path, or run with sudo if it needs root."
    )

  shape = TASK_SHAPES[task]
  x_np = np.random.randn(1, *shape).astype(np.float32)
  step = make_step_onnx(model_path, x_np, use_gpu=use_gpu)
  for _ in range(WARMUP):
    step()

  print(f"[loop] running {total_s}s of inference + sampling "
        f"(sampler core {sampler_core})", flush=True)
  sampler = _collect(step, power_node, total_s, sampler_core)
  n_tot = len(sampler.power_mw)
  if n_tot == 0:
    raise SystemExit(
      f"no samples collected ({sampler.read_errors} read errors; "
      f"last: {sampler.last_error})"
    )
  max_size = max(sizes)
  if n_tot < 2 * max_size:
    raise SystemExit(
      f"only {n_tot} samples; need >= {2 * max_size} for size {max_size}. "
      f"Increase --total-s or drop large sizes."
    )
  print(f"[loop] collected {n_tot} samples, "
        f"{sampler.inf_at_sample[-1]} inferences, "
        f"missed {sampler.missed}", flush=True)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  write_header = (not output_path.exists()) or output_path.stat().st_size == 0
  f_out = open(output_path, "a", newline="", buffering=1)
  writer = csv.writer(f_out)
  if write_header:
    writer.writerow(CSV_HEADER)

  print(f"{'N':>6} {'meas_s':>7} {'chunks':>6} "
        f"{'mean_mJ':>13} {'median_mJ':>13} {'CV%':>7} {'IQR%':>7}", flush=True)

  stable_n = None
  for n in sorted(sizes):
    if n_tot < 2 * n:
      print(f"{n:>6d} skipped (not enough samples)", flush=True)
      continue
    values = _chunk_estimates(
      sampler.t, sampler.power_mw, sampler.inf_at_sample, n,
    )
    s = _stats(values)
    if s is None:
      print(f"{n:>6d} skipped (<2 chunks)", flush=True)
      continue
    measure_s = n / SAMPLE_HZ
    print(f"{n:>6d} {measure_s:>7.3f} {len(values):>6d} "
          f"{s['mean']:>13.6e} {s['median']:>13.6e} "
          f"{s['cv_pct']:>7.3f} {s['iqr_pct']:>7.3f}", flush=True)
    writer.writerow([
      arch_index, task, framework, n, f"{measure_s:.4f}", len(values),
      f"{s['mean']:.9e}", f"{s['median']:.9e}", f"{s['std']:.9e}",
      f"{s['cv_pct']:.4f}", f"{s['iqr_pct']:.4f}",
      f"{s['min']:.9e}", f"{s['max']:.9e}",
    ])
    if stable_n is None and s["cv_pct"] <= tol_pct:
      stable_n = n

  f_out.close()

  if stable_n is None:
    print(f"NOT stable within CV {tol_pct}% at any tested N; "
          f"try larger sizes or longer --total-s", flush=True)
  else:
    print(f"stable at N={stable_n} samples "
          f"({stable_n / SAMPLE_HZ:.3f}s) with CV <= {tol_pct}%", flush=True)


def _parse_sizes(s):
  out = []
  for tok in s.split(","):
    tok = tok.strip()
    if not tok:
      continue
    v = int(tok)
    if v <= 0:
      raise argparse.ArgumentTypeError(f"size must be > 0: {tok}")
    out.append(v)
  if not out:
    raise argparse.ArgumentTypeError("no sizes given")
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--arch-index", type=int, required=True)
  ap.add_argument("--task", type=str, required=True, choices=list(TASK_SHAPES))
  ap.add_argument("--model-path", type=Path, required=True)
  ap.add_argument("--power-node", default=POWER_NODE,
                  help="sysfs power node (mW) for the POM_5V_IN rail")
  ap.add_argument("--total-s", type=float, default=60.0,
                  help="seconds of continuous inference + sampling")
  ap.add_argument("--sizes", type=_parse_sizes, default=DEFAULT_SIZES,
                  help="comma-separated candidate sample counts per measurement")
  ap.add_argument("--tol-pct", type=float, default=2.0,
                  help="target chunk-to-chunk CV%% for stability")
  ap.add_argument("--cpu", action="store_true",
                  help="Use CPUExecutionProvider (default: CUDA GPU)")
  ap.add_argument("--sampler-core", type=int, default=SAMPLER_CORE,
                  help="CPU core to pin the power sampler thread to "
                       "(-1 to disable pinning)")
  ap.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  if args.total_s <= 0:
    raise SystemExit("total-s must be > 0")

  np.random.seed(args.seed)

  sampler_core = None if args.sampler_core < 0 else args.sampler_core

  t0 = time.perf_counter()
  run(
    args.arch_index, args.task, "onnx", args.model_path, args.power_node,
    args.total_s, args.sizes, args.tol_pct, not args.cpu,
    args.output_path.expanduser(), sampler_core,
  )
  print(f"done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
  main()
