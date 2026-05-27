"""Find how many samples per measurement give stable energy/inference on rpi.

A real measurement (scripts/pi/bench_energy.py) collects N power samples at
100Hz during inference, computes sum((p - idle) * dt) / n_inferences. Stability
of a measurement = chunk-to-chunk variability of that quantity when the model
is run continuously and the sample stream is sliced into independent
non-overlapping windows of size N.

Procedure:
  1. Warm up + measure idle baseline.
  2. Run the chosen model in a tight loop for `--total-s` seconds, sampling
     power at 100Hz. Per sample, record cumulative inference count.
  3. For each candidate sample-count N in `--sizes`, slice the stream into
     non-overlapping chunks of N samples. Each chunk yields one synthetic
     measurement using the exact bench_energy formula. Compute mean, median,
     std, CV across chunks.
  4. Report smallest N with CV <= `--tol-pct`.

Run from repo root on the Pi:

  python -m scripts.utils.energy_stability \
    --arch-index 0 --task cifar100 --framework litert \
    --model-path /mnt/usb/archs/arch_0/cifar100_litert.tflite \
    --total-s 60
"""
import argparse
import csv
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from scripts.pi.bench_energy import (  # noqa: E402
  CORE_INFER,
  CORE_SAMPLE,
  CPU_GOVERNOR,
  MAKERS,
  PROCESS_CORES,
  SAMPLE_HZ,
  SAMPLE_INTERVAL_S,
  TASK_SHAPES,
  WARMUP,
  _measure_idle,
  _read_pmic_power_w,
  _set_cpu_governor,
  _set_process_affinity,
  _set_thread_affinity,
)

DEFAULT_OUTPUT = Path.home() / "bep" / "results" / "energy_stability.csv"
DEFAULT_SIZES = [10, 25, 50, 100, 200, 400, 800]

CSV_HEADER = [
  "arch_index", "task", "framework",
  "n_samples", "measure_s",
  "n_chunks",
  "mean_J", "median_J", "std_J", "cv_pct", "iqr_pct",
  "min_J", "max_J",
]


class _Sampler:
  def __init__(self, stop_event, inf_counter):
    self.stop_event = stop_event
    self.inf_counter = inf_counter
    self.samples_w = []
    self.inf_at_sample = []
    self.missed = 0
    self.thread = None

  def start(self):
    self.thread = threading.Thread(target=self._run, daemon=True)
    self.thread.start()

  def _run(self):
    _set_thread_affinity(CORE_SAMPLE, "sample thread")
    deadline = time.perf_counter()
    while not self.stop_event.is_set():
      p = _read_pmic_power_w()
      if p is not None:
        self.samples_w.append(p)
        self.inf_at_sample.append(self.inf_counter[0])
      deadline += SAMPLE_INTERVAL_S
      remaining = deadline - time.perf_counter()
      if remaining > 0:
        time.sleep(remaining)
      else:
        self.missed += 1
        deadline = time.perf_counter()

  def stop(self):
    if self.thread:
      self.thread.join(timeout=1)


def _collect(step, total_s):
  inf_counter = [0]
  stop_event = threading.Event()
  sampler = _Sampler(stop_event, inf_counter)
  sampler.start()
  t_end = time.perf_counter() + total_s
  while time.perf_counter() < t_end:
    step()
    inf_counter[0] += 1
  stop_event.set()
  sampler.stop()
  return sampler


def _chunk_estimates(samples_w, inf_at_sample, idle_w, n):
  """Energy/inference per non-overlapping chunk of n samples."""
  dt = SAMPLE_INTERVAL_S
  total = len(samples_w)
  n_chunks = total // n
  out = []
  for c in range(n_chunks):
    a = c * n
    b = a + n
    e = sum((samples_w[i] - idle_w) * dt for i in range(a, b))
    inf_before = inf_at_sample[a - 1] if a > 0 else 0
    di = inf_at_sample[b - 1] - inf_before
    if di > 0:
      out.append(e / di)
  return out


def _stats(values):
  if len(values) < 2:
    return None
  mean = statistics.fmean(values)
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


def run(
  arch_index, task, framework, model_path,
  total_s, sizes, idle_s, tol_pct, output_path,
):
  if task not in TASK_SHAPES:
    raise SystemExit(f"unknown task: {task}")
  if framework not in MAKERS:
    raise SystemExit(f"unknown framework: {framework}")
  model_path = Path(model_path)
  if not model_path.exists():
    raise SystemExit(f"model not found: {model_path}")

  shape = TASK_SHAPES[task]
  x_np = np.random.randn(1, *shape).astype(np.float32)
  step = MAKERS[framework](model_path, x_np)
  for _ in range(WARMUP):
    step()

  idle_w, _, _ = _measure_idle(idle_s)
  print(f"[loop] running {total_s}s of inference + sampling", flush=True)
  sampler = _collect(step, total_s)
  n_tot = len(sampler.samples_w)
  if n_tot == 0:
    raise SystemExit("no samples collected")
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
        f"{'mean_J':>13} {'median_J':>13} "
        f"{'CV%':>7} {'IQR%':>7}", flush=True)

  stable_n = None
  for n in sorted(sizes):
    if n_tot < 2 * n:
      print(f"{n:>6d} skipped (not enough samples)", flush=True)
      continue
    values = _chunk_estimates(
      sampler.samples_w, sampler.inf_at_sample, idle_w, n,
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
  ap.add_argument("--framework", type=str, required=True, choices=list(MAKERS))
  ap.add_argument("--model-path", type=Path, required=True)
  ap.add_argument("--total-s", type=float, default=60.0,
                  help="seconds of continuous inference + sampling")
  ap.add_argument("--sizes", type=_parse_sizes, default=DEFAULT_SIZES,
                  help="comma-separated candidate sample counts per measurement")
  ap.add_argument("--idle-s", type=float, default=1.0)
  ap.add_argument("--tol-pct", type=float, default=2.0,
                  help="target chunk-to-chunk CV%% for stability")
  ap.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  if args.total_s <= 0:
    raise SystemExit("total-s must be > 0")

  np.random.seed(args.seed)
  _set_thread_affinity(CORE_INFER, "main thread")
  _set_process_affinity(PROCESS_CORES, "process")
  _set_cpu_governor(CPU_GOVERNOR)

  t0 = time.perf_counter()
  run(
    args.arch_index, args.task, args.framework, args.model_path,
    args.total_s, args.sizes, args.idle_s, args.tol_pct,
    args.output_path.expanduser(),
  )
  print(f"done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
  main()
