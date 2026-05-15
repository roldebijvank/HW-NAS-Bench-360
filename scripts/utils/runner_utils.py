"""Shared helpers for Pi and Pixel runners."""
from pathlib import Path
import contextlib
import csv
import logging
import os
import statistics
import sys
import time


WARMUP = 10
TIMED  = 40

CSV_COLS = ["device", "arch_idx", "task", "runtime",
            "lat_ms_median", "lat_ms_var",
            "energy_mj_median",
            "status", "error"]


def time_loop(fn):
  for _ in range(WARMUP): fn()
  samples = []
  for _ in range(TIMED):
    t0 = time.perf_counter_ns()
    fn()
    samples.append((time.perf_counter_ns() - t0) / 1e6)
  return {
    "lat_ms_median": statistics.median(samples),
    "lat_ms_var":    statistics.variance(samples),
  }


def ensure_csv(path):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  if not path.exists():
    with open(path, "w", newline="") as f:
      csv.writer(f).writerow(CSV_COLS)


def append_row(path, row):
  with open(path, "a", newline="") as f:
    csv.writer(f).writerow([row.get(k, "") for k in CSV_COLS])


def read_completed(path):
  path = Path(path)
  if not path.exists(): return set()
  out = set()
  with open(path) as f:
    for line in f:
      s = line.strip()
      if s: out.add(int(s))
  return out


def append_completed(path, arch_idx):
  with open(path, "a") as f:
    f.write(f"{arch_idx}\n")


@contextlib.contextmanager
def suppress_output():
  sys.stdout.flush(); sys.stderr.flush()
  devnull = os.open(os.devnull, os.O_WRONLY)
  old_out, old_err = os.dup(1), os.dup(2)
  try:
    os.dup2(devnull, 1); os.dup2(devnull, 2)
    yield
  finally:
    os.dup2(old_out, 1); os.dup2(old_err, 2)
    os.close(old_out); os.close(old_err); os.close(devnull)


def render_progress(done, total, ok, skipped, err, width=30):
  if total <= 0:
    return f"0/0 [{'-' * width}] 100.0%"
  pct = done / total
  filled = int(round(width * pct))
  bar = "=" * filled + "-" * (width - filled)
  return (f"\r{done}/{total} [{bar}] {pct * 100:5.1f}% "
          f"ok={ok} skip={skipped} err={err}")


def show_progress(done, total, ok, skipped, err, final=False, stream=None):
  end = "\n" if final else ""
  if stream is None:
    stream = sys.stdout
  print(render_progress(done, total, ok, skipped, err),
        end=end, flush=True, file=stream)


def silence_torch_elastic_redirects():
  logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").disabled = True


def parse_arch_list(path):
  archs = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      s = line.strip()
      if not s or s.startswith("#"): continue
      archs.append(int(s))
  return archs
