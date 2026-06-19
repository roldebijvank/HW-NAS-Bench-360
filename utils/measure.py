"""Unified time_loop, temperature gate, and stock temp readers."""
import glob
import re
import statistics
import subprocess
import time


WARMUP = 10
TIMED  = 40
TEMP_POLL_S = 5.0


def time_loop(fn, energy_reader=None, warmup=WARMUP, timed=TIMED, min_window_s=0.0):
  """Warmup + timed passes. Returns (med_ms, var_ms, energy_mj, n_passes)."""
  for _ in range(warmup): fn()
  samples = []
  energy_mj = None
  if energy_reader: energy_reader.start()
  try:
    t_start = time.perf_counter()
    while True:
      t0 = time.perf_counter_ns()
      fn()
      samples.append((time.perf_counter_ns() - t0) / 1e6)
      if len(samples) >= timed and (time.perf_counter() - t_start) >= min_window_s:
        break
  finally:
    if energy_reader:
      energy_mj = energy_reader.stop()
  return statistics.median(samples), statistics.variance(samples), energy_mj, len(samples)


def gate_temp(temp_reader, ceiling, resume, label="", poll_s=TEMP_POLL_S):
  """Block until temp_reader() drops below `resume`. No-op if below `ceiling`."""
  if temp_reader is None: return
  t = temp_reader()
  if t is None or t < ceiling: return
  print(f"  {label}cooling {t:.1f}C -> <{resume}C", flush=True)
  last = time.time()
  while True:
    time.sleep(poll_s)
    t = temp_reader()
    if t is None or t < resume:
      print(f"  {label}cool", flush=True)
      return
    if time.time() - last >= 30:
      print(f"  {label}still {t:.1f}C", flush=True)
      last = time.time()


def vcgencmd_temp():
  try:
    out = subprocess.run(["vcgencmd", "measure_temp"],
                         capture_output=True, text=True, timeout=2).stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return None
  m = re.search(r"temp=([\d.]+)", out)
  return float(m.group(1)) if m else None


def thermal_zone_max_temp(glob_pattern="/sys/class/thermal/thermal_zone*/temp"):
  temps = []
  for p in glob.glob(glob_pattern):
    try:
      with open(p) as f:
        temps.append(float(f.read().strip()) / 1000.0)
    except (OSError, ValueError):
      continue
  return max(temps) if temps else None
