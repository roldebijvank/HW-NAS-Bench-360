"""Phase 2 (Pixel): bench every arch dir already on device.

For each arch_<N>/ under /data/local/tmp/archs (or selection), and for each task,
pre-gates 45C, runs BenchmarkModelActivity (10 warmup + 40 timed, GPU delegate by default),
parses Inference (avg)+std from logcat, appends to results/latency.csv and
results/completed_pixel.txt. Does NOT delete arch dirs.

Run as module so imports work:
  uv run python -m scripts.pixel.bench
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.utils.runner_utils import (append_completed, append_row,
                                        ensure_csv, parse_arch_list,
                                        read_completed)
from scripts.utils.task_specs import TASKS

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "pixel"
DEVICE_ROOT = "/data/local/tmp/archs"
CSV_PATH = ROOT / "results" / "latency.csv"
COMPLETED_PATH = ROOT / "results" / "completed_pixel.txt"

WARMUP = 10
TIMED  = 40

TEMP_CEILING_C = 45.0
TEMP_POLL_S    = 10.0
THERMAL_ZONE   = "/sys/class/thermal/thermal_zone0/temp"

BENCH_ACTIVITY = "org.tensorflow.lite.benchmark/.BenchmarkModelActivity"
BENCH_PKG      = "org.tensorflow.lite.benchmark"
BASE_ARGS = ["--num_threads=1", "--cpu_mask=f0"]
GPU_ARGS = ["--use_gpu=true", *BASE_ARGS]
NPU_ARGS = ["--use_nnapi=true", *BASE_ARGS]

_AVG_RE = re.compile(r"Inference \(avg\):\s*([\d.]+)")
_STD_RE = re.compile(r"Inference \(std deviation\):\s*([\d.]+)")


def adb(*args, check=False):
  return subprocess.run(["adb", *args], capture_output=True, text=True, check=check)


def adb_shell(cmd, check=False):
  return subprocess.run(["adb", "shell", cmd],
                        capture_output=True, text=True, check=check)


def read_device_temp_c():
  r = adb_shell(f"cat {THERMAL_ZONE}")
  try:
    return float(r.stdout.strip()) / 1000.0
  except ValueError:
    return None


def gate_temp(label=""):
  t = read_device_temp_c()
  if t is None or t < TEMP_CEILING_C: return
  print(f"  {label}hot {t:.1f}C >= {TEMP_CEILING_C}C, waiting...", flush=True)
  while True:
    time.sleep(TEMP_POLL_S)
    t = read_device_temp_c()
    if t is None or t < TEMP_CEILING_C:
      print(f"  {label}resumed at {t if t is None else f'{t:.1f}C'}", flush=True)
      return
    print(f"  {label}still hot {t:.1f}C", flush=True)


def device_exists(path):
  return adb_shell(f"test -e {path} && echo Y || echo N").stdout.strip() == "Y"


def list_device_archs():
  r = adb_shell(f"ls -1 {DEVICE_ROOT}")
  out = []
  for name in r.stdout.splitlines():
    name = name.strip()
    if not name.startswith("arch_"): continue
    try:
      out.append(int(name.split("_", 1)[1]))
    except ValueError:
      continue
  return sorted(out)


def wait_for_summary(timeout_s=120, poll_s=1.0):
  t_end = time.time() + timeout_s
  while time.time() < t_end:
    out = adb_shell("logcat -d").stdout
    m_avg = _AVG_RE.search(out)
    m_std = _STD_RE.search(out)
    if m_avg:
      std_ms = float(m_std.group(1)) / 1000.0 if m_std else None
      return (float(m_avg.group(1)) / 1000.0, std_ms)
    time.sleep(poll_s)
  raise TimeoutError("timed out waiting for Inference avg in logcat")


def bench_task(model_path, delegate_args):
  args_str = " ".join([
    f"--graph={model_path}",
    f"--warmup_runs={WARMUP}",
    f"--num_runs={TIMED}",
    *delegate_args,
  ])
  adb_shell("logcat -c", check=True)
  adb_shell(f'am start -S -W -n {BENCH_ACTIVITY} --es args "{args_str}"', check=True)
  try:
    return wait_for_summary()
  finally:
    adb_shell(f"am force-stop {BENCH_PKG}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--use-npu", action="store_true",
                  help="Use NNAPI delegate instead of GPU")
  args = ap.parse_args()

  delegate_args = NPU_ARGS if args.use_npu else GPU_ARGS
  runtime_label = "litert-npu" if args.use_npu else "litert-gpu"

  r = adb("get-state")
  if r.returncode != 0 or "device" not in r.stdout:
    print(f"adb get-state: {r.stdout.strip()} {r.stderr.strip()}", file=sys.stderr)
    sys.exit(2)

  ensure_csv(CSV_PATH)

  on_device = list_device_archs()
  if not on_device:
    print(f"no arch_*/ under {DEVICE_ROOT} on device; run pixel/convert.py first",
          file=sys.stderr)
    sys.exit(2)

  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  else:
    indices = on_device

  indices = indices[args.start:]
  if args.limit: indices = indices[:args.limit]

  done = read_completed(COMPLETED_PATH)
  on_device_set = set(on_device)
  pending = [i for i in indices if i not in done and i in on_device_set]
  missing = [i for i in indices if i not in on_device_set]
  print(f"on device: {len(on_device)}  pending: {len(pending)}  "
        f"completed: {len(indices) - len(pending) - len(missing)}  "
        f"missing on device: {len(missing)}", flush=True)
  if missing[:5]:
    print(f"  missing sample: {missing[:5]}", flush=True)

  total = len(pending)
  for i, arch_idx in enumerate(pending, 1):
    t0 = time.time()
    rows = []
    for task in TASKS:
      gate_temp(label=f"arch {arch_idx} {task}: ")
      remote_model = f"{DEVICE_ROOT}/arch_{arch_idx}/{task}_litert.tflite"
      row = {"device": DEVICE, "arch_idx": arch_idx,
             "task": task, "runtime": runtime_label}
      if not device_exists(remote_model):
        row["status"] = "missing"; row["error"] = "model not on device"
      else:
        try:
          avg_ms, std_ms = bench_task(remote_model, delegate_args)
          row["lat_ms_median"] = avg_ms
          if std_ms is not None:
            row["lat_ms_var"] = std_ms * std_ms
          row["status"] = "ok"
        except Exception as e:
          row["status"] = "error"; row["error"] = str(e)[:200]
      rows.append(row)

    ok = 0
    for r in rows:
      append_row(CSV_PATH, r)
      if r.get("status") == "ok": ok += 1
    append_completed(COMPLETED_PATH, arch_idx)
    dt = time.time() - t0
    print(f"[{i}/{total}] arch {arch_idx}: {ok}/{len(rows)} ok ({dt:.1f}s)",
          flush=True)


if __name__ == "__main__":
  main()
