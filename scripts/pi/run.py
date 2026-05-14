"""Mac-side driver for Pi5 benchmarking.

Producer exports + rsyncs arch dirs to Pi (capped at 300 in-flight via queue
size). Consumer ssh-invokes remote_bench.py per arch, parses JSON rows, gates
on vcgencmd throttle, appends to results/latency.csv + results/completed_pi.txt.

Requires:
  - RPI_HOST env var (e.g. pi@raspberrypi.local)
  - taskset, vcgencmd, python3 + tflite/onnx/torch on Pi
"""
import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.arch_iter import non_iso_indices
from scripts.utils.convert_utils import RUNTIMES
from scripts.utils.runner_utils import (CSV_COLS, append_completed, append_row,
                                        ensure_csv, parse_arch_list,
                                        read_completed,
                                        silence_torch_elastic_redirects)
from scripts.utils.task_specs import TASKS

DEVICE = "pi5"
RPI_HOST = os.environ.get("RPI_HOST")
RPI_BASE = "~/bep"
RPI_QUEUE = f"{RPI_BASE}/queue"
RPI_SCRIPTS = f"{RPI_BASE}/scripts"
RPI_REMOTE_BENCH = f"{RPI_SCRIPTS}/remote_bench.py"

WINDOW = 300
PIN_CORE = 3
TEMP_RESUME_C = 55.0
TEMP_POLL_S = 5.0

LOCAL_STAGE = ROOT / "artifacts" / "exports_pi"
CSV_PATH = ROOT / "results" / "latency.csv"
COMPLETED_PATH = ROOT / "results" / "completed_pi.txt"


def ssh(cmd, check=False, capture=True):
  return subprocess.run(["ssh", RPI_HOST, cmd],
                        capture_output=capture, text=True, check=check)


def rsync_dir(local_dir, remote_path):
  subprocess.run(["rsync", "-a", str(local_dir) + "/", f"{RPI_HOST}:{remote_path}/"],
                 check=True)


def rsync_file(local_file, remote_path):
  subprocess.run(["rsync", "-a", str(local_file), f"{RPI_HOST}:{remote_path}"],
                 check=True)


def read_remote_temp():
  r = ssh("vcgencmd measure_temp")
  m = re.search(r"temp=([\d.]+)", r.stdout)
  return float(m.group(1)) if m else None


def cooldown():
  t = read_remote_temp()
  if t is None or t < TEMP_RESUME_C: return
  print(f"  cooldown: {t:.1f}C, target <{TEMP_RESUME_C}C", flush=True)
  last_log = time.time()
  while True:
    time.sleep(TEMP_POLL_S)
    t = read_remote_temp()
    if t is None or t < TEMP_RESUME_C:
      print(f"  cooldown done at {t if t is None else f'{t:.1f}C'}", flush=True)
      return
    if time.time() - last_log >= 30:
      print(f"  cooling: {t:.1f}C", flush=True)
      last_log = time.time()


def export_arch(arch_idx):
  d = LOCAL_STAGE / f"arch_{arch_idx}"
  if d.exists(): shutil.rmtree(d)
  d.mkdir(parents=True)
  meta = {"arch_idx": arch_idx, "tasks": {}}
  for task, spec in TASKS.items():
    meta["tasks"][task] = {"input_shape": list(spec["input_shape"])}
    for runtime, (ext, fn) in RUNTIMES.items():
      out = d / f"{task}_{runtime}.{ext}"
      fn(arch_idx, spec["input_shape"], spec["num_classes"], out)
  (d / "meta.json").write_text(json.dumps(meta))
  return d


def push_arch(local_dir, arch_idx):
  remote_tmp = f"{RPI_QUEUE}/arch_{arch_idx}.tmp"
  remote_final = f"{RPI_QUEUE}/arch_{arch_idx}"
  ssh(f"rm -rf {remote_tmp} {remote_final} && mkdir -p {remote_tmp}", check=True)
  rsync_dir(local_dir, remote_tmp)
  ssh(f"mv {remote_tmp} {remote_final}", check=True)


def run_remote_bench(arch_idx):
  remote_dir = f"{RPI_QUEUE}/arch_{arch_idx}"
  cmd = (f"taskset -c {PIN_CORE} python3 {RPI_REMOTE_BENCH} "
         f"--arch-dir {remote_dir}")
  r = ssh(cmd)
  if r.returncode != 0:
    raise RuntimeError(f"remote_bench rc={r.returncode}: {r.stderr.strip()[:300]}")
  rows = []
  final = None
  for line in r.stdout.splitlines():
    line = line.strip()
    if not line: continue
    obj = json.loads(line)
    if "throttled" in obj:
      final = obj
    else:
      rows.append(obj)
  if final is None:
    raise RuntimeError("remote_bench missing final throttle line")
  return rows, final["throttled"]


def remove_remote_arch(arch_idx):
  ssh(f"rm -rf {RPI_QUEUE}/arch_{arch_idx}")


def producer(indices, q, stop_evt):
  for arch_idx in indices:
    if stop_evt.is_set(): break
    try:
      d = export_arch(arch_idx)
      push_arch(d, arch_idx)
      shutil.rmtree(d, ignore_errors=True)
    except Exception:
      traceback.print_exc()
      q.put(("error", arch_idx))
      continue
    q.put(("ready", arch_idx))
  q.put(("done", None))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--all", action="store_true",
                  help="iterate 0..15624 instead of non-iso reps")
  args = ap.parse_args()

  if not RPI_HOST:
    print("set RPI_HOST=user@host", file=sys.stderr); sys.exit(2)

  silence_torch_elastic_redirects()
  LOCAL_STAGE.mkdir(parents=True, exist_ok=True)
  ensure_csv(CSV_PATH)
  ssh(f"mkdir -p {RPI_QUEUE} {RPI_SCRIPTS}", check=True)
  rsync_file(Path(__file__).resolve().parent / "remote_bench.py", RPI_REMOTE_BENCH)

  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  elif args.all:
    indices = list(range(15625))
  else:
    print("computing non-iso reps...", flush=True)
    indices = non_iso_indices()
    print(f"  {len(indices)} reps", flush=True)

  indices = indices[args.start:]
  if args.limit: indices = indices[:args.limit]

  done = read_completed(COMPLETED_PATH)
  pending = [i for i in indices if i not in done]
  print(f"pending: {len(pending)} (skipped {len(indices) - len(pending)} completed)",
        flush=True)
  if not pending: return

  q = queue.Queue(maxsize=WINDOW)
  stop_evt = threading.Event()
  prod = threading.Thread(target=producer, args=(pending, q, stop_evt), daemon=True)
  prod.start()

  total = len(pending)
  i = 0
  try:
    while True:
      tag, arch_idx = q.get()
      if tag == "done": break
      if tag == "error":
        i += 1
        print(f"[{i}/{total}] arch {arch_idx}: EXPORT FAIL", flush=True)
        continue

      while True:
        t0 = time.time()
        try:
          rows, throttled = run_remote_bench(arch_idx)
        except Exception as e:
          print(f"[{i+1}/{total}] arch {arch_idx}: BENCH FAIL {e}", flush=True)
          remove_remote_arch(arch_idx)
          rows, throttled = None, False
          break
        if throttled:
          print(f"[{i+1}/{total}] arch {arch_idx}: THROTTLED, cooling down",
                flush=True)
          cooldown()
          continue
        break

      remove_remote_arch(arch_idx)
      i += 1
      if rows is None:
        continue
      ok = 0
      for r in rows:
        r["device"] = DEVICE
        r["arch_idx"] = arch_idx
        append_row(CSV_PATH, r)
        if r.get("status") == "ok": ok += 1
      append_completed(COMPLETED_PATH, arch_idx)
      dt = time.time() - t0
      print(f"[{i}/{total}] arch {arch_idx}: {ok}/{len(rows)} ok ({dt:.1f}s)",
            flush=True)
  except KeyboardInterrupt:
    stop_evt.set()
    print("\ninterrupted", file=sys.stderr)
  finally:
    prod.join(timeout=5)


if __name__ == "__main__":
  main()
