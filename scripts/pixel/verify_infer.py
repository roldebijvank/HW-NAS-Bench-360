"""Verify Pixel tflite models, export missing, then run one-pass inference.

Run as module so imports work:
  uv run python -m scripts.pixel.verify_infer --root /data/local/tmp/archs
"""
import argparse
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from scripts.utils.convert_utils import export_tflite
from scripts.utils.runner_utils import parse_arch_list, show_progress, suppress_output
from scripts.utils.task_specs import TASKS

ROOT = Path(__file__).resolve().parents[2]
DEVICE_ROOT = "/data/local/tmp/archs"
RESULTS_DIR = ROOT / "results"
ERROR_LOG = RESULTS_DIR / "pixel_verify_infer_errors.log"

BASE_ARGS = ["--num_threads=1", "--cpu_mask=f0"]
BENCH_BIN_DEFAULT = "/data/local/tmp/benchmark_model"

BENCH_ACTIVITY = "org.tensorflow.lite.benchmark/.BenchmarkModelActivity"
BENCH_PKG = "org.tensorflow.lite.benchmark"

WARMUP = 0
RUNS = 1
TIMEOUT_S = 60

_AVG_RE = re.compile(r"Inference \(avg\):\s*([\d.]+)")

ADB_LOCK = threading.Lock()


def adb(*args, check=False):
  with ADB_LOCK:
    return subprocess.run(["adb", *args], capture_output=True, text=True, check=check)


def adb_shell(cmd, check=False):
  with ADB_LOCK:
    return subprocess.run(["adb", "shell", cmd],
                          capture_output=True, text=True, check=check)


def device_exists(path):
  return adb_shell(f"test -e {path} && echo Y || echo N").stdout.strip() == "Y"


def device_executable(path):
  return adb_shell(f"test -x {path} && echo Y || echo N").stdout.strip() == "Y"


def list_device_archs(root):
  r = adb_shell(f"ls -1 {root}")
  out = []
  for name in r.stdout.splitlines():
    name = name.strip()
    if not name.startswith("arch_"):
      continue
    try:
      out.append(int(name.split("_", 1)[1]))
    except ValueError:
      continue
  return sorted(out)


def wait_for_summary(timeout_s=TIMEOUT_S, poll_s=1.0):
  t_end = time.time() + timeout_s
  while time.time() < t_end:
    out = adb_shell("logcat -d").stdout
    m_avg = _AVG_RE.search(out)
    if m_avg:
      return True
    time.sleep(poll_s)
  return False


def run_one_pass_activity(remote_model, delegate_args):
  args_str = " ".join([
    f"--graph={remote_model}",
    f"--warmup_runs={WARMUP}",
    f"--num_runs={RUNS}",
    *delegate_args,
  ])
  adb_shell("logcat -c", check=True)
  r = adb_shell(f'am start -S -W -n {BENCH_ACTIVITY} --es args "{args_str}"')
  if r.returncode != 0:
    msg = r.stderr.strip() or r.stdout.strip() or "am start failed"
    return False, msg
  try:
    ok = wait_for_summary()
    return ok, None if ok else "timed out waiting for inference"
  finally:
    adb_shell(f"am force-stop {BENCH_PKG}")


def run_one_pass_cli(remote_model, delegate_args, bench_bin):
  args = [
    bench_bin,
    f"--graph={remote_model}",
    f"--warmup_runs={WARMUP}",
    f"--num_runs={RUNS}",
    *delegate_args,
  ]
  cmd = " ".join(shlex.quote(a) for a in args)
  r = adb_shell(cmd)
  if r.returncode != 0:
    msg = r.stderr.strip() or r.stdout.strip() or "benchmark_model failed"
    return False, msg
  return True, None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default=DEVICE_ROOT,
                  help="Device arch root (default: /data/local/tmp/archs).")
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--tasks", nargs="*", default=list(TASKS.keys()),
                  choices=list(TASKS.keys()))
  ap.add_argument("--use-gpu", action="store_true",
                  help="Use GPU delegate for inference pass.")
  ap.add_argument("--use-npu", action="store_true",
                  help="Use NNAPI delegate for inference pass.")
  ap.add_argument("--benchmark-bin", default=BENCH_BIN_DEFAULT,
                  help="Path to benchmark_model binary on device.")
  ap.add_argument("--use-activity", action="store_true",
                  help="Use BenchmarkModelActivity instead of benchmark_model.")
  args = ap.parse_args()

  r = adb("get-state")
  if r.returncode != 0 or "device" not in r.stdout:
    print(f"adb get-state: {r.stdout.strip()} {r.stderr.strip()}", file=sys.stderr)
    sys.exit(2)

  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  if ERROR_LOG.exists():
    ERROR_LOG.unlink()

  if args.use_gpu and args.use_npu:
    raise SystemExit("choose only one of --use-gpu or --use-npu")

  delegate_args = BASE_ARGS
  if args.use_npu:
    delegate_args = ["--use_nnapi=true", *BASE_ARGS]
  elif args.use_gpu:
    delegate_args = ["--use_gpu=true", *BASE_ARGS]

  if not args.use_activity:
    if not device_executable(args.benchmark_bin):
      raise SystemExit(
        f"benchmark_model not found at {args.benchmark_bin}; "
        "push the binary or use --use-activity"
      )

  device_root = args.root
  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  else:
    indices = list_device_archs(device_root)

  indices = indices[args.start:]
  if args.limit:
    indices = indices[:args.limit]

  tasks = list(args.tasks)

  total = len(indices) * len(tasks)
  done = 0
  ok = 0
  missing = 0
  err = 0
  remaining = total
  progress_lock = threading.Lock()
  stop_event = threading.Event()
  max_retries = 1
  show_progress(done, total, ok, missing, err)

  def update_progress(d_done=0, d_ok=0, d_missing=0, d_err=0):
    nonlocal done, ok, missing, err, remaining
    with progress_lock:
      done += d_done
      ok += d_ok
      missing += d_missing
      err += d_err
      remaining -= d_done
      show_progress(done, total, ok, missing, err)
      if remaining <= 0:
        stop_event.set()

  def model_path(arch_idx, task):
    return f"{device_root}/arch_{arch_idx}/{task}_litert.tflite"

  def export_model(arch_idx, task, tmp_dir):
    remote_dir = f"{device_root}/arch_{arch_idx}"
    adb_shell(f"mkdir -p {remote_dir}")
    remote_model = model_path(arch_idx, task)
    local_path = tmp_dir / f"arch_{arch_idx}_{task}.tflite"
    try:
      with suppress_output():
        export_tflite(arch_idx, TASKS[task]["input_shape"],
                      TASKS[task]["num_classes"], local_path)
      push = adb("push", str(local_path), remote_model)
      if push.returncode != 0:
        raise RuntimeError(push.stderr.strip() or push.stdout.strip())
      return True, None
    except Exception as e:
      return False, str(e)[:200]
    finally:
      if local_path.exists():
        local_path.unlink()

  def run_infer(remote_model):
    if args.use_activity:
      return run_one_pass_activity(remote_model, delegate_args)
    return run_one_pass_cli(remote_model, delegate_args, args.benchmark_bin)

  export_q = queue.Queue()
  infer_q = queue.Queue()

  with tempfile.TemporaryDirectory(prefix="pixel_export_") as tmp:
    tmp_dir = Path(tmp)

    def export_worker():
      while not stop_event.is_set():
        try:
          arch_idx, task, attempt = export_q.get(timeout=0.2)
        except queue.Empty:
          continue
        ok_exp, msg = export_model(arch_idx, task, tmp_dir)
        if ok_exp:
          infer_q.put((arch_idx, task, attempt))
        else:
          with open(ERROR_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{arch_idx}\t{task}\texport\t{msg or 'error'}\n")
          update_progress(d_done=1, d_missing=1)
        export_q.task_done()

    export_thread = threading.Thread(target=export_worker, daemon=True)
    export_thread.start()

    for arch_idx in indices:
      for task in tasks:
        remote_model = model_path(arch_idx, task)
        if device_exists(remote_model):
          infer_q.put((arch_idx, task, 0))
        else:
          export_q.put((arch_idx, task, 0))

    try:
      while not stop_event.is_set():
        try:
          arch_idx, task, attempt = infer_q.get(timeout=0.2)
        except queue.Empty:
          with progress_lock:
            if remaining <= 0:
              break
          continue

        remote_model = model_path(arch_idx, task)
        ok_run, msg = run_infer(remote_model)
        if ok_run:
          update_progress(d_done=1, d_ok=1)
        else:
          if attempt < max_retries:
            adb_shell(f"rm -f {remote_model}")
            export_q.put((arch_idx, task, attempt + 1))
          else:
            with open(ERROR_LOG, "a", encoding="utf-8") as fh:
              fh.write(f"{arch_idx}\t{task}\tinfer\t{msg or 'error'}\n")
            update_progress(d_done=1, d_err=1)
        infer_q.task_done()
    except KeyboardInterrupt:
      stop_event.set()
    finally:
      stop_event.set()
      export_thread.join()

  show_progress(done, total, ok, missing, err, final=True)
  print(f"ok={ok} missing={missing} err={err}")
  if err or missing:
    print(f"see {ERROR_LOG}")


if __name__ == "__main__":
  main()
