"""Phase 1 (Pixel): export tflite per (arch, task) and adb push to device.
No measurement. Skips arch dirs already on device unless --overwrite.

Run as module so imports work:
  uv run python -m scripts.pixel.convert
"""
import argparse
import multiprocessing as mp
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts.utils.arch_iter import non_iso_indices
from scripts.utils.convert_utils import export_tflite
from scripts.utils.runner_utils import (parse_arch_list, render_progress,
                                        show_progress,
                                        silence_torch_elastic_redirects,
                                        suppress_output)
from scripts.utils.task_specs import TASKS

ROOT = Path(__file__).resolve().parents[2]
DEVICE_ROOT = "/data/local/tmp/archs"


def adb(*args, check=False):
  return subprocess.run(["adb", *args], capture_output=True, text=True, check=check)


def adb_shell(cmd, check=False):
  return subprocess.run(["adb", "shell", cmd],
                        capture_output=True, text=True, check=check)


def device_arch_exists(arch_idx, tasks):
  """True iff arch dir contains all expected tflites."""
  remote_dir = f"{DEVICE_ROOT}/arch_{arch_idx}"
  expected = " ".join(f"{remote_dir}/{t}_litert.tflite" for t in tasks)
  r = adb_shell(f"for f in {expected}; do [ -f $f ] || exit 1; done && echo Y || echo N")
  return r.stdout.strip().endswith("Y")


_WORK_TASKS = None
_WORK_TMP = None
_WORK_OVERWRITE = None


def init_worker(tasks, overwrite):
  global _WORK_TASKS, _WORK_TMP, _WORK_OVERWRITE
  _WORK_TASKS = tasks
  _WORK_OVERWRITE = overwrite
  _WORK_TMP = Path(tempfile.mkdtemp(prefix="pixel_convert_"))
  silence_torch_elastic_redirects()
  signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_arch(arch_idx):
  ok = 0; skipped = 0; err = 0
  failures = []
  if not _WORK_OVERWRITE and device_arch_exists(arch_idx, _WORK_TASKS):
    return arch_idx, len(_WORK_TASKS), 0, len(_WORK_TASKS), 0, failures

  remote_dir = f"{DEVICE_ROOT}/arch_{arch_idx}"
  remote_tmp = f"{DEVICE_ROOT}/arch_{arch_idx}.tmp"
  adb_shell(f"rm -rf {remote_tmp} && mkdir -p {remote_tmp}")

  for task in _WORK_TASKS:
    spec = TASKS[task]
    local_path = _WORK_TMP / f"arch_{arch_idx}_{task}.tflite"
    try:
      with suppress_output():
        export_tflite(arch_idx, spec["input_shape"], spec["num_classes"], local_path)
      push = adb("push", str(local_path), f"{remote_tmp}/{task}_litert.tflite")
      if push.returncode != 0:
        raise RuntimeError(push.stderr.strip() or push.stdout.strip())
      ok += 1
    except Exception as e:
      err += 1
      failures.append((task, str(e)[:200]))
    finally:
      if local_path.exists(): local_path.unlink()

  if err == 0:
    adb_shell(f"rm -rf {remote_dir} && mv {remote_tmp} {remote_dir}")
  else:
    adb_shell(f"rm -rf {remote_tmp}")
  return arch_idx, len(_WORK_TASKS), ok, skipped, err, failures


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--overwrite", action="store_true")
  ap.add_argument("--workers", type=int, default=8)
  args = ap.parse_args()

  r = adb("get-state")
  if r.returncode != 0 or "device" not in r.stdout:
    print(f"adb get-state: {r.stdout.strip()} {r.stderr.strip()}", file=sys.stderr)
    sys.exit(2)

  silence_torch_elastic_redirects()
  adb_shell(f"mkdir -p {DEVICE_ROOT}", check=True)

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

  tasks = list(TASKS.keys())
  workers = max(1, args.workers)
  print(f"archs={len(indices)} tasks={tasks} workers={workers}")

  total = len(indices) * len(tasks)
  done = 0; ok = 0; skipped = 0; err = 0
  show_progress(done, total, ok, skipped, err)

  fail_log = ROOT / "results" / "convert_pixel_failed.log"
  fail_log.parent.mkdir(parents=True, exist_ok=True)
  failed_archs = []

  ctx = mp.get_context("spawn")
  with ctx.Pool(processes=workers,
                initializer=init_worker,
                initargs=(tasks, args.overwrite)) as pool:
    try:
      for arch_idx, d, o, s, e, failures in pool.imap_unordered(process_arch, indices):
        done += d; ok += o; skipped += s; err += e
        if failures:
          failed_archs.append(arch_idx)
          with open(fail_log, "a") as fh:
            for task, msg in failures:
              fh.write(f"{arch_idx}\t{task}\t{msg}\n")
          # newline so stderr line appears above progress bar
          print(f"\narch {arch_idx}: {len(failures)} fail "
                f"({', '.join(t for t, _ in failures)})",
                file=sys.stderr, flush=True)
        show_progress(done, total, ok, skipped, err)
    except KeyboardInterrupt:
      pool.terminate(); pool.join()

  show_progress(done, total, ok, skipped, err, final=True)
  print(f"ok / skipped / err : {ok} / {skipped} / {err}")
  if failed_archs:
    print(f"{len(failed_archs)} arch(s) had failures; see {fail_log}")


if __name__ == "__main__":
  main()
