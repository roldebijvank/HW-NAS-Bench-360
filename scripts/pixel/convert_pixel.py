"""Host-side: export TFLite (litert) per arch/task, push to Pixel, delete local.

Default arch set is non-isomorphic NB201 reps (~6466). Models are pushed to:
  /data/local/tmp/feas/arch_<N>/<task>_litert.tflite
Run as module so imports work:
  python -m scripts.pixel.convert_pixel
"""
import argparse
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
  from scripts.utils.arch_iter import non_iso_indices
  from scripts.utils.convert_utils import export_tflite
  from scripts.utils.task_specs import TASKS
except ModuleNotFoundError:
  project_root = Path(__file__).resolve().parents[2]
  if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
  from scripts.utils.arch_iter import non_iso_indices
  from scripts.utils.convert_utils import export_tflite
  from scripts.utils.task_specs import TASKS


DEFAULT_ROOT = "/data/local/tmp/archs"


def adb(*args, check=False):
  return subprocess.run(["adb", *args], capture_output=True, text=True, check=check)


def adb_shell(cmd, check=False):
  return subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, check=check)


def device_exists(path):
  return adb_shell(f"test -f {path} && echo Y || echo N").stdout.strip() == "Y"


def adb_push(src, dst):
  return subprocess.run(["adb", "push", str(src), dst],
                        capture_output=True, text=True)


def parse_arch_list(path):
  archs = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      s = line.strip()
      if not s or s.startswith("#"):
        continue
      archs.append(int(s))
  return archs


@contextlib.contextmanager
def suppress_output():
  # Silence native logs from conversion (stdout/stderr).
  sys.stdout.flush()
  sys.stderr.flush()
  devnull = os.open(os.devnull, os.O_WRONLY)
  old_out = os.dup(1)
  old_err = os.dup(2)
  try:
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    yield
  finally:
    os.dup2(old_out, 1)
    os.dup2(old_err, 2)
    os.close(old_out)
    os.close(old_err)
    os.close(devnull)


def render_progress(done, total, ok, skipped, err, width=30):
  if total <= 0:
    return "0/0 [------------------------------] 100.0%"
  pct = done / total
  filled = int(round(width * pct))
  bar = "=" * filled + "-" * (width - filled)
  return (f"\r{done}/{total} [{bar}] {pct * 100:5.1f}% "
          f"ok={ok} skip={skipped} err={err}")


def show_progress(done, total, ok, skipped, err, final=False):
  end = "\n" if final else ""
  print(render_progress(done, total, ok, skipped, err), end=end, flush=True)


def silence_torch_elastic_redirects():
  logger = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
  logger.disabled = True


def configure_threads(threads):
  if threads and threads > 0:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
      import torch
      torch.set_num_threads(threads)
      if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(threads)
    except Exception:
      pass


_WORK_ROOT = None
_WORK_TASKS = None
_WORK_OVERWRITE = None
_WORK_TMP_DIR = None


def init_worker(root, tasks, overwrite, threads):
  global _WORK_ROOT, _WORK_TASKS, _WORK_OVERWRITE, _WORK_TMP_DIR
  _WORK_ROOT = root
  _WORK_TASKS = tasks
  _WORK_OVERWRITE = overwrite
  _WORK_TMP_DIR = Path(tempfile.mkdtemp(prefix="pixel_conv_worker_"))
  silence_torch_elastic_redirects()
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  configure_threads(threads)


def process_arch(arch_idx):
  ok = 0
  skipped = 0
  err = 0
  for task in _WORK_TASKS:
    spec = TASKS[task]
    remote_dir = f"{_WORK_ROOT}/arch_{arch_idx}"
    remote_path = f"{remote_dir}/{task}_litert.tflite"
    tmp_remote = f"{remote_path}.tmp"

    if not _WORK_OVERWRITE and device_exists(remote_path):
      skipped += 1
      continue

    local_path = _WORK_TMP_DIR / f"arch_{arch_idx}_{task}_litert.tflite"
    try:
      with suppress_output():
        export_tflite(arch_idx, spec["input_shape"],
                      spec["num_classes"], local_path)
      adb_shell(f"mkdir -p {remote_dir}", check=True)
      adb_shell(f"rm -f {tmp_remote}")
      push = adb_push(local_path, tmp_remote)
      if push.returncode != 0:
        raise RuntimeError(push.stderr.strip() or push.stdout.strip())
      adb_shell(f"mv {tmp_remote} {remote_path}", check=True)
      ok += 1
    except Exception:
      err += 1
      adb_shell(f"rm -f {tmp_remote}")
    finally:
      if local_path.exists():
        local_path.unlink()

  return len(_WORK_TASKS), ok, skipped, err


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default=DEFAULT_ROOT,
                  help="on-device dir containing arch_<N>/ subdirs")
  ap.add_argument("--task", action="append", default=[],
                  help="task name (repeatable). default: all")
  ap.add_argument("--arch", action="append", type=int, default=[],
                  help="arch idx (repeatable). default: non-iso set")
  ap.add_argument("--arch-list", type=Path, default=None,
                  help="text file with arch idx per line")
  ap.add_argument("--max", type=int, default=None,
                  help="limit number of archs (after selection)")
  ap.add_argument("--overwrite", action="store_true",
                  help="overwrite on-device files if present")
  ap.add_argument("--workers", type=int, default=8,
                  help="parallel workers (one arch per worker)")
  ap.add_argument("--threads", type=int, default=None,
                  help="torch threads per worker (default: 8 if workers=1, else 1)")
  args = ap.parse_args()

  workers = max(1, args.workers)
  threads = args.threads
  if threads is None:
    threads = 1 if workers > 1 else 8
  print(f"workers={workers} threads={threads}")

  silence_torch_elastic_redirects()

  r = adb("get-state")
  if r.returncode != 0 or "device" not in r.stdout:
    print(f"adb get-state: {r.stdout.strip()} {r.stderr.strip()}", file=sys.stderr)
    sys.exit(2)

  tasks = args.task if args.task else list(TASKS.keys())
  bad = [t for t in tasks if t not in TASKS]
  if bad:
    print(f"unknown task(s): {bad}", file=sys.stderr)
    sys.exit(2)

  if args.arch:
    archs = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    archs = parse_arch_list(args.arch_list)
  else:
    archs = non_iso_indices()

  if args.max is not None:
    archs = archs[:args.max]

  if not archs:
    print("no archs selected", file=sys.stderr)
    sys.exit(2)

  print(f"archs={len(archs)} tasks={tasks}")

  ok = 0
  skipped = 0
  err = 0
  total_items = len(archs) * len(tasks)
  done = 0
  show_progress(done, total_items, ok, skipped, err)
  ctx = mp.get_context("spawn")
  with ctx.Pool(processes=workers,
                initializer=init_worker,
                initargs=(args.root, tasks, args.overwrite, threads)) as pool:
    it = iter(archs)
    active = []
    interrupted = False

    def submit_next():
      try:
        arch_idx = next(it)
      except StopIteration:
        return False
      active.append(pool.apply_async(process_arch, (arch_idx,)))
      return True

    for _ in range(workers):
      if not submit_next():
        break

    while active:
      try:
        progressed = False
        for i in range(len(active) - 1, -1, -1):
          ar = active[i]
          if not ar.ready():
            continue
          progressed = True
          active.pop(i)
          try:
            d, o, s, e = ar.get()
          except Exception:
            d, o, s, e = len(tasks), 0, 0, len(tasks)
          done += d; ok += o; skipped += s; err += e
          show_progress(done, total_items, ok, skipped, err)
          if not interrupted:
            submit_next()
        if not progressed:
          time.sleep(0.05)
      except KeyboardInterrupt:
        interrupted = True
    pool.close()
    pool.join()

  show_progress(done, total_items, ok, skipped, err, final=True)
  print("=" * 80)
  print("SUMMARY")
  print("=" * 80)
  print(f"ok / skipped / err : {ok} / {skipped} / {err}")


if __name__ == "__main__":
  main()
