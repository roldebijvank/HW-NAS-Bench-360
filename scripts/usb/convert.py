"""Export per-arch artifacts for one task to a local directory (e.g., USB).

Run as module so imports work:
  uv run python -m scripts.usb.convert --task cifar100 --dest /Volumes/USB/archs
"""
import argparse
import atexit
import multiprocessing as mp
import os
import shutil
import signal
import tempfile
from pathlib import Path

from scripts.utils.arch_iter import non_iso_indices
from scripts.utils.convert_utils import RUNTIMES
from scripts.utils.runner_utils import (parse_arch_list, show_progress,
                                        silence_torch_elastic_redirects,
                                        suppress_output)
from scripts.utils.task_specs import TASKS

ROOT = Path(__file__).resolve().parents[2]


def _int_env(name, default):
  val = os.environ.get(name)
  if val is None:
    return default
  try:
    return int(val)
  except ValueError:
    return default


def _default_workers():
  cpu = os.cpu_count() or 8
  cap = _int_env("BEP_MAX_WORKERS", 0)
  if cap > 0:
    return max(1, min(cpu, cap))
  return max(1, min(cpu, 4))


def _default_torch_threads(workers):
  cpu = os.cpu_count() or 1
  return max(1, cpu // max(1, workers))


def _apply_thread_env(threads):
  if threads <= 0:
    return
  for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, str(threads))


def _arch_has_task_files(arch_dir, task):
  for runtime, (ext, _) in RUNTIMES.items():
    if not (arch_dir / f"{task}_{runtime}.{ext}").is_file():
      return False
  return True


def local_existing(task, dest_root):
  """Set of arch_idx that have all runtime artifacts for `task`."""
  if not dest_root.exists():
    return set()
  out = set()
  for p in dest_root.iterdir():
    if not (p.is_dir() and p.name.startswith("arch_")):
      continue
    try:
      arch_idx = int(p.name.split("_", 1)[1])
    except ValueError:
      continue
    if _arch_has_task_files(p, task):
      out.add(arch_idx)
  return out


_WORK_TASK = None
_WORK_TMP = None
_WORK_DEST = None
_WORK_TORCH_THREADS = None
_WORK_TORCH_INTEROP_THREADS = None


def init_worker(task, dest_root, torch_threads, torch_interop_threads):
  global _WORK_TASK, _WORK_TMP, _WORK_DEST
  global _WORK_TORCH_THREADS, _WORK_TORCH_INTEROP_THREADS
  _WORK_TASK = task
  _WORK_DEST = dest_root
  _WORK_TORCH_THREADS = torch_threads
  _WORK_TORCH_INTEROP_THREADS = torch_interop_threads
  _WORK_TMP = Path(tempfile.mkdtemp(prefix="usb_convert_"))
  atexit.register(shutil.rmtree, _WORK_TMP, ignore_errors=True)
  silence_torch_elastic_redirects()
  if _WORK_TORCH_THREADS is not None or _WORK_TORCH_INTEROP_THREADS is not None:
    try:
      import torch
      if _WORK_TORCH_THREADS is not None:
        torch.set_num_threads(_WORK_TORCH_THREADS)
      if _WORK_TORCH_INTEROP_THREADS is not None:
        torch.set_num_interop_threads(_WORK_TORCH_INTEROP_THREADS)
    except Exception:
      pass
  signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_arch(arch_idx):
  spec = TASKS[_WORK_TASK]
  stage = _WORK_TMP / f"arch_{arch_idx}"
  if stage.exists():
    shutil.rmtree(stage)
  stage.mkdir()
  failures = []
  try:
    for runtime, (ext, fn) in RUNTIMES.items():
      out = stage / f"{_WORK_TASK}_{runtime}.{ext}"
      try:
        with suppress_output():
          fn(arch_idx, spec["input_shape"], spec["num_classes"], out)
      except Exception as e:
        failures.append((runtime, str(e)[:200]))
    if failures:
      return arch_idx, 0, len(failures), failures

    dest_dir = _WORK_DEST / f"arch_{arch_idx}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for runtime, (ext, _) in RUNTIMES.items():
      src = stage / f"{_WORK_TASK}_{runtime}.{ext}"
      try:
        shutil.copy2(src, dest_dir / src.name)
      except Exception as e:
        failures.append(("copy", str(e)[:200]))
    if failures:
      return arch_idx, 0, len(failures), failures
    return arch_idx, 3, 0, []
  finally:
    shutil.rmtree(stage, ignore_errors=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True, choices=list(TASKS))
  ap.add_argument("--dest", required=True, type=Path,
                  help="Directory to store arch_* folders (e.g., USB mount).")
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--overwrite", action="store_true")
  ap.add_argument("--workers", type=int, default=_default_workers())
  ap.add_argument("--torch-threads", type=int, default=None,
                  help="Torch intraop threads per worker (default: cpu/workers; 0 disables)")
  ap.add_argument("--torch-interop-threads", type=int, default=1,
                  help="Torch interop threads per worker (default: 1; 0 disables)")
  ap.add_argument("--maxtasksperchild", type=int, default=20,
                  help="Recycle worker after N archs (0 disables)")
  args = ap.parse_args()

  dest_root = args.dest.expanduser()
  if dest_root.exists() and not dest_root.is_dir():
    raise SystemExit("--dest must be a directory")
  dest_root.mkdir(parents=True, exist_ok=True)

  silence_torch_elastic_redirects()

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
  if args.limit:
    indices = indices[:args.limit]

  if args.overwrite:
    pending = indices
    skipped = 0
  else:
    print("listing existing arch dirs...", flush=True)
    have = local_existing(args.task, dest_root)
    pending = [i for i in indices if i not in have]
    skipped = len(indices) - len(pending)
    print(f"  {len(have)} complete for {args.task}; pending {len(pending)}")

  workers = max(1, args.workers)
  torch_threads = args.torch_threads
  if torch_threads is None:
    torch_threads = _default_torch_threads(workers)
  if torch_threads is not None and torch_threads <= 0:
    torch_threads = None
  torch_interop = args.torch_interop_threads
  if torch_interop is not None and torch_interop <= 0:
    torch_interop = None
  if torch_threads is not None:
    _apply_thread_env(torch_threads)
  maxtasks = None if args.maxtasksperchild <= 0 else args.maxtasksperchild
  print(f"task={args.task} archs={len(pending)} workers={workers} "
        f"torch_threads={torch_threads if torch_threads is not None else 'default'} "
        f"maxtasksperchild={maxtasks if maxtasks is not None else 'none'}")

  total = len(pending) + skipped
  done = skipped
  ok_a = 0
  err_a = 0
  fail_log = ROOT / "results" / "convert_usb_failed.log"
  fail_log.parent.mkdir(parents=True, exist_ok=True)
  failed_archs = []

  show_progress(done, total, ok_a, skipped, err_a)
  ctx = mp.get_context("spawn")
  with ctx.Pool(processes=workers,
                initializer=init_worker,
                initargs=(args.task, dest_root, torch_threads, torch_interop),
                maxtasksperchild=maxtasks) as pool:
    try:
      for arch_idx, o, e, failures in pool.imap_unordered(process_arch, pending):
        done += 1
        if e == 0:
          ok_a += 1
        else:
          err_a += 1
        if failures:
          failed_archs.append(arch_idx)
          with open(fail_log, "a") as fh:
            for rt, msg in failures:
              fh.write(f"{arch_idx}\t{args.task}\t{rt}\t{msg}\n")
          print(f"\narch {arch_idx}: {len(failures)} fail "
                f"({', '.join(rt for rt, _ in failures)})",
                file=os.sys.stderr, flush=True)
        show_progress(done, total, ok_a, skipped, err_a)
    except KeyboardInterrupt:
      pool.terminate()
      pool.join()

  show_progress(done, total, ok_a, skipped, err_a, final=True)
  print(f"ok / skipped / err : {ok_a} / {skipped} / {err_a}")
  if failed_archs:
    print(f"{len(failed_archs)} arch(s) had failures; see {fail_log}")


if __name__ == "__main__":
  main()
