"""One-pass inference check for local arch_* folders.

Run as module so imports work:
  uv run python -m scripts.usb.infer --root /Volumes/USB/archs
"""
import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.utils.runner_utils import parse_arch_list, show_progress
from scripts.utils.task_specs import TASKS


RUNTIME_EXT = {
  "litert": "tflite",
  "onnx": "onnx",
  "torchmobile": "ptl",
}


def list_arch_dirs(root):
  out = {}
  if not root.exists():
    return out
  for p in root.iterdir():
    if not (p.is_dir() and p.name.startswith("arch_")):
      continue
    try:
      idx = int(p.name.split("_", 1)[1])
    except ValueError:
      continue
    out[idx] = p
  return out


def make_step_litert(path, x_np):
  try:
    from tflite_runtime.interpreter import Interpreter
  except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
  interp = Interpreter(model_path=str(path), num_threads=1)
  interp.allocate_tensors()
  inp = interp.get_input_details()[0]
  out = interp.get_output_details()[0]
  def step():
    interp.set_tensor(inp["index"], x_np)
    interp.invoke()
    interp.get_tensor(out["index"])
  return step


def make_step_onnx(path, x_np):
  import onnxruntime as ort
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1
  so.inter_op_num_threads = 1
  sess = ort.InferenceSession(str(path), sess_options=so,
                              providers=["CPUExecutionProvider"])
  name = sess.get_inputs()[0].name
  def step():
    sess.run(None, {name: x_np})
  return step


def make_step_torchmobile(path, x_np):
  import torch
  torch.set_num_threads(1)
  m = torch.jit.load(str(path))
  m.eval()
  x = torch.from_numpy(x_np)
  def step():
    with torch.no_grad():
      m(x)
  return step


MAKERS = {
  "litert": make_step_litert,
  "onnx": make_step_onnx,
  "torchmobile": make_step_torchmobile,
}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", required=True, type=Path,
                  help="Folder containing arch_* dirs.")
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--tasks", nargs="*", default=list(TASKS.keys()),
                  choices=list(TASKS.keys()))
  ap.add_argument("--runtimes", nargs="*", default=list(RUNTIME_EXT.keys()),
                  choices=list(RUNTIME_EXT.keys()))
  ap.add_argument("--max-list", type=int, default=10)
  ap.add_argument("--show-all", action="store_true")
  ap.add_argument("--stop-on-error", action="store_true")
  ap.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                  help="Thread workers per runtime.")
  ap.add_argument("--no-subprocess", action="store_true",
                  help="Run all runtimes in this process.")
  args = ap.parse_args()

  root = args.root.expanduser()
  arch_dirs = list_arch_dirs(root)
  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  else:
    indices = sorted(arch_dirs.keys())

  indices = indices[args.start:]
  if args.limit:
    indices = indices[:args.limit]

  tasks = list(args.tasks)
  runtimes = list(args.runtimes)

  if not args.no_subprocess and len(runtimes) > 1:
    for rt in runtimes:
      cmd = [sys.executable, "-m", "scripts.usb.infer",
             "--no-subprocess",
             "--root", str(args.root),
             "--runtimes", rt]
      for a in args.arch:
        cmd += ["--arch", str(a)]
      if args.arch_list:
        cmd += ["--arch-list", str(args.arch_list)]
      if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
      if args.start:
        cmd += ["--start", str(args.start)]
      if args.tasks:
        cmd += ["--tasks", *args.tasks]
      if args.max_list != 10:
        cmd += ["--max-list", str(args.max_list)]
      if args.show_all:
        cmd += ["--show-all"]
      if args.stop_on_error:
        cmd += ["--stop-on-error"]
      print(f"running runtime={rt}")
      r = subprocess.run(cmd)
      if r.returncode != 0:
        raise SystemExit(r.returncode)
    return

  input_by_task = {}
  for t in tasks:
    shape = TASKS[t]["input_shape"]
    input_by_task[t] = np.random.randn(1, *shape).astype(np.float32)

  missing_archs = [i for i in indices if i not in arch_dirs]

  def show_list(label, items):
    if not items:
      return
    show = items if args.show_all or len(items) <= args.max_list else items[:args.max_list]
    suffix = "" if len(show) == len(items) else f" (+{len(items) - len(show)} more)"
    print(f"{label}: {show}{suffix}")

  total = len(indices) * len(tasks) * len(runtimes)
  done = 0
  ok = 0
  missing = 0
  err = 0

  if len(runtimes) == 1:
    fail_log = root / f"infer_check_failed_{runtimes[0]}.log"
    arch_fail_log = root / f"infer_check_failed_archs_{runtimes[0]}.txt"
  else:
    fail_log = root / "infer_check_failed.log"
    arch_fail_log = root / "infer_check_failed_archs.txt"
  if fail_log.exists():
    fail_log.unlink()
  if arch_fail_log.exists():
    arch_fail_log.unlink()

  print(f"root={root} archs={len(indices)} tasks={tasks} runtimes={runtimes}")
  show_list("missing_archs sample", missing_archs)
  show_progress(done, total, ok, missing, err)

  failed_archs = set()
  jobs = []
  for arch_idx in indices:
    arch_dir = arch_dirs.get(arch_idx)
    if arch_dir is None:
      missing += len(tasks) * len(runtimes)
      done += len(tasks) * len(runtimes)
      failed_archs.add(arch_idx)
      show_progress(done, total, ok, missing, err)
      continue
    for task in tasks:
      x_np = input_by_task[task]
      for runtime in runtimes:
        ext = RUNTIME_EXT[runtime]
        path = arch_dir / f"{task}_{runtime}.{ext}"
        jobs.append((arch_idx, task, runtime, path, x_np))

  def run_job(job):
    arch_idx, task, runtime, path, x_np = job
    if not path.is_file():
      return "missing", arch_idx, task, runtime, "missing file"
    try:
      step = MAKERS[runtime](path, x_np)
      step()
      return "ok", arch_idx, task, runtime, None
    except Exception as e:
      return "err", arch_idx, task, runtime, str(e)[:200]

  if jobs:
    workers = max(1, min(args.workers, len(jobs)))
    stop = False
    if workers == 1:
      for status, arch_idx, task, runtime, msg in (run_job(j) for j in jobs):
        done += 1
        if status == "ok":
          ok += 1
        elif status == "missing":
          missing += 1
          failed_archs.add(arch_idx)
          with open(fail_log, "a", encoding="utf-8") as fh:
            fh.write(f"{arch_idx}\t{task}\t{runtime}\tmissing file\n")
          if args.stop_on_error:
            stop = True
        else:
          err += 1
          failed_archs.add(arch_idx)
          with open(fail_log, "a", encoding="utf-8") as fh:
            fh.write(f"{arch_idx}\t{task}\t{runtime}\t{msg or 'error'}\n")
          if args.stop_on_error:
            stop = True
        show_progress(done, total, ok, missing, err)
        if stop:
          break
    else:
      with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_job, j) for j in jobs]
        for fut in concurrent.futures.as_completed(futures):
          status, arch_idx, task, runtime, msg = fut.result()
          done += 1
          if status == "ok":
            ok += 1
          elif status == "missing":
            missing += 1
            failed_archs.add(arch_idx)
            with open(fail_log, "a", encoding="utf-8") as fh:
              fh.write(f"{arch_idx}\t{task}\t{runtime}\tmissing file\n")
            if args.stop_on_error:
              stop = True
          else:
            err += 1
            failed_archs.add(arch_idx)
            with open(fail_log, "a", encoding="utf-8") as fh:
              fh.write(f"{arch_idx}\t{task}\t{runtime}\t{msg or 'error'}\n")
            if args.stop_on_error:
              stop = True
          show_progress(done, total, ok, missing, err)
          if stop:
            for f in futures:
              f.cancel()
            break

  if failed_archs:
    with open(arch_fail_log, "w", encoding="utf-8") as fh:
      for arch_idx in sorted(failed_archs):
        fh.write(f"{arch_idx}\n")

  show_progress(done, total, ok, missing, err, final=True)
  print(f"ok={ok} missing={missing} err={err}")
  if err or missing:
    print(f"see {fail_log}")


if __name__ == "__main__":
  main()
