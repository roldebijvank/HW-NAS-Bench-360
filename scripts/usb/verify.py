"""Verify local arch_* folders contain all task/runtime artifacts."""
import argparse
import concurrent.futures
import os
import threading
from pathlib import Path

from scripts.utils.convert_utils import RUNTIMES
from scripts.utils.runner_utils import (parse_arch_list, show_progress,
                                        silence_torch_elastic_redirects,
                                        suppress_output)
from scripts.utils.task_specs import TASKS


_ONNX_LOCK = threading.Lock()


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


def expected_models(tasks, runtimes):
  out = {}
  for t in tasks:
    for r, (ext, fn) in runtimes.items():
      out[f"{t}_{r}.{ext}"] = (t, r, fn)
  return out


def scan_incomplete(indices, arch_dirs, expected, strict):
  missing_archs = [i for i in indices if i not in arch_dirs]
  incomplete = {}
  for i in indices:
    p = arch_dirs.get(i)
    if p is None:
      continue
    present = {f.name for f in p.iterdir() if f.is_file()}
    missing = sorted(expected - present)
    unexpected = sorted(present - expected) if strict else []
    if missing or unexpected:
      incomplete[i] = {"missing": missing, "unexpected": unexpected}
  return missing_archs, incomplete


def build_export_jobs(indices, missing_archs, incomplete, expected, expected_map,
                      arch_dirs, root):
  jobs = []
  for arch_idx in indices:
    rec = incomplete.get(arch_idx)
    if arch_idx in missing_archs:
      arch_dir = root / f"arch_{arch_idx}"
      arch_dir.mkdir(parents=True, exist_ok=True)
      missing = sorted(expected)
    elif rec and rec.get("missing"):
      arch_dir = arch_dirs.get(arch_idx)
      if arch_dir is None:
        continue
      missing = rec["missing"]
    else:
      continue

    for fname in missing:
      spec = expected_map.get(fname)
      if spec is None:
        continue
      task, runtime, fn = spec
      task_spec = TASKS[task]
      jobs.append((arch_idx, arch_dir, fname, runtime, fn, task_spec))
  return jobs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", required=True, type=Path,
                  help="Folder containing arch_* dirs.")
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--max-list", type=int, default=10)
  ap.add_argument("--show-all", action="store_true")
  ap.add_argument("--strict", action="store_true",
                  help="Report unexpected files too.")
  ap.add_argument("--export-missing", action="store_true",
                  help="Export missing models into arch_* folders.")
  ap.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                  help="Thread workers for --export-missing.")
  args = ap.parse_args()

  root = args.root.expanduser()
  arch_dirs = list_arch_dirs(root)
  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  else:
    indices = sorted(arch_dirs.keys())

  tasks = list(TASKS.keys())
  expected_map = expected_models(tasks, RUNTIMES)
  expected = set(expected_map.keys())
  expected_count = len(expected)

  missing_archs, incomplete = scan_incomplete(indices, arch_dirs, expected, args.strict)

  def show_list(label, items):
    if not items:
      return
    show = items if args.show_all or len(items) <= args.max_list else items[:args.max_list]
    suffix = "" if len(show) == len(items) else f" (+{len(items) - len(show)} more)"
    print(f"{label}: {show}{suffix}")

  print(f"root={root} archs={len(indices)} expected_per_arch={expected_count}")
  print(f"missing_archs={len(missing_archs)} incomplete_archs={len(incomplete)}")
  show_list("missing_archs sample", missing_archs)

  if incomplete:
    print("incomplete_models sample:")
    items = list(incomplete.items())
    if not args.show_all and len(items) > args.max_list:
      items = items[:args.max_list]
    for arch_idx, rec in items:
      parts = []
      if rec["missing"]:
        parts.append("missing=" + ",".join(rec["missing"]))
      if rec["unexpected"]:
        parts.append("unexpected=" + ",".join(rec["unexpected"]))
      print(f"  arch_{arch_idx}: " + " ".join(parts))
    if not args.show_all and len(incomplete) > args.max_list:
      print(f"  (+{len(incomplete) - args.max_list} more)")

  if args.export_missing and (missing_archs or incomplete):
    silence_torch_elastic_redirects()
    export_ok = 0
    export_err = 0
    fail_log = root / "export_missing_failed.log"
    if fail_log.exists():
      fail_log.unlink()
    jobs = build_export_jobs(indices, missing_archs, incomplete, expected,
                             expected_map, arch_dirs, root)

    def _export_one(job):
      arch_idx, arch_dir, fname, runtime, fn, task_spec = job
      out_path = arch_dir / fname
      try:
        if runtime == "onnx":
          with _ONNX_LOCK:
            with suppress_output():
              fn(arch_idx, task_spec["input_shape"],
                 task_spec["num_classes"], out_path)
        else:
          with suppress_output():
            fn(arch_idx, task_spec["input_shape"],
               task_spec["num_classes"], out_path)
        return True, None
      except Exception as e:
        return False, str(e)[:200]

    if jobs:
      workers = max(1, min(args.workers, len(jobs)))
      total = len(jobs)
      done = 0
      show_progress(done, total, export_ok, 0, export_err)
      with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(_export_one, job): job for job in jobs}
        for fut in concurrent.futures.as_completed(future_map):
          done += 1
          ok, err = fut.result()
          if ok:
            export_ok += 1
          else:
            export_err += 1
            try:
              job = future_map[fut]
              arch_idx, _, fname, _, _, _ = job
              with open(fail_log, "a", encoding="utf-8") as fh:
                fh.write(f"{arch_idx}\t{fname}\t{err or 'error'}\n")
            except Exception:
              pass
          show_progress(done, total, export_ok, 0, export_err)
      show_progress(done, total, export_ok, 0, export_err, final=True)

    arch_dirs = list_arch_dirs(root)
    missing_archs, incomplete = scan_incomplete(indices, arch_dirs, expected, args.strict)
    print(f"exported_ok={export_ok} exported_err={export_err}")
    print(f"missing_archs={len(missing_archs)} incomplete_archs={len(incomplete)}")

  if not missing_archs and not incomplete:
    print("ok: all archs complete")


if __name__ == "__main__":
  main()