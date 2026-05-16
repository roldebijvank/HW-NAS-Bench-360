"""Verify Pi archs for a single task.

Run as module so imports work:
  uv run python -m scripts.pi.verify --task cifar100
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from scripts.utils.arch_iter import non_iso_indices
from scripts.utils.convert_utils import RUNTIMES
from scripts.utils.runner_utils import parse_arch_list
from scripts.utils.task_specs import TASKS

ROOT = Path(__file__).resolve().parents[2]
RPI_HOST = os.environ.get("RPI_HOST")
RPI_ARCHS = "~/bep/archs"


def ssh(cmd, check=False):
  r = subprocess.run(["ssh", RPI_HOST, cmd], capture_output=True, text=True)
  if check and r.returncode != 0:
    msg = r.stderr.strip() or r.stdout.strip()
    raise RuntimeError(msg or "ssh failed")
  return r


def list_remote_archs():
  cmd = (f"cd {RPI_ARCHS} 2>/dev/null && "
         "for d in arch_*; do [ -d \"$d\" ] && echo $d; done")
  r = ssh(cmd)
  if r.returncode != 0:
    return []
  out = []
  for line in r.stdout.splitlines():
    name = line.strip()
    if not name.startswith("arch_"):
      continue
    try:
      out.append(int(name.split("_", 1)[1]))
    except ValueError:
      continue
  return sorted(out)


def list_task_files(task):
  cmd = (f"cd {RPI_ARCHS} 2>/dev/null && "
         f"find arch_* -maxdepth 1 -type f -name '{task}_*' -print 2>/dev/null")
  r = ssh(cmd)
  if r.returncode != 0 and not r.stdout.strip():
    return []
  return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def build_expected_indices(args):
  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  elif args.all:
    indices = list(range(15625))
  else:
    try:
      indices = non_iso_indices()
    except FileNotFoundError as e:
      print(f"non-iso list missing: {e}", file=sys.stderr)
      print("use --arch-list or --all", file=sys.stderr)
      sys.exit(2)
  indices = indices[args.start:]
  if args.limit:
    indices = indices[:args.limit]
  return list(dict.fromkeys(indices))


def summarize_list(label, items, max_list, show_all=False):
  if not items:
    return
  if show_all or len(items) <= max_list:
    show = items
    suffix = ""
  else:
    show = items[:max_list]
    suffix = f" (+{len(items) - max_list} more)"
  print(f"{label}: {show}{suffix}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True, choices=list(TASKS))
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--max-list", type=int, default=10)
  ap.add_argument("--show-all", action="store_true")
  args = ap.parse_args()

  if not RPI_HOST:
    print("set RPI_HOST=user@host", file=sys.stderr)
    sys.exit(2)

  expected = build_expected_indices(args)
  expected_set = set(expected)
  on_pi = list_remote_archs()
  if not on_pi:
    print(f"no arch_*/ under {RPI_ARCHS} on Pi", file=sys.stderr)
    sys.exit(2)
  on_pi_set = set(on_pi)

  missing_archs = sorted(expected_set - on_pi_set)
  redundant_archs = sorted(on_pi_set - expected_set)

  expected_files = {f"{args.task}_{runtime}.{ext}"
                    for runtime, (ext, _) in RUNTIMES.items()}
  files = list_task_files(args.task)
  files_by_arch = {}
  for path in files:
    p = path[2:] if path.startswith("./") else path
    if "/" not in p:
      continue
    arch_dir, fname = p.split("/", 1)
    if not arch_dir.startswith("arch_"):
      continue
    try:
      arch_idx = int(arch_dir.split("_", 1)[1])
    except ValueError:
      continue
    files_by_arch.setdefault(arch_idx, set()).add(fname)

  incomplete = {}
  for arch_idx in sorted(expected_set & on_pi_set):
    present = files_by_arch.get(arch_idx, set())
    missing = sorted(expected_files - present)
    unexpected = sorted(f for f in present if f not in expected_files)
    if missing or unexpected:
      incomplete[arch_idx] = {"missing": missing, "unexpected": unexpected}

  print(f"task={args.task} expected={len(expected)} on_pi={len(on_pi)}")
  print(f"missing_archs={len(missing_archs)} redundant_archs={len(redundant_archs)}")
  print(f"incomplete_archs={len(incomplete)}")

  summarize_list("missing_archs sample", missing_archs, args.max_list, args.show_all)
  summarize_list("redundant_archs sample", redundant_archs, args.max_list, args.show_all)

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

  if not missing_archs and not redundant_archs and not incomplete:
    print("ok: all expected archs present with expected models")


if __name__ == "__main__":
  main()
