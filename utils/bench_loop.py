"""Shared arch-iteration loop used by all device bench scripts."""
import time
from pathlib import Path

from utils.runner_utils import ensure_csv, append_row, append_completed


def list_arch_dirs(arch_root):
  """Return sorted list of arch indices found under arch_root/arch_*/."""
  if not arch_root.exists(): return []
  out = []
  for p in arch_root.iterdir():
    if not (p.is_dir() and p.name.startswith("arch_")): continue
    try: out.append(int(p.name.split("_", 1)[1]))
    except ValueError: continue
  return sorted(out)


def run_bench_loop(
  *,
  indices,
  on_device_set,
  csv_path,
  csv_cols,
  pending_fn,
  bench_fn,
  done_per_key,
  completion_path_fn,
  mark_done_fn,
):
  """Iterate over arch indices, skip done/absent, call bench_fn, write results.

  Args:
    indices:            ordered list of arch_idx to process
    on_device_set:      set of arch_idx available on device
    csv_path:           Path to results CSV
    csv_cols:           column list for CSV header and row serialisation
    pending_fn:         (arch_idx) -> list of pending items (tasks or (task,fw) pairs)
    bench_fn:           (arch_idx, pending) -> list[row_dict]
    done_per_key:       mutable dict[key -> set[int]]; updated in place
    completion_path_fn: (key) -> Path of completion tracking file
    mark_done_fn:       (rows, pending) -> iterable of keys to mark done
  """
  ensure_csv(csv_path, csv_cols)
  total = len(indices)
  for i, arch_idx in enumerate(indices, 1):
    if arch_idx not in on_device_set:
      print(f"[{i}/{total}] arch {arch_idx}: not on device", flush=True)
      continue
    pending = pending_fn(arch_idx)
    if not pending:
      print(f"[{i}/{total}] arch {arch_idx}: done", flush=True)
      continue
    t0 = time.time()
    rows = bench_fn(arch_idx, pending)
    ok = 0
    for r in rows:
      append_row(csv_path, r, csv_cols)
      if r.get("status") == "ok": ok += 1
    for key in mark_done_fn(rows, pending):
      append_completed(completion_path_fn(key), arch_idx)
      done_per_key[key].add(arch_idx)
    dt = time.time() - t0
    print(f"[{i}/{total}] arch {arch_idx}: {ok}/{len(rows)} ok ({dt:.1f}s)",
          flush=True)
