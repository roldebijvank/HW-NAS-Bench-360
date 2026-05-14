"""Pixel GPU bench via official TFLite BenchmarkModelActivity (host-side, adb-driven).

For each <root>/arch_*/<task>_litert.tflite on-device, runs:
  adb shell am start -S -W -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \\
    --es args "--graph=<path> --use_gpu=true --num_threads=1 --cpu_mask=f0 \\
               --warmup_runs=W --num_runs=R"
Parses `Inference (avg)` (us) from logcat. Writes CSV in bench_pi5.py row schema
minus energy columns: arch_idx,task,runtime,lat_ms,status.
"""
import argparse, csv, re, subprocess, sys, time
from pathlib import Path, PurePosixPath

WARMUP = 5
TIMED  = 30
TOTAL_ARCHS = 6466

BENCH_ACTIVITY = "org.tensorflow.lite.benchmark/.BenchmarkModelActivity"
BENCH_PKG      = "org.tensorflow.lite.benchmark"

DEFAULT_EXTRA_ARGS = ["--use_gpu=true", "--num_threads=1", "--cpu_mask=f0"]

TASKS = ("cifar100", "ninapro", "darcy")


def adb(*args, check=False):
  return subprocess.run(["adb", *args], capture_output=True, text=True, check=check)

def adb_shell(cmd, check=False):
  return subprocess.run(["adb", "shell", cmd],
                        capture_output=True, text=True, check=check)


def fmt_secs(seconds):
  if seconds is None:
    return "-"
  s = int(round(seconds))
  h, rem = divmod(s, 3600)
  m, s = divmod(rem, 60)
  if h:
    return f"{h}h{m:02d}m"
  if m:
    return f"{m}m{s:02d}s"
  return f"{s}s"


def fmt_bytes(num_bytes):
  if num_bytes is None:
    return "-"
  n = float(num_bytes)
  for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
    if n < 1024.0 or unit == "TiB":
      if unit == "B":
        return f"{int(n)}B"
      return f"{n:.1f}{unit}"
    n /= 1024.0


def device_file_size(path):
  r = adb_shell(f"wc -c < {path}")
  try:
    return int(r.stdout.strip().split()[0])
  except (ValueError, IndexError):
    return None


def wait_inference_avg(timeout_s=300, poll_s=0.5):
  # Benchmark activity often stays alive; logcat is the reliable completion signal.
  t_end = time.time() + timeout_s
  last_log = ""
  while time.time() < t_end:
    last_log = adb_shell("logcat -d -s tflite").stdout
    m = re.search(r"Inference \(avg\):\s*([\d.]+)", last_log)
    if m:
      return float(m.group(1)) / 1000.0  # us -> ms
    time.sleep(poll_s)
  tail = last_log[-800:] if last_log else "<empty>"
  raise TimeoutError(f"timed out waiting for Inference avg; logcat tail:\n{tail}")


def bench_one(model_path, *, warmup, runs, extra_args):
  args_str = " ".join([
    f"--graph={model_path}",
    f"--warmup_runs={warmup}",
    f"--num_runs={runs}",
    *extra_args,
  ])
  adb_shell("logcat -c", check=True)
  adb_shell(f'am start -S -W -n {BENCH_ACTIVITY} --es args "{args_str}"', check=True)
  lat_ms = wait_inference_avg()
  adb_shell(f"am force-stop {BENCH_PKG}")
  return lat_ms


def list_arch_models(root):
  """Returns sorted list of (arch_idx, task, on_device_path)."""
  r = adb_shell(f"ls -1 {root}", check=True)
  rows = []
  for name in r.stdout.splitlines():
    name = name.strip()
    if not name.startswith("arch_"): continue
    try: arch_idx = int(name.split("_")[1])
    except (IndexError, ValueError): continue
    for task in TASKS:
      rows.append((arch_idx, task, f"{root}/{name}/{task}_litert.tflite"))
  rows.sort(key=lambda x: (x[0], TASKS.index(x[1])))
  return rows


def device_exists(path):
  return adb_shell(f"test -f {path} && echo Y || echo N").stdout.strip() == "Y"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default="/data/local/tmp/feas",
                  help="on-device dir containing arch_<N>/ subdirs")
  ap.add_argument("--arch", type=int, default=None,
                  help="only run this arch_idx (default: all)")
  ap.add_argument("--warmup", type=int, default=WARMUP)
  ap.add_argument("--runs",   type=int, default=TIMED)
  ap.add_argument("--extra-arg", action="append", default=[],
                  help="extra --foo=bar passed to BenchmarkModel (repeatable)")
  ap.add_argument("--out", type=Path,
                  default=Path(__file__).resolve().parents[2] / "artifacts" / "pixel_bench.csv",
                  help="output CSV path")
  args = ap.parse_args()

  extra_args = DEFAULT_EXTRA_ARGS + args.extra_arg

  r = adb("get-state")
  if r.returncode != 0 or "device" not in r.stdout:
    print(f"adb get-state: {r.stdout.strip()} {r.stderr.strip()}", file=sys.stderr)
    sys.exit(2)

  entries = list_arch_models(args.root)
  if args.arch is not None:
    entries = [e for e in entries if e[0] == args.arch]
  if not entries:
    print(f"no arch_*/<task>_litert.tflite under {args.root}", file=sys.stderr)
    sys.exit(2)
  archs = sorted({e[0] for e in entries})
  print(f"{len(archs)} archs, {len(entries)} (arch,task) combos under {args.root}",
        file=sys.stderr)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  rows = []
  task_durations = {t: [] for t in TASKS}
  task_sizes = {t: [] for t in TASKS}
  with args.out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arch_idx", "task", "runtime", "lat_ms", "status"])
    cur_arch = None
    for i, (arch_idx, task, mp) in enumerate(entries):
      if arch_idx != cur_arch:
        cur_arch = arch_idx
        n_done = sum(1 for a in archs if a < arch_idx)
        print(f"=== arch_{arch_idx} ({n_done + 1}/{len(archs)}) ===", flush=True)
      tag = f"arch_{arch_idx} {task}/litert"
      if not device_exists(mp):
        rows.append([arch_idx, task, "litert", "-", "missing"])
        w.writerow(rows[-1]); f.flush()
        print(f"[skip] {tag}: missing", flush=True); continue
      size_b = device_file_size(mp)
      if size_b is not None:
        task_sizes[task].append(size_b)
      try:
        t0 = time.perf_counter()
        lat_ms = bench_one(mp, warmup=args.warmup, runs=args.runs,
                           extra_args=extra_args)
        dt = time.perf_counter() - t0
        task_durations[task].append(dt)
        rows.append([arch_idx, task, "litert", f"{lat_ms:.3f}", "ok"])
        w.writerow(rows[-1]); f.flush()
        print(f"[ok]   {tag}: lat={lat_ms:.3f}ms ({dt:.1f}s)", flush=True)
      except Exception as e:
        dt = time.perf_counter() - t0
        task_durations[task].append(dt)
        rows.append([arch_idx, task, "litert", "-", f"err: {str(e)[:60]}"])
        w.writerow(rows[-1]); f.flush()
        print(f"[err]  {tag}: {e}", flush=True)

  ok = [r for r in rows if r[4] == "ok"]
  n_miss = sum(1 for r in rows if r[4] == "missing")
  n_err  = sum(1 for r in rows if str(r[4]).startswith("err"))
  print()
  print("=" * 80); print("SUMMARY"); print("=" * 80)
  print(f"archs benched      : {len({r[0] for r in ok})}")
  print(f"ok / missing / err : {len(ok)} / {n_miss} / {n_err}")
  if ok:
    sum_lat = sum(float(r[3]) for r in ok)
    print(f"total lat_ms (sum medians): {sum_lat:.1f}")
  if any(task_durations[t] for t in TASKS):
    avg_parts = []
    est_parts = []
    est_total_s = 0.0
    have_all = True
    for task in TASKS:
      dts = task_durations[task]
      if dts:
        avg = sum(dts) / len(dts)
        avg_parts.append(f"{task}={avg:.1f}s")
        est = avg * TOTAL_ARCHS
        est_parts.append(f"{task}={fmt_secs(est)}")
        est_total_s += est
      else:
        avg_parts.append(f"{task}=-")
        est_parts.append(f"{task}=-")
        have_all = False
    print(f"avg bench time/run : {', '.join(avg_parts)}")
    print(f"est total {TOTAL_ARCHS} archs: {', '.join(est_parts)}")
    if have_all:
      print(f"est total all tasks: {fmt_secs(est_total_s)}")
    else:
      print("est total all tasks: - (need all tasks)")
  if any(task_sizes[t] for t in TASKS):
    avg_parts = []
    est_parts = []
    est_total_b = 0.0
    have_all = True
    for task in TASKS:
      sizes = task_sizes[task]
      if sizes:
        avg = sum(sizes) / len(sizes)
        avg_parts.append(f"{task}={fmt_bytes(avg)}")
        est = avg * TOTAL_ARCHS
        est_parts.append(f"{task}={fmt_bytes(est)}")
        est_total_b += est
      else:
        avg_parts.append(f"{task}=-")
        est_parts.append(f"{task}=-")
        have_all = False
    print(f"avg model size    : {', '.join(avg_parts)}")
    print(f"est storage {TOTAL_ARCHS} archs: {', '.join(est_parts)}")
    if have_all:
      print(f"est storage all tasks: {fmt_bytes(est_total_b)}")
    else:
      print("est storage all tasks: - (need all tasks)")
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
