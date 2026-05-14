"""
RPi daemon: watch ~/bep/queue/incoming/ for arch_<N>.ready sentinels.
For each ready arch, time every (task, runtime) artifact in its dir,
append rows to ~/bep/results/latency_rpi.csv, write arch_<N>.done sentinel,
delete the artifact dir.

Run pinned: taskset -c 3 python pi_daemon.py
"""
from pathlib import Path
import json, time, statistics, csv, os, shutil, sys, argparse
import numpy as np
from tqdm import tqdm

HOME = Path(os.path.expanduser("~/bep"))
INC  = HOME / "queue" / "incoming"
DONE = HOME / "queue" / "done"
RES  = HOME / "results"
CSV  = RES / "latency_rpi.csv"

for p in (INC, DONE, RES): p.mkdir(parents=True, exist_ok=True)

WARMUP = 10
TIMED  = 100

CSV_COLS = ["arch_idx","task","runtime",
            "lat_ms_median","lat_ms_p10","lat_ms_p90","lat_ms_std",
            "status","error"]

EXT = {"litert":"tflite","onnx":"onnx","torchmobile":"ptl"}


def time_loop(fn):
  for _ in range(WARMUP): fn()
  samples = []
  for _ in range(TIMED):
    t0 = time.perf_counter_ns()
    fn()
    samples.append((time.perf_counter_ns() - t0) / 1e6)
  samples.sort()
  return {
    "lat_ms_median": statistics.median(samples),
    "lat_ms_p10":    samples[int(0.1 * TIMED)],
    "lat_ms_p90":    samples[int(0.9 * TIMED) - 1],
    "lat_ms_std":    statistics.stdev(samples),
  }


def bench_tflite(path, x_np):
  try:
    from tflite_runtime.interpreter import Interpreter
  except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
  interp = Interpreter(model_path=str(path), num_threads=1)
  interp.allocate_tensors()
  inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
  def step():
    interp.set_tensor(inp["index"], x_np)
    interp.invoke()
    interp.get_tensor(out["index"])
  return time_loop(step)


def bench_onnx(path, x_np):
  import onnxruntime as ort
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
  sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
  name = sess.get_inputs()[0].name
  def step(): sess.run(None, {name: x_np})
  return time_loop(step)


def bench_torchmobile(path, x_np):
  import torch
  torch.set_num_threads(1)
  m = torch.jit.load(str(path)); m.eval()
  x = torch.from_numpy(x_np)
  def step():
    with torch.no_grad(): m(x)
  return time_loop(step)


BENCH = {"litert": bench_tflite, "onnx": bench_onnx, "torchmobile": bench_torchmobile}


def ensure_csv():
  if not CSV.exists():
    with open(CSV, "w", newline="") as f:
      csv.writer(f).writerow(CSV_COLS)


def append_row(row):
  with open(CSV, "a", newline="") as f:
    csv.writer(f).writerow([row.get(k,"") for k in CSV_COLS])


def process(arch_dir, total_bar=None):
  meta_path = arch_dir / "meta.json"
  meta = json.loads(meta_path.read_text())
  arch_idx = meta["arch_idx"]
  tasks = list(meta["tasks"].items())
  steps = [(t, ti, rt, ext) for t, ti in tasks for rt, ext in EXT.items()]
  rows = []
  bar = tqdm(steps, desc=f"arch_{arch_idx}", position=1, leave=False,
             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
  t_start = time.perf_counter()
  for task, tinfo, runtime, ext in bar:
    bar.set_postfix_str(f"{task}/{runtime}", refresh=True)
    shape = tuple(tinfo["input_shape"])
    x_np = np.random.randn(1, *shape).astype(np.float32)
    art = arch_dir / f"{task}_{runtime}.{ext}"
    base = {"arch_idx": arch_idx, "task": task, "runtime": runtime}
    if not art.exists():
      base.update(status="missing", error="artifact not found")
    else:
      try:
        r = BENCH[runtime](art, x_np)
        base.update(r); base["status"] = "ok"
      except Exception as e:
        base.update(status="error", error=str(e)[:200])
    rows.append(base)
  bar.close()
  for r in rows: append_row(r)
  ok = sum(1 for r in rows if r.get("status") == "ok")
  dt = time.perf_counter() - t_start
  if total_bar is not None:
    total_bar.update(1)
    total_bar.set_postfix_str(f"last=arch_{arch_idx} {dt:.1f}s ok={ok}/{len(rows)}", refresh=True)
  else:
    tqdm.write(f"arch_{arch_idx}: {ok}/{len(rows)} ok in {dt:.1f}s")
  return len(rows)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--total", type=int, default=None, help="expected total archs (for progress bar)")
  args = ap.parse_args()

  ensure_csv()
  done_so_far = len(list(DONE.glob("arch_*.done")))
  total_bar = tqdm(total=args.total, initial=done_so_far, desc="total",
                   position=0, leave=True, unit="arch",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
  tqdm.write(f"daemon up. watching {INC}. csv -> {CSV}")
  while True:
    ready = sorted(INC.glob("arch_*.ready"))
    if not ready:
      time.sleep(0.5); continue
    for r in ready:
      arch_dir = INC / r.stem
      if not arch_dir.is_dir():
        r.unlink(missing_ok=True); continue
      try:
        process(arch_dir, total_bar=total_bar)
      except Exception as e:
        tqdm.write(f"{arch_dir.name}: FAIL {e}")
        if total_bar is not None: total_bar.update(1)
      shutil.rmtree(arch_dir, ignore_errors=True)
      r.unlink(missing_ok=True)
      (DONE / f"{arch_dir.name}.done").touch()


if __name__ == "__main__":
  main()
