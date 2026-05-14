from pathlib import Path
import argparse, re, statistics, subprocess, sys, threading, time
import numpy as np

WARMUP = 5
TIMED  = 30
SAMPLE_INTERVAL_S = 0.024  # ~200 Hz PMIC poll

EXT = {"litert": "tflite", "onnx": "onnx", "torchmobile": "ptl"}

TASK_SHAPES = {
  "cifar100": (3, 32, 32),
  "ninapro":  (1, 52, 16),
  "darcy":    (1, 88, 88),
}

# PMIC power sampling
_PMIC_LINE = re.compile(r"^\s*(\S+?)\s+(?:current|volt)\(\d+\)=([\d.]+)([VA])")

def read_pmic_power_w():
  """Sum I*V across all rails reported by vcgencmd. Returns watts or None."""
  try:
    out = subprocess.run(["vcgencmd", "  "],
                         capture_output=True, text=True, timeout=1).stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return None
  volts, amps = {}, {}
  for line in out.splitlines():
    m = _PMIC_LINE.match(line)
    if not m: continue
    name, val, unit = m.group(1), float(m.group(2)), m.group(3)
    base = name[:-2] if name.endswith(("_A", "_V")) else name
    (volts if unit == "V" else amps)[base] = val
  return sum(volts[k] * amps[k] for k in volts if k in amps)


class PowerSampler(threading.Thread):
  def __init__(self, interval=SAMPLE_INTERVAL_S):
    super().__init__(daemon=True)
    self.interval = interval
    self.stop_evt = threading.Event()
    self.samples = []  # (perf_counter_ns, power_w)

  def run(self):
    while not self.stop_evt.is_set():
      p = read_pmic_power_w()
      t = time.perf_counter_ns()
      if p is not None:
        self.samples.append((t, p))
      self.stop_evt.wait(self.interval)

  def stop(self):
    self.stop_evt.set()
    self.join()


# per-runtime step builders
def make_step_litert(path, x_np):
  from tflite_runtime.interpreter import Interpreter
  
  interp = Interpreter(model_path=str(path), num_threads=1)
  interp.allocate_tensors()
  inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
  def step():
    interp.set_tensor(inp["index"], x_np)
    interp.invoke()
    interp.get_tensor(out["index"])
  return step

def make_step_onnx(path, x_np):
  import onnxruntime as ort
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
  sess = ort.InferenceSession(str(path), sess_options=so,
                              providers=["CPUExecutionProvider"])
  name = sess.get_inputs()[0].name
  def step(): sess.run(None, {name: x_np})
  return step

def make_step_torchmobile(path, x_np):
  import torch
  torch.set_num_threads(1)
  m = torch.jit.load(str(path)); m.eval()
  x = torch.from_numpy(x_np)
  def step():
    with torch.no_grad(): m(x)
  return step

MAKERS = {
  "litert":      make_step_litert,
  "onnx":        make_step_onnx,
  "torchmobile": make_step_torchmobile,
}


# bench
def bench_one(path, runtime, shape, measure_energy=True):
  x_np = np.random.randn(1, *shape).astype(np.float32)
  step = MAKERS[runtime](path, x_np)

  for _ in range(WARMUP): step()

  sampler = PowerSampler() if measure_energy else None
  if sampler: sampler.start()
  iters = []
  for _ in range(TIMED):
    t0 = time.perf_counter_ns()
    step()
    t1 = time.perf_counter_ns()
    iters.append((t0, t1))
  if sampler: sampler.stop()

  lats_ms = [(t1 - t0) / 1e6 for t0, t1 in iters]

  if not sampler:
    return {
      "lat_ms_median":   statistics.median(lats_ms),
      "energy_j_median": float("nan"),
      "power_w_mean":    float("nan"),
      "n_pmic_samples":  0,
    }

  power_mean = (statistics.mean(p for _, p in sampler.samples)
                if sampler.samples else float("nan"))
  energies_j = ([power_mean * (lat_ms / 1e3) for lat_ms in lats_ms]
                if sampler.samples else [])
  return {
    "lat_ms_median":    statistics.median(lats_ms),
    "energy_j_median":  statistics.median(energies_j) if energies_j else float("nan"),
    "power_w_mean":     power_mean,
    "n_pmic_samples":   len(sampler.samples),
  }


def find_arch_dirs(root):
  return sorted((p for p in root.glob("arch_*") if p.is_dir()),
                key=lambda p: int(p.name.split("_")[1]))

EXTRAPOLATE_N = 6466

def print_summary(rows, measure_energy):
  archs = {r[0] for r in rows if r[6] == "ok"}
  n_archs = len(archs)
  ok_rows = [r for r in rows if r[6] == "ok"]
  n_ok = len(ok_rows)
  n_miss = sum(1 for r in rows if r[6] == "missing")
  n_err = sum(1 for r in rows if r[6].startswith("err"))

  sum_lat_ms = sum(float(r[3]) for r in ok_rows)
  iters = WARMUP + TIMED
  bench_s = iters * sum_lat_ms / 1e3
  per_arch_s = bench_s / n_archs if n_archs else float("nan")
  extrap_s = per_arch_s * EXTRAPOLATE_N

  print(f"archs benched      : {n_archs}")
  print(f"ok / missing / err : {n_ok} / {n_miss} / {n_err}")
  print(f"total lat_ms (sum medians): {sum_lat_ms:.1f}")
  if measure_energy:
    sum_e = sum(float(r[4]) for r in ok_rows)
    mean_p = (sum(float(r[5]) for r in ok_rows) / n_ok) if n_ok else float("nan")
    print(f"total energy_J (sum medians): {sum_e:.3f}")
    print(f"mean power_W       : {mean_p:.3f}")
  print(f"inference wall time ({iters} iters/combo): {bench_s:.1f} s  "
        f"({per_arch_s:.2f} s/arch avg)")
  print(f"extrapolated to {EXTRAPOLATE_N} archs: "
        f"{extrap_s:.0f} s = {extrap_s/3600:.2f} h = {extrap_s/86400:.2f} d")
  print("(excludes model load + PMIC setup overhead)")


def main():
  ap = argparse.ArgumentParser()
  default_root = Path(__file__).resolve().parents[2] / "artifacts" / "feas"
  ap.add_argument("--root", type=Path, default=default_root,
                  help="dir containing arch_<N>/ subdirs")
  ap.add_argument("--arch", type=int, default=None,
                  help="only run this arch_idx (default: all)")
  ap.add_argument("--pi4b", action="store_true",
                  help="Pi 4B mode: latency only, skip PMIC energy sampling")
  args = ap.parse_args()

  measure_energy = not args.pi4b
  if measure_energy and read_pmic_power_w() is None:
    print("WARN: vcgencmd pmic_read_adc unavailable; energy will be NaN. "
          "Use --pi4b to skip energy.", file=sys.stderr)

  arch_dirs = find_arch_dirs(args.root)
  if args.arch is not None:
    arch_dirs = [d for d in arch_dirs if d.name == f"arch_{args.arch}"]
  if not arch_dirs:
    print(f"no arch dirs under {args.root}", file=sys.stderr); sys.exit(2)

  rows = []
  for d in arch_dirs:
    arch_idx = int(d.name.split("_")[1])
    for task, shape in TASK_SHAPES.items():
      for runtime, ext in EXT.items():
        art = d / f"{task}_{runtime}.{ext}"
        tag = f"arch_{arch_idx} {task}/{runtime}"
        if not art.exists():
          rows.append([arch_idx, task, runtime, "-", "-", "-", "missing"])
          print(f"[skip] {tag}: missing", flush=True); continue
        try:
          t0 = time.perf_counter()
          r = bench_one(art, runtime, shape, measure_energy=measure_energy)
          dt = time.perf_counter() - t0
          e_str = "-" if not measure_energy else f"{r['energy_j_median']:.4f}"
          p_str = "-" if not measure_energy else f"{r['power_w_mean']:.3f}"
          rows.append([arch_idx, task, runtime,
                       f"{r['lat_ms_median']:.3f}", e_str, p_str, "ok"])
          extra = (f"E={r['energy_j_median']:.4f}J P={r['power_w_mean']:.3f}W "
                   f"({r['n_pmic_samples']} pmic, " if measure_energy
                   else "(")
          print(f"[ok]   {tag}: lat={r['lat_ms_median']:.3f}ms "
                f"{extra}{dt:.1f}s)", flush=True)
        except Exception as e:
          rows.append([arch_idx, task, runtime, "-", "-", "-",
                       f"err: {str(e)[:40]}"])
          print(f"[err]  {tag}: {e}", flush=True)

  print()
  print("=" * 80)
  print("SUMMARY")
  print("=" * 80)
  print_summary(rows, measure_energy)


if __name__ == "__main__":
  main()
