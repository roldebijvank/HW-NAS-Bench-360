import argparse
import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

DATA_ROOT = Path.home() / "bep"
DEFAULT_ARCH_ROOT = Path("/mnt/usb/archs")
DEFAULT_OUTPUT_PATH = DATA_ROOT / "results" / "energy_pi.csv"
DEFAULT_ERROR_LOG = DATA_ROOT / "results" / "energy_errors.txt"

SAMPLE_HZ = 100.0
SAMPLE_INTERVAL_S = 1.0 / SAMPLE_HZ
IDLE_SAMPLE_S = 0.5
MEASURE_S = 0.5
WARMUP = 10

CORE_INFER = 0
CORE_SAMPLE = 1
PROCESS_CORES = {0, 1}
CPU_GOVERNOR = "performance"

TASK_SHAPES = {
  "cifar100": (3, 32, 32),
  "ninapro": (1, 52, 16),
  "darcy": (1, 88, 88),
}

RUNTIME_EXT = {
  "litert": "tflite",
  "onnx": "onnx",
  "torchmobile": "ptl",
}

PMIC_RE = re.compile(
  r"(?P<label>[A-Za-z0-9_/-]+)[^:=]*[:=]\s*"
  r"(?P<value>[-+]?\d*\.?\d+)\s*(?P<unit>mV|V|mA|A)\b",
  re.IGNORECASE,
)

CSV_HEADER = [
  "arch_index",
  "task",
  "framework",
  "energy_per_inference_J",
  "n_samples",
  "n_inferences",
]


def _set_thread_affinity(core, label):
  try:
    tid = threading.get_native_id()
    os.sched_setaffinity(tid, {core})
  except (AttributeError, OSError, ValueError) as e:
    print(f"{label} affinity set failed: {e}", flush=True)


def _set_process_affinity(cores, label):
  try:
    os.sched_setaffinity(0, cores)
  except (AttributeError, OSError, ValueError) as e:
    print(f"{label} affinity set failed: {e}", flush=True)


def _set_cpu_governor(governor):
  cpu_root = Path("/sys/devices/system/cpu")
  paths = list(cpu_root.glob("cpu[0-9]*/cpufreq/scaling_governor"))
  if not paths:
    print("cpu governor not available", flush=True)
    return
  errors = 0
  for path in paths:
    try:
      path.write_text(governor)
    except OSError:
      errors += 1
  if errors:
    print(f"cpu governor set failed on {errors} cores", flush=True)


def _read_temp_c():
  try:
    out = subprocess.run(
      ["vcgencmd", "measure_temp"],
      capture_output=True,
      text=True,
      timeout=1,
    ).stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return None
  m = re.search(r"temp=([\d.]+)", out)
  return float(m.group(1)) if m else None


def _read_throttled():
  try:
    out = subprocess.run(
      ["vcgencmd", "get_throttled"],
      capture_output=True,
      text=True,
      timeout=1,
    ).stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return None
  m = re.search(r"throttled=(0x[0-9a-fA-F]+)", out)
  return m.group(1) if m else None


def _read_pmic_power_w():
  try:
    out = subprocess.run(
      ["vcgencmd", "pmic_read_adc", "VDD_CORE_A", "VDD_CORE_V"],
      capture_output=True,
      text=True,
      timeout=1,
    ).stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return None

  volts = None
  amps = None
  for m in PMIC_RE.finditer(out):
    label = m.group("label").upper()
    val = float(m.group("value"))
    unit = m.group("unit").lower()
    if label == "VDD_CORE_V":
      volts = val / 1000.0 if unit == "mv" else val
    elif label == "VDD_CORE_A":
      amps = val / 1000.0 if unit == "ma" else val

  if volts is None or amps is None:
    return None
  return volts * amps


class PowerSampler:
  def __init__(self, stop_event):
    self.stop_event = stop_event
    self.samples_w = []
    self.thread = None
    self.missed_deadlines = 0

  def start(self):
    self.thread = threading.Thread(target=self._run, daemon=True)
    self.thread.start()

  def _run(self):
    _set_thread_affinity(CORE_SAMPLE, "sample thread")
    deadline = time.perf_counter()
    while not self.stop_event.is_set():
      p_w = _read_pmic_power_w()
      if p_w is not None:
        self.samples_w.append(p_w)
      deadline += SAMPLE_INTERVAL_S
      remaining = deadline - time.perf_counter()
      if remaining > 0:
        time.sleep(remaining)
      else:
        self.missed_deadlines += 1
        deadline = time.perf_counter()

  def stop(self):
    if not self.thread:
      return
    self.thread.join(timeout=1)


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
  sess = ort.InferenceSession(
    str(path),
    sess_options=so,
    providers=["CPUExecutionProvider"],
  )
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


def _sample_for_duration(duration_s):
  stop_event = threading.Event()
  sampler = PowerSampler(stop_event)
  sampler.start()
  time.sleep(duration_s)
  stop_event.set()
  sampler.stop()
  return sampler.samples_w, sampler.missed_deadlines


def _measure_with_inference(step, duration_s):
  stop_event = threading.Event()
  sampler = PowerSampler(stop_event)
  sampler.start()
  n_inferences = 0
  t_end = time.perf_counter() + duration_s
  while time.perf_counter() < t_end:
    step()
    n_inferences += 1
  stop_event.set()
  sampler.stop()
  return sampler.samples_w, n_inferences, sampler.missed_deadlines


def _measure_idle(idle_s):
  exp_idle = int(idle_s * SAMPLE_HZ)
  print(
    f"[idle] sampling {idle_s}s (~{exp_idle} samples)",
    flush=True,
  )
  idle_samples, missed = _sample_for_duration(idle_s)
  if not idle_samples:
    raise SystemExit("no idle samples collected")
  idle_power_mean_w = sum(idle_samples) / len(idle_samples)
  print(
    f"[idle] samples {len(idle_samples)} mean {idle_power_mean_w:.6f} W",
    flush=True,
  )
  print(
    f"[idle] missed deadlines {missed}",
    flush=True,
  )
  return idle_power_mean_w, len(idle_samples), missed


def _append_result(
  output_path,
  arch_index,
  task,
  framework,
  energy_per_inference_j,
  n_samples,
  n_inferences,
):
  output_path.parent.mkdir(parents=True, exist_ok=True)
  write_header = (not output_path.exists()) or output_path.stat().st_size == 0
  with open(output_path, "a", newline="", buffering=1) as f:
    writer = csv.writer(f)
    if write_header:
      writer.writerow(CSV_HEADER)
    writer.writerow([
      arch_index,
      task,
      framework,
      energy_per_inference_j,
      n_samples,
      n_inferences,
    ])
    f.flush()


def _read_completed_rows(path):
  if not path.exists():
    return set()
  completed = set()
  with open(path, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
      if not row:
        continue
      if row[0].strip().lower() == "arch_index":
        continue
      if len(row) < 3:
        continue
      try:
        arch_idx = int(row[0])
      except ValueError:
        continue
      task = row[1].strip()
      framework = row[2].strip()
      if not task or not framework:
        continue
      completed.add((arch_idx, task, framework))
  return completed


def _log_error(error_log_path, arch_index, task, framework, model_path, error_msg):
  error_log_path.parent.mkdir(parents=True, exist_ok=True)
  msg = str(error_msg).replace("\n", " ").replace("\r", " ").strip()
  ts = time.strftime("%Y-%m-%d %H:%M:%S")
  with open(error_log_path, "a", buffering=1) as f:
    f.write(
      f"{ts}\t{arch_index}\t{task}\t{framework}\t{model_path}\t{msg}\n"
    )
    f.flush()


def _list_arch_dirs(arch_root):
  if not arch_root.exists():
    return []
  out = []
  for p in arch_root.iterdir():
    if not (p.is_dir() and p.name.startswith("arch_")):
      continue
    try:
      arch_idx = int(p.name.split("_", 1)[1])
    except ValueError:
      continue
    out.append((arch_idx, p))
  return sorted(out, key=lambda x: x[0])


def _run_single(
  arch_index,
  task,
  framework,
  model_path,
  output_path,
  idle_power_mean_w,
  measure_s,
):
  if task not in TASK_SHAPES:
    raise SystemExit(f"unknown task: {task}")
  if framework not in MAKERS:
    raise SystemExit(f"unknown framework: {framework}")
  if not model_path.exists():
    raise SystemExit(f"model not found: {model_path}")

  tag = f"arch {arch_index} {task}/{framework}"
  shape = TASK_SHAPES[task]
  x_np = np.random.randn(1, *shape).astype(np.float32)
  step = MAKERS[framework](model_path, x_np)

  for _ in range(WARMUP):
    step()

  loaded_samples, n_inferences, missed = _measure_with_inference(step, measure_s)
  if not loaded_samples:
    raise SystemExit("no loaded samples collected")

  p_net = [p - idle_power_mean_w for p in loaded_samples]
  e_total_j = sum(p_net) * SAMPLE_INTERVAL_S
  energy_per_inference_j = e_total_j / n_inferences if n_inferences else 0.0

  print(
    f"[{tag}] samples {len(loaded_samples)} inf {n_inferences} missed {missed} "
    f"energy {energy_per_inference_j:.6f} J",
    flush=True,
  )

  _append_result(
    output_path,
    arch_index,
    task,
    framework,
    energy_per_inference_j,
    len(loaded_samples),
    n_inferences,
  )

  return len(loaded_samples), n_inferences


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("arch_index", type=int, nargs="?")
  ap.add_argument("task", type=str, nargs="?")
  ap.add_argument("framework", type=str, nargs="?")
  ap.add_argument("model_path", type=Path, nargs="?")
  ap.add_argument("output_path", type=Path, nargs="?")
  ap.add_argument("--idle-s", type=float, default=IDLE_SAMPLE_S)
  ap.add_argument("--measure-s", type=float, default=MEASURE_S)
  ap.add_argument("--arch-root", type=Path, default=DEFAULT_ARCH_ROOT)
  ap.add_argument("--task", dest="tasks", choices=list(TASK_SHAPES),
                  action="append", default=[])
  ap.add_argument("--framework", dest="frameworks", choices=list(MAKERS),
                  action="append", default=[])
  ap.add_argument("--output-path", dest="output_path_opt", type=Path,
                  default=None)
  ap.add_argument("--error-log", dest="error_log", type=Path, default=None)
  ap.add_argument("--no-resume", action="store_true",
                  help="Do not skip rows already in output CSV")
  args = ap.parse_args()

  if args.idle_s <= 0 or args.measure_s <= 0:
    raise SystemExit("durations must be positive")

  np.random.seed(0)

  _set_thread_affinity(CORE_INFER, "main thread")
  _set_process_affinity(PROCESS_CORES, "process")
  _set_cpu_governor(CPU_GOVERNOR)

  required_single = [
    args.arch_index,
    args.task,
    args.framework,
    args.model_path,
  ]
  single_base = all(v is not None for v in required_single)
  any_required = any(v is not None for v in required_single)
  output_path = args.output_path_opt or args.output_path
  error_log_path = (args.error_log or DEFAULT_ERROR_LOG).expanduser()
  if any_required and not single_base:
    raise SystemExit("provide arch_index, task, framework, model_path for single run")

  if single_base:
    if output_path is None:
      raise SystemExit("output_path is required for single run")
    if args.task not in TASK_SHAPES:
      raise SystemExit(f"unknown task: {args.task}")
    if args.framework not in MAKERS:
      raise SystemExit(f"unknown framework: {args.framework}")
    if not args.model_path.exists():
      raise SystemExit(f"model not found: {args.model_path}")
    idle_power_mean_w, _, _ = _measure_idle(args.idle_s)
    try:
      _run_single(
        args.arch_index,
        args.task,
        args.framework,
        args.model_path,
        output_path,
        idle_power_mean_w,
        args.measure_s,
      )
      return
    except SystemExit as e:
      _log_error(
        error_log_path,
        args.arch_index,
        args.task,
        args.framework,
        args.model_path,
        e,
      )
      raise
    except Exception as e:
      _log_error(
        error_log_path,
        args.arch_index,
        args.task,
        args.framework,
        args.model_path,
        e,
      )
      raise

  arch_root = args.arch_root.expanduser()
  if not arch_root.exists():
    raise SystemExit(f"arch root not found: {arch_root}")

  tasks = args.tasks or list(TASK_SHAPES.keys())
  frameworks = args.frameworks or list(MAKERS.keys())
  output_path = output_path or DEFAULT_OUTPUT_PATH
  output_path = output_path.expanduser()

  arch_dirs = _list_arch_dirs(arch_root)
  if not arch_dirs:
    raise SystemExit(f"no arch_*/ under {arch_root}")

  completed = set()
  if not args.no_resume and output_path.exists():
    completed = _read_completed_rows(output_path)
    if completed:
      print(f"[resume] skipping {len(completed)} completed rows", flush=True)

  jobs_by_arch = []
  total_jobs = 0
  skipped_jobs = 0
  for arch_idx, arch_dir in arch_dirs:
    jobs = []
    for task in tasks:
      for framework in frameworks:
        if (arch_idx, task, framework) in completed:
          skipped_jobs += 1
          continue
        ext = RUNTIME_EXT[framework]
        model_path = arch_dir / f"{task}_{framework}.{ext}"
        if model_path.exists():
          jobs.append((task, framework, model_path))
    jobs_by_arch.append((arch_idx, jobs))
    total_jobs += len(jobs)

  if total_jobs == 0:
    raise SystemExit("no model artifacts found")

  if skipped_jobs:
    print(f"[resume] skipped jobs {skipped_jobs}", flush=True)

  total_archs = len(arch_dirs)
  job_idx = 0
  idle_power_mean_w = None
  for arch_i, (arch_idx, jobs) in enumerate(jobs_by_arch, 1):
    if idle_power_mean_w is None or (arch_i - 1) % 100 == 0:
      idle_power_mean_w, _, _ = _measure_idle(args.idle_s)
    if not jobs:
      temp_c = _read_temp_c()
      throttled = _read_throttled()
      temp_s = f"{temp_c:.1f}C" if temp_c is not None else "na"
      throttled_s = throttled if throttled is not None else "na"
      print(
        f"[arch {arch_i}/{total_archs}] arch {arch_idx}: 0 jobs "
        f"temp {temp_s} throttle {throttled_s}",
        flush=True,
      )
      continue
    temp_c = _read_temp_c()
    throttled = _read_throttled()
    temp_s = f"{temp_c:.1f}C" if temp_c is not None else "na"
    throttled_s = throttled if throttled is not None else "na"
    print(
      f"[arch {arch_i}/{total_archs}] arch {arch_idx}: {len(jobs)} jobs "
      f"temp {temp_s} throttle {throttled_s}",
      flush=True,
    )
    arch_samples = 0
    for task, framework, model_path in jobs:
      job_idx += 1
      tag = f"arch {arch_idx} {task}/{framework}"
      print(f"[{job_idx}/{total_jobs}] {tag} start", flush=True)
      try:
        samples, _ = _run_single(
          arch_idx,
          task,
          framework,
          model_path,
          output_path,
          idle_power_mean_w,
          args.measure_s,
        )
        arch_samples += samples
      except SystemExit as e:
        _log_error(error_log_path, arch_idx, task, framework, model_path, e)
        print(f"[{tag}] error {e}", flush=True)
      except Exception as e:
        _log_error(error_log_path, arch_idx, task, framework, model_path, e)
        print(f"[{tag}] error {e}", flush=True)
    print(
      f"[arch {arch_i}/{total_archs}] arch {arch_idx}: samples {arch_samples}",
      flush=True,
    )


if __name__ == "__main__":
  main()