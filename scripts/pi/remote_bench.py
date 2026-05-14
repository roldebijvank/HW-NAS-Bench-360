"""Runs on the Pi. Times each (task,runtime) combo in an arch dir and emits
one JSON line per combo to stdout. Pre-gates 60C via vcgencmd measure_temp.
Pin to a core via taskset (caller's responsibility).

Output schema per combo line:
  {"task":..,"runtime":..,"lat_ms_median":..,"lat_ms_var":..,"status":"ok"|"missing"|"error","error":..}
Final line:
  {"throttled": true|false, "raw_throttled": "0x..."}
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


WARMUP = 10
TIMED  = 40

TEMP_CEILING_C = 60.0
TEMP_RESUME_C  = 55.0
TEMP_POLL_S    = 5.0

EXT = {"litert": "tflite", "onnx": "onnx", "torchmobile": "ptl"}


def read_temp_c():
  try:
    out = subprocess.run(["vcgencmd", "measure_temp"],
                         capture_output=True, text=True, timeout=2).stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return None
  m = re.search(r"temp=([\d.]+)", out)
  return float(m.group(1)) if m else None


def read_throttled():
  try:
    out = subprocess.run(["vcgencmd", "get_throttled"],
                         capture_output=True, text=True, timeout=2).stdout
  except (FileNotFoundError, subprocess.TimeoutExpired):
    return None
  m = re.search(r"throttled=(0x[0-9a-fA-F]+)", out)
  return m.group(1) if m else None


def gate_temp(ceiling, resume):
  t = read_temp_c()
  if t is None: return
  if t < ceiling: return
  while True:
    t = read_temp_c()
    if t is None or t < resume: return
    time.sleep(TEMP_POLL_S)


def time_loop(fn):
  for _ in range(WARMUP): fn()
  samples = []
  for _ in range(TIMED):
    t0 = time.perf_counter_ns()
    fn()
    samples.append((time.perf_counter_ns() - t0) / 1e6)
  return statistics.median(samples), statistics.variance(samples)


def make_step_litert(path, x_np):
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


def emit(obj):
  sys.stdout.write(json.dumps(obj) + "\n")
  sys.stdout.flush()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--arch-dir", required=True, type=Path)
  args = ap.parse_args()

  gate_temp(TEMP_CEILING_C, TEMP_RESUME_C)

  meta = json.loads((args.arch_dir / "meta.json").read_text())
  read_throttled()  # clear stale read

  for task, tinfo in meta["tasks"].items():
    shape = tuple(tinfo["input_shape"])
    x_np = np.random.randn(1, *shape).astype(np.float32)
    for runtime, ext in EXT.items():
      art = args.arch_dir / f"{task}_{runtime}.{ext}"
      row = {"task": task, "runtime": runtime}
      if not art.exists():
        row["status"] = "missing"; row["error"] = "artifact not found"
        emit(row); continue
      try:
        step = MAKERS[runtime](art, x_np)
        med, var = time_loop(step)
        row["lat_ms_median"] = med
        row["lat_ms_var"] = var
        row["status"] = "ok"
      except Exception as e:
        row["status"] = "error"; row["error"] = str(e)[:200]
      emit(row)

  raw = read_throttled()
  throttled = bool(raw and int(raw, 16) & 0x7) if raw else False
  emit({"throttled": throttled, "raw_throttled": raw})


if __name__ == "__main__":
  main()
