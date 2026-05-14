"""RPi-side: time arch 0 CIFAR inference on TFLite, ONNX, torch-mobile.
Run after rsync of artifacts/sanity/. Target: ~10.48 ms median (HW-NAS-Bench raspi4_latency).
Pin to one core: taskset -c 3 python sanity_measure.py
"""
from pathlib import Path
import time, statistics, numpy as np

import onnxruntime as ort
from tflite_runtime.interpreter import Interpreter
from tensorflow.lite.python.interpreter import Interpreter


ARCH_IDX = 0
INPUT_SHAPE = (1, 3, 32, 32)
WARMUP = 50
TIMED = 200

ART = Path(__file__).resolve().parent.parent / "artifacts" / "sanity"
x_np = np.random.randn(*INPUT_SHAPE).astype(np.float32)


def time_loop(fn):
  for _ in range(WARMUP): fn()
  samples = []
  for _ in range(TIMED):
    t0 = time.perf_counter_ns()
    fn()
    samples.append((time.perf_counter_ns() - t0) / 1e6)
  samples.sort()
  return {
    "median_ms": statistics.median(samples),
    "p10_ms": samples[int(0.1 * TIMED)],
    "p90_ms": samples[int(0.9 * TIMED) - 1],
    "std_ms": statistics.stdev(samples),
  }


def bench_tflite():
  interp = Interpreter(model_path=str(ART / f"arch{ARCH_IDX}_cifar.tflite"), num_threads=1)
  interp.allocate_tensors()
  inp = interp.get_input_details()[0]
  out = interp.get_output_details()[0]
  interp.set_tensor(inp["index"], x_np)
  def step():
    interp.set_tensor(inp["index"], x_np)
    interp.invoke()
    interp.get_tensor(out["index"])
  return time_loop(step)


def bench_onnx():
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1
  so.inter_op_num_threads = 1
  sess = ort.InferenceSession(str(ART / f"arch{ARCH_IDX}_cifar.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
  name = sess.get_inputs()[0].name
  def step(): sess.run(None, {name: x_np})
  return time_loop(step)


def bench_torchmobile():
  import torch
  torch.set_num_threads(1)
  m = torch.jit.load(str(ART / f"arch{ARCH_IDX}_cifar.ptl"))
  m.eval()
  x = torch.from_numpy(x_np)
  
  def step():
    with torch.no_grad(): m(x)

  return time_loop(step)


if __name__ == "__main__":
  for name, fn in [("litert", bench_tflite), ("onnx", bench_onnx), ("torchmobile", bench_torchmobile)]:
    try:
      r = fn()
      print(f"{name:12s} median={r['median_ms']:.3f} p10={r['p10_ms']:.3f} p90={r['p90_ms']:.3f} std={r['std_ms']:.3f}")
    except Exception as e:
      print(f"{name:12s} FAILED: {e}")
  print("HW-NAS-Bench reference raspi4_latency for arch 0 = 10.481977 ms")
