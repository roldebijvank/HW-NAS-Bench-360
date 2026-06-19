"""Bench arch dirs on Jetson Nano (ONNX only) measuring latency and INA3221 board energy.

  taskset -c 0 python3 ~/HW-NAS-Bench-360/device/jetson/bench.py --energy [--task cifar100] [--arch N]
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.pipeline_config import TASKS, FRAMEWORKS, DEVICES
from utils.measure import gate_temp, time_loop
from utils.runners import make_step_onnx
from utils.runner_utils import read_completed, parse_arch_list
from utils.bench_loop import list_arch_dirs, run_bench_loop

DATA_ROOT = Path.home() / "HW-NAS-Bench-360"
ARCH_ROOT = DATA_ROOT / "archs"
RESULTS_DIR = DATA_ROOT / "results"
CSV_PATH = RESULTS_DIR / "latency_jetson.csv"

DEVICE = "jetson"
TIMED  = 40

POWER_NODE = DEVICES[DEVICE]["energy_rail_path"]
SAMPLE_HZ  = DEVICES[DEVICE]["sampler_hz"]
SAMPLER_CORE = 1

TEMP_CEILING_C = 90.0
TEMP_RESUME_C  = 80.0

TASK_SHAPES = {t: TASKS[t]["input_shape"] for t in TASKS}
RUNTIME_EXT = {n: FRAMEWORKS[n]["ext"] for n in DEVICES[DEVICE]["frameworks"]}
ENERGY_READER_CLS = DEVICES[DEVICE]["energy_reader"]
TEMP_READER = DEVICES[DEVICE]["temp_reader"]

MIN_WINDOW_S = 0.5
IDLE_MEASURE_EVERY = 100
IDLE_MEASURE_SAMPLES = 2
IDLE_SAMPLE_S = 1.0

CSV_COLS = ["device", "arch_idx", "task", "runtime",
            "lat_ms_median", "lat_ms_var",
            "energy_mj_raw", "energy_mj_net", "idle_power_mw",
            "n_passes", "status", "error"]


def measure_idle_power_mw(power_node, sample_hz, duration_s, repeats, core):
  samples = []
  for _ in range(repeats):
    reader = ENERGY_READER_CLS(power_node=power_node, sample_hz=sample_hz, core=core)
    reader.start()
    t0 = time.perf_counter()
    time.sleep(duration_s)
    energy_mj = reader.stop()
    dt = time.perf_counter() - t0
    if energy_mj is not None and dt > 0:
      samples.append(energy_mj / dt)
  if not samples: return None
  return statistics.median(samples)


def make_step(path, x_np, use_gpu):
  import onnxruntime as ort
  providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
               if use_gpu else ["CPUExecutionProvider"])
  return make_step_onnx(
    path, x_np,
    providers=providers,
    graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    log_providers=True,
  )


def bench_arch(arch_root, arch_idx, tasks, energy_enabled, power_node, use_gpu,
               sampler_core, idle_power_mw):
  rows = []
  d = arch_root / f"arch_{arch_idx}"
  for task in tasks:
    shape = TASK_SHAPES[task]
    x_np = np.random.randn(1, *shape).astype(np.float32)
    for runtime, ext in RUNTIME_EXT.items():
      art = d / f"{task}_{runtime}.{ext}"
      row = {"device": DEVICE, "arch_idx": arch_idx,
             "task": task, "runtime": runtime}
      if not art.exists():
        row["status"] = "missing"; row["error"] = "artifact not found"
        rows.append(row); continue
      gate_temp(TEMP_READER, TEMP_CEILING_C, TEMP_RESUME_C,
                label=f"arch {arch_idx} {task}/{runtime}: ")
      try:
        step = make_step(art, x_np, use_gpu=use_gpu)
        reader = (ENERGY_READER_CLS(power_node=power_node,
                                    sample_hz=SAMPLE_HZ, core=sampler_core)
                  if energy_enabled else None)
        med, var, energy_mj, n_passes = time_loop(
          step, reader, timed=TIMED, min_window_s=MIN_WINDOW_S)
        row["lat_ms_median"] = med
        row["lat_ms_var"] = var
        row["n_passes"] = n_passes
        if energy_mj is not None:
          per_inf_mj = energy_mj / n_passes
          row["energy_mj_raw"] = per_inf_mj
          if idle_power_mw is not None:
            idle_mj = idle_power_mw * (med / 1e3)
            row["energy_mj_net"] = per_inf_mj - idle_mj
            row["idle_power_mw"] = idle_power_mw
        row["status"] = "ok"
      except Exception as e:
        row["status"] = "error"; row["error"] = str(e)[:200]
        print(f"  arch {arch_idx} {task}/{runtime}: ERROR {e}",
              file=sys.stderr, flush=True)
      rows.append(row)
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", choices=list(TASK_SHAPES), action="append",
                  default=[], help="repeat to select subset; default all")
  ap.add_argument("--arch", action="append", type=int, default=[])
  ap.add_argument("--arch-list", type=Path, default=None)
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--arch-root", type=Path, default=ARCH_ROOT,
                  help="Directory containing arch_* folders")
  ap.add_argument("--energy", action="store_true",
                  help="Measure energy via INA3221 POM_5V_IN rail")
  ap.add_argument("--power-node", default=POWER_NODE,
                  help="sysfs power node (mW) for the POM_5V_IN rail")
  ap.add_argument("--cpu", action="store_true",
                  help="Use CPUExecutionProvider (default: CUDA GPU)")
  ap.add_argument("--sampler-core", type=int, default=SAMPLER_CORE,
                  help="CPU core to pin the energy sampler thread to "
                       "(-1 to disable pinning)")
  args = ap.parse_args()

  tasks = args.task or list(TASK_SHAPES.keys())

  arch_root = args.arch_root.expanduser()
  on_dev = list_arch_dirs(arch_root)
  if not on_dev:
    print(f"no arch_*/ under {arch_root}; convert/copy archs first",
          file=sys.stderr); sys.exit(2)

  if args.arch:
    indices = list(dict.fromkeys(args.arch))
  elif args.arch_list:
    indices = parse_arch_list(args.arch_list)
  else:
    indices = on_dev
  indices = indices[args.start:]
  if args.limit: indices = indices[:args.limit]

  done_per_key = {t: read_completed(RESULTS_DIR / f"completed_jetson_{t}.txt")
                  for t in tasks}

  energy_enabled = args.energy
  if energy_enabled and not os.access(args.power_node, os.R_OK):
    print(f"power node not readable: {args.power_node}; disabling energy",
          file=sys.stderr)
    energy_enabled = False

  sampler_core = None if args.sampler_core < 0 else args.sampler_core
  idle_power_mw = [None]
  since_idle = [IDLE_MEASURE_EVERY]

  def bench_fn(arch_idx, pending_tasks):
    if energy_enabled and since_idle[0] >= IDLE_MEASURE_EVERY:
      idle_power_mw[0] = measure_idle_power_mw(
        args.power_node,
        sample_hz=SAMPLE_HZ,
        duration_s=IDLE_SAMPLE_S,
        repeats=IDLE_MEASURE_SAMPLES,
        core=sampler_core,
      )
      if idle_power_mw[0] is None:
        print("idle power measure failed; energy_mj_net will be empty",
              file=sys.stderr, flush=True)
      else:
        print(f"idle power: {idle_power_mw[0]:.1f} mW", flush=True)
      since_idle[0] = 0
    rows = bench_arch(arch_root, arch_idx, pending_tasks, energy_enabled,
                      args.power_node, use_gpu=not args.cpu,
                      sampler_core=sampler_core, idle_power_mw=idle_power_mw[0])
    since_idle[0] += 1
    return rows

  run_bench_loop(
    indices=indices,
    on_device_set=set(on_dev),
    csv_path=CSV_PATH,
    csv_cols=CSV_COLS,
    pending_fn=lambda arch_idx: [t for t in tasks
                                 if arch_idx not in done_per_key[t]],
    bench_fn=bench_fn,
    done_per_key=done_per_key,
    completion_path_fn=lambda t: RESULTS_DIR / f"completed_jetson_{t}.txt",
    mark_done_fn=lambda rows, pending: {r["task"] for r in rows
                                        if r.get("status") == "ok"},
  )


if __name__ == "__main__":
  main()
