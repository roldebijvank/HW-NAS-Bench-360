# HW-NAS-Bench-360

Hardware benchmark of all 15,625 NAS-Bench-201 architectures across three
devices (Raspberry Pi 5, Jetson Nano, Pixel 6a), three frameworks (LiteRT,
ONNX Runtime, PyTorch Mobile), and three tasks (CIFAR-100, NinaPro,
DarcyFlow). Latency and energy are measured on-device; the final dataset
is the `results/dataset.parquet` file.

## Dataset

`results/dataset.parquet` is the citable deliverable. It is **not committed
here** — download it from Zenodo/HuggingFace (link TBD). Schema:

| column | description |
|---|---|
| `arch_idx` | NB201 architecture index 0–15624 |
| `device` | `pi5` / `jetson` / `pixel` |
| `framework` | `litert` / `onnx` / `torchmobile` |
| `task` | `cifar100` / `ninapro` / `darcy` |
| `lat_ms` | median latency (ms) |
| `lat_ms_var` | latency variance |
| `energy_mj` | energy per inference (mJ); NaN for Pixel |
| `accuracy` | test accuracy; for darcy: relative L2 error (lower better) |

Isomorphic architectures share measurements with their non-isomorphic
representative (~6466 unique structures measured, expanded to 15625 rows).

## Pipeline

### Stage 1 — Export model artifacts (host)

Export NB201 models for one task to the target device. Requires PyTorch,
litert-torch, onnxruntime, and the data files listed in `data/README.md`.

**Pi (rsync to device over SSH — set `RPI_HOST=user@host` in env):**
```
python -m host.pi.convert --task cifar100
```

**USB (copy to local directory for Jetson):**
```
python -m host.usb.convert --task cifar100 --dest /Volumes/USB/archs
```

**Pixel (adb push — device must be connected):**
```
python -m host.pixel.convert
```

Repeat for `ninapro` and `darcy`. Non-isomorphic representatives only by
default; pass `--all` for all 15625.

### Stage 2 — Measure latency & energy (on-device)

Each device needs:
1. This repo cloned at the path shown below.
2. Device-specific Python packages installed (`device/<device>/requirements.txt`).
3. Model artifacts transferred in Stage 1 (arch dirs in the archs/ path shown).

**Pi 5**

Clone repo to `~/HW-NAS-Bench-360/`. Artifacts land in
`~/HW-NAS-Bench-360/archs/` (via rsync in Stage 1).
```
cd ~/HW-NAS-Bench-360
pip install -r device/pi/requirements.txt
taskset -c 3 python3 device/pi/bench.py [--energy] [--task cifar100]
```
Results: `~/HW-NAS-Bench-360/results/latency.csv`

**Jetson Nano**

Clone repo to `~/HW-NAS-Bench-360/`. Copy USB artifacts to
`~/HW-NAS-Bench-360/archs/`.
```
cd ~/HW-NAS-Bench-360
pip install -r device/jetson/requirements.txt
taskset -c 0 python3 device/jetson/bench.py --energy [--task cifar100]
```
Results: `~/HW-NAS-Bench-360/results/latency.csv`

**Pixel 6a**

Inside proot-distro Ubuntu: clone repo to `~/HW-NAS-Bench-360/`. Artifacts
land in `~/HW-NAS-Bench-360/archs/` (adb-pushed in Stage 1 to
`/data/local/tmp/archs/`, then moved into the proot home before running).
```
cd ~/HW-NAS-Bench-360
pip install -r device/pixel/requirements.txt
python3 device/pixel/bench.py [--task cifar100] [--framework litert]
```
Results: `~/HW-NAS-Bench-360/results/latency_pixel.csv`

### Stage 3 — Export accuracy values (host)

```
python -m utils.export_arch_accuracies --out results/accuracies_by_arch.csv
```

### Stage 4 — Build dataset (host)

```
python -m utils.build_dataset \
  --pi results/pi.csv \
  --jetson results/jetson.csv \
  --pixel results/pixel.csv \
  --accuracy results/accuracies_by_arch.csv \
  --hw-pickle data/hw-nas-bench/HW-NAS-Bench-v1_0.pickle \
  --out results/dataset.parquet
```

### Stage 5 — Query and analyse

```python
import api
df = api.load(non_isomorphic=True)
row = api.query(0, "cifar100", "litert", "pi5")
front = api.get_pareto_front("cifar100", "pi5", "litert")
```

Open `analysis/analyze.ipynb` for RQ analyses (Spearman rank correlations,
Pareto-set Jaccard similarity, op-frequency bias).

## Energy calibration

Run on-device to find stable sample-count N for power measurement:

```
python3 calibration/pi/energy_stability.py \
  --arch-index 0 --task cifar100 --framework litert \
  --model-path ~/HW-NAS-Bench-360/archs/arch_0/cifar100_litert.tflite --total-s 60

python3 calibration/jetson/energy_stability.py \
  --arch-index 0 --task cifar100 \
  --model-path ~/HW-NAS-Bench-360/archs/arch_0/cifar100_onnx.onnx --total-s 60

python3 calibration/jetson/sample_rate_probe.py --dur-s 5
```

## Extending the benchmark

### Add a new task

1. **`config/pipeline_config.py`** — add an entry to `TASKS`:
   ```python
   "mytask": {
       "input_shape": (C, H, W),
       "num_classes": N,
       "acc_source": "nb360_pickle",        # or "nb201_api"
       "acc_pickle": ROOT / "data/..." ,
       "acc_inner_key": ("mytask", 777),
   },
   ```
2. **`utils/task_specs.py`** — if using a new pickle format, add a branch in
   `load_accuracies` to handle it. Existing `nb360_pickle` and `nb201_api`
   sources require no changes.
3. **Stage 1**: re-run `host/*/convert.py --task mytask` to export models.
4. **Stage 2**: pass `--task mytask` to the device bench script.
5. **Stage 3**: re-run `utils/export_arch_accuracies.py`; the new task is
   picked up automatically via `TASKS`.

### Add a new framework

1. **`config/pipeline_config.py`** — add an entry to `FRAMEWORKS`:
   ```python
   "myfw": {"ext": "myext", "exporter": "export_myfw"},
   ```
2. **`utils/convert_utils.py`** — implement `export_myfw(arch_idx, input_shape,
   num_classes, out_path)` and ensure it appears in `RUNTIMES` (auto-built from
   `FRAMEWORKS`).
3. **`utils/runners.py`** — implement `make_step_myfw(path, x_np)` and add it
   to `MAKERS`.
4. **`config/pipeline_config.py`** — add `"myfw"` to the `"frameworks"` list of
   each device that should run it.
5. Stage 1 and Stage 2 pick up the new framework automatically via
   `RUNTIMES` / `RUNTIME_EXT`.

### Add a new device

1. **`config/pipeline_config.py`** — add an entry to `DEVICES`:
   ```python
   "mydevice": {
       "frameworks": ("litert", "onnx"),
       "energy_reader": MyEnergyReader,   # or omit if no energy
       "temp_reader": my_temp_fn,         # or None
   },
   ```
2. **`utils/energy_reader.py`** — if needed, implement a new `EnergyReader`
   subclass with `start()` and `stop() -> float | None` (mJ).
3. **`device/mydevice/bench.py`** — copy the closest existing bench script and
   replace device-specific constants (`DEVICE`, `DATA_ROOT`, `TEMP_*`,
   `CSV_COLS`) and any device-only logic (affinity pinning, idle-power
   resampling, throttle retry). Call `run_bench_loop` with the appropriate
   `pending_fn`, `bench_fn`, and `mark_done_fn`.
4. **`host/mydevice/convert.py`** (or reuse `host/usb/convert.py`) — add a
   transfer script for Stage 1.
5. **`utils/build_dataset.py`** — add a `load_mydevice(path)` function mirroring
   the existing loaders, and include it in `main()`.

## Requirements

```
pip install -r requirements.txt
```

Device scripts also need device-specific packages listed in
`device/<device>/requirements.txt`.
