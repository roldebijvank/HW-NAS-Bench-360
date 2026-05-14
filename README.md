# hwnas360 measurement pipeline

Benchmarks 6466 non-iso NB201 archs across 3 tasks (cifar100, ninapro, darcy) and 3 frameworks (LiteRT, ONNX, TorchMobile) on Pi5; Pixel 6a runs LiteRT/GPU only. Single CSV `results/latency.csv` with a `device` column.

## 1. Pi5 setup (once)

```sh
export RPI_HOST=user@raspberrypi.local
ssh $RPI_HOST 'mkdir -p ~/bep/scripts ~/bep/queue && sudo cpupower frequency-set -g performance'
ssh $RPI_HOST 'pip3 install tflite-runtime==2.14.0 onnxruntime==1.19.2 torch==2.4.1 numpy==1.26.4'
```

`scripts/pi/remote_bench.py` is auto-rsynced by the driver each run.

## 2. Pixel 6a setup (once)

Install the bundled benchmark APK:
```sh
adb install -r android_aarch64_benchmark_model.apk
```
GPU delegate + 45C thermal gate are wired in `scripts/pixel/run.py`.

## 3. Run

### Pi5 (one command, streams)

```sh
uv run python scripts/pi/run.py --limit 5     # smoke test
uv run python scripts/pi/run.py               # full sweep
```

Exports + push + bench + cleanup are interleaved per arch (300-arch sliding window).

### Pixel (two phases: push all, then bench)

```sh
uv run python -m scripts.pixel.convert        # phase 1: export tflite + adb push all archs
uv run python -m scripts.pixel.bench          # phase 2: bench what's on device, append CSV
```

Phase 1 is idempotent (skips arch dirs already on device unless `--overwrite`); phase 2 skips arch idx already in `completed_pixel.txt`. Arch dirs stay on device after benching for restart-friendliness.

Outputs:
- `results/latency.csv` — appended per (arch, task, runtime); columns: `device,arch_idx,task,runtime,lat_ms_median,lat_ms_var,energy_mj_median,status,error`
- `results/completed_pi.txt`, `results/completed_pixel.txt` — finished arch idx per device, used for crash recovery (idempotent restart)

Mac stages Pi exports under `artifacts/exports_pi/` then deletes per-arch on success. Pi keeps at most 300 unmeasured arch dirs in `~/bep/queue/` (sliding window via bounded queue). Pixel keeps all arch dirs in `/data/local/tmp/archs/` permanently (no auto-delete).

## 4. Thermal gating

- Pi5: pre-run gate at 60C via `vcgencmd measure_temp`; post-run `vcgencmd get_throttled`; on throttle, discard rows + cooldown to 55C then re-run same arch.
- Pixel: pre-task gate at 45C via `/sys/class/thermal/thermal_zone0/temp`; polls every 10s while hot.

## 5. Join accuracies

```sh
uv run python -m scripts.utils.join_accuracies
# -> results/hwnas_bench_360_v1.csv
```

cifar100 acc from NB201 `.pth`; ninapro/darcy from NB360 pickles. Set paths in `scripts/utils/task_specs.py`.

## 6. Analysis

`scripts/analyze.ipynb`: Spearman across tasks (Q1) and frameworks (Q2), Jaccard of Pareto-optimal sets (Q3). Set `DEVICE = 'pi5'` or `'pixel'` in cell 1.

## Notes

- Pixel reports mean ± std from `BenchmarkModelActivity`; the median column stores the activity's mean and `lat_ms_var = std**2`. No per-run timing available without rebuilding the APK.
- Energy column reserved; INA226 sampler not wired in this revision.
- Both runners support `--arch <idx>` (repeatable), `--arch-list <file>`, `--limit`, `--start`, `--all`.
