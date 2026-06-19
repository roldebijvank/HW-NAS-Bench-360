# data/

This directory holds upstream data files required to run the pipeline.
**None of these files are redistributed here** — download them from their
original sources and place them at the paths listed below.

## Required files

### NAS-Bench-201 API

Used by Stage 3 (`utils/export_arch_accuracies.py`) to look up CIFAR-100
test accuracies.

- **Source**: https://github.com/D-X-Y/NAS-Bench-201
- **File**: `NAS-Bench-201-v1_1-096897.pth`
- **Expected path**: `data/nas-bench-201/NAS-Bench-201-v1_1-096897.pth`

### HW-NAS-Bench

Used by `utils/model_utils.py` (build models) and `utils/build_dataset.py`
(isomorphism expansion). Contains architecture configs and hardware proxy
metrics for all 15,625 NB201 architectures.

- **Source**: https://github.com/GATECH-EIC/HW-NAS-Bench
- **File**: `HW-NAS-Bench-v1_0.pickle` (plus the `hw_nas_bench_api/` package
  in the same directory)
- **Expected path**: `data/hw-nas-bench/HW-NAS-Bench-v1_0.pickle`

### NB360 NinaPro accuracies

Used by Stage 3 to look up NinaPro test accuracies. This is the NATS-TSS
pickle from the NAS-Bench-360 project keyed to the NinaPro task.

- **Source**: https://github.com/rtu715/NAS-Bench-360
- **File**: `NATS-tss-v1_0-daa55.pickle`
- **Expected path**: `data/nb360_ninapro/NATS-tss-v1_0-daa55.pickle`

### NB360 DarcyFlow accuracies

Same as above but for the DarcyFlow regression task.

- **Source**: https://github.com/rtu715/NAS-Bench-360
- **File**: `NATS-tss-v1_0-48858.pickle`
- **Expected path**: `data/nb360_darcyflow/NATS-tss-v1_0-48858.pickle`
