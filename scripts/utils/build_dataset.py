"""Build results/dataset.parquet: long-form per (arch_idx, device, framework, task).
Maps all 15625 archs to their non-iso rep (~6466) for latency/energy lookups.
Accuracies per-arch are joined from accuracies.csv.
Pixel runtime 'litert-npu' is renamed to 'litert' (device disambiguates).
"""
from pathlib import Path
import sys, pickle, warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
HW_REPO = ROOT / "data" / "hw-nas-bench"
if str(HW_REPO) not in sys.path: sys.path.insert(0, str(HW_REPO))
from hw_nas_bench_api.nas_201_models.cell_searchs.genotypes import Structure


def arch_to_rep_map():
  """Return dict {arch_idx: rep_arch_idx} for all 15625 archs."""
  pkl = HW_REPO / "HW-NAS-Bench-v1_0.pickle"
  with open(pkl, "rb") as f, warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=getattr(np, "VisibleDeprecationWarning", np.exceptions.VisibleDeprecationWarning))
    d = pickle.load(f)
  configs = d["nasbench201"]["cifar10"]["config"]
  iterator = configs.items() if isinstance(configs, dict) else enumerate(configs)
  canon_to_rep, mapping = {}, {}
  for arch_idx, cfg in iterator:
    canon = Structure.str2structure(cfg["arch_str"]).to_unique_str(consider_zero=True)
    if canon not in canon_to_rep:
      canon_to_rep[canon] = arch_idx
    mapping[arch_idx] = canon_to_rep[canon]
  return mapping


def build():
  pi = pd.read_csv(RESULTS / "pi.csv")
  px = pd.read_csv(RESULTS / "latency_pixel.csv")
  acc = pd.read_csv(RESULTS / "accuracies.csv")

  pi = pi.rename(columns={"runtime": "framework"})
  pi["device"] = "pi5"
  pi = pi[["device", "arch_idx", "framework", "task", "lat_ms_median", "lat_ms_var", "energy_mj", "status"]]

  px = px.rename(columns={"runtime": "framework", "energy_mj_median": "energy_mj"})
  px["framework"] = px["framework"].replace({"litert-npu": "litert"})
  px["lat_ms_var"] = np.nan
  px = px[["device", "arch_idx", "framework", "task", "lat_ms_median", "lat_ms_var", "energy_mj", "status"]]

  measured = pd.concat([pi, px], ignore_index=True)
  measured = measured.rename(columns={"arch_idx": "rep_arch_idx"})

  mapping = arch_to_rep_map()
  full = pd.DataFrame({"arch_idx": list(mapping.keys()), "rep_arch_idx": list(mapping.values())})

  df = full.merge(measured, on="rep_arch_idx", how="left")

  acc_long = acc.melt(id_vars="arch_idx", value_vars=["acc_cifar100", "acc_ninapro", "acc_darcy"],
                     var_name="task", value_name="accuracy")
  acc_long["task"] = acc_long["task"].str.removeprefix("acc_")
  df = df.merge(acc_long, on=["arch_idx", "task"], how="left")

  df = df[["arch_idx", "rep_arch_idx", "device", "framework", "task",
           "lat_ms_median", "lat_ms_var", "energy_mj", "accuracy", "status"]]
  out = RESULTS / "dataset.parquet"
  df.to_parquet(out, index=False)
  print(f"wrote {out}  rows={len(df):,}  archs={df['arch_idx'].nunique():,}")
  print(df.head())


if __name__ == "__main__":
  build()
