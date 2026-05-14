"""Iterate non-isomorphic NB201 arch indices (~6466).
Canonicalizes arch_str via Structure.to_unique_str, keeps first occurrence.
"""
from pathlib import Path
import sys, pickle, warnings
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
HW_REPO = ROOT / "data" / "hw-nas-bench"
if str(HW_REPO) not in sys.path: sys.path.insert(0, str(HW_REPO))

from hw_nas_bench_api.nas_201_models.cell_searchs.genotypes import Structure


def non_iso_indices(hw_metrics_pickle=None):
  if hw_metrics_pickle is None:
    hw_metrics_pickle = HW_REPO / "HW-NAS-Bench-v1_0.pickle"
  with open(hw_metrics_pickle, "rb") as f, warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=getattr(np, "VisibleDeprecationWarning", np.exceptions.VisibleDeprecationWarning))
    d = pickle.load(f)
  configs = d["nasbench201"]["cifar10"]["config"]
  iterator = configs.items() if isinstance(configs, dict) else enumerate(configs)
  seen = {}
  reps = []
  for arch_idx, cfg in iterator:
    arch_str = cfg["arch_str"]
    canon = Structure.str2structure(arch_str).to_unique_str(consider_zero=True)
    if canon in seen: continue
    seen[canon] = arch_idx
    reps.append(arch_idx)
  return reps


if __name__ == "__main__":
  reps = non_iso_indices()
  print(f"non-iso reps: {len(reps)}")
  print("first 5:", reps[:5])
