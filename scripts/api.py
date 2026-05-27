"""Read-only API over results/dataset.parquet.

Covers all 15,625 NB201 archs; isomorphism mapping is internal.

Primary:
  query(arch_index, task, framework, device='pi5')

Bulk:
  query_arch(arch_index)          all (device, framework, task)
  query_task(task, framework=None, device=None)
  filter(**kwargs)                generic predicate
  aggregate(by, metric, agg)
  get_pareto_front(task, device='pi5', framework='litert',
                   minimize=('lat_ms_median','energy_mj'),
                   maximize=('accuracy',))
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence
import pandas as pd

PARQUET = Path(__file__).resolve().parent.parent / "results" / "dataset.parquet"

TASKS = ("cifar100", "ninapro", "darcy")
FRAMEWORKS = ("litert", "onnx", "torchmobile")
DEVICES = ("pi5", "pixel")


@lru_cache(maxsize=1)
def _df() -> pd.DataFrame:
  return pd.read_parquet(PARQUET)


def _select(arch_index: int | None = None, task: str | None = None,
            framework: str | None = None, device: str | None = None) -> pd.DataFrame:
  df = _df()
  if arch_index is not None: df = df[df["arch_idx"] == arch_index]
  if task is not None: df = df[df["task"] == task]
  if framework is not None: df = df[df["framework"] == framework]
  if device is not None: df = df[df["device"] == device]
  return df


def query(arch_index: int, task: str, framework: str, device: str = "pi5") -> dict:
  """Single-row lookup. Returns flat dict; latency/energy may be NaN if not measured."""
  r = _select(arch_index, task, framework, device)
  if r.empty:
    raise KeyError(f"no row for arch={arch_index} task={task} framework={framework} device={device}")
  row = r.iloc[0]
  return {
    "arch_index": int(row["arch_idx"]),
    "task": task,
    "framework": framework,
    "device": device,
    "median_latency_ms": _f(row["lat_ms_median"]),
    "median_energy_mj": _f(row["energy_mj"]),
    "accuracy": _f(row["accuracy"]),
  }


def query_arch(arch_index: int) -> pd.DataFrame:
  """All (device, framework, task) rows for one arch."""
  return _select(arch_index=arch_index).reset_index(drop=True)


def query_task(task: str, framework: str | None = None, device: str | None = None) -> pd.DataFrame:
  """All archs for a task, optionally pinned to framework/device."""
  return _select(task=task, framework=framework, device=device).reset_index(drop=True)


def filter(**kwargs) -> pd.DataFrame:
  """Generic filter: arch_index, task, framework, device. Returns DataFrame."""
  return _select(**kwargs).reset_index(drop=True)


def aggregate(by: Sequence[str] = ("device", "framework", "task"),
              metric: str = "lat_ms_median",
              agg: Sequence[str] = ("min", "median", "max")) -> pd.DataFrame:
  """Aggregate `metric` grouped by `by`."""
  df = _df()
  return df.groupby(list(by))[metric].agg(list(agg)).reset_index()


def get_pareto_front(task: str,
                     device: str = "pi5",
                     framework: str = "litert",
                     minimize: Sequence[str] = ("lat_ms_median", "energy_mj"),
                     maximize: Sequence[str] = ("accuracy",)) -> pd.DataFrame:
  """Non-dominated archs across given metrics for one (device, framework, task) slice."""
  df = _select(task=task, framework=framework, device=device).dropna(
    subset=list(minimize) + list(maximize)).reset_index(drop=True)
  if df.empty: return df

  signs = {c: 1 for c in minimize} | {c: -1 for c in maximize}
  cols = list(signs)
  M = df[cols].to_numpy() * [signs[c] for c in cols]  # all minimize

  n = M.shape[0]
  keep = []
  for i in range(n):
    dominated = False
    for j in range(n):
      if i == j: continue
      if all(M[j] <= M[i]) and any(M[j] < M[i]):
        dominated = True
        break
    if not dominated: keep.append(i)
  return df.iloc[keep].sort_values(list(minimize)).reset_index(drop=True)


def _f(x):
  import math
  try:
    if x is None or (isinstance(x, float) and math.isnan(x)): return None
  except TypeError: pass
  return float(x)


if __name__ == "__main__":
  import json
  print("query(0, 'cifar100', 'litert'):")
  print(json.dumps(query(0, "cifar100", "litert"), indent=2))
  print("\nquery_arch(0):"); print(query_arch(0))
  print("\naggregate by (device, framework):")
  print(aggregate(by=("device", "framework"), metric="lat_ms_median"))
  print("\npareto front cifar100 pi5/litert (first 10):")
  print(get_pareto_front("cifar100").head(10))
