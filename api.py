"""Read-only API over results/dataset.parquet (load, query, pareto, aggregate).

  python api.py
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Sequence
import pandas as pd

PARQUET = Path(__file__).resolve().parent / "results" / "results.parquet"

REQUIRED_COLUMNS = (
  "arch_idx",
  "device",
  "framework",
  "task",
  "lat_ms",
  "lat_ms_var",
  "energy_mj",
  "accuracy",
)


@lru_cache(maxsize=1)
def _df() -> pd.DataFrame:
  if not PARQUET.exists():
    raise FileNotFoundError(f"missing parquet file: {PARQUET}")
  df = pd.read_parquet(PARQUET)
  missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
  if missing:
    raise ValueError(f"parquet missing columns: {missing}")
  return df


@lru_cache(maxsize=1)
def _non_iso_indices() -> frozenset[int]:
  """Indices of the non-isomorphic NB201 representatives (~6466)."""
  import sys
  root = Path(__file__).resolve().parent
  if str(root) not in sys.path:
    sys.path.insert(0, str(root))
  from utils.arch_iter import non_iso_indices
  return frozenset(non_iso_indices())


def load(non_isomorphic: bool = False) -> pd.DataFrame:
  """Full dataset, optionally restricted to non-isomorphic arch representatives."""
  df = _df()
  if non_isomorphic:
    df = df[df["arch_idx"].isin(_non_iso_indices())]
  return df.reset_index(drop=True)


def _select(arch_index: int | None = None, task: str | None = None,
            framework: str | None = None, device: str | None = None,
            non_isomorphic: bool = False) -> pd.DataFrame:
  df = _df()
  if non_isomorphic: df = df[df["arch_idx"].isin(_non_iso_indices())]
  if arch_index is not None: df = df[df["arch_idx"] == arch_index]
  if task is not None: df = df[df["task"] == task]
  if framework is not None: df = df[df["framework"] == framework]
  if device is not None: df = df[df["device"] == device]
  return df


def query(arch_index: int, task: str, framework: str, device: str) -> dict:
  """Single-row lookup. Returns a dict with schema columns."""
  r = _select(arch_index, task, framework, device)
  if r.empty:
    raise KeyError(f"no row for arch={arch_index} task={task} framework={framework} device={device}")
  return _row_to_dict(r.iloc[0])


def query_task(task: str, framework: str, device: str,
               non_isomorphic: bool = False) -> pd.DataFrame:
  """All architecture rows for a given (task, framework, device)."""
  return _select(task=task, framework=framework, device=device,
                 non_isomorphic=non_isomorphic).reset_index(drop=True)


def aggregate(by: Sequence[str] = ("device", "framework", "task"),
              metric: str = "lat_ms",
              agg: Sequence[str] = ("min", "median", "max"),
              non_isomorphic: bool = False) -> pd.DataFrame:
  """Aggregate `metric` grouped by `by`."""
  df = load(non_isomorphic=non_isomorphic)
  by_cols = list(_as_tuple(by))
  agg_list = list(_as_tuple(agg))
  return df.groupby(by_cols)[metric].agg(agg_list).reset_index()


# tasks whose `accuracy` column is an error metric (lower is better)
SCORE_LOWER_BETTER_TASKS = frozenset({"darcy"})


def get_pareto_front(task: str,
                     device: str,
                     framework: str,
                     minimise: Sequence[str] = ("lat_ms", "energy_mj"),
                     maximise: Sequence[str] = ("accuracy",),
                     non_isomorphic: bool = False) -> pd.DataFrame:
  """Non-dominated archs across given metrics for one (device, framework, task) slice.

  For tasks in SCORE_LOWER_BETTER_TASKS (darcy: accuracy = rel-L2 error),
  `accuracy` is automatically moved from maximise to minimise.
  """
  minimise = _as_tuple(minimise)
  maximise = _as_tuple(maximise)
  if task in SCORE_LOWER_BETTER_TASKS and "accuracy" in maximise:
    maximise = tuple(c for c in maximise if c != "accuracy")
    if "accuracy" not in minimise:
      minimise = minimise + ("accuracy",)
  df = _select(task=task, framework=framework, device=device,
               non_isomorphic=non_isomorphic).dropna(
    subset=list(minimise) + list(maximise)).reset_index(drop=True)
  if df.empty:
    return df

  signs = {c: 1 for c in minimise} | {c: -1 for c in maximise}
  cols = list(signs)
  M = df[cols].to_numpy() * [signs[c] for c in cols]  # convert to all-minimize

  n = M.shape[0]
  keep = []
  for i in range(n):
    dominated = False
    for j in range(n):
      if i == j:
        continue
      if all(M[j] <= M[i]) and any(M[j] < M[i]):
        dominated = True
        break
    if not dominated:
      keep.append(i)
  return df.iloc[keep].sort_values(list(minimise)).reset_index(drop=True)


def _row_to_dict(row: pd.Series) -> dict:
  out = {}
  for key, value in row.items():
    if pd.isna(value):
      out[key] = None
    elif hasattr(value, "item"):
      out[key] = value.item()
    else:
      out[key] = value
  if "arch_idx" in out and out["arch_idx"] is not None:
    out["arch_idx"] = int(out["arch_idx"])
  return out


def _as_tuple(value: Sequence[str] | str) -> tuple[str, ...]:
  if isinstance(value, str):
    return (value,)
  return tuple(value)


if __name__ == "__main__":
  import json
  print("query(0, 'cifar100', 'litert', 'pi5'):")
  print(json.dumps(query(0, "cifar100", "litert", "pi5"), indent=2))
  print("\nquery_task('cifar100', 'litert', 'pi5'):")
  print(query_task("cifar100", "litert", "pi5").head(3))
  print("\naggregate by (device, framework):")
  print(aggregate(by=("device", "framework"), metric="lat_ms"))
  print("\npareto front cifar100 pi5/litert (first 10):")
  print(get_pareto_front("cifar100", "pi5", "litert").head(10))
