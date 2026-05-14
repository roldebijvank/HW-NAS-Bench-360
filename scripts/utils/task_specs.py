"""Per-task input shape, num_classes, accuracy pickle source.
Used by host (export) and Pi daemon (random input shape).
"""
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parent.parent.parent

NB201_API_FILE = ROOT / "data" / "nas-bench-201" / "NAS-Bench-201-v1_1-096897.pth"

TASKS = {
  "cifar100": {
    "input_shape": (3, 32, 32),
    "num_classes": 100,
    "acc_source": "nb201_api",
    "nb201_dataset": "cifar100",  # for NASBench201API.get_more_info
  },
  "ninapro": {
    "input_shape": (1, 52, 16),
    "num_classes": 18,
    "acc_source": "nb360_pickle",
    "acc_pickle": ROOT / "data" / "nb360_ninapro" / "NATS-tss-v1_0-daa55.pickle",
    "acc_inner_key": ("ninapro", 777),
  },
  "darcy": {
    # native NB360 darcy is (1,85,85); zero-padded to (1,88,88) so NB201 stride-2 cells line up (div by 4).
    "input_shape": (1, 88, 88),
    "num_classes": 1,
    "acc_source": "nb360_pickle",
    "acc_pickle": ROOT / "data" / "nb360_darcyflow" / "NATS-tss-v1_0-48858.pickle",
    "acc_inner_key": ("darcyflow", 777),
  },
}

FINAL_EPOCH_KEY = "ori-test@199"


_cache = {}

def _load_nb201_api():
  if "_nb201" in _cache: return _cache["_nb201"]
  if not NB201_API_FILE.exists():
    _cache["_nb201"] = None
    return None
  import torch
  _orig = torch.load
  torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
  try:
    from nas_201_api import NASBench201API
    api = NASBench201API(str(NB201_API_FILE), verbose=False)
  finally:
    torch.load = _orig
  _cache["_nb201"] = api
  return api


def load_accuracies(task):
  """Return dict arch_idx -> last-epoch eval accuracy. None if source missing."""
  if task in _cache: return _cache[task]
  spec = TASKS[task]
  src = spec["acc_source"]

  if src == "nb201_api":
    api = _load_nb201_api()
    if api is None:
      _cache[task] = None
      return None
    ds = spec["nb201_dataset"]
    out = {}
    for arch_idx in range(15625):
      try:
        info = api.get_more_info(arch_idx, ds, hp="200", is_random=False)
        out[arch_idx] = info.get("test-accuracy")
      except Exception:
        pass
    _cache[task] = out
    return out

  if src == "nb360_pickle":
    path = spec["acc_pickle"]
    if not path.exists():
      _cache[task] = None
      return None
    with open(path, "rb") as f: d = pickle.load(f)
    inner_key = spec["acc_inner_key"]
    out = {}
    for arch_idx, infos in d["arch2infos"].items():
      rec = infos["200"]["all_results"].get(inner_key)
      if rec is None: continue
      out[arch_idx] = rec["eval_acc1es"].get(FINAL_EPOCH_KEY)
    _cache[task] = out
    return out

  raise ValueError(f"unknown acc_source: {src}")


if __name__ == "__main__":
  for t in TASKS:
    a = load_accuracies(t)
    print(f"{t}: {0 if a is None else len(a)} entries")
