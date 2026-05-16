"""Export per-arch accuracies for all tasks into a single CSV."""
from pathlib import Path
import argparse
import csv

from scripts.utils.task_specs import TASKS, load_accuracies

ROOT = Path(__file__).resolve().parent.parent.parent


def _normalize_arch_keys(accs):
    """Return dict with int keys when possible."""
    out = {}
    for k, v in accs.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "accuracies_by_arch.csv"),
        help="Output CSV path.",
    )
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=list(TASKS.keys()),
        help="Task names to include (default: all).",
    )
    args = ap.parse_args()

    tasks = [t for t in args.tasks if t in TASKS]
    if not tasks:
        raise SystemExit("no valid tasks requested")

    accs_by_task = {}
    for t in tasks:
        accs = load_accuracies(t) or {}
        accs_by_task[t] = _normalize_arch_keys(accs)

    archs = set()
    for accs in accs_by_task.values():
        archs.update(accs.keys())

    out_cols = ["arch_idx"] + [f"acc_{t}" for t in tasks]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    total = len(archs)
    last_pct = -1
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for arch_idx in sorted(archs):
            row = {"arch_idx": arch_idx}
            for t in tasks:
                val = accs_by_task[t].get(arch_idx, "")
                row[f"acc_{t}"] = "" if val is None else val
            w.writerow(row)
            row_count += 1
            if total:
                pct = int((row_count / total) * 100)
                if pct != last_pct:
                    print(f"{pct}%", flush=True)
                    last_pct = pct


if __name__ == "__main__":
    main()
