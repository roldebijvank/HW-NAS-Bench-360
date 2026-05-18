"""Join accuracies with Pi results into a single long CSV.

Each output row is one (arch_idx, task, runtime) measurement with
accuracy columns appended.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv

from scripts.utils.arch_iter import non_iso_indices


ROOT = Path(__file__).resolve().parent.parent.parent


def _read_rows(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, (reader.fieldnames or [])


def _load_accuracies(path: Path) -> tuple[dict[int, dict[str, str]], list[str]]:
    rows, fieldnames = _read_rows(path)
    acc_cols = [c for c in fieldnames if c != "arch_idx"]
    accs: dict[int, dict[str, str]] = {}
    for r in rows:
        try:
            arch = int(r["arch_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        accs[arch] = {c: r.get(c, "") for c in acc_cols}
    return accs, acc_cols


def _merge_fieldnames(base_fields: list[str], extra_fields: list[str]) -> list[str]:
    out = list(base_fields)
    for f in extra_fields:
        if f not in out:
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default=str(ROOT / "results" / "accuracies.csv"))
    ap.add_argument("--cifar", default=str(ROOT / "results" / "results_pi_cifar.csv"))
    ap.add_argument("--nina-darcy", default=str(ROOT / "results" / "results_pi_nina_darcy.csv"))
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "merged_pi_results_long.csv"),
        help="Output CSV path.",
    )
    ap.add_argument(
        "--non-iso",
        action="store_true",
        help="Keep only non-isomorphic arch indices.",
    )
    args = ap.parse_args()

    acc_path = Path(args.acc)
    cifar_path = Path(args.cifar)
    nina_darcy_path = Path(args.nina_darcy)
    out_path = Path(args.out)

    accs, acc_cols = _load_accuracies(acc_path)
    print(f"loaded {len(accs)} accuracies from {acc_path}", flush=True)

    rows_all: list[dict] = []
    base_fields: list[str] = []
    for path in (cifar_path, nina_darcy_path):
        rows, fields = _read_rows(path)
        if not base_fields:
            base_fields = fields
        else:
            base_fields = _merge_fieldnames(base_fields, fields)
        rows_all.extend(rows)
    print(f"loaded {len(rows_all)} result rows", flush=True)

    non_iso_set = None
    if args.non_iso:
        try:
            non_iso_set = set(non_iso_indices())
        except FileNotFoundError as e:
            raise SystemExit(f"non-iso list missing: {e}")
        print(f"using non-iso set: {len(non_iso_set)} archs", flush=True)

    out_cols = list(base_fields)
    if "arch_idx" in out_cols:
        arch_pos = out_cols.index("arch_idx")
        out_cols = out_cols[: arch_pos + 1] + acc_cols + out_cols[arch_pos + 1 :]
    else:
        out_cols = ["arch_idx", *acc_cols, *out_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        kept = 0
        for r in rows_all:
            try:
                arch = int(r["arch_idx"])
            except (KeyError, TypeError, ValueError):
                continue
            if non_iso_set is not None and arch not in non_iso_set:
                continue
            out_row = dict(r)
            acc_row = accs.get(arch, {})
            for c in acc_cols:
                out_row[c] = acc_row.get(c, "")
            w.writerow(out_row)
            kept += 1

    print(f"wrote {out_path} ({kept} rows)", flush=True)


if __name__ == "__main__":
    main()
