"""Probe the achievable INA3221 polling rate on the Jetson Nano.

bench.py / energy_stability.py poll the POM_5V_IN power node at a fixed
SAMPLE_HZ (550). This script sweeps a set of candidate target rates and, for
each, polls the node for `--dur-s` seconds, counting how often the loop misses
its deadline (the read + sleep overran the sample interval) and how many reads
errored. Prints an overview so you can pick the highest rate that still keeps
misses low.

A "miss" = after reading the node and accounting for time spent, no time
remained before the next scheduled tick (remaining <= 0); the deadline is reset
to now (same recovery as bench.py's sampler). High miss% means the target rate
is faster than the node can actually be read at.

Self-contained (no scripts.* imports) so it can be copied to the Jetson.

  sudo python3 ~/sample_rate_probe.py --dur-s 5
  python3 ~/sample_rate_probe.py --rates 200,400,550,800,1000 --dur-s 10
"""
import argparse
import statistics
import time
from pathlib import Path

# Matches jetson/bench.py.
POWER_NODE = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input"
DEFAULT_RATES = [100, 200, 400, 550, 800, 1000, 1500, 2000]


def read_power_mw(node):
  with open(node) as f:
    return float(f.read().strip())


def probe(node, target_hz, dur_s):
  """Poll `node` at `target_hz` for `dur_s`. Returns metrics dict.

  Mirrors the timing loop in bench.py's EnergySampler so the measured misses
  reflect what the real sampler would see.
  """
  interval = 1.0 / target_hz
  t = []
  misses = 0
  read_errors = 0
  last_error = None

  t_start = time.perf_counter()
  t_end = t_start + dur_s
  deadline = time.perf_counter()
  while time.perf_counter() < t_end:
    try:
      p = read_power_mw(node)
      t.append(time.perf_counter())
    except (OSError, ValueError) as e:
      read_errors += 1
      last_error = e
    deadline += interval
    remaining = deadline - time.perf_counter()
    if remaining > 0:
      time.sleep(remaining)
    else:
      misses += 1
      deadline = time.perf_counter()
  elapsed = time.perf_counter() - t_start

  n = len(t)
  achieved_hz = n / elapsed if elapsed > 0 else 0.0
  ticks = misses + n + read_errors  # loop iterations ~= scheduled ticks
  miss_pct = (misses / ticks * 100.0) if ticks else 0.0

  dts = [(t[i] - t[i - 1]) * 1e3 for i in range(1, n)]  # inter-sample ms
  if dts:
    dts_sorted = sorted(dts)
    p99 = dts_sorted[min(len(dts_sorted) - 1, int(0.99 * len(dts_sorted)))]
    dt_stats = {
      "median": statistics.median(dts),
      "max": max(dts),
      "p99": p99,
    }
  else:
    dt_stats = {"median": float("nan"), "max": float("nan"), "p99": float("nan")}

  return {
    "target_hz": target_hz,
    "achieved_hz": achieved_hz,
    "n": n,
    "misses": misses,
    "miss_pct": miss_pct,
    "read_errors": read_errors,
    "last_error": last_error,
    "dt_median_ms": dt_stats["median"],
    "dt_p99_ms": dt_stats["p99"],
    "dt_max_ms": dt_stats["max"],
  }


def _parse_rates(s):
  out = []
  for tok in s.split(","):
    tok = tok.strip()
    if not tok:
      continue
    v = float(tok)
    if v <= 0:
      raise argparse.ArgumentTypeError(f"rate must be > 0: {tok}")
    out.append(v)
  if not out:
    raise argparse.ArgumentTypeError("no rates given")
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--power-node", default=POWER_NODE,
                  help="sysfs power node (mW) for the POM_5V_IN rail")
  ap.add_argument("--rates", type=_parse_rates, default=DEFAULT_RATES,
                  help="comma-separated target sample rates (Hz)")
  ap.add_argument("--dur-s", type=float, default=5.0,
                  help="seconds to poll at each rate")
  ap.add_argument("--miss-tol-pct", type=float, default=1.0,
                  help="max miss%% considered sustainable")
  args = ap.parse_args()

  node = Path(args.power_node)
  try:
    probe_mw = read_power_mw(node)
    print(f"[power] {node} -> {probe_mw:.1f} mW", flush=True)
  except (OSError, ValueError) as e:
    raise SystemExit(
      f"cannot read power node {node}: {e}\n"
      f"check the path, or run with sudo if it needs root."
    )

  print(f"[probe] {args.dur_s}s per rate, "
        f"{len(args.rates)} rates\n", flush=True)
  print(f"{'tgtHz':>7} {'gotHz':>8} {'n':>7} {'miss':>7} {'miss%':>7} "
        f"{'rderr':>6} {'dtMed':>7} {'dtP99':>7} {'dtMax':>8}", flush=True)

  results = []
  for hz in sorted(args.rates):
    r = probe(node, hz, args.dur_s)
    results.append(r)
    print(f"{hz:>7.0f} {r['achieved_hz']:>8.1f} {r['n']:>7d} "
          f"{r['misses']:>7d} {r['miss_pct']:>7.2f} {r['read_errors']:>6d} "
          f"{r['dt_median_ms']:>7.2f} {r['dt_p99_ms']:>7.2f} "
          f"{r['dt_max_ms']:>8.2f}", flush=True)

  # Highest target rate that stayed within the miss tolerance and had no
  # read errors -> a safe sustainable SAMPLE_HZ.
  ok = [r for r in results
        if r["miss_pct"] <= args.miss_tol_pct and r["read_errors"] == 0]
  print("", flush=True)
  if ok:
    best = max(ok, key=lambda r: r["target_hz"])
    print(f"sustainable up to ~{best['target_hz']:.0f} Hz "
          f"(achieved {best['achieved_hz']:.1f} Hz, "
          f"miss {best['miss_pct']:.2f}% <= {args.miss_tol_pct}%)", flush=True)
    print(f"recommend SAMPLE_HZ <= {best['target_hz']:.0f}", flush=True)
  else:
    print(f"no tested rate stayed within {args.miss_tol_pct}% misses; "
          f"try lower --rates", flush=True)


if __name__ == "__main__":
  main()
