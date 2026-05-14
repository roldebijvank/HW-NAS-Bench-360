"""Host-side driver: for each non-iso arch_idx, export to 3 runtimes x N tasks,
scp to Pi queue, drop sentinel, wait for done, delete local artifacts.

Requires:
  - RPI_HOST env var (e.g. pi@raspberrypi.local)
  - pi_daemon.py running on Pi at ~/bep/scripts/pi_daemon.py
  - Pi paths: ~/bep/queue/{incoming,done}/, ~/bep/results/latency_rpi.csv
"""
from pathlib import Path
import os, sys, json, shutil, subprocess, time, argparse, traceback
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.utils.task_specs import TASKS
from scripts.utils.convert_utils import RUNTIMES
from scripts.utils.arch_iter import non_iso_indices

RPI_HOST  = os.environ.get("RPI_HOST")
RPI_BASE  = "~/bep"
INC_REM   = f"{RPI_BASE}/queue/incoming"
DONE_REM  = f"{RPI_BASE}/queue/done"

LOCAL_STAGE = ROOT / "artifacts" / "exports"
LOCAL_STAGE.mkdir(parents=True, exist_ok=True)


def ssh(cmd):
  return subprocess.run(["ssh", RPI_HOST, cmd], check=True, capture_output=True, text=True)


def scp_dir(local_dir, remote_path):
  subprocess.run(["scp", "-rq", str(local_dir), f"{RPI_HOST}:{remote_path}"], check=True)


def remote_exists(path):
  r = subprocess.run(["ssh", RPI_HOST, f"test -e {path}"], capture_output=True)
  return r.returncode == 0


def remote_rm(path):
  subprocess.run(["ssh", RPI_HOST, f"rm -f {path}"], check=False)


def export_arch(arch_idx):
  """Build artifacts/exports/arch_<N>/ with meta.json + 3xN files."""
  d = LOCAL_STAGE / f"arch_{arch_idx}"
  if d.exists(): shutil.rmtree(d)
  d.mkdir(parents=True)
  meta = {"arch_idx": arch_idx, "tasks": {}}
  
  for task, spec in TASKS.items():
    meta["tasks"][task] = {"input_shape": list(spec["input_shape"])}
    for runtime, (ext, fn) in RUNTIMES.items():
      out = d / f"{task}_{runtime}.{ext}"
      fn(arch_idx, spec["input_shape"], spec["num_classes"], out)
  (d / "meta.json").write_text(json.dumps(meta))
  
  return d


def push(local_dir, arch_idx):
  scp_dir(local_dir, f"{INC_REM}/")
  ssh(f"touch {INC_REM}/arch_{arch_idx}.ready")


def wait_done(arch_idx, timeout=600, poll=2.0):
  marker = f"{DONE_REM}/arch_{arch_idx}.done"
  t0 = time.time()
  while time.time() - t0 < timeout:
    if remote_exists(marker):
      remote_rm(marker); return True
    time.sleep(poll)
  return False


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--limit", type=int, default=None)
  ap.add_argument("--start", type=int, default=0)
  ap.add_argument("--all", action="store_true", help="iterate 0..15624 instead of non-iso")
  args = ap.parse_args()

  if not RPI_HOST:
    print("set RPI_HOST=user@host"); sys.exit(2)

  ssh(f"mkdir -p {INC_REM} {DONE_REM} {RPI_BASE}/results")

  if args.all:
    indices = list(range(15625))
  else:
    print("computing non-iso reps...", flush=True)
    indices = non_iso_indices()
    print(f"  {len(indices)} reps", flush=True)

  indices = indices[args.start:]
  if args.limit: indices = indices[:args.limit]

  for i, arch_idx in enumerate(indices):
    t0 = time.time()
    try:
      d = export_arch(arch_idx)
      push(d, arch_idx)
      ok = wait_done(arch_idx)
      shutil.rmtree(d, ignore_errors=True)
      dt = time.time() - t0
      print(f"[{i+1}/{len(indices)}] arch {arch_idx}: {'done' if ok else 'TIMEOUT'} ({dt:.1f}s)", flush=True)
    except Exception as e:
      print(f"[{i+1}/{len(indices)}] arch {arch_idx}: FAIL {e}", flush=True)
      traceback.print_exc()


if __name__ == "__main__":
  main()
