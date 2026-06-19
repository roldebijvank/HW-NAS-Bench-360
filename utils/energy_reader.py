"""EnergyReader base class with Pi5 (PMIC) and Jetson (INA3221) subclasses; stop() returns mJ."""
import os
import re
import subprocess
import threading
import time
from pathlib import Path


PMIC_LINE_RE = re.compile(
  r"(?P<label>[A-Za-z0-9_/-]+)\s*[:=]?\s*(?P<value>[-+]?\d*\.?\d+)\s*"
  r"(?P<unit>mV|V|mA|A|mW|W)\b",
  re.IGNORECASE,
)


def integrate_energy_mj(samples):
  """Trapezoidal integral of power (mW) over time (s) -> mJ."""
  if len(samples) < 2: return None
  energy_mj = 0.0
  for i in range(1, len(samples)):
    t0, p0 = samples[i - 1]
    t1, p1 = samples[i]
    dt = t1 - t0
    if dt <= 0: continue
    energy_mj += (p0 + p1) * 0.5 * dt
  return energy_mj


def _normalize_pmic_label(label):
  s = label.upper()
  for suffix in ("_V", "_I", "_A", "_MA", "_MV", "_UA"):
    if s.endswith(suffix):
      return s[:-len(suffix)]
  return s


def parse_pmic_read_adc(lines):
  if not lines: return None
  rails = {}
  power_mw = 0.0
  has_power = False
  for line in lines:
    for m in PMIC_LINE_RE.finditer(line):
      label = _normalize_pmic_label(m.group("label"))
      val = float(m.group("value"))
      unit = m.group("unit").lower()
      if unit == "v":
        rails.setdefault(label, {})["v"] = val
      elif unit == "mv":
        rails.setdefault(label, {})["v"] = val / 1000.0
      elif unit == "a":
        rails.setdefault(label, {})["i"] = val
      elif unit == "ma":
        rails.setdefault(label, {})["i"] = val / 1000.0
      elif unit == "w":
        power_mw += val * 1000.0
        has_power = True
      elif unit == "mw":
        power_mw += val
        has_power = True
  if not has_power:
    for vals in rails.values():
      if "v" in vals and "i" in vals:
        power_mw += vals["v"] * vals["i"] * 1000.0
  return power_mw if power_mw > 0.0 else None


def pin_thread_to_core(core):
  if core is None or not hasattr(os, "sched_setaffinity"):
    return
  try:
    os.sched_setaffinity(0, {core})
  except OSError:
    pass


class EnergyReader:
  def start(self): raise NotImplementedError
  def stop(self): raise NotImplementedError


class PmicEnergyReader(EnergyReader):
  """Pi 5 PMIC sampler: polls vcgencmd pmic_read_adc at ~100 Hz from a thread."""

  SAMPLE_HZ = 100.0

  def __init__(self, vcgencmd_bin="vcgencmd"):
    self.vcgencmd_bin = vcgencmd_bin
    self.samples = []
    self._stop = threading.Event()
    self._thread = None

  def _read_power_mw(self):
    try:
      out = subprocess.run([self.vcgencmd_bin, "pmic_read_adc"],
                           capture_output=True, text=True, timeout=2).stdout
      return parse_pmic_read_adc(out.splitlines())
    except (FileNotFoundError, subprocess.TimeoutExpired):
      return None

  def start(self):
    self.samples = []
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def _run(self):
    interval = 1.0 / self.SAMPLE_HZ
    deadline = time.perf_counter()
    while not self._stop.is_set():
      p = self._read_power_mw()
      if p is not None:
        self.samples.append((time.perf_counter(), p))
      deadline += interval
      remaining = deadline - time.perf_counter()
      if remaining > 0:
        time.sleep(remaining)
      else:
        deadline = time.perf_counter()

  def stop(self):
    self._stop.set()
    if self._thread:
      self._thread.join(timeout=2)
    return integrate_energy_mj(self.samples)


class Ina3221EnergyReader(EnergyReader):
  """Jetson INA3221 sysfs poller (mW). samples = [(t_perf_s, power_mw), ...]."""

  def __init__(self, power_node, sample_hz=550.0, core=None):
    self.power_node = Path(power_node)
    self.interval = 1.0 / sample_hz
    self.core = core
    self.samples = []
    self._stop = threading.Event()
    self._thread = None

  def start(self):
    self.samples = []
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def _run(self):
    pin_thread_to_core(self.core)
    deadline = time.perf_counter()
    with open(self.power_node) as fh:
      while not self._stop.is_set():
        try:
          fh.seek(0)
          p = float(fh.read().strip())
          self.samples.append((time.perf_counter(), p))
        except (OSError, ValueError):
          pass
        deadline += self.interval
        remaining = deadline - time.perf_counter()
        if remaining > 0:
          time.sleep(remaining)
        else:
          deadline = time.perf_counter()

  def stop(self):
    self._stop.set()
    if self._thread:
      self._thread.join(timeout=2)
    return integrate_energy_mj(self.samples)
