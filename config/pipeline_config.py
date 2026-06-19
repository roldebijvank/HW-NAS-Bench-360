"""Central pipeline config: tasks, frameworks, devices.

Importable from host (export, accuracy join) and from device bench scripts
(Pi/Jetson/Pixel).
"""
from pathlib import Path

from utils.energy_reader import PmicEnergyReader, Ina3221EnergyReader
from utils.measure import vcgencmd_temp, thermal_zone_max_temp

ROOT = Path(__file__).resolve().parent.parent


TASKS = {
  "cifar100": {
    "input_shape": (3, 32, 32),
    "num_classes": 100,
    "acc_source": "nb201_api",
    "nb201_dataset": "cifar100",
  },
  "ninapro": {
    "input_shape": (1, 52, 16),
    "num_classes": 18,
    "acc_source": "nb360_pickle",
    "acc_pickle": ROOT / "data" / "nb360_ninapro" / "NATS-tss-v1_0-daa55.pickle",
    "acc_inner_key": ("ninapro", 777),
  },
  "darcy": {
    "input_shape": (1, 88, 88),
    "num_classes": 1,
    "acc_source": "nb360_pickle",
    "acc_pickle": ROOT / "data" / "nb360_darcyflow" / "NATS-tss-v1_0-48858.pickle",
    "acc_inner_key": ("darcyflow", 777),
  },
}


FRAMEWORKS = {
  "litert":      {"ext": "tflite", "exporter": "export_tflite"},
  "onnx":        {"ext": "onnx",   "exporter": "export_onnx"},
  "torchmobile": {"ext": "ptl",    "exporter": "export_torchmobile"},
}


DEVICES = {
  "pi5": {
    "frameworks": ("litert", "onnx", "torchmobile"),
    "energy_reader": PmicEnergyReader,
    "temp_reader": vcgencmd_temp,
  },
  "jetson": {
    "frameworks": ("onnx",),
    "energy_rail_path": "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input",
    "sampler_hz": 550.0,
    "energy_reader": Ina3221EnergyReader,
    "temp_reader": thermal_zone_max_temp,
  },
  "pixel": {
    "frameworks": ("litert", "onnx", "torchmobile"),
  },
}
