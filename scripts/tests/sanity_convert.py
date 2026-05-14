"""Host-side: convert NB201 arch 0 at CIFAR shape to TFLite, ONNX, TorchScript-mobile.
Goal: reproduce HW-NAS-Bench raspi4_latency for arch 0 (~10.48 ms) on the RPi.
Outputs land in artifacts/sanity/ for rsync to the device.
"""
from pathlib import Path
import torch, torch.nn as nn
import litert_torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from scripts.utils.model_utils import build_model


ARCH_IDX = 0
INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 10

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "sanity"
OUT.mkdir(parents=True, exist_ok=True)

net = build_model(ARCH_IDX, INPUT_SHAPE, NUM_CLASSES).eval()
sample = torch.randn(1, *INPUT_SHAPE)

# Some NB201 nets return (logits, features); wrap to logits-only for clean export
class LogitsOnly(nn.Module):
  def __init__(self, m): super().__init__(); self.m = m
  def forward(self, x):
    out = self.m(x)
    return out[1] if isinstance(out, (tuple, list)) and len(out) == 2 else out

net_w = LogitsOnly(net).eval()

# TFLite (FP32) via litert-torch
tflite_path = OUT / f"arch{ARCH_IDX}_cifar.tflite"
edge = litert_torch.convert(net_w, (sample,))
edge.export(str(tflite_path))
print(f"tflite: {tflite_path} {tflite_path.stat().st_size} bytes")

# ONNX
onnx_path = OUT / f"arch{ARCH_IDX}_cifar.onnx"
torch.onnx.export(net_w, sample, str(onnx_path), input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
print(f"onnx:   {onnx_path} {onnx_path.stat().st_size} bytes")

# TorchScript-mobile
ptl_path = OUT / f"arch{ARCH_IDX}_cifar.ptl"
traced = torch.jit.trace(net_w, sample)
optimize_for_mobile(traced)._save_for_lite_interpreter(str(ptl_path))
print(f"ptl:    {ptl_path} {ptl_path.stat().st_size} bytes")
