"""Per-arch export to TFLite, ONNX, TorchScript-mobile at native task shape."""
from pathlib import Path
import gc
import torch, torch.nn as nn
import litert_torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from scripts.utils.model_utils import build_model


class LogitsOnly(nn.Module):
  def __init__(self, m): super().__init__(); self.m = m
  def forward(self, x):
    out = self.m(x)
    return out[1] if isinstance(out, (tuple, list)) and len(out) == 2 else out


def _wrap(arch_idx, input_shape, num_classes):
  net = build_model(arch_idx, input_shape, num_classes).eval()
  return LogitsOnly(net).eval(), torch.randn(1, *input_shape)


def export_tflite(arch_idx, input_shape, num_classes, out_path):
  net, sample = _wrap(arch_idx, input_shape, num_classes)
  edge = litert_torch.convert(net, (sample,))
  edge.export(str(out_path))
  size = out_path.stat().st_size
  del net, sample, edge
  gc.collect()
  return size


def export_onnx(arch_idx, input_shape, num_classes, out_path):
  net, sample = _wrap(arch_idx, input_shape, num_classes)
  torch.onnx.export(net, sample, str(out_path),
                    input_names=["input"], output_names=["logits"], opset_version=17,
                    dynamo=False)
  size = out_path.stat().st_size
  del net, sample
  gc.collect()
  return size


def export_torchmobile(arch_idx, input_shape, num_classes, out_path):
  net, sample = _wrap(arch_idx, input_shape, num_classes)
  traced = torch.jit.trace(net, sample)
  optimized = optimize_for_mobile(traced)
  optimized._save_for_lite_interpreter(str(out_path))
  size = out_path.stat().st_size
  del net, sample, traced, optimized
  gc.collect()
  return size


RUNTIMES = {
  "litert":      ("tflite", export_tflite),
  "onnx":        ("onnx",   export_onnx),
  "torchmobile": ("ptl",    export_torchmobile),
}
