from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
hw_repo = project_root / "data" / "hw-nas-bench"
if not hw_repo.exists():
    raise FileNotFoundError(f"hw-nas-bench data not found at {hw_repo}")
if str(hw_repo) not in sys.path:
    sys.path.insert(0, str(hw_repo))

import torch.nn as nn
from hw_nas_bench_api import HWNASBenchAPI as HW_API
from hw_nas_bench_api.nas_201_models import get_cell_based_tiny_net


hw_api = HW_API(str(hw_repo / "HW-NAS-Bench-v1_0.pickle"), search_space="nasbench201")

def build_model(arch_idx, input_shape, num_classes):
  config = hw_api.get_net_config(arch_idx, "cifar100")
  config['num_classes'] = num_classes                 # type: ignore
  net = get_cell_based_tiny_net(config).eval()
  in_channels = input_shape[0]
  if in_channels != 3:
    old = net.stem[0]
    net.stem[0] = nn.Conv2d(in_channels, old.out_channels,
                            kernel_size=old.kernel_size, stride=old.stride,
                            padding=old.padding, bias=old.bias is not None)
  return net