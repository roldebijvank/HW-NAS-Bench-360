"""Returns a step() callable for each framework (litert, onnx, torchmobile)."""


_ONNX_PROVIDERS_LOGGED = False


def make_step_litert(path, x_np):
  try:
    from tflite_runtime.interpreter import Interpreter
  except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
  interp = Interpreter(model_path=str(path), num_threads=1)
  interp.allocate_tensors()
  inp = interp.get_input_details()[0]
  out = interp.get_output_details()[0]
  def step():
    interp.set_tensor(inp["index"], x_np)
    interp.invoke()
    interp.get_tensor(out["index"])
  return step


def make_step_onnx(path, x_np, providers=("CPUExecutionProvider",),
                   graph_optimization_level=None, log_providers=False):
  import onnxruntime as ort
  global _ONNX_PROVIDERS_LOGGED
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1
  so.inter_op_num_threads = 1
  if graph_optimization_level is not None:
    so.graph_optimization_level = graph_optimization_level
  sess = ort.InferenceSession(str(path), sess_options=so, providers=list(providers))
  if log_providers and not _ONNX_PROVIDERS_LOGGED:
    active = sess.get_providers()
    on_gpu = "CUDAExecutionProvider" in active
    print(f"ONNX providers: {active}  GPU: {'YES' if on_gpu else 'NO'}", flush=True)
    _ONNX_PROVIDERS_LOGGED = True
  name = sess.get_inputs()[0].name
  def step(): sess.run(None, {name: x_np})
  return step


def make_step_torchmobile(path, x_np):
  import torch
  torch.set_num_threads(1)
  m = torch.jit.load(str(path))
  m.eval()
  x = torch.from_numpy(x_np)
  def step():
    with torch.no_grad():
      m(x)
  return step


MAKERS = {
  "litert":      make_step_litert,
  "onnx":        make_step_onnx,
  "torchmobile": make_step_torchmobile,
}
