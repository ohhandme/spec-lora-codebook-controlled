"""Observed evaluator settings used for controlled result comparisons."""
import hashlib
import json
import platform

import torch
import transformers


def model_environment(model):
    layout = []
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        layout.append((name, str(tensor.dtype), str(tensor.device)))
    return {
        "attention": model.config._attn_implementation,
        "embedding_dtype": str(model.get_input_embeddings().weight.dtype),
        "parameter_and_buffer_dtypes": sorted({dtype for _, dtype, _ in layout}),
        "devices": sorted({device for _, _, device in layout}),
        "placement_dtype_sha256": hashlib.sha256(json.dumps(sorted(layout)).encode()).hexdigest(),
    }


def execution_environment(target, draft):
    models = {"target": model_environment(target.model), "draft": model_environment(draft)}
    devices = sorted({device for m in models.values() for device in m["devices"]})
    hardware = {}
    for device in map(torch.device, devices):
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            hardware[str(device)] = {"name": props.name, "capability": [props.major, props.minor],
                                     "multiprocessors": props.multi_processor_count}
        elif device.type == "cpu":
            hardware[str(device)] = {"processor": platform.processor(), "isa": torch.backends.cpu.get_cpu_capability()}
        else:
            raise ValueError(f"Unsupported device for controlled comparison: {device}")
    cuda = torch.backends.cuda
    return {
        "schema": 1, "models": models, "hardware": hardware,
        "python": platform.python_version(), "platform": [platform.system(), platform.machine(), platform.release()],
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "cuda_build": torch.version.cuda, "hip_build": torch.version.hip, "cudnn": torch.backends.cudnn.version(),
        "threads": [torch.get_num_threads(), torch.get_num_interop_threads()],
        "default_dtype": str(torch.get_default_dtype()), "matmul_precision": torch.get_float32_matmul_precision(),
        "autocast": {d: {"enabled": torch.is_autocast_enabled(d), "dtype": str(torch.get_autocast_dtype(d))}
                     for d in sorted({torch.device(name).type for name in devices})},
        "deterministic": torch.are_deterministic_algorithms_enabled(), "mkldnn": torch.backends.mkldnn.enabled,
        "cuda_matmul_tf32": cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "sdpa": {"flash": cuda.flash_sdp_enabled(), "math": cuda.math_sdp_enabled(),
                 "efficient": cuda.mem_efficient_sdp_enabled(), "cudnn": cuda.cudnn_sdp_enabled()},
    }
