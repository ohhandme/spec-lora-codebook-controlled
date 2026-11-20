"""Direct rank-channel modulation of a frozen, unmerged standard LoRA."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math

import torch
from torch import nn
from draft_interventions import LoRALinear
from h_vq64_codec import CausalCodec, ConditionedVQ64
from message_codecs import PCAInt8, MLPInt8

METHODS = ("static", "constant", "local_only", "full_h", "pca_int8", "mlp_int8", "grouped_vq8", "shuffle_h")
COMPRESSED = ("pca_int8", "mlp_int8", "grouped_vq8", "shuffle_h")
TIMELINE = "v1:retained-kv/accepted-old/last-draft-old/correction-or-bonus-new/terminal-no-forward"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(json.dumps((name, str(value.dtype), list(value.shape))).encode() + b"\0")
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class RankGates:
    """The hook is on the original A projection; B, bias, scaling and casts run natively."""
    def __init__(self, model, paths):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("freeze the Draft and original adapters and select eval mode")
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("empty or duplicate LoRA selection")
        self.current = ContextVar("rank_channels", default=None)
        self.schema = []
        self.handles = []
        self.width = 0
        entries = []
        for path in paths:
            layer = model.get_submodule(path)
            if type(layer) is not LoRALinear or not layer.enabled or any(type(m) is not nn.Linear for m in
                    (layer.base, layer.down, layer.up)) or getattr(layer, "merged", False):
                raise ValueError("only original enabled, unmerged standard Linear LoRALinear is supported")
            r = layer.down.out_features
            if r < 1 or r != layer.up.in_features or not math.isfinite(layer.scale):
                raise ValueError("invalid LoRA rank or scale")
            a, b = self.width, self.width + r
            self.schema.append(dict(path=path, rank=r, scale=layer.scale, start=a, stop=b,
                                    adapter_dtype=str(layer.down.weight.dtype)))
            entries.append((layer.down, a, b))
            self.width = b
        for down, a, b in entries:
            self.handles.append(down.register_forward_hook(self._hook(a, b)))

    def _hook(self, a, b):
        def apply(module, args, result):
            value = self.current.get()
            if value is not None:
                if value.ndim == 3 and value.shape[:-1] != result.shape[:-1]:
                    raise ValueError("per-token gate time axis differs from consumed token axis")
                return result * (1 + value[..., a:b].to(result))
        return apply

    @contextmanager
    def use(self, gate):
        if gate is not None and (gate.ndim not in (2, 3) or (gate.ndim == 2 and gate.shape[0] != 1) or
                gate.shape[-1] != self.width or not gate.is_floating_point() or not gate.isfinite().all()):
            raise ValueError("invalid gate layout or value")
        token = self.current.set(gate)
        try:
            yield
        finally:
            self.current.reset(token)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class Continuous(CausalCodec):
    def __init__(self, h, d, r, local):
        super().__init__()
        self.target_hidden_size, self.draft_hidden_size, self.conditioned = h, d, True
        self.local = local
        self.decoder = nn.Linear(2*d+1 if local else h+2*d+1, r)


class Controller(nn.Module):
    def __init__(self, method, h, d, schema, eps=.5):
        super().__init__()
        if method not in METHODS or not 0 < eps <= 1:
            raise ValueError("unsupported method or gate bound")
        self.method, self.h, self.d, self.eps = method, h, d, eps
        self.schema = [dict(row) for row in schema]
        self.r = schema[-1]["stop"]
        if method in ("pca_int8", "mlp_int8", "shuffle_h"):
            factory = PCAInt8 if method == "pca_int8" else MLPInt8
            self.codec = factory(h, d, self.r)
        elif method == "grouped_vq8":
            self.codec = ConditionedVQ64(target_hidden_size=h, draft_hidden_size=d, rank_width=self.r)
        elif method in ("full_h", "local_only"):
            self.codec = Continuous(h, d, self.r, method == "local_only")
        elif method == "constant":
            self.bias = nn.Parameter(torch.zeros(1, self.r))
        self.register_buffer("zero", torch.zeros(1, self.r))
        if hasattr(self, "codec") and hasattr(self.codec, "decoder"):
            nn.init.zeros_(self.codec.decoder.weight)
            nn.init.zeros_(self.codec.decoder.bias)

    def bound(self, u):
        return self.eps * u.tanh()

    def encode(self, inputs):
        if self.method in COMPRESSED:
            return self.codec.encode_payload(*inputs)
        if self.method == "full_h":
            x = self.codec._inputs(*inputs).float().cpu().contiguous()
            return x.numpy().astype("<f4", copy=False).tobytes()
        return b""

    def local_x(self, protocol):
        count, rejected, correction = protocol
        # Known protocol fields only; H never reaches the local network.
        dummy = self.zero.new_zeros(1, self.h)
        return self.codec._inputs(dummy, count, rejected, correction)[:, self.h:]

    def decode(self, payload, protocol):
        if payload is None:
            return self.zero
        if type(payload) is not bytes:
            raise TypeError("expected an immutable byte payload")
        if self.method in COMPRESSED:
            u = self.codec.decode_payload(payload)
        elif self.method == "full_h":
            if len(payload) != 4*(self.h+2*self.d+1):
                raise ValueError("incorrect full input length")
            import numpy as np
            x = torch.from_numpy(np.frombuffer(payload, dtype="<f4").copy()).reshape(1,-1).to(self.zero)
            if not x.isfinite().all():
                raise ValueError("nonfinite full input")
            u = self.codec.decoder(x)
        else:
            if payload:
                raise ValueError("local controls take zero wire bytes")
            if self.method == "static":
                return self.zero
            u = self.bias if self.method == "constant" else self.codec.decoder(self.local_x(protocol))
        return self.bound(u)

    def train_gate(self, inputs):
        if self.method in COMPRESSED:
            u = self.codec.train_control(*inputs)
        elif self.method == "full_h":
            u = self.codec.decoder(self.codec._inputs(*inputs))
        else:
            return self.decode(b"", inputs[1:])
        return self.bound(u)

    def identity(self, model_hash, adapter_hash):
        return dict(version=1, timeline=TIMELINE, method=self.method, target_width=self.h,
                    draft_width=self.d, gate_width=self.r, epsilon=self.eps, schema=self.schema,
                    dtype=str(self.zero.dtype), base_sha256=model_hash, adapter_sha256=adapter_hash,
                    controller_sha256=tensor_digest(self.state_dict()),
                    feedback_bytes=8 if self.method in COMPRESSED else 4*(self.h+2*self.d+1) if self.method=="full_h" else 0,
                    vq_padding=(8*((self.r+7)//8)-self.r) if self.method=="grouped_vq8" else 0)


def restore(package, method, h, d, schema, model_hash, adapter_hash, dtype=torch.float32):
    identity = package["identity"]
    ctrl = Controller(method, h, d, schema, identity["epsilon"]).to(dtype)
    ctrl.load_state_dict(package["state"], strict=True)
    if ctrl.identity(model_hash, adapter_hash) != identity:
        raise ValueError("LoRA, schema, method, dtype or controller binding differs")
    return ctrl
