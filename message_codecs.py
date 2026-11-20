"""Eight-byte causal feedback codecs; endpoints agree on schema and exact state."""
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from h_vq64_codec import CausalCodec, ConditionedVQ64, EDGE_CONTROL_DIM, WIRE_BYTES

METHODS = ("pca_int8", "mlp_int8", "grouped_vq8")


class SignedCodec(CausalCodec):
    def __init__(self, target_hidden_size, draft_hidden_size, rank_width):
        super().__init__()
        if min(target_hidden_size, draft_hidden_size) < 1:
            raise ValueError("hidden sizes must be positive")
        self.target_hidden_size = target_hidden_size
        self.draft_hidden_size = draft_hidden_size
        self.conditioned = True
        self.width = target_hidden_size + 2 * draft_hidden_size + 1
        self.decoder = nn.Linear(WIRE_BYTES, rank_width)
        self.register_buffer("scale", torch.full((WIRE_BYTES,), 1 / 127))

    def _apply(self, fn, recurse=True):
        # The wire uses the same pre-shared FP32 scales when model weights change precision.
        scales = self.scale
        super()._apply(fn, recurse)
        self.scale = scales.to(device=self.scale.device, dtype=torch.float32)
        return self

    def coordinates(self, *inputs):
        return self.project(self._inputs(*inputs))

    def normalized(self, z):
        if not torch.isfinite(z).all() or not torch.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError("invalid coordinates or shared scale")
        return z.float() / self.scale.float()

    def pack(self, z):
        q = self.normalized(z).round().clamp(-127, 127).to(torch.int8)
        return q.reshape(WIRE_BYTES).cpu().contiguous().numpy().tobytes()

    @torch.no_grad()
    def encode_payload(self, *inputs):
        return self.pack(self.coordinates(*inputs))

    def decode_payload(self, payload):
        weight = self.decoder.weight
        if payload is None:
            return weight.new_zeros(1, self.decoder.out_features)
        if type(payload) is not bytes:
            raise TypeError("expected raw bytes")
        if len(payload) != WIRE_BYTES or 128 in payload:
            raise ValueError("invalid symmetric int8 payload")
        signed = [v if v < 128 else v - 256 for v in payload]
        q = weight.new_tensor(signed).reshape(1, WIRE_BYTES)
        return self.decoder((q.float() * self.scale.float()).to(weight.dtype))

    def train_control(self, *inputs, temperature=1.0):
        if not 0 < temperature <= 10:
            raise ValueError("invalid temperature")
        u = self.normalized(self.coordinates(*inputs)).clamp(-127, 127)
        q = u.detach().round() + (u - u.detach())
        return self.decoder((q * self.scale.float()).to(self.decoder.weight.dtype))


class PCAInt8(SignedCodec):
    codec_type = "pca_int8"

    def __init__(self, target_hidden_size, draft_hidden_size, rank_width):
        super().__init__(target_hidden_size, draft_hidden_size, rank_width)
        self.register_buffer("mean", torch.zeros(self.width))
        self.register_buffer("projection", torch.zeros(8, self.width))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, records):
        """Each record is (split, request_id, causal_inputs); one frozen fit."""
        if self.fitted.item():
            raise ValueError("PCA already fitted")
        rows = []
        for split, request_id, inputs in records:
            if split != "train" or not request_id:
                raise ValueError("PCA requires identified train requests")
            rows.append(self._inputs(*inputs).detach().cpu().double())
        if len(rows) < 9 or self.width < 8:
            raise ValueError("insufficient centered PCA samples or features")
        x = torch.cat(rows)
        mean = x.mean(0)
        centered = x - mean
        _, singular, axes = torch.linalg.svd(centered, full_matrices=False)
        if singular[7] <= singular[0] * max(x.shape) * torch.finfo(x.dtype).eps:
            raise ValueError("insufficient PCA rank")
        axes = axes[:8]
        # Orient each component by its largest absolute loading.
        signs = axes[torch.arange(8), axes.abs().argmax(1)].sign()
        self.mean.copy_(mean)
        self.projection.copy_(axes * signs[:, None])
        stored = (x.to(self.mean) - self.mean) @ self.projection.T
        self.scale.copy_(stored.float().abs().amax(0).div(127).clamp_min(1e-8))
        self.fitted.fill_(True)

    def project(self, x):
        if not self.fitted.item():
            raise ValueError("PCA not fitted")
        return F.linear(x - self.mean, self.projection)


class MLPInt8(SignedCodec):
    codec_type = "mlp_int8"

    def __init__(self, target_hidden_size, draft_hidden_size, rank_width):
        super().__init__(target_hidden_size, draft_hidden_size, rank_width)
        self.encoder = nn.Sequential(nn.Linear(self.width, 128), nn.SiLU(), nn.Linear(128, 8), nn.Tanh())

    def project(self, x):
        return self.encoder(x)


