"""Task76 message variants. The receiver always supplies the frozen head's 128 controls."""
import torch
from torch import nn
from torch.nn import functional as F
from h_vq64_codec import CausalCodec
from raw_data import require


ARMS = {7: "T_STAGE_RAND", 8: "T_STAGE_DATA", 9: "L_MARGIN",
        14: "G16_K16", 15: "N_NEAREST", 22: "M_MIX"}


class CachedMessage(CausalCodec):
    def __init__(self, arm, original):
        super().__init__()
        self.arm = arm
        self.target_hidden_size = original.target_hidden_size
        self.draft_hidden_size = original.draft_hidden_size
        self.conditioned = True
        self.groups, self.choices = (16, 16) if arm == 14 else (8, 256)
        self.dim = 128 // self.groups
        self.stages = 2 if arm == 22 else 1
        width = self.target_hidden_size + 2*self.draft_hidden_size + 1
        if arm in (7, 8, 9):
            self.encoder = original.encoder
            self.book = nn.Parameter(original.codebook.detach().unsqueeze(0))
        else:
            outputs = 128 if arm == 15 else self.stages*self.groups*self.choices
            self.encoder = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, outputs))
            self.book = nn.Parameter(torch.randn(self.stages, self.groups, self.choices, self.dim)*.02)
        if arm in (7, 8):
            self.reference = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 128))

    def layout(self, budget):
        if self.arm == 22 and budget in (4, 8, 16):
            return (2 if budget == 16 else 1, 16 if budget == 4 else 256)
        require(budget == 8, "unsupported budget")
        return 1, self.choices

    def encode(self, x, budget):
        stages, choices = self.layout(budget)
        z = self.encoder(x)
        tables = self.book[:stages, :, :choices]
        if self.arm == 15:
            z = z.reshape(-1, self.groups, self.dim)
            distance = z.square().sum(-1, keepdim=True) + tables[0].square().sum(-1)
            distance = distance - 2*torch.einsum("bgd,gkd->bgk", z, tables[0])
            scores = -distance[:, None]
        else:
            scores = z.reshape(-1, self.stages, self.groups, self.choices)[:, :stages, :, :choices]
        return z, tables, scores

    def control(self, x, budget=8):
        z, tables, scores = self.encode(x, budget)
        ids = scores.argmax(-1)
        if self.arm == 15:
            groups = torch.arange(self.groups, device=x.device)
            hard = tables[0, groups, ids[:, 0]]
            penalty = .25*(z-hard.detach()).square().mean() + (hard-z.detach()).square().mean()
            return (hard.detach() + (z-z.detach())).flatten(1), penalty
        soft = scores.softmax(-1)
        weights = F.one_hot(ids, scores.size(-1)).to(soft) + (soft-soft.detach())
        return torch.einsum("bsgk,sgkd->bgd", weights, tables).flatten(1), 0.

    @torch.no_grad()
    def initialize_books(self, x):
        if self.arm != 8:
            return
        points = self.reference(x).reshape(-1, self.groups, self.dim)
        gen = torch.Generator().manual_seed(42)
        for group, pts in enumerate(points.unbind(1)):
            centers = pts[torch.randint(len(pts), (self.choices,), generator=gen)].clone()
            for _ in range(8):
                indices = torch.cdist(pts, centers).argmin(1)
                sums = torch.zeros_like(centers).index_add_(0, indices, pts)
                counts = indices.bincount(minlength=self.choices)[:, None]
                centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
            self.book[0, group].copy_(centers)

    def bootstrap(self, x):
        with torch.no_grad():
            z = self.reference(x).reshape(-1, self.groups, self.dim)
            labels = (z[:, :, None]-self.book[0]).square().sum(-1).argmin(-1)
        return F.cross_entropy(self.encoder(x).reshape(-1, self.choices), labels.flatten())

    @torch.no_grad()
    def wire(self, x, budget):
        require(len(x) == 1, "one message per request boundary")
        _, _, scores = self.encode(x, budget)
        ids = scores.argmax(-1).flatten().tolist()
        if scores.size(-1) == 16:
            ids = [(hi << 4) | lo for hi, lo in zip(ids[::2], ids[1::2])]
        raw = bytes(ids)
        require(len(raw) == budget, "wire budget mismatch")
        return raw

    def unbox(self, raw, budget):
        stages, choices = self.layout(budget)
        require(type(raw) is bytes and len(raw) == budget, "invalid message length")
        ids = list(raw) if choices == 256 else [i for b in raw for i in (b >> 4, b & 15)]
        ix = torch.tensor(ids, device=self.book.device).reshape(stages, self.groups)
        s = torch.arange(stages, device=ix.device)[:, None]
        g = torch.arange(self.groups, device=ix.device)
        return self.book[s, g, ix].sum(0).reshape(1, 128)

    def encode_payload(self, *inputs):
        return self.wire(self._inputs(*inputs), 8)

    def decode_payload(self, payload):
        return self.book.new_zeros(1, 128) if payload is None else self.unbox(payload, 8)
