"""Incremental Qwen3 decoding with immutable token-to-gate history and an independent full replay."""
import torch
from transformers import DynamicCache


class DraftSession:
    def __init__(self, model, gates, *, replay=False):
        self.model, self.gates, self.replay = model, gates, replay
        self.reset()

    def reset(self):
        self.cache = None
        self.committed = []
        self.tokens, self.versions = [], []
        self.values = {0: None}
        self.version = 0
        self.proposal = []
        self.last = self.last_hidden = None
        self.done = False
        self.calls = []
        self.round_log = []

    def validate(self):
        if len(self.tokens) != len(self.versions) or not set(self.versions).issubset(self.values):
            raise ValueError("corrupt gate history")
        prefix = self.committed[:len(self.tokens)]
        if self.tokens[:len(prefix)] != prefix:
            raise ValueError("consumed prefix differs from committed history")
        if self.replay:
            return
        if self.cache is None:
            if self.tokens:
                raise ValueError("missing cache")
            return
        if len(self.cache.layers) != self.model.config.num_hidden_layers:
            raise ValueError("incomplete KV layers")
        for layer in self.cache.layers:
            if layer.keys is None or layer.values is None or layer.keys.shape != layer.values.shape or layer.keys.shape[-2] != len(self.tokens):
                raise ValueError("invalid layer KV length or shape")

    def _consume(self, tokens, version):
        try:
            self._forward_tokens(tokens, version)
        except Exception:
            # DynamicCache updates individual layers in place. Discard a partially executed forward.
            self.reset()
            raise

    def _forward_tokens(self, tokens, version):
        if not tokens:
            return
        seq = self.tokens + tokens
        versions = self.versions + [version]*len(tokens)
        device = self.model.get_input_embeddings().weight.device
        if self.replay:
            gate = None
            if any(self.values[v] is not None for v in versions):
                zero = torch.zeros(1, self.gates.width, device=device)
                gate = torch.stack([self.values[v] if self.values[v] is not None else zero for v in versions], dim=1)
            ids = torch.tensor([seq], device=device)
            with self.gates.use(gate):
                out = self.model(input_ids=ids, use_cache=False, output_hidden_states=True)
        else:
            gate = self.values[version]
            ids = torch.tensor([tokens], device=device)
            if self.cache is None:
                self.cache = DynamicCache(config=self.model.config)
            with self.gates.use(gate):
                out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True, output_hidden_states=True)
            self.cache = out.past_key_values
        self.tokens, self.versions = seq, versions
        self.last = out.logits[:, -1]
        self.last_hidden = out.hidden_states[-1][:, -1]
        self.calls.append(dict(tokens=tokens.copy(), gate=version, actual_input=ids.shape[1]))
        self.validate()

    def start(self, prompt):
        if self.committed or not prompt:
            raise ValueError("reset before starting a new request")
        self.committed = list(prompt)
        self._consume(list(prompt), 0)

    def propose(self, n, eos):
        self.validate()
        if self.proposal or self.done or self.tokens != self.committed or not 1 <= n <= 8:
            raise ValueError("unexpected proposal boundary")
        for i in range(n):
            token = int(self.last.argmax(-1).item())
            self.proposal.append(token)
            if token == eos or i == n-1:
                break
            self._consume([token], self.version)
        return list(self.proposal)

    def finish(self, reply, gate_fn):
        self.validate()
        p = self.proposal
        k, answer, rejected, done = [reply[x] for x in ("accepted","answer","rejected","done")]
        if not p or type(k) != int or not 0 <= k <= len(p) or type(done) is not bool:
            raise ValueError("invalid decision")
        if rejected != (p[k] if k < len(p) else None) or (answer is None and (k<len(p) or not done)):
            raise ValueError("invalid rejection or correction")
        if answer is not None and (type(answer) is not int or not 0<=answer<self.model.config.vocab_size or answer==rejected):
            raise ValueError("invalid authoritative token")
        old = self.version
        nxt = None if done else gate_fn()
        if nxt is not None and (nxt.shape != (1,self.gates.width) or not nxt.is_floating_point() or not nxt.isfinite().all()):
            raise ValueError("invalid decoded gate")
        before = len(self.calls)
        keep = len(self.committed) + k
        consumed = min(keep, len(self.tokens))
        if len(self.tokens) > consumed:
            if not self.replay:
                self.cache.crop(consumed)
            self.tokens = self.tokens[:consumed]
            self.versions = self.versions[:consumed]
        self.committed += p[:k] + ([] if answer is None else [answer])
        self.proposal = []
        if not done:
            if consumed < keep:
                # The last accepted proposal was selected but has not fed the decoder.
                self._consume(self.committed[consumed:keep], old)
            self.version += 1
            self.values[self.version] = nxt
            self._consume([answer], self.version)
        self.done = done
        if done:
            self.last = self.last_hidden = None
        self.validate()
        self.round_log.append(dict(old=old, new=None if done else self.version, accepted=k, terminal=done,
            consumed=len(self.tokens), committed=len(self.committed), pending=self.committed[len(self.tokens):],
            boundary_forwards=self.calls[before:]))
