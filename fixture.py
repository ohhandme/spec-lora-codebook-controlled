"""Deterministic, offline tiny Qwen3 fixtures; original adapters stay frozen and nonzero."""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from draft_interventions import LoRALinear
from gates import RankGates, digest, tensor_digest


def setup(seed=131, *, install_gates=True):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=61, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                      attention_dropout=0., pad_token_id=0, bos_token_id=1, eos_token_id=60)
    cfg._attn_implementation = "eager"
    target = Qwen3ForCausalLM(cfg).requires_grad_(False).eval()
    draft = Qwen3ForCausalLM(cfg)
    chosen = []
    for i, layer in enumerate(draft.model.layers):
        for q, r in (("q_proj",3+i), ("v_proj",5+i)):
            adapter = LoRALinear(getattr(layer.self_attn,q), r, r*2)
            with torch.no_grad():
                adapter.up.weight.normal_(0,.14)
            setattr(layer.self_attn,q,adapter)
            if (i,q) != (1,"q_proj"):
                chosen.append(f"model.layers.{i}.self_attn.{q}")
    draft.requires_grad_(False).eval()
    gates = RankGates(draft, chosen) if install_gates else chosen
    state = draft.state_dict()
    def adaptation(k):
        return ".up." in k or ".down." in k
    hashes = (tensor_digest({k:v for k,v in state.items() if not adaptation(k)}),
              tensor_digest({k:v for k,v in state.items() if adaptation(k)}))
    return target, draft, gates, hashes


@torch.no_grad()
def teacher(target, prefix):
    out = target(torch.tensor([prefix]), use_cache=False, output_hidden_states=True)
    return int(out.logits[0,-1].argmax()), out.hidden_states[-1][:,-1].detach()


def protocol(draft, count, rejected, answer):
    weight = draft.get_input_embeddings().weight
    return count, None if rejected is None else weight[rejected].reshape(1,-1), weight[answer].reshape(1,-1)


def shuffle_row(rows, index):
    row = rows[index]
    # Fixed request derangement within split, accepted count, rejection/bonus and prefix length.
    ids = [i for i, v in enumerate(rows) if (v["split"], v["inputs"][1], v["inputs"][2] is None, len(v["prefix"]))
           == (row["split"], row["inputs"][1], row["inputs"][2] is None, len(row["prefix"]))]
    if len(ids)<2:
        raise ValueError("no independent H donor in this boundary bucket")
    partner = rows[ids[(ids.index(index)+1)%len(ids)]]
    if partner["request"] == row["request"]:
        raise ValueError("invalid same-request donor")
    return (partner["inputs"][0], *row["inputs"][1:]), partner["request"]


def samples(target, draft, split, indices):
    rows = []
    with torch.no_grad():
        for i in indices:
            prefix = [1, 3+i//20, 12+i%20]
            out = draft(torch.tensor([prefix]),use_cache=False)
            rejected = int(out.logits[0,-1].argmax())
            answer, h = teacher(target,prefix)
            if rejected == answer or answer == 60:
                continue
            label, _ = teacher(target, prefix+[answer])
            rows.append(dict(request=digest(prefix), prefix=prefix, split=split, answer=answer,
                             inputs=(h,*protocol(draft,0,rejected,answer)), label=label))
    return rows
