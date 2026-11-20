"""Read-only CPU forward comparison with the actual epoch-1 adapter and unmerged Draft."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import Qwen3ForCausalLM
from draft_interventions import attach_lora
from gates import RankGates, tensor_digest

HERE = Path(__file__).resolve().parent
def sha(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def load_existing(assets, draft_path, manifest_path, training_config=None, adapter_path=None):
    """Load the verified epoch1 delta on the actual BF16 Draft. No optimizer or training."""
    expected = json.loads((HERE/"asset_binding.json").read_bytes())
    training_config = training_config or assets/"training_config.json"
    adapter_path = adapter_path or assets/"epoch1.pt"
    assert sha(adapter_path) == expected["checkpoint_sha256"]
    assert sha(training_config) == expected["training_config_sha256"]
    assert sha(manifest_path) == expected["base_manifest_sha256"]
    assert sha(HERE/"draft_interventions.py") == json.loads(training_config.read_bytes())["implementation_sha256"]["draft_interventions.py"]
    manifest = json.loads(manifest_path.read_bytes())["Qwen3-0.6B"]
    for key in ("config.json","model.safetensors"):
        assert sha(draft_path/key) == manifest[key]["sha256"]
    model = Qwen3ForCausalLM.from_pretrained(draft_path, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True)
    ck = torch.load(adapter_path, map_location="cpu", weights_only=True)
    adapters = attach_lora(model, expected["rank"], expected["alpha"])
    adapters.load_state_dict(ck["adapter"], strict=True)
    model.eval().requires_grad_(False)
    paths = [f"model.layers.{i}.self_attn.{q}" for i in range(len(model.model.layers)) for q in ("q_proj","v_proj")]
    adapter_sha = tensor_digest(adapters.state_dict())
    return model,adapters,paths,expected,manifest,adapter_sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("assets", type=Path)
    p.add_argument("draft", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    torch.set_num_threads(3)
    model,adapters,paths,expected,manifest,adapter_sha = load_existing(a.assets,a.draft,a.manifest)
    with torch.inference_mode():
        ids = torch.tensor([[1, 7, 123, 80, 42]])
        def forward():
            o = model(ids, output_hidden_states=True, use_cache=False)
            return o.logits, o.hidden_states[-1]
        logits, hidden = forward()
        gates = RankGates(model, paths)
        results = {}
        for name, gate in [("disabled",None),("zero",torch.zeros(1,gates.width)),("nonzero",torch.linspace(-.5,.5,gates.width).reshape(1,-1))]:
            with gates.use(gate):
                l, h = forward()
            results[name] = dict(logits_exact=torch.equal(l,logits), hidden_exact=torch.equal(h,hidden),
                                 logits_max=float((l.float()-logits.float()).abs().max()),
                                 hidden_max=float((h.float()-hidden.float()).abs().max()), greedy=int(l[0,-1].argmax()))
        gates.close()
    assert all(results[k]["logits_exact"] and results[k]["hidden_exact"] for k in ("disabled","zero"))
    assert results["nonzero"]["logits_max"] > 0 and adapter_sha == tensor_digest(adapters.state_dict())
    report = dict(scope="actual 0.6B Draft and original epoch1 adapter, CPU BF16 base/FP32 A,B; fixed tokens and test gate",
                  checkpoint=expected["checkpoint_sha256"],base_weights=manifest["model.safetensors"]["sha256"],
                  adapter_sha=adapter_sha, schema=gates.schema, width=gates.width, checks=results,
                  tokens=ids.tolist(), static_greedy=int(logits[0,-1].argmax()), actual_target_8b=False)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2))
    print(results)


if __name__=="__main__":
    main()
