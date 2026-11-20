import sys
import json
import hashlib
from pathlib import Path
from plan import ROOT
sys.path[:0]=[str(ROOT/"formal78"),str(ROOT)]
import torch
from transformers import Qwen3ForCausalLM
from data import Windows,INDEX,MODELS,COUNTS,sha,require,read,read_roster,partition,lines,supervision,digest
from evaluate import Target,Meter,BatchTarget
from fixture import setup
from production_probe import load_existing
from gates import RankGates,tensor_digest
from training import adapter_params


@torch.no_grad()
def tiny_split(target,name,count):
    rows,hidden,roster=[],[],[]
    for i in range(count):
        prompt=[1,5+i,{"train":13,"calibration":17}[name]]
        if name=="calibration" and i==0:
            prompt=[1,2,19]  # Selected solely to exercise a nonzero native acceptance in the fixture.
        prefix=list(prompt)
        y=[]
        # Independent full-prefix native forward provides the deterministic token oracle.
        for _ in range(56):
            tok=int(target.model(target._ids(prefix),use_cache=False).logits[0,-1].argmax())
            y.append(tok)
            prefix.append(tok)
        cloud=BatchTarget(target,prompt,Meter())
        offset=0
        ws=[]
        for k in range(9):
            proposals=y[offset:offset+8]
            if k<8:proposals[k]=(proposals[k]+1)%target.vocab_size
            actual,rejected,answer,commit,h,trace=cloud.verify(proposals,-1,64-offset)
            require(actual==k and commit==y[offset:offset+k+1],"fixture teacher verification mismatch")
            offset+=k+1
            n=1+(i+k)%4
            ws.append(dict(round=k,offset=offset,length=n,labels=y[offset:offset+n],accepted=k,rejected=rejected,
                           correction=answer,observation_key=digest([prompt,k]),layer="model.norm",
                           absolute_position=len(prompt)+offset-2,source_prefix_sha256=digest(cloud.confirmed[:-1])))
            hidden.append(h.cpu())
        rid=digest(prompt)
        rows.append(dict(request_id=rid,family_id=rid,prompt_ids=prompt,output_ids=y,windows=ws))
        roster.append(dict(request_id=rid,family_id=rid,prompt_token_ids=prompt))
    return Windows(rows,torch.stack(hidden)),roster


def tiny_assets():
    target,draft,paths,hashes=setup(install_gates=False)
    return Target(target),draft,paths,dict(fixture="qwen3-local-fp32",base=hashes[0],adapter=hashes[1],
                                          teacher=tensor_digest(target.state_dict()))


def real_assets(a,need_teacher):
    require(__debug__ and sha(a.manifest)==MODELS and sha(a.index)==INDEX,"formal binding differs")
    baseline,adapters,paths,expected,files,adapter_sha=load_existing(a.adapter.parent,a.models/"Qwen3-0.6B",
                      a.manifest,a.training_config,a.adapter)
    target=None
    if need_teacher:
        for file,entry in read(a.manifest)["Qwen3-8B"].items():
            require(sha(a.models/"Qwen3-8B"/file)==entry["sha256"],"Target model mismatch")
        target=Target(Qwen3ForCausalLM.from_pretrained(a.models/"Qwen3-8B",dtype=torch.bfloat16,
                      attn_implementation="eager",local_files_only=True).eval().requires_grad_(False).to(a.device))
    return target,baseline.to(a.device),paths,dict(base=MODELS,adapter_file=expected["checkpoint_sha256"],adapter=adapter_sha,
                      index=INDEX,roster=sha(a.roster),precision="BF16 base and FP32 original A/B")


def real_rows(a,needed):
    roster=read_roster(a.roster)
    cohort="".join(r["request_id"]+"\n" for r in roster[:200]).encode()
    require(hashlib.sha256(cohort).hexdigest()=="8012363ce8a15d0b3330ecba6cae520877cdf335c58e389b9026d7f0fbc6aeb6","bad dev200")
    if not needed:return {},roster
    # Stream split metadata for reserved families; no reserved H or feature file is loaded.
    active=[]
    sealed=set()
    n=0
    for line in a.index.open(encoding="utf8"):
        row=json.loads(line)
        if row["split"] in COUNTS:active.append(row)
        else:
            require(row["split"]=="heldout","unknown split")
            sealed.add(row["family_id"])
            n+=1
    require(n==511 and sealed.isdisjoint(r["family_id"] for r in active),"sealed split overlap")
    rows=partition(active,roster,COUNTS)
    return {s:Windows(rows[s],supervision(s,rows[s],a.shards,a.pack)) for s in needed},roster


def lineage(data):
    import hashlib
    d=hashlib.sha256()
    for chunk in data.hidden.split(1024):
        d.update(chunk.contiguous().view(torch.uint8).numpy())
    return dict(hidden=d.hexdigest(),order=digest([(r["request_id"],r["windows"]) for r in data.rows]))


def schema_for(model,paths):
    out=[]
    offset=0
    for p in paths:
        layer=model.get_submodule(p)
        r=layer.down.out_features
        out.append(dict(path=p,rank=r,start=offset,stop=offset+r,scale=layer.scale,adapter_dtype=str(layer.down.weight.dtype)))
        offset+=r
    return out

