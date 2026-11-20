import argparse
import json
import sys
import time
from pathlib import Path
import torch
from plan import HERE,ROOT,specs,expanded,owner,validate,diff,set_shared,shared
from assets import (tiny_assets,tiny_split,real_assets,real_rows,lineage,schema_for,RankGates,adapter_params,
                    tensor_digest,require,digest,sha,read)
from data import write
from quantizer import Quantizer
from training import fit,Donors,apply_state
from online import evaluate_request,fit_policy


def code_id():
    files=list(HERE.glob("*.py"))+[ROOT/n for n in ("fixture.py","session.py","production_probe.py","gates.py",
                                                   "draft_interventions.py","h_vq64_codec.py","message_codecs.py")]
    files+=[ROOT/"formal78"/n for n in ("objective.py","evaluate.py","batch_target.py","data.py","trainer.py")]
    return digest({p.relative_to(ROOT).as_posix():sha(p) for p in files})


def bind(path,obj):
    if path.exists():require(read(path)==obj,"output already belongs to a different experiment")
    else:write(path,obj)


def identities(model):
    state=model.state_dict()
    adapted=adapter_params(model)
    return dict(base=tensor_digest({k:v for k,v in state.items() if k not in adapted}),adapter=tensor_digest(adapted))


def make(model,paths,h,arm):
    c=expanded(arm)
    if arm=="B0":return None,None
    gates=RankGates(model,paths)
    torch.manual_seed(c["train_recipe.seed"])
    return gates,Quantizer(c,h,model.config.hidden_size,gates.schema).to(model.get_input_embeddings().weight.device)


def train(model,paths,h,arm,data,assets,folder,recipe):
    c=expanded(arm)
    require(owner(arm)==arm and c["training.enabled"],"evaluation-only arm has no optimizer")
    folder.mkdir(parents=True,exist_ok=True)
    gates,ctrl=make(model,paths,h,arm)
    before=identities(model)
    manifest=dict(arm=arm,config=c,source=code_id(),assets=assets,identities=before,schema=gates.schema,
                  data={k:lineage(v) for k,v in data.items()},recipe=recipe)
    try:
        bind(folder/"contract.json",manifest)
        selected=fit(model,gates,ctrl,data["train"],data["calibration"],folder,digest(manifest),**recipe)
        after=identities(model)
        require(before["base"]==after["base"] and (c["original_lora_update"] or before==after),"frozen tensor changed")
        write(folder/"freeze.json",dict(before=before,after=after,params=sum(p.numel() for p in ctrl.parameters()),
                book_storage=ctrl.book.numel(),encoder_output=ctrl.encoder[-1].out_features,
                stages=ctrl.stages,groups=ctrl.g,choices=ctrl.k))
        return selected
    finally:gates.close()


def restore(model,paths,h,arm,assets,ck):
    manifest=read(ck.parent/"contract.json")
    validate(owner(arm),manifest["config"])
    require(manifest["arm"]==owner(arm) and manifest["assets"]==assets and manifest["source"]==code_id()
            and manifest["identities"]==identities(model) and manifest["schema"]==schema_for(model,paths),
            "checkpoint owner, model or code mismatch")
    gates,ctrl=make(model,paths,h,arm)
    state=torch.load(ck,map_location=ctrl.zero.device,weights_only=True)
    try:apply_state(state,digest(manifest),ctrl,model)
    except Exception:
        gates.close()
        raise
    ctrl.eval()
    if arm=="D_FUSED":ctrl.preproject()
    return gates,ctrl


def summarize(rows):
    rr=[r for row in rows for r in row["rounds"]]
    n=len(rr)
    return dict(acceptsum=sum(r["accepted"] for r in rr),roundsum=n,ratio=sum(r["accepted"] for r in rr)/n,
                survival=[sum(r["accepted"]>=i for r in rr)/n for i in range(1,9)],
                zero=sum(r["accepted"]==0 for r in rr)/n,payload=sum(r["wire_bytes"] for r in rr),
                wall_ns=sum(r["wall_ns"] for r in rows))


def paired(rows,ref):
    ref={r["request_id"]:r for r in ref}
    require(set(ref)=={r["request_id"] for r in rows},"paired roster differs")
    groups={}
    diffs={}
    for r in rows:
        other=ref[r["request_id"]]
        delta=r["accepted"]/r["n"]-other["accepted"]/other["n"]
        diffs[r["request_id"]]=delta
        groups.setdefault(r["family_id"],[]).append(delta)
    g=list(groups.values())
    rng=torch.Generator().manual_seed(42)
    boot=[]
    for ids in torch.randint(len(g),(500,len(g)),generator=rng).tolist():
        vals=[v for i in ids for v in g[i]]
        boot.append(sum(vals)/len(vals))
    boot.sort()
    return dict(request_delta=diffs,mean=sum(diffs.values())/len(diffs),family_bootstrap95=[boot[12],boot[487]])


def eval_arm(target,model,paths,arm,asset,ck,rows,limit,eos,dev=None,fit_rows=None,tcp=False):
    gates,ctrl=(None,None) if arm=="B0" else restore(model,paths,target.hidden_size,arm,asset,ck)
    donor=Donors(dev) if arm=="C_SHUFFLE" else None
    budget=int(expanded(arm)["evaluation.budget"][:-1])
    try:
        policy=None
        if arm=="M_POLICY":
            require(fit_rows and {r["family_id"] for r in rows}.isdisjoint(r["family_id"] for r in fit_rows),"policy fitting leakage")
            rec={b:[evaluate_request(target,model,gates,ctrl,arm,r,b,limit,eos) for r in fit_rows] for b in (4,8,16)}
            policy=fit_policy(rec)
        records=[evaluate_request(target,model,gates,ctrl,arm,r,budget,limit,eos,donors=donor,policy=policy,tcp=tcp) for r in rows]
        return dict(arm=arm,checkpoint=sha(ck) if ck else None,policy=policy,summary=summarize(records),rows=records,
                    heldout_requests=0,independent_test=False)
    finally:
        if gates:gates.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shared-profile",type=Path,help="preregistered common seed/loss coefficients for every arm")
    subs=p.add_subparsers(dest="cmd",required=True)
    subs.add_parser("list-arms")
    v=subs.add_parser("validate-plan")
    v.add_argument("--arm",choices=specs)
    v.add_argument("--config",type=Path)
    for command in ("train","evaluate"):
        s=subs.add_parser(command)
        s.add_argument("--arm",choices=specs,required=True)
        s.add_argument("--out",type=Path,required=True)
        s.add_argument("--device",default="cpu")
        for name in ("models","manifest","adapter","index","roster","training-config"):
            s.add_argument("--"+name,type=Path,required=True)
        s.add_argument("--shards",type=Path)
        s.add_argument("--pack",type=Path)
        if command=="train":s.add_argument("--enable-joint-copy",action="store_true")
        else:
            s.add_argument("--checkpoint",type=Path)
            s.add_argument("--baseline-result",type=Path)
            s.add_argument("--parent-result",type=Path)
            s.add_argument("--tcp",action="store_true")
    for command in ("cpu-smoke","benchmark-codec"):
        s=subs.add_parser(command)
        s.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.shared_profile:set_shared(read(a.shared_profile))
    if a.cmd=="list-arms":
        for k in specs:print(k,specs[k]["mode"],owner(k))
        return
    if a.cmd=="validate-plan":
        if a.config:validate(a.arm,read(a.config))
        else:
            for k in specs:validate(k,expanded(k))
        print(json.dumps({k:diff(k) for k in specs},indent=2))
        return
    a.out.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    lock=FileLock(str(a.out/".study.lock"))
    with lock.acquire(timeout=0):return execute(a)


def execute(a):
    if a.cmd in ("cpu-smoke","benchmark-codec"):
        from checks import smoke,benchmark
        return (smoke if a.cmd=="cpu-smoke" else benchmark)(a.out)
    do_train=a.cmd=="train"
    if do_train and expanded(a.arm)["original_lora_update"]:require(a.enable_joint_copy,"enable independent adapter copy explicitly")
    require(a.pack or a.shards or (not do_train and a.arm!="C_SHUFFLE"),"supply --pack or --shards")
    target,model,paths,asset=real_assets(a,not do_train)
    needed=("train","calibration") if do_train else (("calibration",) if a.arm=="C_SHUFFLE" else ())
    data,roster=real_rows(a,needed)
    if do_train:
        return train(model,paths,4096,a.arm,data,asset,a.out,dict(epochs=5,batch=32,micro=2))
    require((a.arm=="B0")== (a.checkpoint is None),"B0 takes no codec checkpoint; trained arms require one")
    bind(a.out/"eval-contract.json",dict(arm=a.arm,model=asset,ck=sha(a.checkpoint) if a.checkpoint else None,
         source=code_id(),roster=sha(a.roster),tcp=a.tcp))
    fam={r["family_id"] for r in roster[:200]}
    fit_rows=[r for r in roster[200:] if r["family_id"] not in fam]
    result=eval_arm(target,model,paths,a.arm,asset,a.checkpoint,roster[:200],128,151645,data.get("calibration"),fit_rows,a.tcp)
    result["comparison_binding"]=dict(source=code_id(),assets=asset,profile=dict(shared),tcp=a.tcp,
                                      eval="development200",policy="native_greedy_batched",limit=128)
    for name,other,expected in (("baseline",a.baseline_result,"B0"),("parent",a.parent_result,specs[a.arm]["parent"])):
        if other:
            reference=read(other)
            require(reference["arm"]==expected and reference["comparison_binding"]==result["comparison_binding"],
                    "incorrect comparator, source, shared profile, execution mode or input data")
            result[name]=paired(result["rows"],reference["rows"])
    write(a.out/"result.json",result)


if __name__=="__main__":main()

