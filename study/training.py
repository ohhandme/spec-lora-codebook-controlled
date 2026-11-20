import json
import math
import time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from data import digest,write,require
from gates import tensor_digest
from replay import forecast
from trainer import save_pt


class Donors:
    """A deterministic derangement of RAW decision H within split and accepted-count bucket."""
    def __init__(self,data):
        self.data=data
        self.pool=defaultdict(dict)
        for idx,(i,j) in enumerate(data.entries):
            row=data.rows[i]
            key=row["windows"][j]["accepted"]
            self.pool[key].setdefault(row["request_id"],[]).append(idx)
        self.mapping=[None]*len(data.entries)
        for k,groups in self.pool.items():
            names=sorted(groups)
            require(len(names)>=2,"insufficient independent requests in ACK bucket "+str(k))
            for i,name in enumerate(names):
                other=groups[names[(i+1)%len(names)]]
                for j,idx in enumerate(groups[name]):self.mapping[idx]=other[j%len(other)]

    def live(self,rid,k,round_index):
        names=sorted(n for n in self.pool[k] if n!=rid)
        require(bool(names),"missing cross-request donor at evaluation")
        name=names[round_index%len(names)]
        group=self.pool[k][name]
        idx=group[round_index%len(group)]
        return self.data.hidden[idx],name,idx


def current_x(model,data,indices,donor):
    dev=model.get_input_embeddings().weight.device
    ws=[data.rows[r]["windows"][j] for r,j in (data.entries[ix] for ix in indices)]
    rejected=torch.tensor([-1 if w["rejected"] is None else w["rejected"] for w in ws],device=dev)
    corr=torch.tensor([w["correction"] for w in ws],device=dev)
    k=torch.tensor([w["accepted"] for w in ws],device=dev)
    emb=model.get_input_embeddings()
    hrows=[donor.mapping[ix] for ix in indices] if donor else indices
    return torch.cat((data.hidden[hrows].to(dev),emb(rejected.clamp_min(0)).masked_fill(rejected[:,None]<0,0),
                      emb(corr),k[:,None]/8),dim=-1).float()


def token_loss(logits,labels,c,ce_only=False):
    mask=labels!=-100
    n=mask.sum(1)
    require(bool((n>0).all()),"empty supervised window")
    z=logits.float()
    ce=F.cross_entropy(z.transpose(1,2),labels,ignore_index=-100,reduction="none").sum(1)/n
    if ce_only or c["loss.auxiliary"]=="none":return ce
    y=labels.clamp_min(0)[:,:,None]
    correct=z.gather(-1,y).squeeze(-1)
    competitor=z.scatter(-1,y,-float("inf")).amax(-1)
    margin=(correct-competitor)/c["loss.tau"]
    if c["loss.auxiliary"]=="target_vs_max_competitor_margin":
        aux=F.softplus(c["loss.delta"]/c["loss.tau"]-margin)
        return ce+c["loss.lambda_margin"]*(aux*mask).sum(1)/n
    survival=F.logsigmoid(margin).masked_fill(~mask,0).cumsum(1).exp()
    return ce-c["loss.lambda_prefix"]*(survival*mask).sum(1)/n


def entropies(p):
    mean=p.mean(0)
    entropy=-(p*p.clamp_min(1e-30).log()).sum(-1).mean()
    balance=(mean*(mean.clamp_min(1e-30).log()+math.log(p.shape[-1]))).sum(-1).mean()
    return balance,entropy


def phase_of(c,step,total):
    ratio=step/total
    if c.get("training.schedule"):
        return "continuous" if ratio<.25 else ("bootstrap" if ratio<.5 else "hard")
    return "soft" if c.get("training.quantization_schedule") and ratio<.25 else "hard"


def budget_of(c,step):
    dist=c.get("training.budget_distribution",{"8B":1})
    rng=torch.Generator().manual_seed(420003+step)
    i=torch.multinomial(torch.tensor(list(dist.values()),dtype=torch.float),1,generator=rng).item()
    return int(list(dist)[i][:-1])


def adapter_params(model):
    return {k:v for k,v in model.named_parameters() if k.endswith((".down.weight",".up.weight"))}


@torch.no_grad()
def calibrate(model,gates,ctrl,data,micro,donor,phase="hard"):
    sums=torch.zeros(len(data.rows),dtype=torch.float64)
    counts=torch.zeros_like(sums)
    budget=int(ctrl.c["evaluation.budget"][:-1])
    for i in range(0,len(data.entries),micro):
        ix=list(range(i,min(i+micro,len(data.entries))))
        z,y=forecast(model,gates,ctrl,data,ix,budget,phase,donor)
        vals=token_loss(z,y,ctrl.c,True).double().cpu()
        ids=torch.tensor([data.entries[j][0] for j in ix])
        sums.index_add_(0,ids,vals)
        counts.index_add_(0,ids,torch.ones_like(vals))
    return float((sums/counts).mean())


def apply_state(state,key,ctrl,model):
    require(state["key"]==key,"foreign arm, data, precision, model or budget")
    ctrl.load_state_dict(state["codec"])
    with torch.no_grad():
        for k,v in state["adapter"].items():adapter_params(model)[k].copy_(v)


def fit(model,gates,ctrl,train,dev,out,key,epochs,batch,micro,*,interval=64):
    c=ctrl.c
    joint=adapter_params(model) if c["original_lora_update"] else {}
    for p in joint.values():p.requires_grad_(True)
    params=list(ctrl.parameters())+list(joint.values())
    opt=torch.optim.AdamW(params,lr=c["train_recipe.lr"],weight_decay=c["train_recipe.weight_decay"])
    shuffle=c["input.teacher_h"]=="cross_request_per_split_matched_bucket"
    td,vd=(Donors(train),Donors(dev)) if shuffle else (None,None)
    if td:write(out/"donors.json",dict(train=td.mapping,calibration=vd.mapping))
    total=math.ceil(len(train.entries)/batch)*epochs
    rng=c["train_recipe.seed"]
    ep=cursor=step=0
    best=float("inf")
    logs=[]
    initialized=False
    stats={}
    named=dict(ctrl.named_parameters())
    named.update(joint)
    ever=set()
    def state():
        return dict(codec=ctrl.state_dict(),adapter={k:p.detach() for k,p in joint.items()},key=key)
    def checkpoint():
        save_pt(out/"resume.pt",dict(**state(),opt=opt.state_dict(),ep=ep,cursor=cursor,step=step,best=best,
                logs=logs,initialized=initialized,stats=stats,ever=sorted(ever)))
    if (out/"resume.pt").exists():
        s=torch.load(out/"resume.pt",map_location=ctrl.zero.device,weights_only=True)
        apply_state(s,key,ctrl,model)
        opt.load_state_dict(s["opt"])
        ep,cursor,step,best,logs,initialized,stats=[s[k] for k in ("ep","cursor","step","best","logs","initialized","stats")]
        ever.update(s["ever"])
    while ep<epochs:
        order=torch.randperm(len(train.entries),generator=torch.Generator().manual_seed(rng+ep)).tolist()
        while cursor<len(order):
            start=time.perf_counter()
            ix=order[cursor:cursor+batch]
            phase=phase_of(c,step,total)
            budget=budget_of(c,step)
            if phase=="bootstrap" and not initialized:
                # Same deterministic training sample for RAND and DATA, with no calibration access.
                sample=torch.randperm(len(train.entries),generator=torch.Generator().manual_seed(rng))[:c["cluster.samples"]].tolist()
                ctrl.init_prototypes(current_x(model,train,sample,td))
                initialized=True
            opt.zero_grad(set_to_none=True)
            if phase=="bootstrap":
                ctrl.bootstrap(current_x(model,train,ix,td)).mean().backward()
            else:
                for j in range(0,len(ix),micro):
                    subset=ix[j:j+micro]
                    z,y=forecast(model,gates,ctrl,train,subset,budget,phase,td)
                    value=token_loss(z,y,c).sum()/len(ix)
                    require(bool(value.isfinite()),"nonfinite token objective")
                    value.backward()
            info={}
            if phase in ("hard","soft"):
                # One contribution per supervised window across the entire optimizer batch.
                _,info=ctrl.quantize(current_x(model,train,ix,td),budget,phase)
                auxiliary=sum(v.mean() for v in info["costs"])
                for p in info["prob"]:
                    balance,sharp=entropies(p)
                    auxiliary+=c["loss.lambda_entropy"]*(int(c["loss.balance"])*balance+int(c["loss.sharp"])*sharp)
                if isinstance(auxiliary,torch.Tensor) and auxiliary.requires_grad:auxiliary.backward()
            torch.nn.utils.clip_grad_norm_(params,float("inf"),error_if_nonfinite=True)
            for name in named.keys()-ever:
                p=named[name]
                if p.grad is not None and bool(p.grad.detach().abs().amax()>0):ever.add(name)
            opt.step()
            if ctrl.ema:ctrl.ema_step([info])
            tokens=sum(train.rows[r]["windows"][k]["length"] for r,k in (train.entries[i] for i in ix))
            st=stats.setdefault(phase,dict(updates=0,windows=0,task_tokens=0,prototype_labels=0,seconds=0.))
            st["updates"]+=1
            st["windows"]+=len(ix)
            st["task_tokens"]+=tokens if phase!="bootstrap" else 0
            st["prototype_labels"]+=len(ix)*ctrl.g if phase=="bootstrap" else 0
            st["seconds"]+=time.perf_counter()-start
            if info:
                counts=st.setdefault("hard_counts",{})
                for level,p in enumerate(info["prob"]):
                    k=p.shape[-1]
                    name=f"{budget}B_stage{level}"
                    hist=F.one_hot(info["ids"][:,level],k).sum(0).cpu()
                    if name in counts:hist+=torch.tensor(counts[name])
                    counts[name]=hist.tolist()
                    st["soft_entropy_sum"]=st.get("soft_entropy_sum",0.)+float(entropies(p.detach())[1])*len(ix)
                    st["soft_entropy_samples"]=st.get("soft_entropy_samples",0)+len(ix)
            cursor+=len(ix)
            step+=1
            if step==1 or step%interval==0:checkpoint()
        score=calibrate(model,gates,ctrl,dev,micro,vd)
        require(math.isfinite(score),"nonfinite calibration")
        ep+=1
        cursor=0
        scores=dict(epoch=ep,ce=score)
        if c.get("training.quantization_schedule"):scores["soft_ce"]=calibrate(model,gates,ctrl,dev,micro,vd,"soft")
        logs.append(scores)
        save_pt(out/f"epoch{ep}.pt",dict(**state(),epoch=ep))
        if score<best:best=score
        checkpoint()
    selected=min(logs,key=lambda v:v["ce"])["epoch"]
    require(any(n.startswith("encoder.") for n in ever),"encoder had zero gradient throughout training")
    for st in stats.values():
        st["usage"]={}
        for name,hist in st.get("hard_counts",{}).items():
            values=torch.tensor(hist)
            freq=values/values.sum(1,keepdim=True)
            seen=values>0
            pairs=torch.triu_indices(len(hist),len(hist),1)
            a,b=seen[pairs[0]],seen[pairs[1]]
            st["usage"][name]=dict(dead=(~seen).sum(1).tolist(),max_share=freq.amax(1).tolist(),
                    perplexity=(-(freq*freq.clamp_min(1e-30).log()).sum(1)).exp().tolist(),
                    mean_group_index_jaccard=float(((a&b).sum(1)/(a|b).sum(1)).mean()))
    write(out/"training.json",dict(selected=selected,last=ep,key=key,scores=logs,stages=stats,
          trainable=list(named),nonzero_gradients=sorted(ever),updates=step))
    return selected

