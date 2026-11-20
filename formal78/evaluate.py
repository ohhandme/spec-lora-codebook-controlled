"""Actual native greedy Target on one packed causal window, then frozen LoRA Draft evaluation."""
from contextlib import nullcontext
import time
import torch
from batch_target import BatchTarget
from data import require, digest
from fixture import protocol
from session import DraftSession


class Target:
    def __init__(self, model):
        self.model=model
        self.vocab_size=model.config.vocab_size
        self.hidden_size=model.config.hidden_size
        self.device=model.get_input_embeddings().weight.device

    def _ids(self,tokens):
        require(tokens and all(type(t) is int and 0<=t<self.vocab_size for t in tokens),"invalid token IDs")
        return torch.tensor([tokens],device=self.device)


class Meter:
    def timed(self,name):
        return nullcontext()


@torch.inference_mode()
def preflight(target,prompt,eos):
    observed=[]
    hook=target.model.model.register_forward_pre_hook(
        lambda m,args,kw: observed.append(kw["input_ids"].shape[-1]),with_kwargs=True)
    try:
        state=BatchTarget(target,prompt,Meter())
        before=len(observed)
        tokens=[(eos+i+1)%target.vocab_size for i in range(8)]
        k,rejected,answer,commit,h,trace=state.verify(tokens,eos,16)
        require(observed[before:]==[9] and trace["target_cache_kept"]==len(prompt)+len(commit)-1
                and h is not None and h.numel()==target.hidden_size,"real Target batch/KV preflight failed")
        return {"input_lengths":observed,"accepted":k,"answer":answer,"commit":commit,
                "prefix_sha256":digest(prompt),"trace":trace}
    finally:
        hook.remove()


@torch.inference_mode()
def evaluate_one(target,model,gates,ctrl,row,eos,limit=128,shadow=False):
    prompt=row["prompt_token_ids"]
    cloud=BatchTarget(target,prompt,Meter())
    draft=DraftSession(model,gates)
    draft.start(prompt)
    output,rounds=[],[]
    while len(output)<limit:
        remaining=limit-len(output)
        prop=draft.propose(min(8,remaining),eos)
        k,rejected,answer,commit,h,trace=cloud.verify(prop,eos,remaining)
        output.extend(commit)
        done=output[-1]==eos or len(output)==limit
        reply={"accepted":k,"rejected":rejected,"answer":answer,"done":done}
        raw=b""
        elapsed=0.
        gate=None
        if ctrl is not None and not done:
            args=protocol(model,k,rejected,answer)
            if target.device.type=="cuda":torch.cuda.synchronize(target.device)
            start=time.perf_counter()
            raw=ctrl.encode((h.reshape(1,-1),*args))
            gate=ctrl.decode(raw,args)
            if target.device.type=="cuda":torch.cuda.synchronize(target.device)
            elapsed=time.perf_counter()-start
            expected=8 if ctrl.method=="grouped_vq8" else 4*(ctrl.h+2*ctrl.d+1)
            require(len(raw)==expected,"wrong feedback byte width")
        draft.finish(reply,lambda:None if shadow else gate)
        require(draft.committed==cloud.confirmed,"Draft and batch Target commits differ")
        rounds.append(dict(candidates=prop,accepted=k,committed=commit,feedback_bytes=len(raw),
                           codec_wall_seconds=elapsed,**trace))
        if done:break
    return {"request_id":row["request_id"],"family_id":row["family_id"],"output":output,"rounds":rounds,
            "accepted":sum(r["accepted"] for r in rounds),"verification_rounds":len(rounds),
            "feedback_bytes":sum(r["feedback_bytes"] for r in rounds)}
