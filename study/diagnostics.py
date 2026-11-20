import copy
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import torch
from assets import tiny_assets,schema_for
from quantizer import Quantizer
from plan import expanded
from training import fit,entropies
from batch_target import BatchTarget
from evaluate import Meter
from session import DraftSession
from online import Plain,evaluate_request
from gates import RankGates,tensor_digest


def run(data,rows):
    target,model,paths,_=tiny_assets()
    schema=schema_for(model,paths)
    x=torch.randn(8,97)
    results={}
    for name in ("N_NEAREST","N_EMA","N_RVQ","M_MIX","G16_K16"):
        q=Quantizer(expanded(name),32,32,schema)
        for b in ((4,8,16) if name=="M_MIX" else (8,)):
            z,info=q.quantize(x,b)
            wire=q.wire_ids(x[:1],b)
            assert len(wire)==b
            q.receiver.weight.data.normal_(0,.02)
            torch.testing.assert_close(.5*q.receiver(z[:1]).tanh(),q.decode(wire,b),rtol=1e-5,atol=1e-6)
            if name=="N_RVQ":
                ix=info["ids"]
                first=q.book[0,torch.arange(q.g),ix[:,0]]
                raw=q.encoder(x).reshape(-1,q.g,q.v)
                reference=((raw-first)[:,:,None]-q.book[1]).square().sum(-1).argmin(-1)
                assert torch.equal(ix[:,1],reference)
            if q.ema:
                assert "book" not in dict(q.named_parameters())
                old=q.book.clone()
                q.ema_step([info])
                assert not torch.equal(q.book,old)
                old=q.book.clone()
                q.eval()
                q.wire_ids(x[:1],b)
                assert torch.equal(old,q.book)
        results[name]=True
    # Batch-mean KL gradient checks across two microbatch layouts.
    logits=torch.randn(5,8,16,requires_grad=True)
    p=logits.softmax(-1)
    ref=torch.autograd.grad(sum(entropies(p)),logits)[0]
    p=torch.cat([part.softmax(-1) for part in logits.split(2)])
    check=torch.autograd.grad(sum(entropies(p)),logits)[0]
    torch.testing.assert_close(ref,check)
    q=Quantizer(expanded("C0"),32,32,schema)
    gates=RankGates(model,paths)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for side in ("ref","retry"):(root/side).mkdir()
        init=copy.deepcopy(q)
        def fit_one(ctrl,side):
            fit(model,gates,ctrl,data["train"],data["calibration"],root/side,"resume-test",1,9,3,interval=1)
        fit_one(q,"ref")
        import training
        save=training.save_pt
        def injected(path,state):
            save(path,state)
            if path.name=="resume.pt" and state["step"]==1:raise InterruptedError()
        with patch("training.save_pt",side_effect=injected):
            try:fit_one(init,"retry")
            except InterruptedError:pass
            else:raise AssertionError("interruption not reached")
        fit_one(init,"retry")
        assert tensor_digest(q.state_dict())==tensor_digest(init.state_dict())
    gates.close()
    assert all(not model.get_submodule(p).down._forward_hooks for p in paths)
    prompt=rows[0]["prompt_token_ids"]
    with torch.inference_mode():
        d=DraftSession(model,Plain())
        d.start(prompt)
        direct=model(torch.tensor([prompt]),use_cache=False,output_hidden_states=True)
        assert torch.equal(d.last,direct.logits[:,-1]) and torch.equal(d.last_hidden,direct.hidden_states[-1][:,-1])
        before=tensor_digest(model.state_dict())
        r=evaluate_request(target,model,None,None,"B0",rows[0],limit=12,eos=-1)
        assert tensor_digest(model.state_dict())==before
        t=time.perf_counter_ns()
        next_ids=target._ids(prompt)
        cache=None
        ar=[]
        for _ in range(12):
            o=target.model(next_ids,past_key_values=cache,use_cache=True)
            cache=o.past_key_values
            token=o.logits[0,-1].argmax().item()
            ar.append(token)
            next_ids=target._ids([token])
        elapsed=time.perf_counter_ns()-t
        assert ar==r["output"]
        target.model.bfloat16()
        prop=ar[:8]
        state=BatchTarget(target,prompt,Meter())
        k,_,_,_,h,tr=state.verify(prop,-1,12)
        serial=[target.model(target._ids(prompt+prop[:i]),use_cache=False,output_hidden_states=True) for i in range(9)]
        observed=[]
        hook=target.model.register_forward_hook(lambda module,args,out:observed.append(out.logits.detach()))
        try:
            fresh=BatchTarget(target,prompt,Meter())
            fresh.verify(prop,-1,12)
        finally:hook.remove()
        assert len(observed)==1 and observed[0].shape[1]==len(prop)+1
        errors=[(observed[0][:,i].float()-o.logits[:,-1].float()).abs().max().item() for i,o in enumerate(serial)]
        results["bf16_max_abs_serial_vs_packed_cached"]=max(errors)
        results["bf16_serial_argmax_disagreements"]=sum(t!=o.logits[0,-1].argmax().item() for t,o in zip(tr["target_argmax_ids"],serial))
        results["bf16_decision_h_abs"]=float((h.float()-serial[k].hidden_states[-1][0,-1].float()).abs().max())
    results.update(resume_exact=True,micro_gradient_equal=True,b0_native_exact=True,target_alone_cpu_12_tokens_ns=elapsed)
    return results

