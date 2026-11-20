import copy
import json
import socket
import struct
import time
from pathlib import Path
import torch
from experiment import (expanded,owner,specs,validate,train,tiny_assets,tiny_split,eval_arm,read,write,
                       identities,make,paired,restore,summarize,code_id,sha)
from assets import schema_for
from quantizer import Quantizer,pack,unpack
from training import token_loss,entropies
from objective import FeedbackController
from online import evaluate_request,receive
from batch_target import BatchTarget
from evaluate import Meter


def raises(f):
    try:f()
    except (ValueError,RuntimeError,EOFError):return True
    raise AssertionError("expected rejection")


def defenses(target,model,paths):
    c=expanded("C0")
    for k in ("train_recipe.lr","train_recipe.seed","original_lora_update","target_policy","data.train_requests",
              "codec.latent_dim","codec.payload_bits","train_recipe.selection","compute.base_dtype"):
        wrong=dict(c)
        wrong[k]="tampered"
        raises(lambda:validate("C0",wrong))
    for bits,size in ((4,16),(8,8)):
        values=torch.tensor([0,2**bits-1]*(size//2))
        raw=pack(values,bits)
        assert torch.equal(unpack(raw,bits,size,"cpu"),values)
        raises(lambda:unpack(raw[:-1],bits,size,"cpu"))
    uniform=torch.full((2,1,2),.5)
    mixed=torch.tensor([[[1.,0.]],[[0.,1.]]])
    collapsed=mixed[:1].expand_as(mixed)
    assert entropies(uniform)[0]==0 and entropies(collapsed)[0]>0 and entropies(mixed)[1]<entropies(uniform)[1]
    z=torch.tensor([[[0.,8.],[8.,0.],[8.,0.]],[[8.,0.],[8.,0.],[0.,8.]]])
    y=torch.zeros(2,3,dtype=torch.long)
    assert token_loss(z,y,expanded("L_PREFIX"))[0]>token_loss(z,y,expanded("L_PREFIX"))[1]
    for id in ("L_MARGIN","L_PREFIX"):
        z=torch.randn(2,3,5)*1000
        y=torch.tensor([[0,-100,-100],[2,1,4]])
        vals=token_loss(z,y,expanded(id))
        assert vals.isfinite().all() and torch.equal(vals[0],token_loss(z[:1,:1],y[:1,:1],expanded(id))[0])
        raises(lambda:token_loss(z,torch.full_like(y,-100),expanded(id)))
    schema=schema_for(model,paths)
    torch.manual_seed(42)
    canonical=FeedbackController("grouped_vq8",32,32,schema)
    torch.manual_seed(42)
    current=Quantizer(c,32,32,schema)
    assert torch.equal(current.book[0],canonical.codec.codebook)
    for p,q in zip(current.encoder.parameters(),canonical.codec.encoder.parameters()):assert torch.equal(p,q)
    with torch.no_grad():
        current.receiver.weight.normal_()
        canonical.receiver.weight.copy_(current.receiver.weight)
        x=torch.randn(3,97)
        assert torch.equal(current.gates(x,8,"hard")[0],canonical.batch_gates(x))
        # Every reference verifies its own native forward, including full-accept/bonus and terminal EOS.
        seq=[1,2]
        ar=[]
        for i in range(12):
            ar.append(target.model(target._ids(seq+ar),use_cache=False).logits[0,-1].argmax().item())
        for k in (0,1,7,8):
            prop=ar[:8]
            if k<8:prop[k]=(prop[k]+1)%61
            state=BatchTarget(target,seq,Meter())
            n,_,ans,comm,h,t=state.verify(prop,-1,12)
            oracle=target.model.model(target._ids(seq+prop[:k]),use_cache=False).last_hidden_state[0,-1]
            torch.testing.assert_close(oracle,h,atol=2e-5,rtol=2e-5)
            assert n==k and comm==ar[:k+1] and t["target_cache_kept"]==len(seq)+len(comm)-1
            state.cache.crop(0)
            raises(lambda:state.verify([1],-1,3))
        assert BatchTarget(target,seq,Meter()).verify([ar[0]],ar[0],3)[2] is None
        assert BatchTarget(target,seq,Meter()).verify(ar[:2],-1,2)[2] is None
        tgt=copy.deepcopy(target)
        tgt.model.get_output_embeddings().weight.zero_()
        assert BatchTarget(tgt,seq,Meter()).verify([1],-1,2)[:3]==(0,1,0)
    with socket.socket() as server:
        server.bind(("127.0.0.1",0))
        server.listen(1)
        with socket.create_connection(server.getsockname()) as client,server.accept()[0] as peer:
            client.sendall(struct.pack("!II",2,17))
            raises(lambda:receive(peer))


def smoke(out):
    torch.set_num_threads(1)
    target,model,paths,asset=tiny_assets()
    defenses(target,model,paths)
    datasets={}
    datasets["train"],_=tiny_split(target,"train",3)
    datasets["calibration"],roster=tiny_split(target,"calibration",4)
    from diagnostics import run
    write(out/"diagnostics.json",run(datasets,roster))
    summary={}
    results={}
    root=out/"weights"
    root.mkdir(exist_ok=True)
    for arm in specs:
        target,model,paths,asset=tiny_assets()
        authority=owner(arm)
        folder=root/authority
        selected=None
        if arm!="B0":
            if arm==authority:
                train(model,paths,32,arm,datasets,asset,folder,dict(epochs=2,batch=9,micro=3))
            target,model,paths,asset=tiny_assets()
            training=read(folder/"training.json")
            selected=folder/f"epoch{training['selected']}.pt"
        result=eval_arm(target,model,paths,arm,asset,selected,roster[:2],12,-1,datasets["calibration"],roster[2:])
        results[arm]=result
        if arm!="B0":
            result["vs_B0"]=paired(result["rows"],results["B0"]["rows"])
            result["vs_parent"]=paired(result["rows"],results[specs[arm]["parent"]]["rows"])
            last=folder/f"epoch{training['last']}.pt"
            if last!=selected:
                target,model,paths,asset=tiny_assets()
                last_result=eval_arm(target,model,paths,arm,asset,last,roster[:2],12,-1,datasets["calibration"],roster[2:])
                write(out/(arm+"-last.json"),last_result)
            result["last_summary"]=last_result["summary"] if last!=selected else result["summary"]
        write(out/(arm+".json"),result)
        summary[arm]=result["summary"]
        print(arm,result["summary"]["acceptsum"],result["summary"]["roundsum"],flush=True)
    checks={}
    for arm in ("B0","C0","G16_K16","M_MIX_E4","M_MIX_E8","M_MIX_E16","D_FUSED"):
        target,model,paths,asset=tiny_assets()
        ctrl=gates=None
        budget=int(expanded(arm)["evaluation.budget"][:-1])
        if arm!="B0":
            folder=root/owner(arm)
            ck=folder/f"epoch{read(folder/'training.json')['selected']}.pt"
            gates,ctrl=restore(model,paths,32,arm,asset,ck)
        actual=evaluate_request(target,model,gates,ctrl,arm,roster[0],budget,12,-1,tcp=True)
        def trace(r):
            return [(t["candidates"],t["accepted"],t["committed"],t["payload_hex"]) for t in r["rounds"]]
        assert trace(actual)==trace(results[arm]["rows"][0])
        write(out/(arm+"-tcp.json"),actual)
        if arm=="C0":
            shadow=evaluate_request(target,model,gates,ctrl,arm,roster[0],budget,12,-1,shadow=True)
            replay=evaluate_request(target,model,gates,ctrl,arm,roster[0],budget,12,-1,replay=True)
            assert trace(replay)==trace(actual)
            assert [r["candidates"] for r in shadow["rounds"]]==[r["candidates"] for r in results["B0"]["rows"][0]["rounds"]]
            raises(lambda:restore(model,paths,32,"C_LOCAL",asset,ck))
            ctrl.preproject()
            frame=next(bytes.fromhex(r["payload_hex"]) for r in actual["rounds"] if r["wire_bytes"])
            regular,fused=ctrl.decode(frame,8),ctrl.decode(frame,8,True)
            with torch.no_grad():
                with gates.use(regular):l=model(torch.tensor([[1,2,3]])).logits
                with gates.use(fused):lf=model(torch.tensor([[1,2,3]])).logits
            write(out/"fused.json",dict(gate_max_abs=float((regular-fused).detach().abs().max()),
                  logits_max_abs=float((l-lf).abs().max()),candidate_trace_equal=
                  trace(results["D_FUSED"]["rows"][0])==trace(results["C0"]["rows"][0])))
        if gates:gates.close()
        checks[arm]=True
    write(out/"summary.json",dict(status="passed",arms=summary,tcp=checks,source=code_id(),actual_gpu=False))


@torch.no_grad()
def benchmark(out):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    q=Quantizer(expanded("C0"),4096,1024,[{"stop":448}]).eval()
    q.receiver.weight.normal_(0,.01)
    x=torch.randn(1,6145)
    raw=q.wire_ids(x,8)
    t=time.perf_counter_ns()
    q.preproject()
    prep=time.perf_counter_ns()-t
    scores=q.encoder(x)
    ix=scores.reshape(8,256).argmax(-1)
    def measure(fn):
        for i in range(10):fn()
        vals=[]
        for i in range(100):
            t=time.perf_counter_ns()
            fn()
            vals.append(time.perf_counter_ns()-t)
        vals.sort()
        return dict(p50_ns=vals[50],p95_ns=vals[95],n=len(vals))
    sections=dict(encoder=measure(lambda:q.encoder(x)),argmax=measure(lambda:scores.reshape(8,256).argmax(-1)),
                  pack=measure(lambda:pack(ix,8)),unpack=measure(lambda:unpack(raw,8,8,"cpu")),
                  encode_and_pack=measure(lambda:q.wire_ids(x,8)),decode=measure(lambda:q.decode(raw,8)),
                  fused=measure(lambda:q.decode(raw,8,True)))
    write(out/"benchmark.json",dict(scope="synthetic production shape, CPU FP32",timings=sections,
          precompute_ns=prep,fused_bytes=q.projected.numel()*q.projected.element_size(),
          regular_bytes=(q.book.numel()+q.receiver.weight.numel())*4,
          gate_max_abs=float((q.decode(raw,8)-q.decode(raw,8,True)).abs().max())))

