import base64
import json
import socket
import struct
import threading
import time
from contextlib import nullcontext
import torch
from batch_target import BatchTarget
from evaluate import Target,Meter
from fixture import protocol
from data import digest,require
from session import DraftSession
from training import Donors


class Plain:
    width=0
    def use(self,gate):
        require(gate is None,"unexpected feedback in untouched DraftLoRA")
        return nullcontext()


def read_n(sock,n):
    result=bytearray()
    while len(result)<n:
        block=sock.recv(n-len(result))
        if not block:raise EOFError("short TCP frame")
        result.extend(block)
    return bytes(result)


def send(sock,obj,raw=b""):
    js=json.dumps(obj,separators=(",",":"),allow_nan=False).encode()
    buf=struct.pack("!II",len(js),len(raw))+js+raw
    sock.sendall(buf)
    return len(buf)


def receive(sock):
    n,m=struct.unpack("!II",read_n(sock,8))
    require(0<n<1<<20 and m<=16,"invalid TCP header")
    return json.loads(read_n(sock,n)),read_n(sock,m),8+n+m


class Remote:
    """Actual localhost TCP, raw codec bytes follow the length-delimited JSON header."""
    def __init__(self,handle):
        self.listener=socket.socket()
        self.listener.bind(("127.0.0.1",0))
        self.listener.listen(1)
        self.error=None
        def server():
            try:
                with self.listener.accept()[0] as conn,torch.inference_mode():
                    while True:
                        req,payload,_=receive(conn)
                        if req=="end":break
                        require(not payload,"candidate request has unexpected payload")
                        obj,raw=handle(req)
                        send(conn,obj,raw)
            except Exception as exc:self.error=exc
        self.thread=threading.Thread(target=server,daemon=True)
        self.thread.start()
        self.conn=socket.create_connection(self.listener.getsockname())
        self.conn.settimeout(60)

    def call(self,req):
        up=send(self.conn,req)
        answer,raw,down=receive(self.conn)
        return answer,raw,up,down

    def close(self):
        try:send(self.conn,"end")
        finally:
            self.conn.close()
            self.thread.join(5)
            self.listener.close()
        if self.error:raise self.error


@torch.inference_mode()
def evaluate_request(target,model,gates,ctrl,arm,row,budget=8,limit=128,eos=151645,*,donors=None,policy=None,
                     tcp=False,shadow=False,replay=False):
    prompt=row["prompt_token_ids"]
    cloud=BatchTarget(target,prompt,Meter())
    draft=DraftSession(model,gates or Plain(),replay=replay)
    draft.start(prompt)
    rounds=[]
    rid=row["request_id"]
    def handle(req):
        proposals,left=req["tokens"],req["left"]
        before=list(cloud.confirmed)
        actual=[]
        hook=target.model.model.register_forward_pre_hook(lambda m,a,k:actual.append(k["input_ids"].shape[-1]),with_kwargs=True)
        try:k,rejected,ans,comm,h,trace=cloud.verify(proposals,eos,left)
        finally:hook.remove()
        require(actual==[len(proposals)+1],"Target did not perform one packed forward")
        done=comm[-1]==eos or len(comm)==left
        raw=b""
        rate=budget
        donor=None
        encode_ns=policy_ns=0
        if ctrl is not None and not done:
            if donors:
                h,name,idx=donors.live(rid,k,len(rounds))
                donor=[name,idx]
            if policy:
                t=time.perf_counter_ns()
                rate=policy[str(k)]
                policy_ns=time.perf_counter_ns()-t
            _,r,c=protocol(model,k,rejected,ans)
            if r is None:r=torch.zeros_like(c)
            x=torch.cat((h.to(c.device).reshape(1,-1),r,c,c.new_tensor([[k/8]],dtype=torch.float32)),1).float()
            t=time.perf_counter_ns()
            raw=ctrl.wire_ids(x,rate)
            encode_ns=time.perf_counter_ns()-t
        trace.update(actual_inputs=actual,legal_prefix_sha=digest(before+proposals[:k]),h_donor=donor,
                     encode_pack_ns=encode_ns,policy_ns=policy_ns,rate=rate)
        return dict(accepted=k,rejected=rejected,answer=ans,committed=comm,done=done,trace=trace),raw
    conn=Remote(handle) if tcp else None
    started=time.perf_counter_ns()
    output=[]
    try:
        while len(output)<limit:
            left=limit-len(output)
            t=time.perf_counter_ns()
            proposed=draft.propose(min(left,8),eos)
            draft_ns=time.perf_counter_ns()-t
            req={"tokens":proposed,"left":left}
            if conn:reply,raw,up,down=conn.call(req)
            else:
                reply,raw=handle(req)
                up=down=None
            rate=reply["trace"]["rate"]
            t=time.perf_counter_ns()
            gate=None
            if ctrl is not None and not reply["done"]:
                gate=ctrl.decode(raw,rate,arm=="D_FUSED")
            else:require(raw==b"","terminal or static feedback")
            decode_ns=time.perf_counter_ns()-t
            t=time.perf_counter_ns()
            draft.finish(reply,lambda:None if shadow else gate)
            boundary_ns=time.perf_counter_ns()-t
            output.extend(reply["committed"])
            require(draft.committed==cloud.confirmed,"Draft and Target diverged")
            rounds.append(dict(**reply,candidates=proposed,payload_hex=raw.hex(),wire_bytes=len(raw),
                               app_up=up,app_down=down,decode_ns=decode_ns,boundary_draft_ns=boundary_ns,draft_ns=draft_ns))
            if reply["done"]:break
    finally:
        if conn:conn.close()
    return dict(request_id=rid,family_id=row["family_id"],output=output,rounds=rounds,
                accepted=sum(r["accepted"] for r in rounds),n=len(rounds),wall_ns=time.perf_counter_ns()-started)


def fit_policy(records):
    """Calibrate a cheap decision table using the next round, including observed encoding costs."""
    by_count={}
    for budget,rows in records.items():
        for row in rows:
            for r,next_round in zip(row["rounds"],row["rounds"][1:]):
                k=r["accepted"]
                cost=(r["trace"]["encode_pack_ns"]+r["decode_ns"])*1e-6
                value=next_round["accepted"]-.0001*r["wire_bytes"]-.001*cost
                by_count.setdefault((k,budget),[]).append(value)
    return {str(k):max(records,key=lambda b:sum(by_count[k,b])/len(by_count[k,b]) if (k,b) in by_count else -1e99)
            if any((k,b) in by_count for b in records) else 8 for k in range(9)}

