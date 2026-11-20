"""Deterministically shuffled legal windows, 32 windows/update, atomic mid-epoch resume."""
import time
from pathlib import Path
import torch
from data import require,write,read
from objective import losses


def save_pt(path,value):
    tmp=path.with_suffix(".pending")
    torch.save(value,tmp)
    tmp.replace(path)


@torch.no_grad()
def calibration(model,gates,ctrl,data,micro):
    sums=torch.zeros(len(data.rows),dtype=torch.float64)
    count=torch.zeros_like(sums)
    for start in range(0,len(data.entries),micro):
        ids=list(range(start,min(start+micro,len(data.entries))))
        values=losses(model,gates,ctrl,data,ids).double().cpu()
        requests=torch.tensor([data.entries[i][0] for i in ids])
        sums.index_add_(0,requests,values)
        count.index_add_(0,requests,torch.ones_like(values))
    require(bool((count>0).all()),"empty calibration request")
    return float((sums/count).mean())


def fit(model,gates,ctrl,train,dev,out,key,micro,epochs=5,batch=32,save_every=64):
    out.mkdir(exist_ok=True)
    opt=torch.optim.AdamW(ctrl.parameters(),lr=1e-4,weight_decay=.01)
    epoch,cursor,updates,best_score,selected=0,0,0,float("inf"),None
    state_path=out/"resume.pt"
    if state_path.exists():
        s=torch.load(state_path,map_location=ctrl.zero.device,weights_only=True)
        require(s["key"]==key,"resume input binding differs")
        ctrl.load_state_dict(s["controller"])
        opt.load_state_dict(s["optimizer"])
        epoch,cursor,updates,best_score,selected=[s[k] for k in ("epoch","cursor","updates","best_score","selected")]
    def checkpoint():
        save_pt(state_path,dict(key=key,controller=ctrl.state_dict(),optimizer=opt.state_dict(),
                epoch=epoch,cursor=cursor,updates=updates,best_score=best_score,selected=selected))
    while epoch<epochs:
        order=torch.randperm(len(train.entries),generator=torch.Generator().manual_seed(42+epoch)).tolist()
        ctrl.train()
        while cursor<len(order):
            group=order[cursor:cursor+batch]
            opt.zero_grad(set_to_none=True)
            for start in range(0,len(group),micro):
                value=losses(model,gates,ctrl,train,group[start:start+micro]).sum()/len(group)
                require(bool(value.isfinite()),"nonfinite training loss")
                value.backward()
            torch.nn.utils.clip_grad_norm_(ctrl.parameters(),float("inf"),error_if_nonfinite=True)
            opt.step()
            cursor+=len(group)
            updates+=1
            if updates==1 or updates%save_every==0:
                checkpoint()
                write(out/"progress.json",dict(epoch=epoch+1,windows=cursor,updates=updates,time=time.time()))
        ctrl.eval()
        score=calibration(model,gates,ctrl,dev,micro)
        require(torch.isfinite(torch.tensor(score)).item(),"nonfinite calibration CE")
        epoch+=1
        cursor=0
        save_pt(out/f"epoch{epoch}.pt",dict(key=key,controller=ctrl.state_dict(),epoch=epoch,score=score,updates=updates))
        if score<best_score:
            best_score,selected=score,epoch
        checkpoint()
    ck=torch.load(out/f"epoch{selected}.pt",map_location=ctrl.zero.device,weights_only=True)
    require(ck["key"]==key and ck["score"]==best_score,"wrong selected checkpoint")
    ctrl.load_state_dict(ck["controller"])
    ctrl.eval()
    write(out/"selected.json",dict(key=key,epoch=selected,calibration_ce=best_score,updates=updates))
    return ctrl
