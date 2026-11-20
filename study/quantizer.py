import math
import torch
from torch import nn
from torch.nn import functional as F
from objective import FeedbackController


def pack(ids, bits):
    values=ids.reshape(-1).tolist()
    if bits not in (4,8) or len(values)*bits%8 or any(type(x)!=int or x<0 or x>=2**bits for x in values):
        raise ValueError("invalid quantizer indices")
    if bits==8:return bytes(values)
    return bytes((a<<4)|b for a,b in zip(values[::2],values[1::2]))


def unpack(raw,bits,count,device):
    if type(raw)!=bytes or len(raw)*8!=bits*count:raise ValueError("invalid raw payload length")
    values=list(raw) if bits==8 else [v for x in raw for v in (x>>4,x&15)]
    return torch.tensor(values,device=device,dtype=torch.long)


class Quantizer(nn.Module):
    def __init__(self,c,h,d,schema):
        super().__init__()
        self.c,self.h,self.d,self.r=c,h,d,schema[-1]["stop"]
        self.g,self.k,self.v,self.stages=[c["codec."+k] for k in ("groups","choices","code_dim","stages")]
        self.kind=c["codec.kind"]
        self.nearest="nearest" in self.kind
        self.multi=self.kind=="parallel_additive_multirate"
        self.ema=c["codec.update"]=="ema_count_vector"
        # Reuse exactly the RNG consumption, tensors, initialization and receiver of C0.
        ref=FeedbackController("grouped_vq8",h,d,schema)
        self.receiver=ref.receiver
        self.register_buffer("zero",ref.zero)
        shape=(self.stages,self.g,self.k,self.v)
        if self.kind=="categorical_st" and self.g==8:
            self.encoder=ref.codec.encoder
            book=ref.codec.codebook.detach().unsqueeze(0)
        else:
            out=self.g*self.v if self.nearest else self.stages*self.g*self.k
            self.encoder=nn.Sequential(nn.Linear(h+2*d+1,128),nn.SiLU(),nn.Linear(128,out))
            book=torch.randn(shape)*.02
        if self.ema:
            self.register_buffer("book",book)
            self.register_buffer("ema_count",torch.ones(shape[:-1]))
            self.register_buffer("ema_sum",book.clone())
        else:self.book=nn.Parameter(book)
        if c.get("training.schedule"):
            self.reference=nn.Sequential(nn.Linear(h+2*d+1,128),nn.SiLU(),nn.Linear(128,128))
        self.register_buffer("projected",None,persistent=False)

    def inputs(self,x):
        if self.c["input.all_conditioning"]=="constant_zero":return torch.zeros_like(x)
        if self.c["input.teacher_h"]=="zero":return torch.cat((torch.zeros_like(x[:,:self.h]),x[:,self.h:]),1)
        return x

    def layout(self,budget):
        if self.multi:
            if budget not in (4,8,16):raise ValueError("unsupported multi-rate budget")
            return (2 if budget==16 else 1,16 if budget==4 else 256)
        return self.stages,self.k

    def quantize(self,x,budget=8,phase="hard"):
        x=self.inputs(x)
        if phase=="continuous":
            return self.reference(x),{}
        n,choices=self.layout(budget)
        encoded=self.encoder(x)
        if self.nearest:
            residual=encoded.reshape(-1,self.g,self.v)
        else:scores=encoded.reshape(-1,self.stages,self.g,self.k)
        vectors,indices,probs,costs,residuals=[],[],[],[],[]
        group=torch.arange(self.g,device=x.device)
        for s in range(n):
            table=self.book[s,:,:choices]
            if self.nearest:
                distances=(residual.square().sum(-1,keepdim=True)+table.square().sum(-1)[None]
                           -2*torch.einsum("bgd,gkd->bgk",residual,table))
                logits=-distances
            else:logits=scores[:,s,:,:choices]
            probability=logits.softmax(-1)
            index=logits.argmax(-1)
            hard=table[group,index]
            if self.nearest:
                residuals.append(residual.detach())
                penalty=(residual-hard.detach()).square().mean((1,2))*self.c["loss.commitment"]
                if not self.ema:penalty=penalty+(hard-residual.detach()).square().mean((1,2))
                costs.append(penalty)
                vectors.append(hard)
                residual=residual-hard.detach()
            else:
                weights=probability if phase=="soft" else F.one_hot(index,choices).to(probability)+(probability-probability.detach())
                vectors.append(torch.einsum("bgk,gkd->bgd",weights,table))
            indices.append(index)
            probs.append(probability)
        decoded=torch.stack(vectors).sum(0)
        if self.nearest:decoded=decoded.detach()+(encoded.reshape_as(decoded)-encoded.reshape_as(decoded).detach())
        return decoded.flatten(1),dict(ids=torch.stack(indices,1),prob=probs,costs=costs,residuals=residuals)

    def gates(self,x,budget,phase):
        latent,info=self.quantize(x,budget,phase)
        return .5*torch.tanh(self.receiver(latent)),info

    def wire_ids(self,x,budget):
        n,k=self.layout(budget)
        encoded=self.encoder(self.inputs(x))
        if self.nearest:
            res=encoded.reshape(-1,self.g,self.v)
            indices=[]
            for stage in range(n):
                table=self.book[stage,:,:k]
                dist=res.square().sum(-1,keepdim=True)+table.square().sum(-1)[None]-2*torch.einsum("bgd,gkd->bgk",res,table)
                index=dist.argmin(-1)
                indices.append(index)
                if stage+1<n:res=res-table[torch.arange(self.g,device=x.device),index]
            ids=torch.stack(indices,1)
        else:ids=encoded.reshape(-1,self.stages,self.g,self.k)[:,:n,:,:k].argmax(-1)
        return pack(ids,int(math.log2(k)))

    def preproject(self):
        w=self.receiver.weight.reshape(self.r,self.g,self.v)
        self.projected=torch.einsum("sgkv,rgv->sgkr",self.book,w).detach()

    def decode(self,raw,budget,fused=False):
        n,k=self.layout(budget)
        ix=unpack(raw,int(math.log2(k)),n*self.g,self.book.device).reshape(n,self.g)
        s=torch.arange(n,device=ix.device)[:,None]
        g=torch.arange(self.g,device=ix.device)
        if fused:
            if self.projected is None:raise ValueError("prepare fused tables before decoding")
            u=self.projected[s,g,ix].sum((0,1)).unsqueeze(0)
        else:u=self.receiver(self.book[s,g,ix].sum(0).reshape(1,-1))
        return .5*torch.tanh(u)

    @torch.no_grad()
    def ema_step(self,infos):
        if not self.ema:raise ValueError("EMA on gradient codebook")
        sums,counts=torch.zeros_like(self.book),torch.zeros_like(self.ema_count)
        for info in infos:
            for s,res in enumerate(info["residuals"]):
                ix=info["ids"][:,s].T
                counts[s].scatter_add_(1,ix,torch.ones_like(ix,dtype=counts.dtype))
                sums[s].scatter_add_(1,ix[:,:,None].expand(-1,-1,self.v),res.transpose(0,1))
        decay=self.c["ema.decay"]
        self.ema_count.mul_(decay).add_(counts,alpha=1-decay)
        self.ema_sum.mul_(decay).add_(sums,alpha=1-decay)
        self.book.copy_(self.ema_sum/self.ema_count.clamp_min(self.c["ema.epsilon"])[...,None])

    @torch.no_grad()
    def init_prototypes(self,x):
        z=self.reference(self.inputs(x)).reshape(-1,self.g,self.v)
        if self.c["initialization.prototype_rule"]=="random":return
        rng=torch.Generator().manual_seed(42)
        for g in range(self.g):
            pts=z[:,g]
            centers=pts[torch.randint(len(pts),(self.k,),generator=rng)].clone()
            for _ in range(self.c["cluster.iterations"]):
                idx=torch.cdist(pts,centers).argmin(1)
                sums=torch.zeros_like(centers).index_add_(0,idx,pts)
                counts=idx.bincount(minlength=self.k)[:,None]
                centers=torch.where(counts>0,sums/counts.clamp_min(1),centers)
            self.book[0,g].copy_(centers)

    def bootstrap(self,x):
        with torch.no_grad():
            z=self.reference(self.inputs(x)).reshape(-1,self.g,self.v)
            labels=((z[:,:,None]-self.book[0])**2).sum(-1).argmin(-1)
        logits=self.encoder(self.inputs(x)).reshape(-1,self.k)
        return F.cross_entropy(logits,labels.flatten(),reduction="none").reshape(-1,self.g).mean(1)

