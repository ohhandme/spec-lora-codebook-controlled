"""Teacher-forced windows with the original per-position LoRA gate timeline."""
import torch
from torch import nn
from torch.nn import functional as F
from h_vq64_codec import ConditionedVQ64
from gates import Controller


class FeedbackController(Controller):
    """The 128 code coordinates have their own receiver into the 448 frozen LoRA channels."""
    def __init__(self, method, h, d, schema):
        super().__init__(method,h,d,schema)
        if method == "grouped_vq8":
            self.codec = ConditionedVQ64(target_hidden_size=h,draft_hidden_size=d,rank_width=128)
            nn.init.normal_(self.codec.codebook,std=.02)
            self.receiver = nn.Linear(128,self.r,bias=False)
            nn.init.zeros_(self.receiver.weight)

    def batch_gates(self, x):
        if self.method == "full_h":
            return self.bound(self.codec.decoder(x))
        scores = self.codec.encoder(x).reshape(-1,8,256)
        soft = scores.softmax(-1)
        hard = F.one_hot(scores.argmax(-1),256).to(soft)
        code = torch.einsum("bgc,gcd->bgd",hard+(soft-soft.detach()),self.codec.codebook)
        return self.bound(self.receiver(code.flatten(1)))

    def train_gate(self, inputs):
        return self.batch_gates(self.codec._inputs(*inputs))

    def decode(self, payload, protocol):
        if self.method=="grouped_vq8" and payload is not None:
            return self.bound(self.receiver(self.codec.decode_payload(payload)))
        return super().decode(payload,protocol)


def losses(model, gates, ctrl, data, indices):
    """Each example replays its own earlier receipts. Its forecast sees no later receipt."""
    tokens, owners, positions, labels = [], [], [], []
    hrows, accepted, rejected, correction = [], [], [], []
    for index in indices:
        ri,j = data.entries[index]
        row = data.rows[ri]
        prompt,output = row["prompt_ids"],row["output_ids"]
        ws=row["windows"][:j+1]
        w=ws[-1]
        end=w["offset"]+w["length"]-1
        seq=prompt+output[:end]
        first=len(hrows)+1
        owner=[0]*len(seq)
        for i,v in enumerate(ws):
            start=len(prompt)+v["offset"]-1
            stop=len(prompt)+ws[i+1]["offset"]-1 if i<j else len(seq)
            owner[start:stop]=[first+i]*(stop-start)
            hrows.append(data.starts[ri]+i)
            accepted.append(v["accepted"])
            rejected.append(-1 if v["rejected"] is None else v["rejected"])
            correction.append(v["correction"])
        tokens.append(seq)
        owners.append(owner)
        positions.append([len(prompt)+w["offset"]-1+k for k in range(w["length"])])
        labels.append(w["labels"])
    device=ctrl.zero.device
    def ids(v):return torch.tensor(v,device=device,dtype=torch.long)
    def pad(values, fill):
        width=max(map(len,values))
        return ids([v+[fill]*(width-len(v)) for v in values])
    word=model.get_input_embeddings()
    r=ids(rejected)
    embeddings=word(r.clamp_min(0)).masked_fill((r<0)[:,None],0)
    x=torch.cat((data.hidden[hrows].to(device),embeddings,word(ids(correction)),ids(accepted)[:,None]/8),-1).float()
    table=torch.cat((ctrl.zero,ctrl.batch_gates(x)))
    gate=table[pad(owners,0)]
    token_ids=pad(tokens,0)
    # Right padding cannot affect earlier positions under causal attention.
    with gates.use(gate):
        hidden=model.model(input_ids=token_ids,use_cache=False,return_dict=True).last_hidden_state
    selected=hidden[torch.arange(len(tokens),device=device)[:,None],pad(positions,0)]
    logits=model.get_output_embeddings()(selected)
    target=pad(labels,-100)
    per_token=F.cross_entropy(logits.float().transpose(1,2),target,reduction="none")
    return per_token.sum(1)/target.ne(-100).sum(1)
