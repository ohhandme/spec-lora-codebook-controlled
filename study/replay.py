import torch

def forecast(model,gates,ctrl,data,indices,budget,phase,donors=None):
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
            hrows.append(donors.mapping[data.starts[ri]+i] if donors else data.starts[ri]+i)
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
    values,_=ctrl.gates(x,budget,phase)
    table=torch.cat((ctrl.zero,values))
    gate=table[pad(owners,0)]
    token_ids=pad(tokens,0)
    # Right padding cannot affect earlier positions under causal attention.
    with gates.use(gate):
        hidden=model.model(input_ids=token_ids,use_cache=False,return_dict=True).last_hidden_state
    selected=hidden[torch.arange(len(tokens),device=device)[:,None],pad(positions,0)]
    logits=model.get_output_embeddings()(selected)
    target=pad(labels,-100)
    return logits,target
