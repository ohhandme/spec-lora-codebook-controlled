"""Exercise cached training, staged restart, packed payloads and original Task76 loopback."""
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import torch
import runner as six
from test_formal import fixture, scalar
from objective import features
from verify503 import loopback


def main():
    torch.set_num_threads(1)
    draft, target, cfg, initial, data = fixture()
    table = draft.model.get_input_embeddings()
    device = torch.device("cpu")
    x = features(data["train"], table)[-1]
    row = {"request_id":"tiny", "prompt_token_ids":[1,4,10], "expected_target_token_ids":[]}
    for _ in range(10):
        sequence = row["expected_target_token_ids"]
        sequence.append(target.next(row["prompt_token_ids"] + sequence).token_id)
        if sequence[-1] == 40:
            break
    results = {}
    warm = {}
    opts = dict(epochs=3,batch_size=4,requests=3,interval=1)
    with tempfile.TemporaryDirectory() as tmp:
        for arm in six.ARMS:
            folder = Path(tmp)/str(arm)
            folder.mkdir()
            head = six.make_head(draft,cfg,initial,arm)
            if arm in (7,8):
                warm[arm] = {k:v.clone() for k,v in head.message.state_dict().items()}
            with patch.object(draft.model, "forward", side_effect=AssertionError("Draft forward during cached fit")):
                state = six.fit(head,table,data,{"arm":arm},folder,device,**opts)
            assert state["windows"] == 57 and state["step"] == 15
            assert all(torch.equal(dict(head.base.named_parameters())[k],v) for k,v in initial.items())
            assert all(p.grad is None for p in draft.model.parameters())
            if arm in (7,8):
                assert set(state["phases"]) == {"continuous","bootstrap","hard"}
                assert state["phases"]["bootstrap"]["task_tokens"] == 0
            if arm == 22:
                assert set(state["budgets"]) == {"4","8","16"}
            if arm == 8:
                other = folder/"restart"
                other.mkdir()
                fresh = six.make_head(draft,cfg,initial,arm)
                with patch.object(six, "calibrate", side_effect=InterruptedError):
                    try:
                        six.fit(fresh,table,data,{"arm":arm},other,device,**opts)
                    except InterruptedError:
                        pass
                    else:
                        raise AssertionError("interruption missed")
                six.fit(fresh,table,data,{"arm":arm},other,device,**opts)
                assert all(torch.equal(v,fresh.message.state_dict()[k]) for k,v in head.message.state_dict().items())
            head.eval()
            for b in ((4,8,16) if arm == 22 else (8,)):
                packet = head.message.wire(x[:1],b)
                assert len(packet) == b
                torch.testing.assert_close(head.message.unbox(packet,b),head.message.control(x[:1],b)[0])
                try:
                    head.message.unbox(packet[:-1],b)
                except ValueError:
                    pass
                else:
                    raise AssertionError("truncated packet accepted")
            batch = data["train"]
            ops = features(batch,table)
            logits = six.cached_logits(head,batch,ops,head.message.control(ops[-1])[0])
            for i in (0,8):
                torch.testing.assert_close(logits[i:i+1],scalar(head,table,batch,i))
            contract = dict(max_new=10,window=8,eos_token_id=40,codec_binding={"arm":arm,"budget":8})
            totals = loopback([row],target,draft,head.requires_grad_(False).eval(),device,folder,contract)
            for mode in ("raw","shadow","active"):
                six.screen.check_trace(six.read(folder/mode/"tiny.json"),row,mode,contract)
            assert totals["raw"]["accepted"] == totals["shadow"]["accepted"]
            results[arm] = {"steps":state["step"],"phases":state["phases"],"budgets":state["budgets"],
                            "tcp_arms":list(totals),"active_feedback_bytes":totals["active"]["feedback_bytes"]}
        assert all(torch.equal(v,warm[8][k]) for k,v in warm[7].items())
    z = torch.tensor([[[1.,1.,0.],[0.,99.,0.]]],requires_grad=True)
    y = torch.tensor([[1,-100]])
    loss = six.window_loss(z,y,True)
    torch.testing.assert_close(loss-six.window_loss(z,y),torch.tensor([.1*0.69314718]))
    loss.backward()
    assert z.grad[0,1].count_nonzero() == 0
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
