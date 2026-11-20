"""Six controlled methods on the original Task76 cached features and frozen C head."""
import argparse
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "formal"), str(ROOT / "eval200")]
import run as native
import evaluate as screen
import torch
from torch.nn import functional as F
from objective import features, take
from raw_data import read, write, require, sha, digest
from variants import ARMS, CachedMessage


def make_head(draft, cfg, initial, arm):
    head = native.init_head(draft, "grouped_vq8", initial, cfg)
    head.message = CachedMessage(arm, head.message).to(draft.device)
    return head


def stage(arm, step, total):
    if arm in (7, 8):
        if step*4 < total:
            return "continuous"
        if step*2 < total:
            return "bootstrap"
    return "hard"


def rate(arm, step):
    if arm != 22:
        return 8
    gen = torch.Generator().manual_seed(420003 + step)
    idx = torch.multinomial(torch.tensor([1/3, 1/3, 1/3]), 1, generator=gen).item()
    return (4, 8, 16)[idx]


def cached_logits(head, batch, operands, message):
    local, rejected, correction, _ = operands
    base = head.base
    codes = base.core.local_encode(local, rejected, correction, batch["accepted"])
    control = sum(base.core.repair._lookup(codes[:, i], len(local)) for i in range(2))
    local_fp = local.to(base.local_hidden_lift[0].weight.dtype)
    control = control + base.hidden_projection(base.local_hidden_lift(local_fp))
    control = control + base.top32_projection(base.local_top32_surrogate(local_fp))
    return base.core.repair.repair_from_control(
        batch["hidden"], control + message, torch.arange(1, 9, device=local.device))


def window_loss(logits, labels, margin=False):
    z = logits.float()
    valid = labels != -100
    counts = valid.sum(-1)
    require(bool((counts > 0).all()), "empty window")
    token = F.cross_entropy(z.flatten(0, 1), labels.flatten(), reduction="none").reshape_as(labels)
    if margin:
        idx = labels.clamp_min(0).unsqueeze(-1)
        gold = z.gather(-1, idx).squeeze(-1)
        competitor = z.scatter(-1, idx, -float("inf")).amax(-1)
        token = token + .1 * F.softplus(competitor - gold) * valid
    return token.sum(-1) / counts


@torch.no_grad()
def calibrate(head, table, data, device, batch_size, requests):
    head.eval()
    sums = torch.zeros(requests, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for start in range(0, len(data["labels"]), batch_size):
        b = take(data, slice(start, start + batch_size), device)
        operands = features(b, table)
        message, _ = head.message.control(operands[-1], 8)
        ce = window_loss(cached_logits(head, b, operands, message), b["labels"]).cpu().double()
        sums.index_add_(0, b["request"].cpu(), ce)
        counts.index_add_(0, b["request"].cpu(), torch.ones_like(ce))
    require(bool((counts > 0).all()), "missing calibration request")
    return (sums/counts).mean().item()


def fit(head, table, data, key, out, device, *, epochs=5, batch_size=32, requests=503, interval=128):
    msg = head.message
    opt = torch.optim.AdamW(msg.parameters(), lr=1e-4, weight_decay=.01)
    train = data["train"]
    n = len(train["labels"])
    total = math.ceil(n/batch_size)*epochs
    state = dict(key=key, epoch=1, cursor=0, step=0, windows=0, labels=0, history=[],
                 initialized=False, phases={}, budgets={})
    resume = out/"resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        require(saved["state"]["key"] == key, "foreign training checkpoint")
        msg.load_state_dict(saved["codec"])
        opt.load_state_dict(saved["optimizer"])
        state = saved["state"]

    def persist():
        native.save_pt(resume, dict(state=state, codec=msg.state_dict(), optimizer=opt.state_dict()))
        write(out/"progress.json", state)

    for ep in range(state["epoch"], epochs+1):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(42+ep))
        head.train()
        head.base.eval()
        for start in range(state["cursor"], n, batch_size):
            batch = take(train, order[start:start+batch_size], device)
            operands = features(batch, table)
            x = operands[-1]
            phase = stage(msg.arm, state["step"], total)
            budget = rate(msg.arm, state["step"])
            opt.zero_grad(set_to_none=True)
            if phase == "bootstrap":
                if not state["initialized"]:
                    if msg.arm == 8:
                        ids = torch.randperm(n, generator=torch.Generator().manual_seed(42))[:4096]
                        sample = {k: train[k][ids].to(device) for k in ("target","rejected","accepted","correction")}
                        sample["hidden"] = train["hidden"][ids, :1].to(device)
                        msg.initialize_books(features(sample, table)[-1])
                    state["initialized"] = True
                loss = msg.bootstrap(x)
            else:
                if phase == "continuous":
                    control, penalty = msg.reference(x), 0.
                else:
                    control, penalty = msg.control(x, budget)
                z = cached_logits(head, batch, operands, control)
                loss = window_loss(z, batch["labels"], msg.arm == 9).mean() + penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(msg.parameters(), float("inf"), error_if_nonfinite=True)
            opt.step()
            count = len(x)
            state["cursor"] = start+count
            state["step"] += 1
            state["windows"] += count
            valid = int(batch["labels"].ne(-100).sum())
            state["labels"] += valid
            stats = state["phases"].setdefault(phase, dict(steps=0, windows=0, task_tokens=0, prototype_labels=0))
            stats["steps"] += 1
            stats["windows"] += count
            stats["task_tokens"] += valid if phase != "bootstrap" else 0
            stats["prototype_labels"] += count*msg.groups if phase == "bootstrap" else 0
            bucket = str(budget)
            state["budgets"][bucket] = state["budgets"].get(bucket, 0)+1
            if state["step"] == 1 or state["step"] % interval == 0:
                persist()
        score = calibrate(head, table, data["calibration"], device, batch_size, requests)
        require(math.isfinite(score), "nonfinite calibration CE")
        epoch_path = out/f"epoch{ep}.pt"
        native.save_pt(epoch_path, dict(key=key, epoch=ep, codec=msg.state_dict()))
        state["history"].append(dict(epoch=ep, request_equal_ce=score, sha256=sha(epoch_path)))
        state.update(epoch=ep+1, cursor=0)
        persist()
    require(state["step"] == total and state["windows"] == n*epochs
            and state["labels"] == epochs*int(train["labels"].ne(-100).sum()), "incomplete budget")
    best = min(state["history"], key=lambda r:(r["request_equal_ce"], r["epoch"]))
    require(sha(out/f"epoch{best['epoch']}.pt") == best["sha256"], "selected weights differ")
    write(out/"selected.json", dict(**best, key=key, updates=total, epochs=epochs,
          tokens=state["labels"], phases=state["phases"], budgets=state["budgets"]))
    return state


def evaluate(args, head, draft, key, device):
    from feedback_codec.hf_backend import HFTargetBackend, assert_compatible_pair
    from verify503 import loopback, ARMS as trials
    best = read(args.out/"selected.json")
    require(best["key"] == key and (best["epochs"],best["updates"],best["tokens"]) ==
            (5,24485,6087440), "foreign or incomplete selection")
    ck = args.out/f"epoch{best['epoch']}.pt"
    require(sha(ck) == best["sha256"], "selected checkpoint hash mismatch")
    saved = torch.load(ck, map_location="cpu", weights_only=True)
    require(saved["key"] == key and saved["epoch"] == best["epoch"], "wrong checkpoint identity")
    head.message.load_state_dict(saved["codec"])
    head.requires_grad_(False).eval()
    rows = screen.cohort(args.roster)
    contract = dict(max_new=128, window=8, eos_token_id=151645, roster=sha(args.roster),
                    checkpoint_sha256=best["sha256"], training=key, ordered_ids_sha256=screen.IDS_SHA,
                    codec_binding=dict(arm=key["arm"], budget=8, control_dim=128, checkpoint=best["sha256"]),
                    scope="cached-six-eval200", transport="single-GPU local TCP", draft_implementation="reference")
    out = args.out/"eval200"
    out.mkdir(exist_ok=True)
    screen.recover_rows(out, rows, contract, None)
    target = HFTargetBackend(str(args.models/"Qwen3-8B"), None, device=str(device), dtype=torch.bfloat16)
    assert_compatible_pair(target, draft)
    totals = loopback(rows, target, draft, head, device, out, contract)
    for row in rows:
        for trial in trials:
            screen.check_trace(read(out/trial/(row["request_id"]+".json")), row, trial, contract)
    write(out/"completion.json", dict(requests=len(rows), totals=totals, contract=digest(contract)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["train","eval"])
    p.add_argument("--arm", type=int, choices=list(ARMS), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/data/Task76-Codecs-Matched-A10080G-20261120"))
    for flag in ("models", "manifest", "base", "roster"):
        p.add_argument("--"+flag, type=Path)
    args = p.parse_args()
    inputs = args.cache/"inputs"
    args.models = args.models or inputs/"models"
    args.manifest = args.manifest or inputs/"model_manifest.json"
    args.base = args.base or inputs/"shared-repair-C-epoch1.pt"
    args.roster = args.roster or inputs/"calibration_roster.json"
    import fcntl
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/"worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(torch.cuda.device_count() == 1, "expose one assigned GPU to this process")
        meta = read(args.cache/"binding.json")
        require(meta == read(HERE/"cache_binding.json"), "not the recorded Task76 feature cache")
        key = dict(arm=ARMS[args.arm], id=args.arm, cache=digest(meta), base=native.BASE_C,
                   models=native.MODELS, native_source=native.code_binding(),
                   source={p.name:sha(p) for p in (HERE/"runner.py",HERE/"variants.py",ROOT/"eval200/evaluate.py")},
                   epochs=5, batch=32, seed=42, receiver="Task76 frozen C plus 128-D additive control")
        if (args.out/"contract.json").exists():
            require(read(args.out/"contract.json")==key, "output belongs to another study")
        write(args.out/"contract.json", key)
        native.check_models(args, ["Qwen3-0.6B"] + (["Qwen3-8B"] if args.phase=="eval" else []))
        device = torch.device("cuda:0")
        draft = native.Draft(str(args.models/"Qwen3-0.6B"), None, device=str(device), dtype=torch.bfloat16)
        head = make_head(draft, native.CodecConfig(4096,1024,draft.vocab_size), native.base_only(args.base), args.arm)
        if args.phase == "train":
            data, data_key = native.read_data(args.cache)
            require(data_key==key["cache"], "cache changed during read")
            table = draft.model.get_input_embeddings()
            del draft
            fit(head, table, data, key, args.out, device)
        else:
            evaluate(args, head, draft, key, device)


if __name__ == "__main__":
    main()
