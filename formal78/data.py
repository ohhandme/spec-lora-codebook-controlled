"""Bind the original 3986/503 index to the frozen validation roster and RAW caches."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


INDEX = "6b8797ea5155e4fd9bd01474fd3c669e6902e2025b5723e03e2a5faf311c3ff6"
ROSTER = "a75803ceae25fcf21a4ae42cb87470bca375096d9e0676ed52645a727ae4a836"
ORIGINAL_ROSTER = "576c46525d252bfeeb108206f0197b27d8aa27e923205cbec738c0e2f5b6326c"
MODELS = "3aa01f3c1f4b9bf488fbe696a374775e4ebf371be1a1836f5a1bda02256c3ee4"
COUNTS = {"train": (3986, 156688), "calibration": (503, 19736)}
TRAIN_TOKENS = 1217488


def require(test, reason):
    if not test:
        raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def lines(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_roster(path):
    source = sha(path)
    if source == ROSTER:
        return lines(path)
    require(source == ORIGINAL_ROSTER, "unrecognized 503 roster")
    return [
        {
            "request_id": r["request_id"],
            "family_id": r["family_id"],
            "prompt_token_ids": r["prompt_ids"],
            "expected_target_token_ids": r["frozen_target_ids"],
        }
        for r in read(path)
    ]


def partition(records, roster, counts):
    parts = {split: [r for r in records if r["split"] == split] for split in counts}
    ids, families = set(), set()
    for split, rows in parts.items():
        unique = {r["request_id"] for r in rows}
        fam = {r["family_id"] for r in rows}
        require(len(rows) == len(unique) and not unique & ids and not fam & families, "split leakage")
        ids.update(unique)
        families.update(fam)
        require((len(rows), sum(len(r["windows"]) for r in rows)) == counts[split], "split size differs")
        for r in rows:
            for w in r["windows"]:
                off, n = w["offset"], w["length"]
                require(
                    0 < off < len(r["output_ids"])
                    and 1 <= n <= 8
                    and len(w["labels"]) == n
                    and w["labels"] == r["output_ids"][off : off + n]
                    and r["output_ids"][off - 1] == w["correction"],
                    "saved Target labels/offset differ",
                )
    dev = {r["request_id"]: r for r in parts["calibration"]}
    require(len(dev) == len(roster) == len({r["request_id"] for r in roster}), "roster size/duplicates")
    for row in roster:
        r = dev[row["request_id"]]
        require(
            (r["prompt_ids"], r["output_ids"], r["family_id"])
            == (row["prompt_token_ids"], row["expected_target_token_ids"], row["family_id"]),
            "503 roster mismatch",
        )
    return parts


def check_round(r, w, rr):
    offset = w["offset"]
    corrected = r["prompt_ids"] + r["output_ids"][:offset]
    before_answer = corrected[:-1]
    count, proposal = w["accepted"], rr["candidate_ids"]
    require(type(count) is int and 0 <= count <= len(proposal) <= 8, "invalid offered count")
    answer = rr["correction_id"] if count < len(proposal) else rr["bonus_id"]
    rejected = proposal[count] if count < len(proposal) else None
    h = rr["hidden"]
    require(
        rr["observation_key"] == w["observation_key"]
        and rr["accepted_count"] == count
        and w["correction"] == answer == corrected[-1]
        and w["rejected"] == rr["rejected_id"] == rejected
        and rr["committed_ids"] == proposal[:count] + [answer]
        and corrected[-(count + 1) :] == rr["committed_ids"]
        and rr["stop_reason"] == "continue"
        and rr["prefix_after_commit_length"] == len(corrected)
        and h["present"]
        and h["same_effective_target_forward_as_decision"] is True
        and h["available_before_receipt"] is True
        and h["layer"] == "model.norm"
        and h["absolute_position"] == len(corrected) - 2
        and h["source_prefix_sha256"] == digest(before_answer)
        and h["predicted_token_id"] == answer
        and w["hidden_key"] == rr["tensor_prefix"] + ".hidden",
        "illegal RAW decision prefix/answer/H",
    )



def supervision(split, rows, shards, packed=None):
    """Only Target H can be reused. No Draft activation tensor is opened."""
    total = sum(len(row["windows"]) for row in rows)
    if packed:
        meta = read(packed / "binding.json")
        require(meta["index_sha256"] == INDEX and meta["roster_sha256"] in (ROSTER, ORIGINAL_ROSTER),
                "wrong packed index or roster")
        path = packed / (split + ".safetensors")
        require(sha(path) == meta["splits"][split]["packed_sha256"], "packed supervision hash differs")
        with safe_open(str(path), framework="pt") as f:
            h = f.get_tensor("target")
            ordered = [(i, w) for i, row in enumerate(rows) for w in row["windows"]]
            for key in ("request", "accepted", "correction", "rejected"):
                values = [i if key == "request" else (-1 if w[key] is None else w[key]) for i,w in ordered]
                require(torch.equal(f.get_tensor(key), torch.tensor(values)), "packed window order differs: "+key)
            require(torch.equal(f.get_tensor("labels"), torch.tensor([
                w["labels"] + [-100]*(8-w["length"]) for _,w in ordered])), "packed labels differ")
    else:
        h = torch.empty(total, 4096, dtype=torch.bfloat16)
        index = 0
        for row in rows:
            folder = shards / row["request_id"]
            trace = read(folder / "trace.json")
            require(trace["trajectory_policy"] == "RAW" and not trace["repair_enabled"]
                    and trace["prompt_ids"] == row["prompt_ids"] and trace["output_ids"] == row["output_ids"],
                    "wrong source trace")
            with safe_open(str(folder / "features.safetensors"), framework="pt") as src:
                for w in row["windows"]:
                    rr = trace["rounds"][w["round"]]
                    check_round(row,w,rr)
                    value = src.get_tensor(w["hidden_key"]).reshape(-1).contiguous()
                    require(value.dtype == h.dtype and value.numel() == 4096 and
                            hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest() ==
                            rr["hidden"]["sha256"], "source H digest differs")
                    h[index] = value
                    index += 1
    require(h.shape == (total,4096) and h.dtype == torch.bfloat16
            and all(x.isfinite().all() for x in h.split(2048)), "invalid Target supervision")
    return h


class Windows:
    def __init__(self, rows, hidden):
        self.rows, self.hidden = rows, hidden
        self.entries, self.starts = [], []
        for ri,row in enumerate(rows):
            self.starts.append(len(self.entries))
            previous = 0
            for j,w in enumerate(row["windows"]):
                require(w["offset"] == previous+w["accepted"]+1 and w["round"] == j,
                        "missing or reordered historical gate boundary")
                previous = w["offset"]
                self.entries.append((ri,j))
        require(len(self.entries)==len(hidden), "window H count differs")
