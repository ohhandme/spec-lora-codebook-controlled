import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MATRIX=HERE/"arm_matrix.json"
assert hashlib.sha256(MATRIX.read_bytes()).hexdigest()=="1e19feb6d60606e265e861c15083488ef1e111689db7b007d9f8d3c80a782a8f"
registry=json.loads(MATRIX.read_bytes())
specs={s["id"]:s for s in registry["arms"]}
shared={}


def set_shared(values):
    allowed={"train_recipe.seed","loss.lambda_margin","loss.lambda_prefix","loss.lambda_entropy","loss.delta","loss.tau"}
    if set(values)-allowed or any(type(v) not in (int,float) for v in values.values()):
        raise ValueError("invalid paired study profile")
    import math
    if any(not math.isfinite(v) for v in values.values()) or values.get("loss.tau",1)<=0 or any(
            v<0 for k,v in values.items() if k.startswith("loss.lambda")):
        raise ValueError("invalid temperature or loss coefficient")
    if type(values.get("train_recipe.seed",42)) is not int:raise ValueError("expected integer seed")
    shared.clear()
    shared.update(values)


def flatten(node,prefix=""):
    result={}
    for k,v in node.items():
        name=prefix+k
        if isinstance(v,dict):
            result.update(flatten(v,name+"."))
        else:
            result[name]=v
    return result


def expanded(arm):
    s=specs[arm]
    c=expanded(s["parent"]) if s["parent"] else flatten(registry["common"])
    if not s["parent"]:
        c.update({"loss.lambda_margin":.1,"loss.lambda_prefix":.1,"loss.lambda_entropy":.01,
                  "loss.delta":0.,"loss.tau":1.,"loss.commitment":.25,"ema.decay":.99,"ema.epsilon":1e-5,
                  "cluster.iterations":8,"cluster.samples":4096,"compute.base_dtype":"bfloat16",
                  "compute.adapter_dtype":"float32","order":"seed+zero_based_epoch","codec.stages":1,
                  "codec.pretrain":"included_in_updates","input.teacher_h":"same_decision",
                  "input.all_conditioning":"native","training.schedule":None,"loss.balance":False,
                  "loss.sharp":False,"loss.auxiliary":"none","evaluation.budget":"8B",
                  "decode.backend":"original","codec.update":"gradient","policy.cost_per_byte":.0001,
                  "policy.cost_per_ms":.001})
    c.update(copy.deepcopy(s["overrides"]))
    c.update(shared)
    return c


def validate(arm,actual):
    expected=expanded(arm)
    different={k:(expected.get(k),actual.get(k)) for k in set(actual)|set(expected)
               if k not in expected or k not in actual or actual[k]!=expected[k]}
    if different:raise ValueError("undeclared changes in "+arm+": "+str(different))


def owner(arm):
    return {"D_FUSED":"C0","M_POLICY":"M_MIX"}.get(arm,"M_MIX" if arm.startswith("M_MIX_E") else arm)


def diff(arm):
    s=specs[arm]
    current=expanded(arm)
    parent=expanded(s["parent"]) if s["parent"] else {}
    return {k:[parent.get(k),v] for k,v in current.items() if parent.get(k)!=v}

