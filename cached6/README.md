# Six methods on the original Task76 cache

From the fixed checkout, use the original Task76 Python environment. The existing new-80G execution owner starts the training queue once. Each of the first two processes has its own model, optimizer, output directory and advisory lock on the assigned GPU0. The launcher waits for both exit codes, then trains 9, 14, 15 and 22 serially. The separate eval invocation evaluates the six selected checkpoints serially.

~~~sh
export PY=/data/specenv/bin/python HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=GPU-fe8adf09-4c85-6d12-d0fa-448c457b2a7a
export RUN_ROOT=/data/task76-cached-six-v1-results
bash cached6/queue.sh train
# Once training is complete, to evaluate the selected weights:
bash cached6/queue.sh eval
~~~

An individual entry uses `"$PY" -u cached6/runner.py train --arm 7 --out "$RUN_ROOT/arm-7"`. Replace train with eval for evaluation. Reissue the same training command to recover its own optimizer, minibatch cursor and phase. After a process exits, the OS releases that run's file lock. Outputs include progress.json, resume.pt, epoch checkpoints, calibration history in progress.json and selected.json. Evaluation resumes sealed requests and saves per-round data under eval200/{raw,shadow,active}.

| Order ID | Method | Change relative to the frozen Task76 additive repair interface |
|---|---|---|
| 7 | T_STAGE_RAND | Allocate 1/4 of updates to continuous control supervised by token CE, 1/4 to classifier bootstrap with random prototypes, remainder to hard-code CE. |
| 8 | T_STAGE_DATA | Same initialization and schedule as 7. Initialize prototypes at bootstrap using train-only reference latents, up to 4,096 sampled windows and eight groupwise k-means iterations. |
| 9 | L_MARGIN | Add 0.1 times softplus of the highest competing logit minus the teacher token logit to valid-token CE. An exact tie contributes log(2) to the unweighted margin penalty. |
| 14 | G16_K16 | Sixteen groups, sixteen choices and eight coordinates per group. Pack two indices per byte, high nibble first. |
| 15 | N_NEAREST | Eight nearest-neighbor groups of 256 vectors with 16 coordinates. Train encoder and codebook using straight-through task gradients, commitment weight .25 and a codebook loss. |
| 22 | M_MIX | Sample 4/8/16 raw bytes uniformly per update on one additive two-stage codebook. The original matched evaluation and calibration use 8 bytes. |

Each arm adds a decoded 128-coordinate vector to the frozen original C repair control. The six arms train new codecs from their registered initialization with the original C-secondary weights. The selected historical VQ checkpoint remains a comparison result. The comparison of 7 and 8 isolates prototype initialization under this receiver. Arm 22 alone does not establish a gain over a separately trained fixed-rate, architecture-matched model.

Training reads the already prepared cache from `/data/Task76-Codecs-Matched-A10080G-20261120`: binding.json, train.safetensors and calibration.safetensors. The checked metadata and both split hashes are in cache_binding.json. The original Task76 evaluation wrapper used the following files under that directory:

- `inputs/models/{Qwen3-0.6B,Qwen3-8B}` and `inputs/model_manifest.json`
- `inputs/shared-repair-C-epoch1.pt` (SHA-256 `6edc4cd191daf7123b8407ee2876f2164dd1c4038bc20ae00268225157eaa91e`) and its original adjacent completion/selection metadata
- `inputs/calibration_roster.json` (503 requests)

The CLI accepts path overrides. No cache preparation command is required. Training uses the frozen vocabulary head, token table, cached Draft hidden states and cached decision-time Target H. Only evaluation loads the 8B Target. Five epochs, effective batch 32, original permutation, per-window normalization, equal-request calibration CE and earliest tied epoch match Task76. Bootstrap prototype labels and task-supervised tokens are counted separately in progress.json. The development evaluation uses the original ordered first 200 requests, three Task76 arms, native greedy decisions, eight-byte nonterminal replies and the reference loopback implementation.

Run `python cached6/check_cpu.py` for six small training runs, frozen-weight and zero Draft-forward checks, stage restart, packed-wire vs training controls, tied-margin and padding checks and the tiny native TCP evaluations. Formal GPU throughput and quality are measured by the execution owner.
