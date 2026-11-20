# Controlled feedback codebooks for frozen DraftLoRA

This experiment tests whether a small message from the verifier improves the next proposal window of an unmerged DraftLoRA model. B0 runs the original model without a dynamic rank hook, message or optimizer. C0 uses eight 256-way indices, 16 coordinates per index and a zero-initialized 128-to-rank linear receiver before the bounded gate. The target verifies the pending token and the complete proposed window in one causal call. It retains the committed KV prefix and takes feedback from the native decision row.

## Reproduce locally

The tested stack was Python 3.13, PyTorch 2.9.1 CPU, Transformers 4.57.3, safetensors 0.8.0 and filelock. The CPU suite creates two tiny Qwen3 models with nonzero custom LoRA weights. It generates legal teacher states locally. Run these commands from the repository root.

~~~sh
python study/experiment.py list-arms
python study/experiment.py validate-plan
python study/experiment.py cpu-smoke --out ./results/tiny
python study/experiment.py benchmark-codec --out ./results/codec
~~~

The smoke trains 22 configurations, including two explicitly tested joint-training copies. It evaluates 28 entries, including B0 and five same-checkpoint variants. Its two evaluation prompts include an intentionally selected nonzero first acceptance to exercise the accepted-token path. This fixture tests computation and protocol behavior. The two-prompt results do not support a quality ranking.

The original 0.6B CPU probe is in [verification/actual-original.json](verification/actual-original.json). To reproduce it, put the original `epoch1.pt` and `training_config.json` in the adapter directory and run the following command from the repository root. The probe checks the hashes in `asset_binding.json` before loading the model.

~~~sh
python production_probe.py "$ADAPTER_DIR" "$DRAFT_0_6B_DIR" "$MODEL_MANIFEST" ./results/actual-original.json
~~~

For the five recorded input tokens, disabled and zero gates reproduced the static final hidden state and full logits exactly. A nonzero test gate changed both tensors. The recorded JSON and the CPU rerun were byte-identical (SHA-256 `0cf7721494c4b7ed185bb5d8a394f8cd1e89ce3aa74826bbdcb6e5c2d979fa68`). This probe used the Draft and original adapter only.

## Run with bound external assets

Supply the original model directories, model manifest, adapter checkpoint, original adapter training configuration, pinned RAW index and development roster. Supply either a verified prepared teacher pack or the original RAW shards. No model download is needed by these commands.

~~~sh
python study/experiment.py train --arm C0 --out ./runs/C0 --device cuda:0 \
  --models "$MODELS" --manifest "$MODEL_MANIFEST" --adapter "$LORA_EPOCH1" \
  --training-config "$ORIGINAL_TRAINING_CONFIG" --index "$INDEX3986" --roster "$ROSTER503" \
  --pack "$PREPARED_TEACHER_H"
~~~

Repeat the same command to resume. The output binds the arm, code, model, split order, hidden-state hashes, dtype, gate schema and training recipe. The runner rejects a foreign checkpoint or changed configuration. Optional J arms require the additional flag `--enable-joint-copy`. Their adapter updates are saved in per-arm checkpoints. The input weight file is opened read-only.

~~~sh
python study/experiment.py evaluate --arm B0 --out ./eval/B0 --device cuda:0 \
  --models "$MODELS" --manifest "$MODEL_MANIFEST" --adapter "$LORA_EPOCH1" \
  --training-config "$ORIGINAL_TRAINING_CONFIG" --index "$INDEX3986" --roster "$ROSTER503"
python study/experiment.py evaluate --arm C0 --checkpoint "$SELECTED_C0" --out ./eval/C0 --device cuda:0 \
  --models "$MODELS" --manifest "$MODEL_MANIFEST" --adapter "$LORA_EPOCH1" \
  --training-config "$ORIGINAL_TRAINING_CONFIG" --index "$INDEX3986" --roster "$ROSTER503" \
  --baseline-result ./eval/B0/result.json --parent-result ./eval/B0/result.json
~~~

Choose `epochN.pt` using the `selected` field in `training.json`. Repeat evaluation with the fixed last epoch in a separate output directory. Pass C0's checkpoint for D_FUSED and M_MIX's checkpoint for its budget and policy variants. For C_SHUFFLE pass the calibration teacher pack or shards during evaluation. The donor must come from a different request in the same split with the same accepted count. Policy fitting uses family-disjoint requests from the remainder of calibration. Training uses only the train split. The first 200 of calibration503 constitute a development diagnostic. The 511 reserved requests are excluded from fitting and evaluation.

The controlled differences are registered in study/arm_matrix.json. A common JSON profile can preregister `train_recipe.seed`, `loss.tau`, `loss.delta` and the three `loss.lambda_*` coefficients. Pass `--shared-profile FILE` before the subcommand for every arm in the study. Model identity, optimizer, selection and data changes are rejected. The selected and last checkpoint, parameter list, actual phase-update counts, supervised tokens, prototype labels, elapsed time and codebook assignment statistics are recorded separately. Stage timings and parameter counts are available for comparing compute. No FLOP counter is included.

For a paired report, supply the matched B0 and direct parent results. The report includes per-request differences, a family-resampled 95% interval, accepted tokens per round, zero acceptance fraction and survival probabilities. FP32 and BF16 can produce different serial and packed logits. Native greedy argmax decides acceptance for every arm.

Each comparison checks the environment recorded from the loaded Target and Draft before starting evaluation. Both results must match on tensor placement and dtype layout, attention implementation, hardware description, Python, Torch and Transformers versions, CUDA build, cuDNN, backend precision and kernel settings, and thread counts. A missing or different binding requires a fresh evaluation in the same environment. Training checkpoints retain their existing code and model checks and load onto the evaluation device. The environment binding applies to evaluation outputs only. The CPU suite records same-environment pairing, rejection of altered or legacy metadata, and rejection after an actual FP32-to-BF16 Draft conversion in [verification/environment-regression.json](verification/environment-regression.json). Matching fields do not measure host load or guarantee reproducible wall time.

## Wire and timing

The TCP option runs a real loopback socket between the Draft loop and a Target service thread. Frames use a big-endian pair of uint32 lengths followed by UTF-8 JSON and the untouched raw payload. The reply header identifies the current rate. Four-bit indices use high nibble first. Zero-length feedback is used at a terminal decision and for B0. The application-frame sizes include diagnostic JSON. They cannot be used as compact production packet sizes. Decode is invoked once per applicable reply. Synthetic CPU component timing, localhost transport and application payload lengths measure different quantities.

D_FUSED caches receiver-projected group entries, sums selected vectors and then applies tanh. At production width the FP32 cache needs 3,670,016 bytes. Compare the gate, logits and candidates with C0 at the same checkpoint and precision before interpreting the timing. The tiny continuous-generation Target reference has no simulated network delay.

The source is derived from frozen rank-gate training revision 5001e60cbbf0342198d2ea855dd2e59b9c9edc54 and its cached greedy verifier. Original rank-hook implementation and model bindings are documented in the [rank-gate repository](https://github.com/ohhandme/spec-lora-feedback-gates).

