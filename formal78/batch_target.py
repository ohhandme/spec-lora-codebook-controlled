"""One causal Target forward per proposal window, with one committed token pending."""
import torch



def check_length(target, cache, length):
    if cache is None:
        if length != 0:
            raise RuntimeError("missing Target KV")
        return
    layers = target.model.config.num_hidden_layers
    if len(cache.layers) != layers or any(cache.get_seq_length(i) != length for i in range(layers)):
        raise RuntimeError(f"Target KV layer length differs from {length}")


def decision(proposals, predictions, eos, remaining):
    if not proposals or len(predictions) != len(proposals) + 1 or len(proposals) > remaining:
        raise ValueError("invalid decision window")
    accepted = next((i for i, t in enumerate(proposals) if t != predictions[i]), len(proposals))
    rejected = proposals[accepted] if accepted < len(proposals) else None
    answer = predictions[accepted] if rejected is not None or (
        proposals[-1] != eos and accepted < remaining) else None
    committed = proposals[:accepted] + ([] if answer is None else [answer])
    return accepted, rejected, answer, committed


class BatchTarget:
    @torch.inference_mode()
    def __init__(self, target, prompt, meter):
        self.target, self.meter = target, meter
        self.confirmed = list(prompt)
        target._ids(self.confirmed)
        self.cache = None
        if len(prompt) > 1:
            with meter.timed("target_prefill"):
                result = target.model.model(input_ids=target._ids(prompt[:-1]), use_cache=True,
                                            return_dict=True)
                self.cache = result.past_key_values
        check_length(target, self.cache, len(prompt) - 1)

    @torch.inference_mode()
    def verify(self, proposals, eos, remaining):
        before = len(self.confirmed)
        check_length(self.target, self.cache, before - 1)
        if before + len(proposals) > self.target.model.config.max_position_embeddings:
            raise ValueError("candidate window exceeds context")
        chunk = [self.confirmed[-1], *proposals]
        ids = self.target._ids(chunk)
        # Capture only the final norm output from the SAME forward that provides native logits.
        captured = []
        with self.meter.timed("target_verify"):
            hook = self.target.model.model.norm.register_forward_hook(
                lambda module, inputs, output: captured.append(output))
            try:
                out = self.target.model(input_ids=ids, past_key_values=self.cache, use_cache=True,
                                        output_hidden_states=False, return_dict=True)
            finally:
                hook.remove()
            if (len(captured) != 1 or captured[0].shape != (1, len(chunk), self.target.hidden_size)
                    or out.logits.shape != (1, len(chunk), self.target.vocab_size)):
                raise RuntimeError("bad batch logits or final hidden shape")
            if not torch.isfinite(out.logits).all() or not torch.isfinite(captured[0]).all():
                raise FloatingPointError("nonfinite Target result")
            predictions = out.logits[0].argmax(-1).tolist()
        raw = before + len(proposals)
        check_length(self.target, out.past_key_values, raw)
        accepted, rejected, answer, committed = decision(proposals, predictions, eos, remaining)
        # Row i has seen the confirmed prefix and exactly proposals[:i].
        hidden = None if answer is None else captured[0][0, accepted].detach().clone()
        kept = before + len(committed) - 1
        with self.meter.timed("target_kv_crop"):
            self.cache = out.past_key_values
            self.cache.crop(kept)
            check_length(self.target, self.cache, kept)
        self.confirmed.extend(committed)
        trace = {"target_argmax_ids": predictions, "first_reject_index": accepted if rejected is not None else None,
                 "decision_hidden_row": accepted if answer is not None else None,
                 "decision_prefix_length": before + accepted if answer is not None else None,
                 "target_forward_count": 1, "target_input_tokens": len(chunk),
                 "target_cache_before": before - 1, "target_cache_raw": raw, "target_cache_kept": kept}
        return accepted, rejected, answer, committed, hidden, trace
