#!/usr/bin/env python3
"""S4 offline prototype; generate a model continuation and a FAIL-CLOSED worker guard.

Usage: make_resume.py REFERENCE_ROOT NEW_OUTPUT_DIRECTORY
Writes model.py.v1, model_runner.py.v1 and a source/output hash manifest. Existing
outputs are never overwritten. No deployment or engine integration is performed.

The original Qwen4ExpModel.forward is byte-for-byte untouched. The additional
flash_prefill_segments entry point is for a future dedicated harness/adapter only.
FLASH_PREFILL_SEGMENT_LAYERS=0 retains the stock path. Positive values fail at
GPUModelRunner construction because the scheduler/PP/KV transaction is NOT wired.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import textwrap


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"anchor count {text.count(old)}, wanted 1: {old[:100]!r}")
    return text.replace(old, new, 1)


def method(source, cls, name):
    tree = ast.parse(source)
    classes = [c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls]
    if len(classes) != 1:
        raise ValueError(f"expected one class {cls}")
    methods = [m for m in classes[0].body if isinstance(m, ast.FunctionDef) and m.name == name]
    if len(methods) != 1:
        raise ValueError(f"expected one method {cls}.{name}")
    m = methods[0]
    return "".join(source.splitlines(keepends=True)[m.lineno - 1:m.end_lineno])


def transform_model(source):
    if "def flash_prefill_segments(" in source:
        raise ValueError("model is already patched")
    original = method(source, "Qwen4ExpModel", "forward")
    body_start = "        if get_pp_group().is_first_rank:\n"
    prefetch_start = "        if self.start_layer < self.end_layer:\n"
    loop_start = "        for layer_idx, layer in islice(\n"
    final_start = "        if not get_pp_group().is_last_rank:\n"
    for anchor in (body_start, prefetch_start, loop_start, final_start):
        if original.count(anchor) != 1:
            raise ValueError(f"forward boundary anchor changed: {anchor!r}")
    header, body = original.split(body_start)
    body = body_start + body
    init, rest = body.split(prefetch_start)
    prefetch, rest = (prefetch_start + rest).split(loop_start)
    loop, tail = (loop_start + rest).split(final_start)
    tail = final_start + tail
    header = replace_once(header, "def forward(", "def flash_prefill_segments(")
    header = replace_once(header,
        "        deepstack_input_embeds: IntermediateTensors | None = None,\n",
        "        deepstack_input_embeds: IntermediateTensors | None = None,\n"
        "        *,\n        flash_chunk_key: tuple[str, int, int],\n"
        "        flash_num_requests: int,\n")
    header = replace_once(header, "-> torch.Tensor | IntermediateTensors:",
                          "-> Iterable[tuple[int, int, object | None]]:")
    prefetch = replace_once(prefetch, "        if self.start_layer < self.end_layer:\n", "")
    prefetch = textwrap.dedent(prefetch)
    prefetch = replace_once(prefetch, "self.layers[self.start_layer]", "self.layers[_flash_lo]")
    loop = replace_once(loop, "enumerate(self.layers), self.start_layer, self.end_layer",
                         "enumerate(self.layers), _flash_lo, _flash_hi")
    loop = replace_once(loop, "if layer_idx + 1 < self.end_layer:", "if layer_idx + 1 < _flash_hi:")
    # Do not materialize delayed HC at a segment boundary: keep all three tensors.
    park = '''
            if _flash_hi < self.end_layer:
                hidden_states = hidden_states.clone()
                block_output = block_output.clone() if block_output is not None else None
                injection = injection.clone() if injection is not None else None
                yield (_flash_lo, _flash_hi, None)
'''
    tail = replace_once(tail,
        '            return IntermediateTensors({"hidden_states": hidden_states})',
        '            _flash_output = IntermediateTensors({"hidden_states": hidden_states.clone()})\n'
        '            hidden_states = block_output = injection = None\n'
        '            yield (_flash_lo, _flash_hi, _flash_output)\n'
        '            return')
    tail = replace_once(tail, "        return sample_hidden_states",
        "        _flash_output = sample_hidden_states.clone()\n"
        "        hidden_states = block_output = injection = multi_hidden = sample_hidden_states = None\n"
        "        yield (_flash_lo, _flash_hi, _flash_output)\n        return")
    checks = '''        """EXPERIMENTAL: layer continuation only, NOT an engine execution path.

        Use a StageSlot with privately bound forward metadata and completion fences.
        Caller supplies one text-only prefill chunk; no MTP, deepstack or overlap.
        A yield means kernels were submitted, not that their work has completed.
        """
        import os
        _flash_n = int(os.environ.get("FLASH_PREFILL_SEGMENT_LAYERS", "0"))
        if _flash_n <= 0:
            raise ValueError("segmented entry point requires FLASH_PREFILL_SEGMENT_LAYERS > 0")
        if type(flash_num_requests) is not int or flash_num_requests != 1:
            raise ValueError("prototype supports exactly one prefilling request")
        if (not isinstance(flash_chunk_key, tuple) or len(flash_chunk_key) != 3
                or not isinstance(flash_chunk_key[0], str) or not flash_chunk_key[0]
                or type(flash_chunk_key[1]) is not int or flash_chunk_key[1] < 1
                or type(flash_chunk_key[2]) is not int or flash_chunk_key[2] < 0):
            raise ValueError("invalid request/generation/chunk identity")
        if deepstack_input_embeds is not None or self._mtp_hidden_buffer is not None:
            raise NotImplementedError("prototype is text-only, MTP=0")
        if self.start_layer >= self.end_layer:
            raise ValueError("empty pipeline stage")
        if getattr(self, "_flash_prefill_live", None) is not None:
            raise RuntimeError("one parked prefill per model instance")
        self._flash_prefill_live = flash_chunk_key
        _flash_failed = False
        try:
'''
    private = '''        # These are runner scratch views in the ordinary path. Clone once before yielding.
        positions = positions.clone()
        input_ids = input_ids.clone() if input_ids is not None else None
        inputs_embeds = inputs_embeds.clone() if inputs_embeds is not None else None
        query_start_loc = query_start_loc.clone() if query_start_loc is not None else None
        ngram_context = ngram_context.clone() if ngram_context is not None else None
        if intermediate_tensors is not None:
            intermediate_tensors = IntermediateTensors(
                {"hidden_states": intermediate_tensors["hidden_states"].clone()})
'''
    segmented = private + init
    segmented += "        inputs_embeds = intermediate_tensors = None  # do not retain consumed input buffers\n"
    segmented += "        for _flash_lo in range(self.start_layer, self.end_layer, _flash_n):\n"
    segmented += "            _flash_hi = min(_flash_lo + _flash_n, self.end_layer)\n"
    # Start PLE only inside this segment: no shared prefetch result may cross a yield.
    segmented += textwrap.indent(prefetch, "            ")
    segmented += textwrap.indent(loop, "    ")
    segmented += park + tail
    extra = header + checks + textwrap.indent(segmented, "    ")
    extra += ("        except GeneratorExit:\n"
              "            # StageSlot closes only after the compute/consumer fences drain.\n"
              "            raise\n"
              "        except BaseException:\n            _flash_failed = True\n            raise\n"
              "        finally:\n            if not _flash_failed:\n"
              "                self._flash_prefill_live = None\n")
    # Other exceptions deliberately keep the owner quarantined. A half-executed
    # real layer is not resumable; worker recovery is outside this prototype.
    result = replace_once(source, original, original + "\n" + extra)
    ast.parse(result)
    if method(result, "Qwen4ExpModel", "forward") != original:
        raise ValueError("disabled-path forward changed")
    return result


RUNNER_GUARD = '''        # S4 v1: fail closed until the scheduler/PP/KV completion adapter exists.
        _flash_segment_layers = int(__import__("os").environ.get("FLASH_PREFILL_SEGMENT_LAYERS", "0"))
        if _flash_segment_layers < 0:
            raise ValueError("FLASH_PREFILL_SEGMENT_LAYERS must be nonnegative")
        if _flash_segment_layers:
            raise NotImplementedError(
                "FLASH_PREFILL_SEGMENT_LAYERS: offline model-continuation prototype only; "
                "scheduler, private metadata, PP transport and KV commit adapter are not implemented")
'''


def transform_runner(source):
    if RUNNER_GUARD in source:
        raise ValueError("runner is already patched")
    anchor = "    def __init__(self, vllm_config: VllmConfig, device: torch.device):\n"
    result = replace_once(source, anchor, anchor + RUNNER_GUARD)
    ast.parse(result)
    return result


def generate(root, destination):
    sources = {
        "model.py.v1": (root / "models/qwen4_exp/nvidia/model.py", transform_model),
        "model_runner.py.v1": (root / "v1/worker/gpu/model_runner.py", transform_runner),
    }
    outputs, manifest = {}, {"status": "OFFLINE_PROTOTYPE_NOT_DEPLOYABLE", "files": {}}
    sha = lambda b: hashlib.sha256(b).hexdigest()
    for name, (path, transform) in sources.items():
        raw = path.read_bytes()
        output = transform(raw.decode()).encode()
        outputs[name] = output
        manifest["files"][name] = {"source": str(path.relative_to(root)),
                                   "source_sha256": sha(raw), "output_sha256": sha(output)}
    # All anchors/ASTs validated before any output is written; never replace existing work.
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in outputs.items():
        (destination / name).write_bytes(data)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(json.dumps(generate(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
