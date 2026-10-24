#!/usr/bin/env python3
"""Build the vision tower only on the first pipeline stage (vLLM 0.29.1rc1 image, Qwen4Exp nvidia/model.py).

With vision enabled, Qwen4ExpForConditionalGeneration builds Qwen3_VisionTransformer on every PP rank, although only
the first stage runs the encoder: ~0.42 GiB of BF16 weights per GPU wasted on the second TP pair. Later stages now
build it under no_init_weights (meta device, no allocation) wrapped in StageMissingLayer -- the same mechanism the
stock code uses when every image/video limit is 0 -- so metadata (dtype, merge size, ...) stays readable, the
weight loader skips the module, and calling it raises.
usage: make_vision_guard.py model.image.py out_model.py
"""
import difflib, sys

src = open(sys.argv[1]).read()
orig = src


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


src = sub(src, """    StageMissingLayer,
""", """    StageMissingLayer,
    no_init_weights,
""")
src = sub(src, """            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(""", """            # flash-next: only the first PP stage runs the encoder; later stages keep a metadata-only placeholder.
            from contextlib import nullcontext

            later_stage = not get_pp_group().is_first_rank
            with self._mark_tower_model(vllm_config, {"image", "video"}), (
                no_init_weights(self, lambda mod: StageMissingLayer("vision_tower", mod))
                if later_stage else nullcontext()
            ):
                self.visual = Qwen3_VisionTransformer(""")
open(sys.argv[2], "w").write(src)
open(sys.argv[2] + ".diff", "w").write("".join(difflib.unified_diff(
    orig.splitlines(True), src.splitlines(True), "a/model.py", "b/model.py")))
print("wrote", sys.argv[2])
