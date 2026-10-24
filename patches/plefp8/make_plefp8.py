#!/usr/bin/env python3
"""FP8 PLE table (item: FP8 PLE phase): run the pinned-table UVA lookup on byte views for 1-byte table dtypes.

The model already selects an FP8 PLE format from the text-config key ple_embedding_dtype="float8_e4m3fn"
(Qwen4ExpPLEFp8EmbeddingMethod: float8_e4m3fn table + one global float32 weight_scale, dequantized after the lookup;
the ETP reduce sums int8 views). The lookup itself, _lookup_ple_embedding_from_pinned_kernel, is a Triton row copy in
the table's dtype, and Triton may refuse fp8e4nv pointers below sm_89 (RTX 3090 = sm_86). The kernel only moves
bytes, so for a 1-byte table dtype the patched _lookup passes uint8 views of the UVA table and of the output buffer:
the same bytes land in the same places (masked rows load 0x00 = +0.0 in e4m3fn), no arithmetic is involved.
BF16 tables (itemsize 2) take the unchanged call: with the BF16 checkpoint the patched file behaves exactly like its
input. harness/ple_fp8_check.py tests both kernel variants on the GPU against torch indexing.
usage: make_plefp8.py ngram_embedding.py.in ngram_embedding.py.out
"""
import difflib, sys

src = orig = open(sys.argv[1]).read()


def sub(old, new):
    global src
    assert src.count(old) == 1, (old[:80], src.count(old))
    src = src.replace(old, new)


sub('''        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self._uva_weight,
                flat_ids,
                output,
''', '''        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            # make_plefp8.py: a 1-byte table (FP8) is copied as uint8 bytes; Triton may not take fp8e4nv on sm_86
            byte_rows = self._uva_weight.element_size() == 1
            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self._uva_weight.view(torch.uint8) if byte_rows else self._uva_weight,
                flat_ids,
                output.view(torch.uint8) if byte_rows else output,
''')
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), "in", "out", n=1))
