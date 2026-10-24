#!/usr/bin/env python3
"""S6a: staged-prefill request offsets/lengths from the metadata's CPU copies instead of a GPU read per layer.

_qsa_staged_prefill (qsa.py) starts with `query_start_loc.tolist()` and `seq_lens.tolist()` on GPU tensors: two
device->host syncs on every QSA layer of every prefill step, although the values are the same for all layers of the
step and the runner built them on the CPU (CommonAttentionMetadata.query_start_loc_cpu / seq_lens_cpu_upper_bound).
Each sync stops the CPU from launching ahead -- the decode submission S5 needs (Codex review, S5 sketch).
QSA_HOST_META=1: the metadata builder keeps the CPU copies on the metadata object, the staged prefill reads them.
seq_lens_cpu_upper_bound is exact for prefill rows (vLLM's own note; MTP=0 here), and only the largest prefill row's
length is used. QSA_HOST_META_VERIFY=1 (test only) also reads the GPU values and counts mismatches of the offsets or
of the chosen row's length; the first mismatches are logged as errors and the GPU values are used for that call.
Unset, the patched file behaves exactly like its input.
usage: make_hostmeta.py qsa.py.in qsa.py.out
"""
import difflib, sys

src = orig = open(sys.argv[1]).read()


def sub(old, new):
    global src
    assert src.count(old) == 1, (old[:80], src.count(old))
    src = src.replace(old, new)


sub('''class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
''', '''_QSA_HOST_META = os.environ.get("QSA_HOST_META", "0") == "1"
_QSA_HOST_META_VERIFY = os.environ.get("QSA_HOST_META_VERIFY", "0") == "1"
_QSA_HOST_META_STATS = [0, 0]  # verified calls, mismatches


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if _QSA_HOST_META:
            # S6a (make_hostmeta.py): CPU copies for the staged prefill, no per-layer GPU read
            qsl = getattr(common_attn_metadata, "query_start_loc_cpu", None)
            sl = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
            if qsl is not None and sl is not None:
                # copied to lists now: the runner's CPU buffers are reused by the next step's preparation
                md._qsa_host_meta = (qsl.tolist(), sl.tolist())
        return md
''')

sub('''    offsets = starts[: num_requests + 1].tolist()
    lengths = seq_lens[:num_requests].tolist()
    best = max(range(num_requests), key=lambda i: offsets[i + 1] - offsets[i])
''', '''    host = getattr(attn_metadata, "_qsa_host_meta", None)
    if host is not None and len(host[0]) >= num_requests + 1 and len(host[1]) >= num_requests:
        offsets = host[0][: num_requests + 1]  # host lists: no device sync
        lengths = host[1][:num_requests]
        best = max(range(num_requests), key=lambda i: offsets[i + 1] - offsets[i])
        if _QSA_HOST_META_VERIFY:
            g_off, g_len = starts[: num_requests + 1].tolist(), seq_lens[:num_requests].tolist()
            _QSA_HOST_META_STATS[0] += 1
            if g_off != offsets or int(g_len[best]) != int(lengths[best]):
                _QSA_HOST_META_STATS[1] += 1
                if _QSA_HOST_META_STATS[1] <= 5:
                    from vllm.logger import init_logger
                    init_logger(__name__).error("QSA host meta MISMATCH (%d of %d): cpu %s / %s gpu %s / %s",
                                                _QSA_HOST_META_STATS[1], _QSA_HOST_META_STATS[0], offsets[:8],
                                                lengths[best], g_off[:8], g_len[best])
                offsets, lengths = g_off, g_len
                best = max(range(num_requests), key=lambda i: offsets[i + 1] - offsets[i])
            elif _QSA_HOST_META_STATS[0] % 5000 == 0:
                from vllm.logger import init_logger
                init_logger(__name__).info("QSA host meta verify: %d calls, %d mismatches", *_QSA_HOST_META_STATS)
    else:
        offsets = starts[: num_requests + 1].tolist()
        lengths = seq_lens[:num_requests].tolist()
        best = max(range(num_requests), key=lambda i: offsets[i + 1] - offsets[i])
''')
open(sys.argv[2], 'w').write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), 'qsa.py.in', 'qsa.py.out'))
