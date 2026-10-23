#!/usr/bin/env python3
"""Request identity for the QSA row cache (vLLM 0.29.1rc1 image, Model Runner V2:
vllm/v1/worker/gpu/model_runner.py; HiSparse forces V2, the old gpu_model_runner.py is not used).

The row cache in qsa.py first keyed its rows on a request's first physical KV page. Under prefix
caching, sessions with the same prompt start share that page, so concurrent sessions read each
other's cached KV rows (cross-session leak found on 23 Oct 2026). This patch gives every request a
unique int32 id when it is added (preempted requests are removed and re-added, so they get a new
one), keeps it per request-state slot on the GPU, and in prepare_attn gathers the batch's ids by
idx_mapping into vllm.forward_context.QSA_REQ_UID (a fixed GPU buffer allocated by qsa.py before
graph capture). Dummy batches write -2. qsa.py only caches when QSA_REQ_UID_WRITER is set, which
this module does on import.

usage: make_requid.py model_runner.image.py out_model_runner.py
"""
import difflib, sys

src = open(sys.argv[1]).read()
orig = src


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


src = sub(src, """from vllm.forward_context import BatchDescriptor, set_forward_context
""", """from vllm.forward_context import BatchDescriptor, set_forward_context
import vllm.forward_context as _qsa_fc

# QSA row cache: this runner writes per-request ids (see _qsa_write_req_uids).
_qsa_fc.QSA_REQ_UID_WRITER = True
""")

src = sub(src, """            req_index = self.req_states.req_id_to_index[req_id]
            if self.adaptive_verification is not None:
                self.adaptive_verification.add_request(req_index)
""", """            req_index = self.req_states.req_id_to_index[req_id]
            self._qsa_new_uid(req_index)
            if self.adaptive_verification is not None:
                self.adaptive_verification.add_request(req_index)
""")

src = sub(src, """        # Block tables: num_kv_cache_groups x [num_reqs_padded, max_num_blocks].
        block_tables = self.block_tables.gather_block_tables(
""", """        self._qsa_write_req_uids(input_batch)
        # Block tables: num_kv_cache_groups x [num_reqs_padded, max_num_blocks].
        block_tables = self.block_tables.gather_block_tables(
""")

src = sub(src, """        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        if valid_state_slots:
""", """        buf = getattr(_qsa_fc, "QSA_REQ_UID", None)
        if buf is not None:
            buf.fill_(-2)  # dummy requests never own a real request's row
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        if valid_state_slots:
""")

src = sub(src, """    def _remove_request(self, req_id: str) -> bool:
""", """    def _qsa_new_uid(self, req_index: int) -> None:
        slots = getattr(self, "_qsa_uid_slots", None)
        if slots is None:
            slots = self._qsa_uid_slots = torch.full(
                (self.max_num_reqs,), -2, dtype=torch.int32, device=self.device)
            self._qsa_uid_next = 0
        self._qsa_uid_next = self._qsa_uid_next % (2**31 - 2) + 1  # ids >= 1; -1 empty, -2 dummy
        slots[req_index] = self._qsa_uid_next

    def _qsa_write_req_uids(self, input_batch: InputBatch) -> None:
        buf = getattr(_qsa_fc, "QSA_REQ_UID", None)
        if buf is None:
            return
        n = input_batch.num_reqs
        buf.fill_(-2)
        if n and getattr(self, "_qsa_uid_slots", None) is not None:
            # GPU gather in batch order, stream-ordered before the forward pass that reads buf.
            buf[:n].copy_(self._qsa_uid_slots[input_batch.idx_mapping[:n]])

    def _remove_request(self, req_id: str) -> bool:
""")

open(sys.argv[2], "w").write(src)
open(sys.argv[2] + ".diff", "w").write("".join(difflib.unified_diff(
    orig.splitlines(True), src.splitlines(True), "a/model_runner.py", "b/model_runner.py")))
print("wrote", sys.argv[2])
