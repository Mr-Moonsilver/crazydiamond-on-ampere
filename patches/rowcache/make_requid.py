#!/usr/bin/env python3
"""Request identity for the QSA row cache (vLLM 0.29.1rc1 image, gpu_model_runner.py).

The row cache in qsa.py first keyed its rows on a request's first physical KV page. Under prefix
caching, sessions with the same prompt start share that page, so concurrent sessions read each
other's cached KV rows (cross-session leak found on 23 Oct 2026). This patch gives every request a
unique int32 id when the runner first sees it (and a new one when it resumes after preemption) and
writes the ids of the current batch, in batch order, into vllm.forward_context.QSA_REQ_UID (a fixed
GPU buffer allocated by qsa.py before graph capture). Dummy runs write -2. qsa.py only caches when
QSA_REQ_UID_WRITER is set, which this module does on import.

usage: make_requid.py gpu_model_runner.image.py out_gpu_model_runner.py
"""
import difflib, sys

src = open(sys.argv[1]).read()
orig = src


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


src = sub(src, """    set_forward_context,
)
""", """    set_forward_context,
)
import vllm.forward_context as _qsa_fc

# QSA row cache: this runner writes per-request ids (see _qsa_write_req_uids).
_qsa_fc.QSA_REQ_UID_WRITER = True
""")

# New ids: on creation and on resume after preemption; dropped when the request finishes.
src = sub(src, """            req_state = self.requests.pop(req_id, None)
            self._on_request_state_removed(req_id, req_state)
""", """            req_state = self.requests.pop(req_id, None)
            self._on_request_state_removed(req_id, req_state)
            self._qsa_uid_map().pop(req_id, None)
""")
src = sub(src, """        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
""", """        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        for req_id in resumed_req_ids:
            self._qsa_new_uid(req_id)
""")
src = sub(src, """            self.requests[req_id] = req_state
            self.late_interaction_runner.register_request(req_id, pooling_params)
""", """            self.requests[req_id] = req_state
            self._qsa_new_uid(req_id)
            self.late_interaction_runner.register_request(req_id, pooling_params)
""")

# Write the batch's ids right after the block table, in the same batch order.
src = sub(src, """        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)
""", """        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)
        self._qsa_write_req_uids(num_reqs)
""")
src = sub(src, """                self.input_batch.block_table.commit_block_table(num_reqs_padded)
""", """                self.input_batch.block_table.commit_block_table(num_reqs_padded)
                buf = getattr(_qsa_fc, "QSA_REQ_UID", None)
                if buf is not None:
                    buf.fill_(-2)  # dummy requests never own a real request's row
""")

src = sub(src, """    def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
""", """    def _qsa_uid_map(self) -> dict[str, int]:
        m = getattr(self, "_qsa_req_uid", None)
        if m is None:
            m = self._qsa_req_uid = {}
            self._qsa_uid_next = 0
        return m

    def _qsa_new_uid(self, req_id: str) -> None:
        m = self._qsa_uid_map()
        self._qsa_uid_next = self._qsa_uid_next % (2**31 - 2) + 1  # ids >= 1; -1 empty, -2 dummy
        m[req_id] = self._qsa_uid_next

    def _qsa_write_req_uids(self, num_reqs: int) -> None:
        buf = getattr(_qsa_fc, "QSA_REQ_UID", None)
        if buf is None:
            return
        m = self._qsa_uid_map()
        cpu = getattr(self, "_qsa_uid_cpu", None)
        if cpu is None:
            cpu = self._qsa_uid_cpu = torch.empty(buf.shape[0], dtype=torch.int32, pin_memory=True)
        ids = cpu.numpy()
        ids.fill(-2)
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            if req_id not in m:
                self._qsa_new_uid(req_id)
            ids[i] = m[req_id]
        # Same pattern as the runner's other pinned inputs: _prepare_inputs runs inside
        # synchronize_input_prep, so the previous step's copy has finished before we refill.
        buf.copy_(cpu, non_blocking=True)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
""")

open(sys.argv[2], "w").write(src)
open(sys.argv[2] + ".diff", "w").write("".join(difflib.unified_diff(
    orig.splitlines(True), src.splitlines(True), "a/gpu_model_runner.py", "b/gpu_model_runner.py")))
print("wrote", sys.argv[2])
