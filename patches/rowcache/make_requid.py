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

src = sub(src, """                scheduler_output.kv_cache_block_copies,
            )

    def gather_batch_req_state(
""", """                scheduler_output.kv_cache_block_copies,
            )
            self._qsa_copy_host_pages(scheduler_output)

    def gather_batch_req_state(
""")

src = sub(src, """    def _remove_request(self, req_id: str) -> bool:
""", """    def _qsa_copy_host_pages(self, scheduler_output: SchedulerOutput) -> None:
        # QSA keeps its KV history in pinned host RAM: bind_kv_cache rebinds layer.kv_cache to a UVA
        # view of _qsa_host_kv, while self.kv_caches still holds the layer's 2-byte GPU placeholders.
        # The partial-hit copy-on-write above therefore never copied the history; the private page
        # kept whatever its previous owner wrote (cross-session leak, 23 Oct 2026). Copy the host pages
        # of the pairs whose destination is a QSA-group block, on the same stream, before forward.
        views = getattr(self, "_qsa_host_views", None)
        if views is None:
            ctx = self.compilation_config.static_forward_context
            views, groups = [], []
            for g, group in enumerate(self.kv_cache_config.kv_cache_groups):
                hosts = [ctx[n].kv_cache for n in group.layer_names
                         if n in ctx and getattr(ctx[n], "_qsa_host_kv", None) is not None]
                if hosts:
                    groups.append(g)
                    views.extend(hosts)
            self._qsa_host_views, self._qsa_host_groups = views, groups
            self._qsa_cow_verify = os.environ.get("QSA_COW_VERIFY", "0") == "1"
            self._qsa_cow_stats = [0, 0, 0, 0]  # copies, page pairs, unequal before, unequal after
            logger.info("QSA host-page CoW: %d host KV views in kv-cache groups %s (verify=%s)",
                        len(views), groups, self._qsa_cow_verify)
        if not views:
            return
        groups = self._qsa_host_groups
        dst_ids: set[int] = set()
        for new_req in scheduler_output.scheduled_new_reqs:
            for g in groups:
                dst_ids.update(new_req.block_ids[g])
        for ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
            if ids is not None:
                for g in groups:
                    dst_ids.update(ids[g])
        for ids in (scheduler_output.block_table_updates or {}).values():
            for g in groups:
                dst_ids.update(ids[g])
        copies = [c for c in scheduler_output.kv_cache_block_copies if c.dst_block_id in dst_ids]
        if not copies:
            return
        num_blocks = self.kv_cache_config.num_blocks
        before = self._qsa_cow_unequal(views, copies) if self._qsa_cow_verify else 0
        copy_kv_cache_blocks_inplace(views, num_blocks, copies)
        stats = self._qsa_cow_stats
        stats[0] += len(copies)
        if self._qsa_cow_verify:
            after = self._qsa_cow_unequal(views, copies)
            stats[1] += len(copies) * len(views)
            stats[2] += before
            stats[3] += after
            logger.info("QSA host-page CoW verify: %d copies x %d layers, unequal before %d, after %d "
                        "(total copies %d, page pairs %d, unequal before %d, after %d)",
                        len(copies), len(views), before, after, *stats)

    @staticmethod
    def _qsa_cow_unequal(views, copies) -> int:
        # Test-only (QSA_COW_VERIFY=1): byte-compare source and destination host pages of every
        # copy on every local QSA layer. Synchronizes, so nothing in flight still writes them.
        torch.cuda.synchronize()
        src = [c.src_block_id for c in copies]
        dst = [c.dst_block_id for c in copies]
        n = 0
        for view in views:
            a = view[src].view(len(copies), -1).view(torch.int16)
            b = view[dst].view(len(copies), -1).view(torch.int16)
            n += int((a != b).any(dim=1).sum())
        return n

    def _remove_request(self, req_id: str) -> bool:
""")

if "\nimport os\n" not in src:
    src = sub(src, "import functools\n", "import functools\nimport os\n")

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
