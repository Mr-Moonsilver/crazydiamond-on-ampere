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
            # verify mode: watched sources {block id: (prefix tokens, [crc per layer])}, extra invariant counters
            self._qsa_watch: dict[int, tuple[int, list[int]]] = {}
            self._qsa_vstats = dict(src_rechecks=0, src_mutated=0, controls=0, controls_changed=0,
                                    src_changed_by_copy=0, prefix_unknown=0, src_reallocated=0)
            self._qsa_srcs: set[int] = set()  # copied source block ids (reuse evidence, verify off)
            self._qsa_src_reused = 0
            self._qsa_vstep = 0
            logger.info("QSA host-page CoW: %d host KV views in kv-cache groups %s (verify=%s)",
                        len(views), groups, self._qsa_cow_verify)
        if not views or (not scheduler_output.kv_cache_block_copies and not self._qsa_cow_verify
                         and not self._qsa_srcs):
            return
        groups = self._qsa_host_groups
        dst_ids: set[int] = set()
        prefix_of: dict[int, int] = {}  # dst block id -> reusable tokens in that page (consumer's hit % page)
        block_size = views[0].shape[2]
        for new_req in scheduler_output.scheduled_new_reqs:
            for g in groups:
                dst_ids.update(new_req.block_ids[g])
                for b in new_req.block_ids[g]:
                    prefix_of[b] = new_req.num_computed_tokens % block_size
        for ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
            if ids is not None:
                for g in groups:
                    dst_ids.update(ids[g])
        for ids in (scheduler_output.block_table_updates or {}).values():
            for g in groups:
                dst_ids.update(ids[g])
        copies = [c for c in (scheduler_output.kv_cache_block_copies or []) if c.dst_block_id in dst_ids]
        num_blocks = self.kv_cache_config.num_blocks
        if self._qsa_cow_verify:
            self._qsa_verify_step(views, dst_ids, copies, prefix_of, num_blocks)
            return
        reused = self._qsa_srcs & dst_ids  # a copied source block handed to a new owner (eviction + reuse)
        if reused:
            self._qsa_src_reused += len(reused)
            self._qsa_srcs -= reused
        if not copies:
            return
        copy_kv_cache_blocks_inplace(views, num_blocks, copies)
        self._qsa_srcs.update(c.src_block_id for c in copies)
        if len(self._qsa_srcs) > 4096:
            self._qsa_srcs = set(list(self._qsa_srcs)[-2048:])
        stats = self._qsa_cow_stats
        stats[0] += len(copies)
        now = time.monotonic()
        if now - getattr(self, "_qsa_cow_logged", 0.0) >= 5.0:
            # cheap coverage counter for release checks (verify off): cumulative copies on this rank
            self._qsa_cow_logged = now
            logger.info("QSA host-page CoW count: total copies %d (x %d layers), copy sources later reallocated %d",
                        stats[0], len(views), self._qsa_src_reused)

    @staticmethod
    def _qsa_page(view, block_id: int, ntok: int):
        # host page of one block, first ntok tokens of every head, as int16 (bit-exact compare)
        return view[block_id, :, :ntok].contiguous().view(torch.int16)

    def _qsa_crc(self, view, block_id: int, ntok: int) -> int:
        x = self._qsa_page(view, block_id, ntok).flatten().to(torch.int64)
        w = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.int64) % 65521
        return int((x * w).sum())

    def _qsa_verify_step(self, views, dst_ids, copies, prefix_of, num_blocks) -> None:
        # Test-only (QSA_COW_VERIFY=1), synchronizes every step. Invariants on every local QSA layer:
        #  1. after a copy, dst's reusable prefix (consumer hit % page tokens) equals src's, byte for byte;
        #  2. the copy changes neither src nor up to 4 unrelated blocks (checksums before/after the copy);
        #  3. a copied source's prefix stays unchanged while that block is not reallocated: re-checked on every later
        #     copy from it and every 20 steps (covers the consumer appending into its own copy).
        torch.cuda.synchronize()
        vs, st = self._qsa_vstats, self._qsa_cow_stats
        self._qsa_vstep += 1
        for b in dst_ids:  # reallocated blocks: their old contents are legitimately gone
            if self._qsa_watch.pop(b, None) is not None:
                vs["src_reallocated"] += 1
        recheck = [b for b in self._qsa_watch if self._qsa_vstep % 20 == 0] + \
                  [c.src_block_id for c in copies if c.src_block_id in self._qsa_watch]
        for b in dict.fromkeys(recheck):
            ntok, crcs = self._qsa_watch[b]
            vs["src_rechecks"] += 1
            if [self._qsa_crc(v, b, ntok) for v in views] != crcs:
                vs["src_mutated"] += 1
        if not copies:
            return
        endpoints = {c.src_block_id for c in copies} | {c.dst_block_id for c in copies} | set(self._qsa_watch)
        controls = [b for b in random.sample(range(1, num_blocks), min(32, num_blocks - 1)) if b not in endpoints][:4]
        ntoks = []
        for c in copies:
            n = prefix_of.get(c.dst_block_id)
            if not n:
                vs["prefix_unknown"] += 1
                n = views[0].shape[2]
            ntoks.append(n)
        src_before = [[self._qsa_crc(v, c.src_block_id, n) for v in views] for c, n in zip(copies, ntoks)]
        ctl_before = [[self._qsa_crc(v, b, views[0].shape[2]) for v in views] for b in controls]
        unequal_before = sum(int(not torch.equal(self._qsa_page(v, c.src_block_id, n), self._qsa_page(v, c.dst_block_id, n)))
                             for c, n in zip(copies, ntoks) for v in views)
        copy_kv_cache_blocks_inplace(views, num_blocks, copies)
        torch.cuda.synchronize()
        unequal_after = sum(int(not torch.equal(self._qsa_page(v, c.src_block_id, n), self._qsa_page(v, c.dst_block_id, n)))
                            for c, n in zip(copies, ntoks) for v in views)
        src_after = [[self._qsa_crc(v, c.src_block_id, n) for v in views] for c, n in zip(copies, ntoks)]
        vs["src_changed_by_copy"] += sum(a != b for a, b in zip(src_before, src_after))
        vs["controls"] += len(controls)
        vs["controls_changed"] += sum([self._qsa_crc(v, b, views[0].shape[2]) for v in views] != ctl
                                      for b, ctl in zip(controls, ctl_before))
        for c, n, crcs in zip(copies, ntoks, src_after):
            self._qsa_watch[c.src_block_id] = (n, crcs)
        while len(self._qsa_watch) > 64:
            self._qsa_watch.pop(next(iter(self._qsa_watch)))
        st[0] += len(copies); st[1] += len(copies) * len(views); st[2] += unequal_before; st[3] += unequal_after
        logger.info("QSA host-page CoW verify: %d copies x %d layers, prefix tokens %s, unequal before %d, after %d "
                    "(total copies %d, page pairs %d, unequal before %d, after %d) invariants %s",
                    len(copies), len(views), ntoks, unequal_before, unequal_after, *st, vs)

    def _remove_request(self, req_id: str) -> bool:
""")

if "\nimport os\n" not in src:
    src = sub(src, "import functools\n", "import functools\nimport os\n")
if "\nimport random\n" not in src:
    src = sub(src, "import functools\n", "import functools\nimport random\n")

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
