"""append-pieces lever A on the real stock KV cache manager (pinned container only; the hub has no vllm).

Stock's KVCacheManager (hybrid coordinator: one align-mode Mamba group and one full-attention group, block 4048, hash unit
176, retention 0 as the box's default, or dense), stock's Scheduler._mamba_block_aligned_split bound to a host carrying
the production split fields, and the real PrefillAdmissionController's _aligned_split and commit-side guard.  Turns run
as the native lane does: each piece planned by staged_chunk_plan from its start (admit's plan), allocated with
delay_cache_blocks, then commit's guard and cache_blocks; stock runs the handback rows and a short reply; the request
finishes.  A shadow of the align kernels tracks the state each physical Mamba block holds: the CoW copies, the pre-copy
from the running block into the window block, the write at the piece's end; a piece must resume from the state after
exactly its start.  Checked after every publication: each hashed Mamba block holds the state its key names.

Readings: pieces per turn, the next turn's hit position (flag on equals off), the shadow invariant (on and off hold it; the
mutant without the guard breaks it), and a sibling diverging between the crossed boundary and the tail.
"""
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'patches/s5_engine')]

try:
    import torch  # noqa: F401
    from vllm.v1.kv_cache_interface import MambaSpec, FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, get_group_id
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    STACK = True
except Exception:          # the hub: no vllm
    STACK = False

import scheduler_hooks as sh  # noqa: E402

BLOCK, UNIT, CAP, FLOOR, BUDGET = 4048, 176, 2048, 64, 1024
ENV = sh.APPEND_NO_BLOCK_STOP_ENV


def cdiv(a, b):
    return -(-a // b)


class Stack:
    """One stock manager, the split host, a real controller and the shadow of the Mamba state slots."""

    def __init__(self, flag, retention=0, guard=True, blocks=96):
        m = MambaSpec(block_size=BLOCK, shapes=((4,),), dtypes=(torch.float32,), mamba_type=Kind.GDN_ATTN,
                      mamba_cache_mode='align')
        a = FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
        cfg = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], prefix_cache_retention_interval=retention,
                            kv_cache_groups=[KVCacheGroupSpec(['m'], m), KVCacheGroupSpec(['a'], a)])
        self.manager = KVCacheManager(cfg, max_model_len=65536, scheduler_block_size=BLOCK, hash_block_size=UNIT,
                                      enable_caching=True)
        assert self.manager.coordinator.enable_partial_hash_hits
        self.mamba = self.manager.coordinator.single_type_managers[0]
        assert self.mamba.mamba_cache_mode == 'align'
        self.host = NS(cache_config=NS(block_size=BLOCK), hash_block_size=UNIT, use_eagle_block_drop=False,
                       mamba_has_prefill_checkpoint_blocks=False, mamba_prefill_checkpoint_alignment=None,
                       mamba_partial_cache_hit=True, mamba_fine_grained_prefix_cache=True,
                       max_num_scheduled_tokens=BUDGET, scheduler_config=NS(long_prefill_token_threshold=0),
                       kv_cache_manager=self.manager)
        self.host._mamba_block_aligned_split = types.MethodType(Scheduler._mamba_block_aligned_split, self.host)
        with mock.patch.dict(os.environ, {ENV: flag}):
            self.controller = sh.PrefillAdmissionController(
                self.host, leases=sh.SchedulerLeases(), frontier=__import__('async_handoff').CommittedFrontier(),
                engine_epoch=3, chunk_tokens=CAP, ranks=(0, 1))
        self.guard = guard
        hash_fn = get_hash_fn_by_name('sha256')
        init_none_hash(hash_fn)
        self.hasher = get_request_block_hasher(UNIT, hash_fn)
        self.shadow = {}            # physical Mamba block id -> tokens its state is after (None: never written)
        self.violations = []
        self.seq = 0

    def request(self, tokens):
        self.seq += 1
        return Request('r%d' % self.seq, list(tokens), SamplingParams(max_tokens=64), None, block_hasher=self.hasher)

    # -- the align kernels, shadowed -----------------------------------------------------------------------------------
    def _forward(self, request, computed, rows, new_ids):
        for bid in new_ids:
            self.shadow[bid] = None
        copies, retained = self.manager.take_kv_cache_block_copies()
        for copy in copies:                                   # queued CoW copies run before the forward
            self.shadow[copy.dst_block_id] = self.shadow.get(copy.src_block_id)
        self.manager.block_pool.free_blocks(retained)
        blocks = self.mamba.req_to_blocks[request.request_id]
        src = cdiv(computed, BLOCK) - 1
        dst = cdiv(computed + rows, BLOCK) - 1
        if src >= 0 and src != dst:                           # pre-copy: running block -> window block
            self.shadow[blocks[dst].block_id] = self.shadow.get(blocks[src].block_id)
        if computed and self.shadow.get(blocks[dst].block_id) != computed:
            self.violations.append(('resumed from a wrong state', request.request_id, computed,
                                    self.shadow.get(blocks[dst].block_id)))
        self.shadow[blocks[dst].block_id] = computed + rows

    def check_published(self, where):
        group = 0
        for block in self.manager.block_pool.blocks:
            key = block.block_hash
            if key is None or block.is_null or get_group_id(key) != group:
                continue
            if self.shadow.get(block.block_id) != block.block_hash_num_tokens:
                self.violations.append(('hashed state differs from its key', where, block.block_id,
                                        block.block_hash_num_tokens, self.shadow.get(block.block_id)))

    def _ids(self, allocated):
        return [b.block_id for group in allocated.blocks for b in group] if allocated is not None else []

    # -- one turn -------------------------------------------------------------------------------------------------------
    def native_turn(self, request, reply=3):
        """Native pieces to the native end (one-row handback, or r rows on a geometry tail), stock's rest, a reply."""
        manager, c = self.manager, self.controller
        manager.new_step_starts()
        blocks, hit, _junction = manager.get_computed_blocks(request)
        P = request.num_tokens
        r = P % UNIT
        native_end = P - (r if 2 <= r <= FLOOR else 1)
        pieces, position, first = [], hit, True
        while position < native_end:
            def split(at, limit, start=position, first=first):
                limit = sh.bounded_limit((97152, 2024), at, limit)
                if at == start:
                    return c._aligned_split(self.host, request, limit,
                                            num_new_local_computed_tokens=hit if first else 0)
                return c._aligned_split(self.host, sh._ProjectedPrefillRequest(request, at), limit)
            plan = sh.staged_chunk_plan(position, native_end, CAP, FLOOR, split)
            assert plan, (position, native_end)
            rows = plan[0]
            manager.new_step_starts()                          # schedule()'s per-step reset; each piece is a step
            allocated = manager.allocate_slots(request, rows, num_new_computed_tokens=hit if first else 0,
                                               new_computed_blocks=blocks if first else None, delay_cache_blocks=True)
            assert allocated is not None, ('allocation', position, rows, hit, P, manager.block_pool.get_num_free_blocks())
            self._forward(request, position, rows, self._ids(allocated))
            end = position + rows
            txn = NS(chunk_start=position, chunk_end=end)
            if c.append_no_block_stop and self.guard:          # commit's order: the guard, then cache_blocks
                fused = c._append_unpublish_stale(request, txn)
                c._append_note_commit(request, txn, end == native_end, fused)
            manager.cache_blocks(request, end)
            request.num_computed_tokens = end
            self.check_published('piece %d-%d' % (position, end))
            pieces.append((position, end))
            position, first = end, False
        left = P - request.num_computed_tokens               # stock: the handback rows, then the reply tokens
        manager.new_step_starts()
        allocated = manager.allocate_slots(request, left)
        self._forward(request, request.num_computed_tokens, left, self._ids(allocated))
        request.num_computed_tokens += left
        for i in range(reply):
            request.append_output_token_ids(900000 + self.seq * 100 + i)
            manager.new_step_starts()
            allocated = manager.allocate_slots(request, 1)
            self._forward(request, request.num_computed_tokens, 1, self._ids(allocated))
            request.num_computed_tokens += 1
            self.check_published('reply')
        manager.free(request)
        self.check_published('free')
        return hit, pieces


def conversation(start_tokens, appends):
    """Token lists of a linear conversation: each turn is the previous prompt, its 3-token reply, then new rows."""
    turns, prompt, seq = [], list(range(1, start_tokens + 1)), 0
    turns.append(list(prompt))
    for new in appends:
        seq += 1
        prompt = prompt + [900000 + seq * 100 + i for i in range(3)] + [500000 + seq * 10000 + i for i in range(new)]
        turns.append(list(prompt))
    return turns


@unittest.skipUnless(STACK, 'pinned container only (stock vllm)')
class Conversation(unittest.TestCase):
    """A fresh turn then four appends sized like V1's (1.6k-4.4k new rows), run with the flag off and on."""
    APPENDS = (2600, 3300, 1700, 4400)

    def run_conversation(self, flag, retention=0, guard=True):
        stack = Stack(flag, retention=retention, guard=guard)
        out = []
        prompts = conversation(13000, self.APPENDS)
        # the reply ids are the conversation's: the turn's request appends them, the next prompt repeats them
        for i, tokens in enumerate(prompts):
            request = stack.request(tokens)
            stack.seq = i + 1  # reply ids of turn i are conversation()'s seq i + 1 in the next prompt
            out.append(stack.native_turn(request))
        return stack, out

    def test_on_and_off_hold_the_invariant_and_hit_the_same_positions(self):
        for retention in (0, None):
            with self.subTest(retention=retention):
                off, turns_off = self.run_conversation('0', retention)
                on, turns_on = self.run_conversation('1', retention)
                self.assertEqual(off.violations, [])
                self.assertEqual(on.violations, [])
                self.assertEqual([hit for hit, _ in turns_on], [hit for hit, _ in turns_off])
                # every append resumes at the previous prompt's last hash boundary
                prompts = conversation(13000, self.APPENDS)
                for i, (hit, _) in enumerate(turns_on[1:], 1):
                    self.assertEqual(hit, len(prompts[i - 1]) // UNIT * UNIT)
                n_off = sum(len(p) for _, p in turns_off[1:])
                n_on = sum(len(p) for _, p in turns_on[1:])
                self.assertLess(n_on, n_off)
                a = on.controller.append_pieces
                self.assertGreaterEqual(a['fused_pieces'], n_off - n_on)     # a fused piece can move the next cut too
                self.assertEqual(a['appends'], len(self.APPENDS) + (1 if 13000 <= 8192 else 0))
                self.assertGreaterEqual(a['stale_states_unpublished'], 1)
                print('retention', retention, 'pieces off', [len(p) for _, p in turns_off],
                      'on', [len(p) for _, p in turns_on], 'hits', [h for h, _ in turns_on], 'block', a)

    def test_mutant_without_the_guard_publishes_a_wrong_state(self):
        stack, _turns = self.run_conversation('1', 0, guard=False)
        kinds = {v[0] for v in stack.violations}
        self.assertIn('hashed state differs from its key', kinds)

    def test_a_sibling_diverging_past_the_crossed_boundary_resumes_from_a_right_state(self):
        # a sibling shares the last append's prompt through the crossed block boundary, then differs: its hit is the
        # block boundary state off (stock materialised it) and an earlier, right state on; never the stale block
        for flag in ('0', '1'):
            with self.subTest(flag=flag):
                stack, turns = self.run_conversation(flag, 0)
                prompts = conversation(13000, self.APPENDS)
                last = prompts[-1]
                start = turns[-1][0]
                crossed = (start // BLOCK + 1) * BLOCK
                self.assertLess(crossed, len(last) // UNIT * UNIT)
                sibling = stack.request(last[:crossed + 50] + [7] * 500)
                stack.manager.new_step_starts()
                _blocks, hit, _ = stack.manager.get_computed_blocks(sibling)
                print('flag', flag, 'sibling hit', hit, 'crossed', crossed, 'previous tail', start)
                if flag == '0':
                    self.assertEqual(hit, crossed)
                else:
                    self.assertLess(hit, crossed)
                    self.assertLessEqual(hit, start)        # at most the previous turn's tail
                stack.native_turn(sibling)
                self.assertEqual(stack.violations, [])


if __name__ == '__main__':
    unittest.main()
