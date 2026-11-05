"""Decode-phase dp5: native chunks above 1024 rows, FLASH_S5_CHUNK_TOKENS. CPU only: planning, sizing, funding. No model and no GPU; what a real 2048-row forward does is the box smoke's."""
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness')]
import inwait_arms as T
import lane_state
import model_hooks as mh
import scheduler_hooks as sh
import test_e3_lane_private as private_fixture
import test_e3_staged_chunk_plan as plan_fixture


def stock_split(prompt_tokens, budget):
    """The deployed _mamba_block_aligned_split at this model's blocks (4048 rows, hash 176), the budget it reads being stock's."""
    last = prompt_tokens - 1

    def split(start, limit):
        end = start + limit
        if end < last:
            aligned = end // T.MAMBA_BLOCK_ROWS * T.MAMBA_BLOCK_ROWS
            if aligned > start or T.MAMBA_BLOCK_ROWS <= budget:
                end = aligned
        stop = last // T.HASH_BLOCK_ROWS * T.HASH_BLOCK_ROWS
        if start < stop < end:
            end = stop
        return end - start
    return split


class Cap(unittest.TestCase):
    def test_at_or_below_1024_it_is_what_it_always_was(self):
        self.assertEqual(sh.native_chunk_cap(1024, 1024), 1024)
        self.assertEqual(sh.native_chunk_cap(1024, 63), 63)
        self.assertEqual(sh.native_chunk_cap(8, 1024), 8)

    def test_above_1024_stock_budget_does_not_clamp_it(self):
        self.assertEqual(sh.native_chunk_cap(2048, 1024), 2048)
        self.assertEqual(sh.native_chunk_cap(1536, 1024), 1536)

    def test_every_site_reads_it(self):
        for name, text, sites in (('scheduler_hooks', 'cap = native_chunk_cap(self.chunk_tokens, budget)', 1),
                                  ('native_auto_arm', 'cap = native_chunk_cap(c.chunk_tokens, s.max_num_scheduled_tokens)', 2),     # the plan and, since compose-5, the re-entry plan
                                  ('stage_overlap', "cap = native_chunk_cap(self.chunk_tokens, getattr(scheduler, 'max_num_scheduled_tokens', self.chunk_tokens))", 1)):
            self.assertEqual((ROOT / 'patches/s5_engine' / (name + '.py')).read_text().count(text), sites, name)


class Plan(unittest.TestCase):
    def sizes(self, prompt, cap):
        plan = sh.staged_chunk_plan(0, prompt - 1, cap, 64, stock_split(prompt, 1024))
        self.assertEqual(sum(plan), prompt - 1)
        return plan

    def test_the_count_the_smoke_expects_is_the_planners(self):
        for prompt in (65445, 131094):
            for cap in (1024, 1536, 2048):
                self.assertEqual(len(self.sizes(prompt, cap)), T.expected_chunks(prompt, cap), (prompt, cap))

    def test_the_count_at_1024_is_what_the_box_ran(self):
        self.assertEqual(T.expected_chunks(65445, 1024), 66)         # build-4c cf.g2-live-0
        self.assertEqual(T.expected_chunks(131094, 1024), 131)       # build-4c cf.g2-live-2

    def test_2048_runs_2048_and_2000_up_to_the_forced_stops(self):
        self.assertEqual(self.sizes(65445, 2048), (2048, 2000) * 16 + (528, 148))
        self.assertEqual(self.sizes(131094, 2048), (2048, 2000) * 32 + (1408, 149))


class Scratch(unittest.TestCase):
    def topk(self, chunk_tokens):
        return next(s.nbytes for s in lane_state.prefill_scratch_specs(chunk_tokens) if s.name == 'prefill_qsa_topk')

    def test_unchanged_at_1024_and_below(self):
        self.assertEqual(self.topk(1024), lane_state.TOPK_BYTES)
        self.assertEqual(self.topk(8), lane_state.TOPK_BYTES)

    def test_scales_above_it(self):
        self.assertEqual(self.topk(2048), 2 * lane_state.TOPK_BYTES)


class Funding(unittest.TestCase):
    """The private lane in proportion when the native chunk exceeds stock's bound. Refusing below 1024 stays: test_e3_lane_private's own test."""

    def fixture(self):
        class Fixture(private_fixture.PrivateBindingsTests):
            chunk_tokens = 2048
        fixture = Fixture('test_workspace_lane_must_be_the_extra_lane')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_the_lane_is_funded_in_proportion(self):
        f = self.fixture()
        other = f.discover()
        f.ws.manager.unlock()
        f.runner.max_num_tokens = 1024
        stock = f.ws.manager._current_workspaces[0]
        wanted = private_fixture.lane_private._numel(stock) * private_fixture.lane_private._element_size(stock) * 2
        self.assertGreaterEqual(other.prepare_workspace(), wanted)

    def test_equal_bound_stays_a_mirror(self):
        f = self.fixture()
        stock = f.ws.manager._current_workspaces[0]
        self.assertEqual(f.bindings._workspace_bytes, private_fixture.lane_private._numel(stock) * private_fixture.lane_private._element_size(stock))


class Admission(unittest.TestCase):
    def make(self, chunk_tokens, tokens=98312):
        ad = plan_fixture.admission
        manager = ad.Manager(cached=0)
        request = ad.Request(tokens=tokens)
        scheduler = ad.Scheduler(manager, [request])
        scheduler.max_num_scheduled_tokens = 1024
        controller = sh.PrefillAdmissionController(scheduler, leases=sh.SchedulerLeases(), frontier=ad.ah.CommittedFrontier(), engine_epoch=7,
                                                   chunk_tokens=chunk_tokens, ranks=(0, 1), minimum_stage_rows=64).attach()
        return manager, scheduler, controller

    def test_a_chunk_of_2048_is_admitted_past_stock_budget_of_1024(self):
        manager, scheduler, controller = self.make(2048)
        record = controller.admit(scheduler)
        self.assertEqual(record.chunk_tokens, 2048)
        self.assertEqual(manager.alloc_calls[0][1], 2048)

    def test_at_1024_it_stays_1024(self):
        manager, scheduler, controller = self.make(1024)
        self.assertEqual(controller.admit(scheduler).chunk_tokens, 1024)
        self.assertEqual(manager.alloc_calls[0][1], 1024)

    def test_a_prompt_under_the_chunk_goes_stock(self):
        import native_auto_arm
        self.assertIn('end - start < max(MIN_NATIVE_ROWS, c.chunk_tokens)', Path(native_auto_arm.__file__).read_text())


class Ple(unittest.TestCase):
    """The deployed path to the embedding; the pinned source and the real shape, dtype and device check run in test_dp5_ple (torch)."""

    def model(self, real=True, rows=1024):
        class Tensor:
            shape, dtype, device = (rows, 8, 160), 'bf16', 'cuda:0'
        embedding = types.SimpleNamespace(supports_prefetch=True, etp_data_parallel_size=1, _prefetch_buffer=Tensor(), _prefetch_stream=types.SimpleNamespace(cuda_stream=7))
        ple = types.SimpleNamespace(ple_embedding=types.SimpleNamespace(ngram_embedding=embedding)) if real else types.SimpleNamespace(ngram_embedding=embedding)
        return types.SimpleNamespace(layers=[types.SimpleNamespace(ple=None), types.SimpleNamespace(ple=ple)]), embedding

    def test_above_1024_the_real_path_resolves(self):
        model, embedding = self.model()
        self.assertIs(mh.ple_embedding(model, 1, 2048), embedding)
        self.assertIsNone(mh.ple_embedding(model, 0, 2048))

    def test_at_1024_nothing_changes_on_the_real_path(self):
        model, _ = self.model()
        self.assertIsNone(mh.ple_embedding(model, 1))
        self.assertIsNone(mh.ple_embedding(model, 1, 1024))

    def test_the_flag_takes_the_real_path_at_1024(self):
        model, embedding = self.model()
        self.assertIs(mh.ple_embedding(model, 1, 1024, True), embedding)

    def test_the_old_path_still_resolves_at_either(self):
        model, embedding = self.model(real=False)
        self.assertIs(mh.ple_embedding(model, 1), embedding)
        self.assertIs(mh.ple_embedding(model, 1, 2048), embedding)


if __name__ == '__main__':
    unittest.main()
