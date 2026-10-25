"""e2_state_grade: startup geometry and drained-state rules on synthetic engine log lines (CPU only)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_state_grade as g

RANKS = g.RANKS


def startup_line(rank, **over):
    pp = int(rank[9])
    d = dict(pp=pp, local_qsa_layers=[f'model.language_model.layers.{i}.self_attn.attn' for i in range(3 + 24 * pp, 24 + 24 * pp, 4)],
             local_qsa_groups={f'model.language_model.layers.{i}.self_attn.attn': k for k, i in enumerate(range(3 + 24 * pp, 24 + 24 * pp, 4))},
             pool_blocks=1340, page_tokens=4048, arena_pages=24, arena_budget_bytes=100663296, baseline_arena_bytes=99483648,
             incremental_kv_bytes=0, host_sources_bytes=6 * 1340 * 4048 * 1024, dma_env_enabled=True, off_file='/work/e2-arena.off',
             exactness_enabled=False, reserve_bytes=536870912)
    d.update(over)
    return f'2026-10-25T13:00:00Z (Worker_{rank[7:]} pid=7) INFO startup.py:198] FLASH_E2_STARTUP ' + json.dumps(d, sort_keys=True)


def state_line(rank, **over):
    d = dict(state='idle', active_owners=0, source_leases=0, pending_dma=0, pending_consumers=0, valid_arena_pages=0, arena_epoch=3,
             registered_requests=0, quarantined=False, incremental_kv_bytes=0, reused_arena_bytes=99483648, rpc_sequence=120,
             rpc_active=False, pending_producer_pages=2, metadata_pages=9, failed=False, cow_pairs=0, cow_views=0, dma={}, routes_by_layer={})
    d.update(over)
    return f'2026-10-25T13:01:00Z ({rank} pid=7) INFO runner_lifecycle.py:80] FLASH_E2_STATE ' + json.dumps(d, sort_keys=True)


class StartupTests(unittest.TestCase):
    def test_four_workers_pass_and_verify_flag_matches(self):
        lines = [startup_line(r) for r in RANKS]
        self.assertEqual(g.startup(lines)['verdict'], 'PASS')
        self.assertEqual(g.startup(lines, verify=0)['verdict'], 'PASS')
        self.assertEqual(g.startup(lines, verify=1)['verdict'], 'FAIL')

    def test_missing_worker_inconclusive_duplicate_or_geometry_fail(self):
        self.assertEqual(g.startup([startup_line(r) for r in RANKS[:3]])['verdict'], 'INCONCLUSIVE')
        self.assertEqual(g.startup([startup_line(r) for r in RANKS] + [startup_line(RANKS[0])])['verdict'], 'FAIL')
        for over in (dict(arena_pages=23), dict(incremental_kv_bytes=4096), dict(baseline_arena_bytes=95338496),
                     dict(local_qsa_layers=[f'model.layers.{i}.attn' for i in range(2, 24, 4)]), dict(pp=1)):
            lines = [startup_line(RANKS[0], **over)] + [startup_line(r) for r in RANKS[1:]]
            self.assertEqual(g.startup(lines)['verdict'], 'FAIL', over)

    def test_malformed_or_unmarked_record_fails(self):
        lines = [startup_line(r) for r in RANKS[1:]] + ['(Worker_PP0_TP0 pid=7) FLASH_E2_STARTUP {not json']
        self.assertEqual(g.startup(lines)['verdict'], 'FAIL')
        lines = [startup_line(r) for r in RANKS] + ['INFO FLASH_E2_STARTUP {"pp": 0}']
        self.assertEqual(g.startup(lines)['verdict'], 'FAIL')


class DrainedTests(unittest.TestCase):
    def test_last_record_decides(self):
        busy = [state_line(r, state='consuming', active_owners=1, valid_arena_pages=24, rpc_active=True) for r in RANKS]
        idle = [state_line(r) for r in RANKS]
        self.assertEqual(g.drained(busy + idle)['verdict'], 'PASS')
        self.assertEqual(g.drained(idle + busy)['verdict'], 'FAIL')
        self.assertEqual(g.drained(idle[:3])['verdict'], 'INCONCLUSIVE')
        self.assertEqual(g.drained([])['verdict'], 'INCONCLUSIVE')

    def test_pending_producer_and_metadata_pages_are_reported_not_required(self):
        out = g.drained([state_line(r, pending_producer_pages=5, metadata_pages=40) for r in RANKS])
        self.assertEqual(out['verdict'], 'PASS')
        self.assertEqual(out['workers'][RANKS[0]]['pending_producer_pages'], 5)

    def test_quarantine_failure_or_leases_fail(self):
        for over in (dict(quarantined=True, state='quarantined'), dict(failed=True), dict(source_leases=1), dict(pending_dma=1),
                     dict(registered_requests=1), dict(rpc_active=True)):
            lines = [state_line(RANKS[0], **over)] + [state_line(r) for r in RANKS[1:]]
            self.assertEqual(g.drained(lines)['verdict'], 'FAIL', over)


if __name__ == '__main__':
    unittest.main()
