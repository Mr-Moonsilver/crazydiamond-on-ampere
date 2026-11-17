#!/usr/bin/env python3
"""Alpha M3 receipts (CPU). Leaves are declared in harness/alpha_m3_fixture_leaves.json."""
import ast
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
sys.path.insert(0, str(ROOT / 'harness'))
sys.path.insert(0, str(ROOT / 'tools'))
import alpha_m3_queue_deltas as deltas  # noqa: E402
import native43_allocator_phase_deltas as tool  # noqa: E402
import qsa_driver  # noqa: E402
import qsa_operation as qo  # noqa: E402
import worker_bridge  # noqa: E402

REFERENCE = ROOT / 'reference/vllm-src'
INDEXER = REFERENCE / 'models/qwen4_exp/nvidia/ops/qsa_indexer.py'
PREFIX = '(Worker_PP0_TP0_EP0 pid=96) INFO 11-01 14:39:45 [worker_bridge.py:625] '


class Sized:
    """A shape-only tensor: the reference's sizing reads .shape, .dtype and slices."""
    dtype = 'bf16'

    def __init__(self, *shape):
        self.shape = shape
        self.device = 'cuda:0'

    def __getitem__(self, key):
        return self


class SizingAgainstTheReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not INDEXER.is_file():
            raise RuntimeError('required reference absent; sizing proof NOT_RUN')
        tree = ast.parse(INDEXER.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'qsa_select_paged_prefill')
        node = ast.Module(body=[node], type_ignores=[])
        cls.code = compile(node, str(INDEXER), 'exec')

    def reference(self, rows, max_seq_len, ratio, columns, page_size, megabytes):
        calls = []

        def logits(q, k, table, starts, visible, max_query, width, query_offset, num_queries):
            calls.append((num_queries, width))
            return Sized(num_queries, width)
        names = dict(
            torch=SimpleNamespace(Tensor=object, empty=lambda *a, **k: Sized(1), uint8='uint8'),
            triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
            envs=SimpleNamespace(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=megabytes),
            _TOPK_WORKSPACE_BYTES=1 << 20, _prefill_logits=logits, _topk=lambda *a: None)
        exec(self.code, names)
        topk = 2048
        names['qsa_select_paged_prefill'](
            Sized(rows, 8, 64), Sized(100, page_size, 1, 64), Sized(1, columns), Sized(2), Sized(rows), topk, ratio,
            rows, Sized(rows, topk // ratio), max_seq_len)
        return calls

    def test_the_formula_matches_the_reference_on_every_case(self):
        cases = dict(typical=(1024, 100000, 8, 4096, 64, 512),
                     history_200k=(1024, 200000, 8, 4096, 64, 512),
                     short_history_floor=(1024, 10, 8, 4096, 64, 512),
                     table_clamp=(1024, 10_000_000, 8, 10, 64, 512),
                     chunked_by_the_bound=(4096, 200000, 8, 4096, 64, 8),
                     smaller_ratio=(7, 262144, 4, 4096, 64, 512),
                     one_row_per_chunk=(5, 262144, 1, 4096, 64, 1))
        chunked = 0
        for name, case in cases.items():
            with self.subTest(case=name):
                calls = self.reference(*case)
                width, _, chunks, peak, total = qo.indexer_score_shape(*case)
                self.assertEqual({w for _, w in calls}, {width})
                self.assertEqual(chunks, len(calls))
                self.assertEqual(peak, max(n for n, _ in calls) * width * 4)
                self.assertEqual(total, sum(n for n, _ in calls) * width * 4)
                chunked += len(calls) > 1
        self.assertGreaterEqual(chunked, 2)

    def test_the_allocation_is_the_float32_tensor_of_those_shapes(self):
        text = INDEXER.read_text()
        self.assertIn('logits = torch.empty(\n        (num_queries, logits_width), dtype=torch.float32, device=q.device\n', text)
        envs = (REFERENCE / 'envs.py').read_text()
        self.assertIn('"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": lambda: int(\n        os.getenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")', envs)


class DriverCallSite(unittest.TestCase):
    def call(self, *, num_tokens=1024, decode=0, skip=False, broken=False):
        compressed = SimpleNamespace(num_decode_tokens=decode, max_seq_len=100000,
                                     block_table=SimpleNamespace(shape=(1, 4096)))
        cache = SimpleNamespace(prefix='side.c', kv_cache=SimpleNamespace(shape=(100, 64, 1, 64)))
        attn = SimpleNamespace(indexer=SimpleNamespace(skip_topk=skip, compress_ratio=8, compressed_key_cache=cache))
        metadata = [{} if broken else {'side.c': compressed}]
        operation = SimpleNamespace(stats=dict(indexer_layers=0, indexer_rows=0, indexer_width=0, indexer_chunks=0,
                                               indexer_peak_bytes=0, indexer_total_bytes=0, indexer_errors=0))
        operation.note_indexer_scores = lambda *a: qo.QsaPrefillOperation.note_indexer_scores(operation, *a)
        qsa_driver._note_indexer_scores(operation, lambda: SimpleNamespace(attn_metadata=metadata), attn, num_tokens)
        return operation.stats

    def test_prefill_rows_are_sized_from_the_compressed_metadata(self):
        stats = self.call(num_tokens=1024, decode=24)
        width = qo.indexer_score_shape(1000, 100000, 8, 4096, 64, 512)[0]
        self.assertEqual((stats['indexer_layers'], stats['indexer_rows'], stats['indexer_width']), (1, 1000, width))
        self.assertEqual(stats['indexer_total_bytes'], 1000 * width * 4)

    def test_no_scores_means_no_row_and_a_failure_is_counted_not_raised(self):
        self.assertEqual(self.call(skip=True)['indexer_layers'], 0)
        self.assertEqual(self.call(num_tokens=24, decode=24)['indexer_layers'], 0)
        stats = self.call(broken=True)
        self.assertEqual((stats['indexer_layers'], stats['indexer_errors']), (0, 1))


class FakeCuda:
    def __init__(self, **counters):
        self.counters = counters

    def is_initialized(self):
        return True

    def memory_stats(self):
        return {'num_' + name: value for name, value in self.counters.items()}

    def mem_get_info(self):
        return 300, 24 * 1024


class RealFormatsThroughTheTool(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='alpha-m3-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def memory_line(self, wall, point='open', rank=0, **counters):
        bridge = SimpleNamespace(rank=rank, _memory_cuda=FakeCuda(**counters), memory_rows=[], memory_receipt_errors=0)
        tx = SimpleNamespace(txn_seq=1, chunk_start=0, chunk_end=1024)
        with self.assertLogs('worker_bridge', level='INFO') as logs:
            worker_bridge.NativeWorkerBridge._memory_receipt(bridge, point, tx)
        self.assertEqual(bridge.memory_receipt_errors, 0)
        return wall + ' ' + PREFIX + logs.records[0].getMessage()

    def alloc_line(self, wall, rank=0, **stats):
        row = dict(union_scratch_sets=6, union_scratch_retired=5, union_scratch_bytes=10_000_000,
                   union_scratch_live_max_bytes=20_000_000, indexer_layers=6, indexer_rows=6144, indexer_width=12544,
                   indexer_chunks=6, indexer_peak_bytes=51_380_224, indexer_total_bytes=308_281_344, indexer_errors=0,
                   alloc_receipt_errors=0)
        row.update(stats)
        op = SimpleNamespace(stats=row, transaction=SimpleNamespace(txn_seq=1, chunk_start=0, chunk_end=1024),
                             lane=SimpleNamespace(worker_rank=rank), union_retire=True)
        with self.assertLogs('qsa_operation', level='INFO') as logs:
            qo.QsaPrefillOperation._alloc_receipt(op)
        self.assertEqual(row['alloc_receipt_errors'], 0)
        return wall + ' ' + PREFIX + logs.records[0].getMessage()

    def run_tool(self, lines, phases=()):
        stream = self.root / 'engine.log'
        stream.write_text('\n'.join(lines) + '\n')
        for name, started, finished in phases:
            (self.root / ('engine-watch-%s.json' % name)).write_text(json.dumps(dict(started=started, finished=finished)))
        out = self.root / 'deltas.json'
        rc = tool.main(['--stream', str(stream), '--watch-dir', str(self.root), '--out', str(out)])
        return rc, (json.loads(out.read_text()) if out.exists() else None)

    def test_deltas_per_phase_from_the_real_line_formats(self):
        def wall(second):
            return '2026-11-02T01:00:%02d.5Z' % second

        def memory(second, retries, syncs, allocs):
            return self.memory_line(wall(second), alloc_retries=retries, sync_all_streams=syncs, ooms=0,
                                    device_alloc=allocs, device_free=allocs - 1)
        lines = [memory(1, 5, 1, 100), memory(3, 7, 2, 110), self.alloc_line(wall(3)),
                 memory(10, 8, 2, 120), memory(12, 20, 9, 150), self.alloc_line(wall(12), indexer_width=20000),
                 'garbage line', '2026-11-02T01:00:13Z (x) INFO flash-s5 memory point=open norank=1']
        rc, record = self.run_tool(lines, [('row1', '2026-11-02T01:00:00Z', '2026-11-02T01:00:05Z'),
                                           ('lifecycle', '2026-11-02T01:00:09Z', '2026-11-02T01:00:20Z'),
                                           ('empty', '2026-11-02T01:00:30Z', '2026-11-02T01:00:31Z')])
        self.assertEqual(rc, 0)
        self.assertEqual(record['verdict'], 'RECEIPT')
        phases = {p['name']: p['ranks']['0'] for p in record['phases']}
        self.assertEqual(set(phases), {'whole-run', 'row1', 'lifecycle', 'empty'})
        first = phases['row1']['counters']
        self.assertEqual((first['baseline'], first['alloc_retries'], first['sync_all_streams'], first['device_allocs']),
                         ('first-in-window', 2, 1, 10))
        # The later phase starts from the last line before its window.
        later = phases['lifecycle']['counters']
        self.assertEqual((later['baseline'], later['alloc_retries'], later['sync_all_streams'], later['device_frees']),
                         ('before-window', 13, 7, 40))
        self.assertEqual(phases['lifecycle']['allocations']['indexer_width_max'], 20000)
        self.assertEqual(phases['lifecycle']['allocations']['union_live_max'], 20_000_000)
        self.assertEqual(phases['empty'], dict(counters=dict(lines=0), allocations=dict(chunks=0)))
        self.assertEqual(phases['whole-run']['counters']['alloc_retries'], 15)
        self.assertEqual(phases['whole-run']['allocations']['chunks'], 2)

    def test_line_written_by_held_f_keeps_its_counters_and_marks_the_new_ones_unknown(self):
        old = ('2026-11-01T14:39:45.188560889Z ' + PREFIX + 'flash-s5 memory point=open rank=0 txn_seq=%d chunk=0-1024 '
               'allocated=22696461824 reserved=23337107456 inactive_split=124746240 alloc_retries=%d ooms=0 free=69382 total=25769803776')
        rc, record = self.run_tool([old % (1, 5), old.replace('14:39:45.188560889Z', '14:39:50Z', 1) % (2, 9)])
        self.assertEqual(rc, 0)
        counters = record['phases'][0]['ranks']['0']['counters']
        self.assertEqual((counters['alloc_retries'], counters['sync_all_streams']), (4, None))

    def test_torch_without_a_counter_reports_unknown_not_zero(self):
        lines = [self.memory_line('2026-11-02T01:00:01Z', alloc_retries=1), self.memory_line('2026-11-02T01:00:02Z', alloc_retries=2)]
        rc, record = self.run_tool(lines)
        self.assertEqual(rc, 0)
        counters = record['phases'][0]['ranks']['0']['counters']
        self.assertEqual((counters['alloc_retries'], counters['sync_all_streams'], counters['device_allocs']), (1, None, None))

    def test_an_unreadable_watch_record_is_named_and_no_lines_exits_2(self):
        (self.root / 'engine-watch-bad.json').write_text('{')
        rc, record = self.run_tool(['no receipts here'])
        self.assertEqual((rc, record['lines'], record['phases']), (2, 0, []))
        self.assertEqual(record['invalid_watch_records'], [dict(name='bad', error='JSONDecodeError')])
        missing = tool.main(['--stream', str(self.root / 'absent.log'), '--out', str(self.root / 'x.json')])
        self.assertEqual(missing, 2)
        self.assertFalse((self.root / 'x.json').exists())

    def test_the_memory_line_stays_bounded(self):
        line = self.memory_line('2026-11-02T01:00:01Z', alloc_retries=10**6, sync_all_streams=10**6,
                                device_alloc=10**9, device_free=10**9)
        self.assertLess(len(line.split(PREFIX)[1]), 400)


class QueueDeltas(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='alpha-m3-q-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_the_deltas_apply_once_on_the_l_f_texts_and_parse(self):
        runs = ROOT / 'harness/runs'
        draft = deltas.apply_draft((runs / 'q18a_native43_lifecycle_f.draft.sh').read_text())
        declare = deltas.apply_declare((runs / 'q18a_native43_lifecycle_f_declare.sh').read_text())
        self.assertEqual(draft.count('native43_allocator_phase_deltas.py'), 2)
        self.assertEqual(declare.count('native43_allocator_phase_deltas.py'), 1)
        for name, text in (('draft.sh', draft), ('declare.sh', declare)):
            path = self.root / name
            path.write_text(text)
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr[:300])

    def test_a_missing_anchor_is_refused(self):
        with self.assertRaises(ValueError):
            deltas.apply_draft('no anchors')
        with self.assertRaises(ValueError):
            deltas.apply_declare('no anchors')

    def test_the_receipt_step_runs_on_a_shell_snippet_and_cannot_fail_the_queue(self):
        snippet = deltas.M3_NEW.split('\n', 1)[1]
        harness = self.root / 'h'
        harness.mkdir()
        (harness / 'native43_allocator_phase_deltas.py').write_text('raise SystemExit(7)\n')
        script = 'set -u\nH=%s\nQ=%s\nlrc=0\n%secho reached\n' % (harness, self.root, snippet)
        result = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
        self.assertEqual((result.returncode, result.stdout.strip()), (0, 'reached'))
        self.assertIn('allocator-phase-deltas rc=7', (self.root / 'allocator-phase-deltas.log').read_text())
        unset = 'set -u\nQ=%s\nlrc=0\n%secho reached\n' % (self.root, snippet)
        result = subprocess.run(['bash', '-c', unset], capture_output=True, text=True)
        self.assertEqual((result.returncode, result.stdout.strip()), (0, 'reached'))


if __name__ == '__main__':
    logging.disable(logging.NOTSET)
    unittest.main()
