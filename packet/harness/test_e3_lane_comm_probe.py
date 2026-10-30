#!/usr/bin/env python3
"""CPU-only tests for the standalone E3 lane-communicator probe and its queue.

No torch, no CUDA, no docker and no engine. The frozen verdict is driven with
synthetic phase records; the queue is parsed with ``bash -n`` and checked for its
pinned mounts, labels and final line. These tests cannot establish NCCL
behaviour, real communicator construction or GPU timing.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
QUEUE = HERE / 'runs' / 'q17_lane_comm.draft.sh'
sys.path.insert(0, str(HERE))

from e3_lane_comm_probe import (  # noqa: E402
    DEFAULT_TOKEN_COUNTS, LABELS, MATMUL_SIZE, NATIVE_PINS, PP_TOPOLOGY,
    main, parse_devices, verdict, verify_native, write_json)

WIDTH = 4096
ITERS = 20


def tp_record(rank=0, **overrides):
    cases = [{'chunk_tokens': tokens, 'hidden_width': WIDTH, 'dtype': 'bfloat16',
              'exact': True, 'timed_exact': True, 'ms': [1.0] * ITERS, 'total_ms': float(ITERS),
              'median_ms': 1.0, 'min_ms': 1.0, 'max_ms': 1.0, 'collectives': 100}
             for tokens in DEFAULT_TOKEN_COUNTS]
    overlap = {'matmul': [MATMUL_SIZE] * 3, 'dtype': 'bfloat16',
               'chunk_tokens': DEFAULT_TOKEN_COUNTS[-1], 'hidden_width': WIDTH,
               'iterations': ITERS, 'wall_ms': 30.0, 'matmul_stream_ms': 25.0,
               'allreduce_stream_ms': 24.0, 'matmul_finite': True, 'allreduce_exact': True,
               'basis': ('interleaved host submissions; lane-local spans, not '
                         'simultaneous-kernel proof')}
    scope = {'inside': {'is_lane': True, 'installed_is_lane': True,
                        'lane_collectives_delta': 1, 'stock_calls': 0, 'exact': True},
             'restored': True, 'reentrant_refused': True, 'stock_calls': 0}
    base = {'schema': 1, 'phase': 'tp', 'rank': rank, 'label': LABELS['tp'],
            'device': rank, 'status': 'ok', 'capability': [8, 6],
            'hidden_width': WIDTH, 'max_tokens': 2048, 'iterations': ITERS,
            'torch': 'mock', 'vllm': 'mock',
            'tp_world_size': 2, 'pp_world_size': 1, 'pp_channel': False,
            'stream_basis': ('explicit torch.cuda.Stream stands in for the green '
                             'prefill stream'),
            'hidden_width_basis': 'passed by the queue',
            'warm': True, 'tp_warmed': True, 'allreduce': cases,
            'overlap': overlap, 'scope': scope, 'describe': {'warmed': True},
            'cleanup': 'ok', 'errors': []}
    base.update(overrides)
    return base


def pp_record(rank=0, **overrides):
    cases = []
    for index, tokens in enumerate(DEFAULT_TOKEN_COUNTS):
        case = {'num_tokens': tokens, 'seq': index, 'stage': 0, 'released': True}
        if rank == 0:
            case.update(payload_sent=True, header_sent=True,
                        receipt_verified=True, receipt_queries=True)
        else:
            case.update(payload_equal=True, header_equal=True,
                        consumer_event_query=True, receipt_written=True, consumer_finite=True)
        cases.append(case)
    ownership = ({'consumer_event_query': True} if rank == 1 else
                 {'wrong_seq_refused': True, 'wrong_route_refused': True,
                  'wrong_kind_refused': True, 'stayed_unreleased': True,
                  'correct_release': True})
    mismatch = {'expected_seq': len(cases), 'sent_seq': len(cases) + 1,
                'txn_seq': 900 + len(cases), 'peer_txn_seq': 900 + len(cases),
                'deadlock': False, 'bounded': True}
    if rank == 0:
        mismatch.update(completed=True, outstanding_after=1)
    else:
        mismatch.update(refused=True, timed_out=False, quarantined=True,
                        elapsed_s=0.05)
    owned = 2 * (2048 * WIDTH * 2 + 64)
    base = {'schema': 1, 'phase': 'pp', 'rank': rank, 'label': LABELS['pp'],
            'device': rank, 'status': 'ok', 'capability': [8, 6],
            'hidden_width': WIDTH, 'max_tokens': 2048, 'iterations': ITERS,
            'torch': 'mock', 'vllm': 'mock',
            'tp_world_size': 1, 'pp_world_size': 2, 'topology': PP_TOPOLOGY,
            'role': 'sender' if rank == 0 else 'receiver', 'peer_rank': 1 - rank,
            'bind_before_live': True, 'warm': True, 'warmed': True,
            'cases': cases, 'ownership': ownership, 'mismatch': mismatch,
            'bytes_owned': owned, 'bytes_owned_expected': owned,
            'bytes_owned_matches': True, 'cleanup': 'ok', 'errors': []}
    base.update(overrides)
    return base


def all_records():
    return [tp_record(0), tp_record(1), pp_record(0), pp_record(1)]


class VerdictRules(unittest.TestCase):
    def assertVerdict(self, records, want):
        result = verdict(records)
        self.assertEqual(result['verdict'], want, result['reasons'])
        if want != 'PASS':
            self.assertTrue(result['reasons'])
        return result

    def test_both_phases_on_both_ranks_pass(self):
        self.assertVerdict(all_records(), 'PASS')

    def test_no_records_is_inconclusive(self):
        self.assertVerdict([], 'INCONCLUSIVE')

    def test_partial_coverage_is_inconclusive(self):
        self.assertVerdict([tp_record(0), tp_record(1)], 'INCONCLUSIVE')
        self.assertVerdict(all_records() + [tp_record(0)], 'INCONCLUSIVE')

    def test_tp_correctness_and_timing_fail(self):
        bad = tp_record()
        bad['allreduce'][0]['exact'] = False
        self.assertVerdict(all_records()[:1] + [bad] + all_records()[2:], 'FAIL')
        for value in (float('nan'), float('inf'), 0, -1, True):
            bad = tp_record()
            bad['allreduce'][1]['ms'][0] = value
            self.assertVerdict([bad], 'FAIL')
        bad = tp_record()
        bad['allreduce'][2]['ms'].pop()
        self.assertVerdict([bad], 'FAIL')
        bad = tp_record()
        del bad['allreduce'][3]['median_ms']
        self.assertVerdict([bad], 'FAIL')
        bad = tp_record()
        bad['allreduce'] = bad['allreduce'][:3]
        self.assertVerdict([bad], 'FAIL')

    def test_tp_scope_bookkeeping_fails(self):
        for field, value in (('stock_calls', 1), ('restored', False),
                             ('reentrant_refused', False)):
            bad = tp_record()
            bad['scope'][field] = value
            self.assertVerdict([bad], 'FAIL')
        bad = tp_record()
        bad['scope']['inside']['lane_collectives_delta'] = 0
        self.assertVerdict([bad], 'FAIL')
        bad = tp_record()
        bad['scope']['inside']['is_lane'] = False
        self.assertVerdict([bad], 'FAIL')
        bad = tp_record()
        bad['scope']['inside']['stock_calls'] = 2
        self.assertVerdict([bad], 'FAIL')

    def test_tp_overlap_and_warm_fail(self):
        for key in ('matmul_finite', 'allreduce_exact'):
            bad = tp_record()
            bad['overlap'][key] = False
            self.assertVerdict([bad], 'FAIL')
        bad = tp_record()
        bad['overlap']['wall_ms'] = 0
        self.assertVerdict([bad], 'FAIL')
        bad = tp_record()
        del bad['overlap']
        self.assertVerdict([bad], 'FAIL')
        bad = tp_record()
        bad['warm'] = False
        self.assertVerdict([bad], 'FAIL')

    def test_pp_round_trip_failures(self):
        mutations = [
            (1, lambda record: record['cases'][0].update(payload_equal=False)),
            (1, lambda record: record['cases'][1].update(header_equal=False)),
            (1, lambda record: record['cases'][2].update(consumer_event_query=False)),
            (1, lambda record: record['cases'][3].update(released=False)),
            (0, lambda record: record['cases'][0].update(payload_sent=False)),
            (0, lambda record: record['cases'][0].update(receipt_verified=False)),
            (0, lambda record: record['cases'][1].update(released=False)),
        ]
        for rank, mutate in mutations:
            records = all_records()
            mutate(records[2 + rank])
            self.assertVerdict(records, 'FAIL')

    def test_pp_ownership_and_mismatch_failures(self):
        for key in ('wrong_seq_refused', 'wrong_route_refused', 'wrong_kind_refused',
                    'stayed_unreleased', 'correct_release'):
            records = all_records()
            records[2]['ownership'][key] = False
            self.assertVerdict(records, 'FAIL')
        records = all_records()
        records[3]['ownership']['consumer_event_query'] = False
        self.assertVerdict(records, 'FAIL')
        for key, value in (('deadlock', True), ('bounded', False)):
            records = all_records()
            records[3]['mismatch'][key] = value
            self.assertVerdict(records, 'FAIL')
        records = all_records()
        records[3]['mismatch']['refused'] = False
        self.assertVerdict(records, 'FAIL')
        records = all_records()
        records[3]['mismatch']['quarantined'] = False
        self.assertVerdict(records, 'FAIL')
        records = all_records()
        records[2]['mismatch']['completed'] = False
        self.assertVerdict(records, 'FAIL')

    def test_pp_inventory_identity_and_roles_fail(self):
        for key, value in (('bytes_owned', 0), ('label', 'other'),
                           ('topology', 'other'), ('role', 'receiver'),
                           ('bind_before_live', False), ('warm', False),
                           ('tp_world_size', 2), ('pp_world_size', 1)):
            records = all_records()
            records[2][key] = value
            self.assertVerdict(records, 'FAIL')

    def test_hard_errors_and_cleanup_fail(self):
        for kind in ('nccl', 'protocol', 'launch', 'inventory', 'cleanup'):
            records = all_records()
            records[0]['errors'] = [{'kind': kind, 'message': 'synthetic'}]
            self.assertVerdict(records, 'FAIL')
        records = all_records()
        records[3]['cleanup'] = 'failed'
        self.assertVerdict(records, 'FAIL')

    def test_inconclusive_classes(self):
        for status in ('missing', 'timeout'):
            records = all_records()
            records[1]['status'] = status
            self.assertVerdict(records, 'INCONCLUSIVE')
        records = all_records()
        records[0]['status'] = 'interrupted'
        self.assertVerdict(records, 'INCONCLUSIVE')
        for kind in ('internal', 'import', 'config', 'capability', 'interrupted'):
            records = all_records()
            records[2]['errors'] = [{'kind': kind, 'message': 'synthetic'}]
            self.assertVerdict(records, 'INCONCLUSIVE')
        for capability in ([8, 0], None, (7, 5)):
            records = all_records()
            records[3]['capability'] = capability
            self.assertVerdict(records, 'INCONCLUSIVE')

    def test_unmeasured_rank_reports_the_real_cause_not_capability(self):
        # A rank that dies before it can measure anything must say why; the
        # absent capability is a symptom, never the whole reason.
        records = all_records()
        records[0].update(status='error', capability=None,
                          errors=[{'kind': 'import', 'message': 'vllm missing'}])
        result = self.assertVerdict(records, 'INCONCLUSIVE')
        text = ' '.join(result['reasons'])
        self.assertIn('import: vllm missing', text)
        self.assertNotIn('capability', text)
        records = all_records()
        records[1].update(status='config', capability=None,
                          errors=[{'kind': 'config',
                                   'message': 'native identity refused: inventory'}])
        result = self.assertVerdict(records, 'INCONCLUSIVE')
        self.assertIn('native identity refused', ' '.join(result['reasons']))
        # A measured rank with a recorded capability keeps the explicit reason.
        records = all_records()
        records[3]['capability'] = [8, 0]
        result = self.assertVerdict(records, 'INCONCLUSIVE')
        self.assertIn('[8, 0] != [8, 6]', ' '.join(result['reasons']))

    def test_hard_failure_outranks_inconclusive_and_coverage(self):
        records = all_records()
        records[0]['allreduce'][0]['exact'] = False
        records[1]['status'] = 'missing'
        records.pop(3)
        self.assertVerdict(records, 'FAIL')

    def test_devices_must_be_two_unique_ordinals(self):
        self.assertEqual(parse_devices('0,1'), [0, 1])
        for text in ('0', '0,1,2', '0,0', '-1,0', 'a,b'):
            with self.assertRaises((ValueError, TypeError)):
                parse_devices(text)


class ArtifactIdentity(unittest.TestCase):
    def native_copy(self, root):
        # q17c is an immutable older standalone receipt, not the current
        # decode helper. Keep its declared source instead of moving its pin.
        frozen = {'green_context.py': 'q17c_original_green_context.py',
                  'e3_types.py': 'q17c_original_e3_types.py'}
        for name in NATIVE_PINS:
            source = (HERE / 'fixtures' / frozen[name] if name in frozen
                      else HERE.parent / 'patches/s5_engine' / name)
            shutil.copyfile(source, root / name)
        (root / 'SHA256SUMS').write_text(
            ''.join(f'{sha}  {name}\n' for name, sha in sorted(NATIVE_PINS.items())))

    def test_exact_pins_extra_file_and_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.native_copy(root)
            self.assertEqual(verify_native(root), NATIVE_PINS)
            (root / 'extra.py').write_text('')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                verify_native(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.native_copy(root)
            (root / 'lane_comm.py').write_text('different')
            with self.assertRaisesRegex(ValueError, 'source pin'):
                verify_native(root)

    def test_symlink_native_source_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.native_copy(root)
            (root / 'extra.py').symlink_to(root / 'lane_comm.py')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                verify_native(root)

    def test_json_publication_does_not_clobber_a_racing_writer(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'result.json'
            original = os.link

            def race(src, dst):
                target.write_text('other writer')
                return original(src, dst)

            with patch('e3_lane_comm_probe.os.link', side_effect=race):
                with self.assertRaises(FileExistsError):
                    write_json(target, {'ok': True})
            self.assertEqual(target.read_text(), 'other writer')


class PlanAndImportIsolation(unittest.TestCase):
    def test_import_has_no_torch_or_vllm_side_effect(self):
        code = ('import sys; sys.path.insert(0, %r); import e3_lane_comm_probe; '
                'assert "torch" not in sys.modules; '
                'assert "vllm" not in sys.modules' % str(HERE))
        done = subprocess.run([sys.executable, '-c', code], capture_output=True,
                              text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_plan_is_cpu_only_and_carries_both_labels(self):
        done = subprocess.run([sys.executable, str(HERE / 'e3_lane_comm_probe.py'),
                               '--plan'], capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        plan = json.loads(done.stdout)
        self.assertEqual(plan['phases'], ['tp', 'pp'])
        self.assertEqual(plan['labels'], LABELS)
        self.assertEqual(plan['labels']['tp'], 'tp-pair-1,3-real')
        self.assertEqual(plan['labels']['pp'], 'two-card-PP-synthetic')
        self.assertEqual(plan['devices'], [0, 1])
        self.assertEqual(plan['token_counts'], list(DEFAULT_TOKEN_COUNTS))
        self.assertEqual(plan['matmul'], [4096, 4096, 4096])
        self.assertIn('init_distributed_environment', ' '.join(plan['routes']))
        self.assertIn('initialize_model_parallel', ' '.join(plan['routes']))

    def test_plan_does_not_import_torch(self):
        code = ('import sys; sys.path.insert(0, %r); '
                'import e3_lane_comm_probe as p; p.main(["--plan"])' % str(HERE))
        done = subprocess.run([sys.executable, '-c', code], capture_output=True,
                              text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn('torch', done.stderr)


class EarlyRefusal(unittest.TestCase):
    def test_native_refusal_writes_a_well_formed_four_record_summary(self):
        # The queue's grader requires phases == ['tp', 'pp'] and exactly four
        # record names; an identity refusal must still satisfy that shape.
        with tempfile.TemporaryDirectory() as temp:
            native = Path(temp) / 'native'
            native.mkdir()
            out = Path(temp) / 'out'
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                rc = main(['--native', str(native), '--out', str(out),
                           '--hidden-width', '4096'])
            self.assertEqual(rc, 2, buffer.getvalue())
            self.assertIn('E3_LANE_COMM_PROBE INCONCLUSIVE', buffer.getvalue())
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['verdict'], 'INCONCLUSIVE')
            self.assertEqual(summary['phases'], ['tp', 'pp'])
            self.assertEqual(summary['records'], [
                'tp-rank-0.json', 'tp-rank-1.json',
                'pp-rank-0.json', 'pp-rank-1.json'])
            reasons = ' '.join(summary['reasons'])
            self.assertIn('native identity refused', reasons)
            self.assertNotIn('capability', reasons)
            self.assertEqual(sorted(path.name for path in out.iterdir()),
                             ['probe.claim', 'probe.claim.tmp', 'summary.json', 'summary.json.tmp'])


class QueueScript(unittest.TestCase):
    NEEDLES = (
        'fail_early engine-active',
        'echo "INCONCLUSIVE $1"',
        'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b',
        'systemctl show vllm-flash -p ActiveState --value',
        '[[ ! -e $Q && ! -L $Q ]]',
        'timeout --signal=TERM --kill-after=30s 2700s env Q17_LANE_COMM_BOUNDED=1 bash "$0"',
        '--gpus all',
        '--ipc=host',
        '--network none',
        '-e CUDA_VISIBLE_DEVICES=1,3',
        '-v "$H:/h:ro"',
        '-v "$E3_NATIVE:/n:ro"',
        '-v "$Q:/q"',
        'PYTHONPATH=/n:/h',
        '--hidden-width "$E3_HIDDEN_WIDTH"',
        '--phase both',
        'tp-pair-1,3-real',
        'two-card-PP-synthetic',
        'Q17_LANE_COMM DONE rc=$rc engine-untouched containers-retained',
    )
    FORBIDDEN = ('docker rm', 'docker start', 'docker restart', 'systemctl start',
                 'systemctl stop', 'systemctl restart', 'systemctl enable',
                 'rm -rf', 'pkill', 'killall')

    def test_bash_syntax(self):
        done = subprocess.run(['bash', '-n', str(QUEUE)], capture_output=True,
                              text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_pinned_inputs_mounts_labels_and_final_line(self):
        text = QUEUE.read_text()
        for needle in self.NEEDLES:
            self.assertIn(needle, text)
        for forbidden in self.FORBIDDEN:
            self.assertNotIn(forbidden, text)

    def test_shellcheck_when_available(self):
        shellcheck = shutil.which('shellcheck')
        if shellcheck is None:
            self.skipTest('shellcheck not installed')
        done = subprocess.run([shellcheck, '-S', 'warning', str(QUEUE)],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_missing_input_prints_the_final_line_without_touching_the_box(self):
        env = {key: value for key, value in os.environ.items()
               if key not in ('H', 'Q', 'E3_NATIVE', 'E3_HIDDEN_WIDTH')}
        env['Q17_LANE_COMM_BOUNDED'] = '0'
        done = subprocess.run(['bash', str(QUEUE)], capture_output=True, text=True,
                              env=env, timeout=30)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn('INCONCLUSIVE missing-input:', done.stdout)
        self.assertEqual(done.stdout.strip().splitlines()[-1],
                         'Q17_LANE_COMM DONE rc=2 engine-untouched containers-retained')

    def test_summary_launch_consistency_and_abort(self):
        text = QUEUE.read_text()
        code = text.split("<<'PYGRADE'\n", 1)[1].split('\nPYGRADE', 1)[0]
        good = {'verdict': 'PASS', 'phases': ['tp', 'pp'], 'labels': dict(LABELS),
                'records': ['tp-rank-0.json', 'tp-rank-1.json',
                            'pp-rank-0.json', 'pp-rank-1.json']}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'summary.json'
            for verdict_name, raw, expected in [('PASS', 0, 0), ('PASS', 1, 2),
                                               ('PASS', 124, 2), ('FAIL', 0, 1),
                                               ('FAIL', 137, 1), ('INCONCLUSIVE', 0, 2),
                                               ('INCONCLUSIVE', 2, 2)]:
                record = dict(good, verdict=verdict_name)
                path.write_text(json.dumps(record))
                done = subprocess.run([sys.executable, '-c', code, str(path), str(raw)],
                                      capture_output=True, text=True, timeout=10)
                self.assertEqual(done.returncode, expected,
                                 (verdict_name, raw, done.stderr))
            for broken in ({}, dict(good, phases=['tp']),
                           dict(good, records=good['records'][:2]),
                           dict(good, labels={'tp': 'tp-pair-1,3-real'})):
                path.write_text(json.dumps(broken))
                done = subprocess.run([sys.executable, '-c', code, str(path), '0'],
                                      capture_output=True, text=True, timeout=10)
                self.assertEqual(done.returncode, 2, broken)



class ReviewedRuntimePaths(unittest.TestCase):
    def test_cross_rank_mismatch_identity_and_inventory(self):
        records = all_records()
        records[3]['mismatch']['txn_seq'] += 1
        self.assertEqual(verdict(records)['verdict'], 'FAIL')
        records = all_records()
        records[0]['allreduce'][0]['timed_exact'] = False
        self.assertEqual(verdict(records)['verdict'], 'FAIL')
        records = all_records()
        records[3]['bytes_owned_matches'] = False
        self.assertEqual(verdict(records)['verdict'], 'FAIL')
        records = all_records()
        records[3]['cases'][0]['consumer_finite'] = False
        self.assertEqual(verdict(records)['verdict'], 'FAIL')

    def test_worker_exit_cannot_be_hidden_by_a_good_record(self):
        from e3_lane_comm_probe import _with_process_exit
        for rc in (-11, 1, 2, None):
            records = all_records()
            records[0] = _with_process_exit(records[0], rc)
            self.assertEqual(verdict(records)['verdict'], 'INCONCLUSIVE')
        records[0]['allreduce'][0]['exact'] = False
        # Existing arithmetic evidence must survive a later abnormal exit.
        self.assertEqual(verdict(records)['verdict'], 'FAIL')

    def test_partial_timeout_not_confused_with_observed_mismatch(self):
        records = all_records()
        records[3]['cases'] = records[3]['cases'][:1]
        records[3]['errors'] = [{'kind': 'timeout', 'message': 'bounded poll'}]
        self.assertEqual(verdict(records)['verdict'], 'INCONCLUSIVE')
        records[3]['cases'][0]['payload_equal'] = False
        self.assertEqual(verdict(records)['verdict'], 'FAIL')

    def test_receipts_are_exclusive_and_rerun_refused(self):
        import e3_lane_comm_probe as p
        with tempfile.TemporaryDirectory() as tmp:
            p._write_receipt(tmp, 'pp', 0, {'status': 'ok'})
            with self.assertRaises(FileExistsError):
                p._write_receipt(tmp, 'pp', 0, {'status': 'different'})
            with self.assertRaises(SystemExit):
                p.main(['--native', tmp, '--out', tmp, '--hidden-width', '10240'])

    def test_pp_executes_real_channel_both_roles_and_records_proof(self):
        # Exercise actual _phase_pp + actual lane_comm over the existing CPU wire
        # fakes. This catches bookkeeping bugs synthetic PASS records miss.
        import types
        from unittest.mock import patch
        import e3_lane_comm_probe as p
        from test_e3_lane_comm import fake_torch, FakeStream, FakeTensor, FakeWire, FakePyNccl
        import lane_comm as lc
        import e3_types as et
        class Scalar:
            def __init__(self, value): self.value = value
            def item(self): return self.value
        width, counts = 4, p.DEFAULT_TOKEN_COUNTS
        def pattern(torch, n, w, dev, seed=0):
            return FakeTensor([(i + seed) % 200 for i in range(n * w)], (n,w))
        wire = FakeWire()
        wire.messages.append([lc.HEADER_MAGIC] * lc.HEADER_LEN)
        for seq, n in enumerate(counts):
            txn = et.TransactionId(1, p.SYNTHETIC_UID, 0, 0, n, seq)
            wire.messages += [lc.header_values(txn, 0, seq, n), pattern(None,n,width,0,seq).values]
        seq = len(counts)
        txn = et.TransactionId(1,p.SYNTHETIC_UID,0,0,1,900+seq)
        wire.messages += [lc.header_values(txn,0,seq+1,1),pattern(None,1,width,0,seq).values]
        records=[]
        with tempfile.TemporaryDirectory() as tmp, patch.object(p,'_pattern',pattern), \
                patch.object(FakeTensor,'float',lambda x:x,create=True), \
                patch.object(FakeTensor,'sum',lambda x:Scalar(sum(x.values)),create=True):
            for rank in (1,0):
                torch=fake_torch()
                torch.equal=lambda a,b:a.values==b.values
                torch.isfinite=lambda x:Scalar(True)
                group=types.SimpleNamespace(world_size=2,is_last_rank=bool(rank),rank_in_group=rank,cpu_group=object())
                tp=types.SimpleNamespace(world_size=1)
                factory=lambda **kw:FakePyNccl(rank,2,wire=wire)
                record=p._new_record('pp',rank,rank,width,2048,20)
                p._phase_pp(torch,lc,et,factory,lambda:tp,lambda:group,rank,rank,
                            FakeStream(50),tmp,width,2048,counts,1,1,record)
                record.update(status='ok',capability=[8,6],cleanup='ok')
                records.append(record)
                self.assertEqual(p._phase_pp_failures(record), [], record)
            self.assertTrue(records[0]['ownership']['consumer_event_query'])
            self.assertEqual([r['num_tokens'] for r in records[0]['cases']],list(counts))
            self.assertTrue(all(r['consumer_finite'] for r in records[0]['cases']))
            self.assertEqual(records[1]['ownership']['correct_release'],True)

    def test_invalid_bounds_rejected_before_import(self):
        for name, value in (('--phase-timeout','nan'),('--poll-timeout','inf'),
                            ('--receipt-timeout','0'),('--max-tokens','1')):
            with self.assertRaises(SystemExit):
                main(['--plan',name,value])

class WorkerConfigContext(unittest.TestCase):
    def exercise(self, phase, fail_at=None):
        import types
        from unittest.mock import patch
        import e3_lane_comm_probe as p
        expected = (2, 1) if phase == 'tp' else (1, 2)
        previous = object()
        active = [previous]
        calls = []
        class ParallelConfig:
            def __init__(self, **kw):
                self.kw = kw
                assert kw == dict(tensor_parallel_size=expected[0],
                                  pipeline_parallel_size=expected[1])
                calls.append('parallel_config')
        class VllmConfig:
            def __init__(self, *, parallel_config):
                assert isinstance(parallel_config, ParallelConfig)
                self.parallel_config = parallel_config
                calls.append('vllm_config')
        @contextlib.contextmanager
        def set_current_vllm_config(config):
            self.assertIs(active[0], previous)
            active[0] = config
            calls.append('enter')
            try:
                yield
            finally:
                calls.append('exit')
                active[0] = previous
        def step(name):
            self.assertIsInstance(active[0], VllmConfig, name)
            calls.append(name)
            if name == fail_at:
                raise RuntimeError('injected ' + name)
        def distributed(**kw):
            step('distributed')
            self.assertEqual(kw['world_size'], 2)
            self.assertEqual(kw['backend'], 'nccl')
        def model(**kw):
            step('model')
            self.assertEqual(kw, dict(tensor_model_parallel_size=expected[0],
                                     pipeline_model_parallel_size=expected[1]))
        def body(*args): step('body')
        config_module=types.ModuleType('vllm.config')
        config_module.ParallelConfig=ParallelConfig
        config_module.VllmConfig=VllmConfig
        config_module.set_current_vllm_config=set_current_vllm_config
        vllm=types.ModuleType('vllm')
        vllm.__version__='fake'
        cuda=types.SimpleNamespace(set_device=lambda device:None,
            get_device_capability=lambda device:(8,6),
            Stream=lambda **kw:object(), synchronize=lambda:step('synchronize'))
        torch=types.SimpleNamespace(cuda=cuda,__version__='fake',version=types.SimpleNamespace(cuda='fake'))
        imported=(torch,vllm,object(),distributed,model,object(),object(),
                  lambda:step('destroy_model'),lambda:step('destroy_dist'),object(),object())
        with patch.dict(sys.modules, {'vllm':vllm,'vllm.config':config_module}), \
                patch.object(p,'_imports',return_value=imported), \
                patch.object(p,'_phase_tp',side_effect=body), \
                patch.object(p,'_phase_pp',side_effect=body):
            record=p._run_rank(phase,0,'unused','unused',0,10240,2048,20,
                               p.DEFAULT_TOKEN_COUNTS,30,60)
        self.assertIs(active[0],previous)
        self.assertEqual(calls[:4], ['parallel_config','vllm_config','enter','distributed'])
        self.assertEqual(calls[-1], 'exit')
        self.assertIn('destroy_model',calls)
        self.assertIn('destroy_dist',calls)
        self.assertEqual(record['vllm_config'], {
            'construction':'VllmConfig(parallel_config=ParallelConfig(...))',
            'tensor_parallel_size':expected[0], 'pipeline_parallel_size':expected[1],
            'model_config':None,'context_entered':True,'context_closed':True})
        return record,calls

    def test_real_worker_enters_phase_config_through_cleanup(self):
        for phase in ('tp','pp'):
            with self.subTest(phase=phase):
                record,calls=self.exercise(phase)
                self.assertEqual(record['status'],'ok',record)
                self.assertEqual(record['cleanup'],'ok')
                self.assertEqual(calls[4:-1],['model','body','synchronize','destroy_model','destroy_dist'])

    def test_failed_init_or_body_restores_config_after_cleanup(self):
        for phase in ('tp','pp'):
            for where in ('distributed','model','body'):
                with self.subTest(phase=phase,where=where):
                    record,_=self.exercise(phase,where)
                    self.assertEqual(record['status'],'error')
                    self.assertEqual(record['cleanup'],'ok')
                    self.assertTrue(any('injected '+where in e['message'] for e in record['errors']))

    def test_cleanup_failure_still_exits_config_and_preserves_failure(self):
        record,calls=self.exercise('pp','destroy_model')
        self.assertEqual(record['cleanup'],'failed')
        self.assertIn('destroy_dist',calls)
        self.assertTrue(any(e['kind']=='cleanup' for e in record['errors']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
