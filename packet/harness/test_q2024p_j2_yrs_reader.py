#!/usr/bin/env python3
"""Yield re-entry settle pair: the exploratory off/on reader (tools/q2024p_j2_yrs_reader.py). CPU only.

Synthetic records in fresh mkdtemp directories (left in place, never removed). The engine-log lines are the formats of a
real V1 j2 record (memory point=released/open, native-gap JSON events, copied with their field names and order kept;
values shortened) and of native_auto_arm._yrs_refusal's proof-miss line. The status block's shape is the real policy's
yrs_record() (test_append_yield.Rig). No engine, torch or box access.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent / 'tools', HERE.parent / 'patches' / 's5_engine'):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import q2024p_j2_yrs_reader as reader  # noqa: E402

T0 = 1794436591.0          # 2026-11-11T22:36:31Z


def at(s):
    whole = int(T0 + s)
    frac = ('%.9f' % ((T0 + s) - whole))[2:]
    import datetime
    return datetime.datetime.fromtimestamp(whole, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + '.' + frac + 'Z'


def mem(s, point, rank, txn):
    return ('%s flash-s5 memory point=%s rank=%d txn_seq=%d chunk=0-2048 allocated=1 reserved=1 inactive_split=1 '
            'alloc_retries=1 ooms=0 free=1 total=1 sync_all_streams=45 device_allocs=1 device_frees=1 pid=95 seq=1 sha=x\n'
            % (at(s), point, rank, txn))


def gap(s, event, **fields):
    doc = dict(pid=85, seq=1, sha='x', clock_basis='clock-monotonic', coverage_complete=True, engine_epoch=1, event=event,
               gap_seq=1, observed_ns=1, recording=False, route_generation=32, schema='s5-native-gap/1')
    doc.update(fields)
    return '%s flash-s5 native-gap %s\n' % (at(s), json.dumps(doc, separators=(',', ':')))


def miss(s, stage, field, retry):
    return ('%s flash-s5 yield-reentry proof-miss stage=%s field=%s rank=0 got=1 want=0 retry=%d bound=4 ends=0 '
            'elapsed_ms=170.2 settle=0\n' % (at(s), stage, field, retry))


def yield_episode(base, txn, refusals, key='k1'):
    """One FULL yield return as the V1 log shows it: release (4 ranks) -> commit -> disarm -> yield_full -> reentry ->
    first drain (+0.168) -> refusals x (barrier, drain) -> barrier, identity -> armed -> rank0 open of txn+1."""
    lines = [mem(base, 'released', r, txn) for r in range(4)]
    lines += [gap(base + .02, 'chunk_commit', txn=txn, request_key=key),
              gap(base + .04, 'admission_disarmed', reason='route-returned-full', request_key=key),
              gap(base + .05, 'append_yield_full', request_key=key),
              gap(base + .051, 'auto_reentry', request_key=key)]
    t = base + .051 + .168
    lines.append(gap(t, 'drain_empty', request_key=key))
    for n in range(refusals):
        lines.append(miss(t + .01, 'barrier', 'prefill-requests', n + 1))
        t += .031
        lines.append(gap(t, 'drain_empty', request_key=key))
    lines += [gap(t + .03, 'admission_armed', request_key=key), mem(t + .05, 'open', 0, txn + 1),
              gap(t + .051, 'chunk_open', txn=txn + 1, request_key=key)]
    return lines


def block(rig_record, **changes):
    value = json.loads(json.dumps(rig_record))
    for key, change in changes.items():
        value[key] = change
    return value


def real_record():
    import test_append_yield as ay
    return ay.Rig(cached={}).p.yrs_record()


def arm(label, before, after, served=(0, 0), start=T0 - 1, end=T0 + 100):
    rows = []
    for tag, b, s in (('before', before, served[0]), ('after', after, served[1])):
        ay = dict(counters=dict(handbacks_served=s), declines={} if b is None else dict(yield_reentry=b))
        rows.append(dict(label=tag, append_yield=ay))
    return dict(statuses=rows, replay_start_s=start, replay_end_s=end, complete=True, reasons=[],
                replay_config=dict(DURATION=600, WARMUP=120), replay_summary=dict(warm_turns=10, turns_finished_after_window=0))


def record(arms, log_lines, settle='0', spb='0'):
    root = Path(tempfile.mkdtemp(prefix='yrs-reader-'))
    for label, value in arms.items():
        (root / ('arm-' + label + '.json')).write_text(json.dumps(value))
    for phase in ('start', 'end'):
        env = {'FLASH_S5_YIELD_REENTRY_SETTLE': settle, 'FLASH_S5_SMALL_PIECE_BATCH': spb}
        (root / ('engine.identity-%s.json' % phase)).write_text(json.dumps(dict(identity=dict(env=env))))
    (root / 'engine-stream.log').write_text(''.join(log_lines))
    return root


class Shape(unittest.TestCase):
    def test_reader_keys_are_the_policy_record(self):
        rec = real_record()
        self.assertEqual(rec['schema'], 's5-yield-reentry/1')
        for key in reader.SCALARS:
            self.assertIs(type(rec[key]), int, key)
        for key in reader.MAPS:
            self.assertIs(type(rec[key]), dict, key)
        self.assertEqual(rec['armed_ms'], [0, 0.0, 0.0])
        self.assertEqual(set(rec) - set(reader.SCALARS) - set(reader.MAPS),
                         {'schema', 'settle_on', 'retry_bound', 'armed_ms'})

    def test_proof_miss_line_is_the_policy_format(self):
        source = (HERE.parent / 'patches/s5_engine/native_auto_arm.py').read_text()
        self.assertIn("'flash-s5 yield-reentry proof-miss stage=%s field=%s rank=%s got=%s want=%s retry=%d bound=%s '", source)
        m = reader.MISS.match(miss(0, 'identity', 'capability-cover', 3))
        self.assertEqual((m[2], m[3], m[4]), ('identity', 'capability-cover', '3'))


class Delta(unittest.TestCase):
    def test_delta_after_minus_before(self):
        rec = real_record()
        before = block(rec, refused={'barrier:prefill-requests': 8}, refused_retry={'1': 2, '2': 2, '3': 2, '4': 2},
                       retries_until_armed={'4': 2}, barrier_pass=2, armed_ms=[2, 600.0, 320.0])
        after = block(rec, refused={'barrier:prefill-requests': 12, 'barrier:live-uids': 1},
                      refused_retry={'1': 3, '2': 3, '3': 3, '4': 3, '5': 1}, retries_until_armed={'4': 3, 'exceeded': 1},
                      exceeded=1, barrier_pass=3, barrier_first_try=0, armed_ms=[3, 900.0, 330.0])
        d = reader.reentry_delta(before, after)
        self.assertEqual(d['refused'], {'barrier:live-uids': 1, 'barrier:prefill-requests': 4})
        self.assertEqual(d['refused_retry'], {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1})
        self.assertEqual(d['retries_until_armed'], {'4': 1, 'exceeded': 1})
        self.assertEqual((d['exceeded'], d['barrier_pass'], d['barrier_first_try']), (1, 1, 0))
        self.assertEqual(d['armed'], dict(n=1, mean_ms=300.0, max_ms_lifetime=330.0))
        self.assertEqual(d['settle_on'], (0, 0))
        self.assertIsNone(reader.reentry_delta(None, after))


class Record(unittest.TestCase):
    def test_off_record_reads_counters_lines_and_handoff(self):
        rec = real_record()
        after = block(rec, refused={'barrier:prefill-requests': 4}, refused_retry={'1': 1, '2': 1, '3': 1, '4': 1},
                      retries_until_armed={'4': 1}, barrier_pass=1, armed_ms=[1, 391.0, 391.0])
        log = yield_episode(1, 10, 4) + [miss(200, 'barrier', 'route', 1)]   # outside the arm window
        root = record({'n3.g2-live-0': arm('n3.g2-live-0', rec, after)}, log)
        result = reader.collect(root, 'off')
        a = result['arms']['n3.g2-live-0']
        self.assertEqual(a['yield_reentry']['refused'], {'barrier:prefill-requests': 4})
        self.assertEqual(a['proof_miss_lines'], dict(n=4, by_field={'barrier:prefill-requests': 4},
                                                     by_retry={'1': 1, '2': 1, '3': 1, '4': 1}))
        h = a['full_handoff']['yield']
        self.assertEqual(h['n'], 1)
        self.assertEqual(h['drains_per_episode'], {'5': 1})
        self.assertAlmostEqual(h['loops']['mean_ms'], 4 * 31.0, places=0)
        self.assertAlmostEqual(h['mean_ms'], 1000 * (.051 + .168 + 4 * .031 + .05), delta=1.0)
        self.assertNotIn('run', a['full_handoff'])
        self.assertFalse([n for n in result['notes'] if 'yield_reentry' in n or 'settle' in n], result['notes'])
        self.assertIn('missing-arm:sp.g2-live-0', result['notes'])

    def test_run_path_return_is_not_a_yield(self):
        rec = real_record()
        log = [mem(1, 'released', r, 5) for r in range(4)]
        log += [gap(1.02, 'chunk_commit', txn=5, request_key='a'),
                gap(1.08, 'admission_disarmed', reason='route-returned-full', request_key='a'),
                gap(1.3, 'drain_empty', request_key='b'), gap(1.35, 'admission_armed', request_key='b'),
                mem(1.37, 'open', 0, 6)]
        root = record({'n3.g2-live-0': arm('n3.g2-live-0', rec, rec)}, log)
        h = reader.collect(root, 'off')['arms']['n3.g2-live-0']['full_handoff']
        self.assertEqual(set(h), {'run'})
        self.assertEqual(h['run']['n'], 1)
        self.assertAlmostEqual(h['run']['mean_ms'], 370.0, delta=1.0)

    def test_notes_never_verdicts(self):
        rec = real_record()
        on = block(rec, settle_on=1, settle_waits=1, settled=1, settle_ticks=3, barrier_pass=1, barrier_first_try=1,
                   retries_until_armed={'0': 1})
        arms = {'n3.g2-live-0': arm('n3.g2-live-0', rec, on, served=(0, 1)),
                'n3.g2-live-1': arm('n3.g2-live-1', None, None)}
        # an off record with an on block: settle_on and settled are noted; the missing block is noted; identity mismatch noted
        result = reader.collect(record(arms, [], settle='1'), 'off')
        self.assertIn('n3-record-lacks-yield_reentry:n3.g2-live-1', result['notes'])
        self.assertIn('off-record-settled:n3.g2-live-0', result['notes'])
        self.assertIn('identity-settle-off-start-1-end-1', result['notes'])
        self.assertTrue(any(n.startswith('settle-on-off-') for n in result['notes']), result['notes'])
        self.assertEqual(result['arms']['n3.g2-live-0']['handbacks_served'], 1)
        text = reader.render([result])
        self.assertIn('settle waits=1 ticks=3 settled=1', text)
        for word in ('PASS', 'FAIL', 'verdict=', 'GO'):
            self.assertNotIn(word, text.split('-- yield re-entry', 1)[1])

    def test_on_record_without_notes(self):
        rec = block(real_record(), settle_on=1)
        on = block(rec, settle_waits=2, settled=1, settle_ticks=5, barrier_pass=1, barrier_first_try=1,
                   retries_until_armed={'0': 1})
        arms = {label: arm(label, rec, on) for label in ('n3.g2-live-0', 'n3.g2-live-1')}
        result = reader.collect(record(arms, [], settle='1', spb='1'), 'on')
        self.assertFalse([n for n in result['notes'] if 'settle' in n or 'yield_reentry' in n], result['notes'])
        self.assertEqual(result['arms']['n3.g2-live-1']['yield_reentry']['retries_until_armed'], {'0': 1})

    def test_json_out_is_fresh_only(self):
        rec = real_record()
        root = record({'n3.g2-live-0': arm('n3.g2-live-0', rec, rec)}, [])
        out = Path(tempfile.mkdtemp(prefix='yrs-reader-out-')) / 'read.json'
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reader.main(['off=' + str(root), '--json-out', str(out)]), 0)
            self.assertEqual(json.loads(out.read_text())['schema'], 'q2024p-j2-yrs-read/1')
            with self.assertRaises(FileExistsError):
                reader.main(['off=' + str(root), '--json-out', str(out)])


if __name__ == '__main__':
    unittest.main()
