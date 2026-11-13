#!/usr/bin/env python3
"""Busy/far decode-forward reader (tools/q2024p_j2_busyfar_reader.py; Codex's 13 Nov 15:36 recipe). CPU only.

Synthetic receipts written as engine-log lines (docker stamp + ``flash-s5 <kind> {json}``) into fresh mkdtemp
record roots (left in place, never removed). One n3 arm, TP2 x PP2: a busy execute whose bracket lies inside the
native command envelope of every rank, and a far execute 400 ms clear of every envelope, in one stratum. Each
refusal of the recipe has a case; each rule has a mutant the suite must catch (neighbour-line join, far without the
200 ms guard, sum instead of max over TP peers, null timings filled with 0).
"""
import contextlib
import datetime
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent / 'tools'):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import q2024p_j2_busyfar_reader as reader  # noqa: E402

T = 10 ** 12                      # monotonic ns origin of the synthetic run
W0 = 1794507900.0                 # wall clock of the arm start
MS = 1_000_000


def iso(wall):
    return datetime.datetime.fromtimestamp(wall, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + '000Z'


def us(ms):
    return int(ms * 1000)


class Record:
    """Builds one record root. Times in ms from T; every receipt gets pid / per-pid seq / sha like the writer."""

    def __init__(self):
        self.lines, self.rseq, self.lineno = [], {}, {}
        self.skip_rseq = set()          # (pid, seq) receipts to leave out: a lost write

    def emit(self, kind, body, wall, pid, stream=None, parts=None, raw=None):
        key = (pid,) + (stream or (kind,))
        line_no = self.lineno.get(key, 0)
        self.lineno[key] = line_no + 1
        bodies = parts or [body]
        out = []
        for i, b in enumerate(bodies):
            seq = self.rseq.get(pid, 0)
            self.rseq[pid] = seq + 1
            doc = dict(b, schema=reader.SCHEMAS[{'handler': 'handler', 'progress': 'progress', 'anatomy': 'anatomy',
                                                  'native-gap': 'gap'}[kind]])
            if kind != 'native-gap':
                doc.update(line=line_no, part=i, parts=len(bodies))
            if (pid, seq) in self.skip_rseq:
                continue
            ident = '{"pid":%d,"seq":%d,"sha":"s%d-%d",' % (pid, seq, pid, line_no)   # the writer's identity prefix
            out.append('%s flash-s5 %s %s' % (iso(wall), kind, ident + json.dumps(doc, separators=(',', ':'))[1:]))
        self.lines.extend(out)
        return out

    def write(self, arms=(('n3.g2-live-0', W0, W0 + 100),), zst=False):
        root = Path(tempfile.mkdtemp(prefix='busyfar-'))
        for name, start, end in arms:
            (root / ('arm-%s.json' % name)).write_text(json.dumps(dict(label=name, wall_start_s=start, wall_end_s=end)))
        text = '\n'.join(self.lines) + '\n'
        (root / 'engine.log').write_text(text)
        if zst:
            subprocess.run(['zstd', '-q', '--rm', str(root / 'engine.log')], check=True)
        return root


SEAMS = ['execute', 'forward', 'prep_states']
DEF = dict(D=(15.0, 10.0), H=0.2, obs=0.1)


def standard(rec=None, busy_D=15.0, far_D=10.0, far_at=2500.0, busy_at=300.0, peers=None, drop_rank_seq=(),
             null_dev=(), first_nonempty=False, far=True, env_core=(100.0, 900.0), h_ns=200_000, nulls_H=()):
    """Chunk A (txn 1): native envelope [100, 900] ms on every rank, busy execute seq 11 at busy_at. Chunk B (txn 2):
    envelope [2000, 2100] ms, far execute seq 21 at far_at. Samples 12 / 22; rank 2 returns the outputs."""
    rec = rec or Record()
    peers = peers or {}
    steps = [(10, 11, 12, busy_at, busy_D, 1)]
    if far:
        steps.append((20, 21, 22, far_at, far_D, 2))
    # engine progress chunks (pid 1)
    for (e0, e1, sm, at, _, txn) in steps:
        lo = env_core[0] if txn == 1 else 2000.0
        hi = env_core[1] if txn == 1 else 2100.0
        s_rows = [[e0, 'execute_model', us(at - 2), us(at - 1.5), us(at - 1)],
                  [e1, 'execute_model', us(at - 1), us(at - 0.5), us(at)],
                  [sm, 'sample_tokens', us(at + 20), us(at + 20.5), us(at + 21)]]
        n_rows = [[100 * txn + k, 'flash_s5_submit', 0, k, 0, k, 'o', us(lo + k), us(lo + k), None, None, None, None,
                   us(hi - k), 'ok', False] for k in range(2)]
        rec.emit('progress', dict(side='engine', rank=None, base_ns=T, txn=txn, request_generation=1,
                                  chunk_start=0 if txn == 1 else 2048, chunk_end=2048 if txn == 1 else 4096,
                                  decode_live=4, mode='burst', commit_ns=T + us(hi) * 1000, dropped=0,
                                  fields=dict(s=['seq', 'kind', 't_submit', 't_recv', 't_take'],
                                              n=['seq', 'kind', 'segment', 'op_seq', 'stage', 'layer', 'origin',
                                                 't_submit', 't_sent', 't_recv0', 't_recv1', 't_recv2', 't_recv3',
                                                 't_apply', 'outcome', 'direct']),
                                  rows=dict(s=s_rows, n=n_rows)), W0 + 10 * txn, pid=1)
    # worker progress: the native commands between the steps (so the seq space is complete) -- rank 0
    native = [x for x in range(13, 20)] + [x for x in range(100, 102)] + [x for x in range(200, 202)]
    rec.emit('progress', dict(side='worker', rank=0, base_ns=T, dropped=0,
                              fields=dict(r=['seq', 'kind', 't_enter', 't_exit', 't_emit', 'direct', 'replied']),
                              rows=dict(r=[[x, 'flash_s5_submit', us(150), us(151), us(151), True, True] for x in native])),
             W0 + 30, pid=10)
    # worker anatomy per rank and chunk
    for rank in range(4):
        for txn, (lo, hi) in ((1, env_core), (2, (2000.0, 2100.0))):
            rec.emit('anatomy', dict(side='worker', rank=rank, stage=rank // 2, base_ns=T, txn=txn, request_generation=1,
                                     chunk_start=0 if txn == 1 else 2048, chunk_end=2048 if txn == 1 else 4096,
                                     dropped=0, rows=dict(c=[[10, 10, 0, 0, us(lo), us(hi), 500, 10, 0, 10, 0, 0, 0]])),
                     W0 + 31 + txn, pid=10 + rank)
    # handler rows per rank
    for rank in range(4):
        m, a = [], []
        for (e0, e1, sm, at, D, txn) in steps:
            dev = peers.get((rank, e1), D)
            m.append([e0, 'e', int(rank >= 2), 0, us(at - 1), us(at - 0.8), 200000, 190000, 1000, None, None, None, None,
                      None, [[0, 100000, 100000, 90000, 1]] + ([[1, 1, 1, 1, 1]] if first_nonempty else [])])
            if (rank, e1) not in drop_rank_seq:
                m.append([e1, 'e', int(rank >= 2), 0, us(at), us(at + 3), 3000000, 2900000, 100000,
                          None if (rank, e1) in null_dev else int(dev * MS), None, 0, 0, 5000,
                          [[0, 3000000, 2000000, 2900000, 1]] +
                          ([[1, None if (rank, e1) in nulls_H else h_ns, h_ns, h_ns, 1]])])
            m.append([sm, 's', int(rank >= 2), 0, us(at + 21), us(at + 23), 2000000, 1900000, 2000, None, 1000000, None,
                      None, 4000, [[0, 2000000, 2000000, 1900000, 1]]])
            if rank == 2:
                a.append([e1, 0, us(at + 3), None, None, 30000, us(at + 3.1), 1000])
                a.append([sm, 0, us(at + 23), 6 * MS, 50000, 90000, us(at + 30), 2000])
            if rank == 3:
                a.append([sm, 0, us(at + 23), None, None, 30000, us(at + 23.1), 1000])
        rec.emit('handler', dict(side='worker', rank=rank, base_ns=T, head=dict(seams=SEAMS, counters=dict(rows_dropped=0)),
                                 rows=dict(m=m, a=a)), W0 + 40, pid=10 + rank)
    # native-gap: route generation 1 from before the run
    rec.emit('native-gap', dict(event='chunk_open', observed_ns=T + 50 * MS, route_generation=1, engine_epoch=1,
                                gap_seq=0, recording=True, coverage_complete=True, txn=1), W0 + 5, pid=1)
    return rec


def run(rec, **kw):
    root = rec.write(**kw)
    data = reader.read(str(root))
    items, est, meta = reader.analyse(data)
    return data, items, est, meta, reader.report(data, items, est, meta)


def item(items, rank, seq):
    return next(i for i in items if i['rank'] == rank and i['seq'] == seq)


def table(doc, rank):
    return next(t for t in doc['tables'] if t['rank'] == str(rank))


class Receipts(unittest.TestCase):
    def test_atomic_assembly_counts_and_refusals(self):
        rec = standard()
        part = dict(side='worker', rank=1, base_ns=T + 7, rows=dict(m=[]), head=dict(seams=SEAMS))
        rec.emit('handler', None, W0 + 50, pid=11, parts=[part, part])[1:]
        rec.lines.pop()                                            # part 1 of 2 lost: missing part
        good = rec.emit('handler', None, W0 + 51, pid=11, parts=[dict(part, base_ns=T + 8)])
        rec.lines.append(good[0])                                  # identical repeat: deduplicated
        bad = rec.emit('handler', None, W0 + 52, pid=11, parts=[dict(part, base_ns=T + 9)])
        rec.lines.append(bad[0].replace('"rows":{"m":[]}', '"rows":{"m":[[1]]}'))   # conflicting duplicate
        rec.lines.append(iso(W0 + 53) + ' flash-s5 handler {"pid": 11, "seq": 9' + iso(W0 + 53) + ' flash-s5 progress {')
        data, items, est, meta, doc = run(rec)
        loss = data['loss']
        self.assertEqual(loss['handler']['missing_parts'], 1)
        self.assertEqual(loss['handler']['duplicate_parts'], 1)
        self.assertEqual(loss['handler']['conflicting_duplicates'], 1)
        self.assertEqual(loss['torn_lines']['handler'], 1)
        body = reader.text(doc)
        self.assertIn('handler missing_parts 1 conflicts 1', body)
        self.assertGreaterEqual(body.count('      loss: torn'), len(doc['tables']) + 1)   # beside every table

    def test_zst_read_through_zstd_and_fresh_output_required(self):
        rec = standard()
        root = rec.write(zst=True)
        self.assertTrue((root / 'engine.log.zst').is_file())
        out = Path(tempfile.mkdtemp(prefix='busyfar-out-')) / 'fresh'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reader.main(['--root', str(root), '--out', str(out)]), 0)
        doc = json.loads((out / 'busyfar.json').read_text())
        self.assertTrue(doc['engine_log'].endswith('engine.log.zst'))
        self.assertTrue((root / 'engine.log.zst').is_file())       # never recompressed or removed
        with self.assertRaises(SystemExit):
            reader.main(['--root', str(root), '--out', str(out)])

    def test_no_anatomy_refuses(self):
        rec = standard()
        rec.lines = [l for l in rec.lines if 'flash-s5 anatomy' not in l]
        with self.assertRaises(SystemExit):
            reader.read(str(rec.write()))


class Bracket(unittest.TestCase):
    def test_busy_inside_far_certified_and_paired(self):
        data, items, est, meta, doc = run(standard())
        self.assertEqual(meta['pairing']['paired'], 2)
        for rank in (0, 1, 2):
            self.assertEqual(item(items, rank, 11)['cls'], 'inside')
            self.assertEqual(item(items, rank, 21)['cls'], 'far')
        self.assertEqual(item(items, 3, 11)['cls'], 'unlocated')        # no get_output proof on rank 3
        k = table(doc, 2)['contrast']
        self.assertEqual(k['pairs'], 1)
        self.assertAlmostEqual(k['paired_D_diff_ms']['p50'], 5.0)
        self.assertAlmostEqual(k['host_only_lower_bound_ms']['p50'], 15.0 - 0.2 - 0.1 - 10.0)
        self.assertEqual(item(items, 2, 11)['queue'], 1 * MS)               # worker_enter - engine_submit

    def test_far_needs_the_200ms_guard(self):
        rec = standard(far_at=2230.0)      # [L,U] = [2230, 2260]: 130 ms after the envelope, inside the guard
        data, items, *_ = run(rec)
        self.assertNotEqual(item(items, 2, 21)['cls'], 'far')

    def test_ambiguous_when_bracket_leaves_the_core(self):
        rec = standard(busy_at=895.0)      # [895, 925] ms: 25 ms of 30 outside the core, D 15 ms: < 0.8 D provable
        data, items, *_ = run(rec)
        self.assertEqual(item(items, 2, 11)['cls'], 'ambiguous')

    def test_insufficient_same_arm_support(self):
        data, items, est, meta, doc = run(standard(far=False))
        k = table(doc, 2)['contrast']
        self.assertEqual(k['pairs'], 0)
        self.assertEqual(k['support'][0]['verdict'], 'INSUFFICIENT_SAME_ARM_SUPPORT')
        self.assertIn('INSUFFICIENT_SAME_ARM_SUPPORT', reader.text(doc))

    def test_estimated_never_feeds_contrast(self):
        data, items, est, meta, doc = run(standard(far_at=2230.0))
        self.assertTrue(est)
        self.assertEqual(table(doc, 2)['contrast']['pairs'], 0)
        self.assertIn('ESTIMATED', reader.text(doc))

    def test_record_loss_in_window_unsupported(self):
        rec = Record()
        rec.skip_rseq.add((10, 1))         # rank 0's chunk-1 anatomy receipt lost; no line gap shows it (line 0)
        data, items, *_ = run(standard(rec))
        self.assertEqual(data['loss']['receipt_seq_missing'], 1)
        self.assertEqual(data['loss']['anatomy']['line_gaps'], 0)
        self.assertEqual(item(items, 0, 11)['cls'], 'unsupported_loss')
        self.assertEqual(item(items, 2, 21)['cls'], 'far')          # outside the loss window

    def test_cannot_say_header_in_codex_words(self):
        *_, doc = run(standard())
        body = reader.text(doc)
        self.assertTrue(body.startswith('What this contrast cannot say'))
        for phrase in ('24 decode SMs', 'copy-stream activity', 'x8 link is fixed', 'matched strata'):
            self.assertIn(phrase, body.split('\n\n')[0])


class Pairing(unittest.TestCase):
    def test_two_nonempty_executes_unpaired(self):
        data, items, est, meta, doc = run(standard(first_nonempty=True))
        self.assertEqual(meta['pairing']['several_pending'], 2)    # 10 and 11 both ran a forward, in both steps
        self.assertEqual(meta['pairing']['paired'], 0)
        self.assertEqual(item(items, 2, 11)['cls'], 'unlocated')

    def test_missing_rank_row_unknown_emptiness(self):
        rec = standard()                   # rank 3's rows gone: the empty executes 10 / 20 can't be proven empty
        rec.lines = [l for l in rec.lines if not ('flash-s5 handler' in l and '"rank":3' in l)]
        data, items, est, meta, doc = run(rec)
        self.assertEqual(meta['pairing']['unknown_emptiness'], 2)
        self.assertEqual(meta['pairing']['paired'], 0)
        self.assertEqual(item(items, 2, 11)['cls'], 'unlocated')

    def test_seq_gap_unpaired(self):
        rec = standard()
        rec.lines = [l.replace('[19,"flash_s5_submit"', '[119,"flash_s5_submit"') for l in rec.lines]
        data, items, est, meta, doc = run(rec)
        self.assertEqual(meta['pairing']['seq_gap'], 1)


def stage_rule_holds():
    """stage D = max of TP peers on the same seq; a missing peer row leaves that seq out."""
    rec = standard(peers={(0, 21): 10.0, (1, 21): 14.0}, drop_rank_seq={(1, 11)})
    data, items, *_ = run(rec)
    stage = {i['seq']: i for i in reader.stage_items(items) if i['rank'] == 'stage0'}
    return set(stage) == {21} and stage[21]['D'] == 14 * MS and stage[21]['peers'] == {0: 10.0, 1: 14.0}


def guard_rule_holds():
    data, items, *_ = run(standard(far_at=2230.0))
    return item(items, 2, 21)['cls'] != 'far'


def null_rule_holds():
    data, items, est, meta, doc = run(standard(null_dev={(2, 11)}, nulls_H={(0, 11)}))
    return (not any(i['rank'] == 2 and i['seq'] == 11 for i in items) and item(items, 0, 11)['H'] is None
            and table(doc, 0)['contrast']['host_only_lower_bound_ms']['n'] == 0)


class Rules(unittest.TestCase):
    def test_rules_hold(self):
        self.assertTrue(stage_rule_holds())
        self.assertTrue(guard_rule_holds())
        self.assertTrue(null_rule_holds())


class Mutants(unittest.TestCase):
    def test_neighbour_line_join_caught(self):
        def positional(by_rank, peer, it):
            mine = sorted(by_rank.get(it['rank'], {}))
            theirs = [by_rank[peer][s] for s in sorted(by_rank.get(peer, {}))]
            index = mine.index(it['seq'])
            return theirs[index] if index < len(theirs) else None
        with mock.patch.object(reader, 'join_peer', positional):
            self.assertFalse(stage_rule_holds())

    def test_far_without_guard_caught(self):
        with mock.patch.object(reader, 'GUARD_NS', 0):
            self.assertFalse(guard_rule_holds())

    def test_sum_instead_of_max_caught(self):
        with mock.patch.object(reader, 'stage_value', sum):
            self.assertFalse(stage_rule_holds())

    def test_zero_filled_nulls_caught(self):
        real = reader.num
        with mock.patch.object(reader, 'num', lambda v: 0 if v is None else real(v)):
            self.assertFalse(null_rule_holds())


if __name__ == '__main__':
    unittest.main()
