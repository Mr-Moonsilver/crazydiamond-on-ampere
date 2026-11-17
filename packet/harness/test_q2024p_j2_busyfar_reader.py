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
                                                  'native-gap': 'gap', 'decode-timeline': 'timeline'}[kind]])
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


META = dict(actual_rows=8, actual_tokens=8, uid_digest='u' * 64, history_tokens=[1000] * 8, bank='split',
            route_generation=1, decode_sms=24, bank_sms=24, stream=7, cg_mode='FULL', num_tokens=8, num_reqs=8,
            uniform_token_count=1, max_query_len=1, num_ubatches=None, forward_calls=1, handler_flags=0,
            completion='own-forward-end-event-query')


TL_FIELDS = dict(d=list(reader.D_FIELDS), p=['phase', 'layer', 'group', 'bucket', 'stream', 'start_ms', 'end_ms'],
                 s=['complete', 'counters'])


def standard(rec=None, busy_D=15.0, far_D=10.0, far_at=2500.0, busy_at=300.0, peers=None, drop_rank_seq=(),
             null_dev=(), first_nonempty=False, far=True, env_core=(100.0, 900.0), h_ns=200_000, nulls_H=(),
             extra_busy=False, outside=False, outside_at=1400.0, timeline=False, tl_drop=(), tl_meta=None, tl_dev=None,
             qnative=None, idle=None, tl_stamp=None):
    """Chunk A (txn 1): native envelope [100, 900] ms on every rank, busy execute seq 11 at busy_at. Chunk B (txn 2):
    envelope [2000, 2100] ms, far execute seq 21 at far_at. Samples 12 / 22; rank 2 returns the outputs.

    Timeline options (decode-timeline stamp): ``extra_busy`` a second busy step 14/15/16 in chunk A (cadence 11 -> 15);
    ``outside`` an outside-native-progress pair 17/18 at outside_at between the chunks (no engine progress rows);
    ``timeline`` d rows on every rank: one native session per (rank, txn) and one outside session per rank, own
    completion at t_enter + D + 5 ms; ``tl_drop`` (rank, seq) d rows left out; ``tl_meta`` / ``tl_dev`` per-row overrides;
    ``qnative`` {rank: ms} a native handler on that rank's worker thread from busy_at - 1 for ms; ``idle``
    {(rank, seq): idle_before_launch}; ``tl_stamp`` wall stamp of the outside timeline lines."""
    rec = rec or Record()
    peers = peers or {}
    idle = idle or {}
    steps = [(10, 11, 12, busy_at, busy_D, 1)]
    if extra_busy:
        steps.append((14, 15, 16, busy_at + 30, busy_D, 1))
    if far:
        steps.append((20, 21, 22, far_at, far_D, 2))
    if outside:
        steps.append((None, 17, 18, outside_at, far_D, None))
    # engine progress chunks (pid 1), one flush per txn
    for txn in sorted({s[5] for s in steps if s[5] is not None}):
        lo = env_core[0] if txn == 1 else 2000.0
        hi = env_core[1] if txn == 1 else 2100.0
        s_rows = []
        for (e0, e1, sm, at, _, x) in steps:
            if x == txn:
                s_rows += [[e0, 'execute_model', us(at - 2), us(at - 1.5), us(at - 1)],
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
    for rank, q in sorted((qnative or {}).items()):
        rec.emit('progress', dict(side='worker', rank=rank, base_ns=T, dropped=0,
                                  fields=dict(r=['seq', 'kind', 't_enter', 't_exit', 't_emit', 'direct', 'replied']),
                                  rows=dict(r=[[13, 'flash_s5_submit', us(busy_at - 1), us(busy_at - 1 + q),
                                                us(busy_at - 1 + q), True, True]])), W0 + 30, pid=10 + rank)
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
            if e0 is not None:
                m.append([e0, 'e', int(rank >= 2), 0, us(at - 1), us(at - 0.8), 200000, 190000, 1000, None, None, None,
                          None, None, [[0, 100000, 100000, 90000, 1]] + ([[1, 1, 1, 1, 1]] if first_nonempty else [])])
            if (rank, e1) not in drop_rank_seq:
                m.append([e1, 'e', int(rank >= 2), 0, us(at), us(at + 3), 3000000, 2900000, 100000,
                          None if (rank, e1) in null_dev else int(dev * MS), None, idle.get((rank, e1), 0), 0, 5000,
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
    if timeline:
        tl_meta, tl_dev = tl_meta or {}, tl_dev or {}

        def d_row(rank, e1, at, D, start, end):
            meta = dict(META, **tl_meta.get((rank, e1), {}))
            dev = tl_dev.get((rank, e1), int(peers.get((rank, e1), D) * MS))
            return [e1, meta, start, end, dev, 3000000, 100000, us(at), us(at + 3), us(at + D + 5)]
        for rank in range(4):
            for txn in sorted({s[5] for s in steps if s[5] is not None}):
                ds = [d_row(rank, e1, at, D, 1.0, 1.0 + D) for (e0, e1, sm, at, D, x) in steps
                      if x == txn and (rank, e1) not in tl_drop]
                rec.emit('decode-timeline', dict(
                    side='worker', rank=rank, base_ns=T, engine_epoch=1, request_generation=1, txn=txn,
                    chunk_start=0 if txn == 1 else 2048, chunk_end=2048 if txn == 1 else 4096, sample=txn,
                    sample_every=16, session_limit=256, byte_limit=4 << 20, anchor='native-phase-event',
                    anchor_sample=txn, stage=rank // 2, route_generation=1, phases=1, forwards=len(ds), cancelled=False,
                    timing_backend_available=True, dropped=0, fields=TL_FIELDS,
                    rows=dict(d=ds, p=[['attention', 'layer.0', 0, 8, 7, 0.5, 2.0]], s=[[True, {}]])),
                    W0 + 41 + txn, pid=10 + rank, stream=('tl', txn))
            for (e0, e1, sm, at, D, x) in steps:
                if x is None and (rank, e1) not in tl_drop:
                    rec.emit('decode-timeline', dict(
                        side='worker', rank=rank, base_ns=T, scope='outside-native-progress', engine_epoch=1,
                        stage=rank // 2, sample='outside-1', sample_every=64, anchor_sample='outside-1',
                        engine_send_ns=T + us(at - 1) * 1000, execute_seq=e1, byte_limit=4 << 20, sample_limit=1024,
                        anchor='this-decode-forward-first-event', basis='local', dropped=0, fields=TL_FIELDS,
                        rows=dict(d=[d_row(rank, e1, at, D, 0.0, D)], s=[[True, {}]])),
                        tl_stamp if tl_stamp is not None else W0 + 45, pid=10 + rank, stream=('tl', 'outside-1'))
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


def table(doc, rank, group=None):
    return next(t for t in doc['tables'] if t['rank'] == str(rank) and (group is None or t['group'] == group))


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


# --- decode-timeline stamp ------------------------------------------------------------------------------------------
def tl_run(**kw):
    arms = kw.pop('arms', None)
    rec = standard(timeline=True, **kw)
    return run(rec, **({'arms': arms} if arms else {}))


def own_u_rule_holds():
    """A rank without its own d row stays unlocated even though rank 2's get_output would bound it (no borrowed U)."""
    data, items, est, meta, doc = tl_run(tl_drop={(0, 11)})
    i = item(items, 0, 11)
    return (i['cls'] == 'unlocated' and i['why'] == 'no_own_completion' and meta['excluded']['no_own_completion'] == 1
            and item(items, 3, 11)['cls'] == 'inside')


def outside_arm_rule_holds():
    """The outside pair at 1400 ms lies between chunk A (arm n3.a) and chunk B (arm n2.b): an arm boundary, even
    though its timeline line was logged inside n3.a's wall window."""
    arms = (('n3.a', W0, W0 + 15), ('n2.b', W0 + 15, W0 + 100))
    data, items, est, meta, doc = tl_run(outside=True, arms=arms, tl_stamp=W0 + 12)
    return (not any(i['seq'] == 17 for i in items) and meta['excluded']['arm_boundary'] >= 4
            and item(items, 2, 11)['arm'] == 'n3.a')


def session_key_rule_holds():
    """Two outside sessions on one rank with no forward (s row only, base_ns 0, line 0) are two flushes, not a
    conflicting duplicate."""
    rec = standard(timeline=True)
    for n in (2, 3):
        rec.emit('decode-timeline', dict(side='worker', rank=1, base_ns=0, scope='outside-native-progress',
                                         engine_epoch=1, stage=0, sample='outside-%d' % n, anchor_sample='outside-%d' % n,
                                         engine_send_ns=T, execute_seq=40 + n, dropped=0, fields=TL_FIELDS,
                                         rows=dict(s=[[False, {'unresolved_forward': 1}]])),
                 W0 + 46, pid=11, stream=('tl', 'outside-%d' % n))
    data, items, est, meta, doc = run(rec)
    s = data['loss']['timeline_stamp']
    return (data['loss']['timeline']['conflicting_duplicates'] == 0 and s['sessions_outside'] == 2
            and s['counters'].get('unresolved_forward') == 2 and s['sessions_incomplete'] == 2)


def exposed_rule_holds():
    """Critical PP path of queue_native = max over stages of max over TP peers; a null idle_before stays null."""
    data, items, est, meta, doc = tl_run(extra_busy=True, qnative={0: 0.6, 1: 0.3, 2: 0.4, 3: 0.2},
                                         idle={(0, 11): 1, (2, 11): 1, (3, 11): 1})
    e = meta['exposed_wait']
    rows = {r['bin_ms']: r for r in e['table']}
    hot = rows.get('0.5-1')
    data2, items2, est2, meta2, doc2 = tl_run(extra_busy=True, qnative={0: 0.6, 1: 0.3, 2: 0.4, 3: 0.2},
                                              idle={(0, 11): 1, (1, 11): None, (2, 11): 1, (3, 11): 1})
    hot2 = {r['bin_ms']: r for r in meta2['exposed_wait']['table']}.get('0.5-1')
    return (hot is not None and hot['n'] == 1 and abs(hot['queue_native_crit_ms']['p50'] - 0.6) < 1e-6
            and abs(hot['cadence_ms']['p50'] - 30.0) < 1e-6 and abs(hot['exposed_candidate_ms']['p50'] - 0.6) < 1e-6
            and hot2 is not None and hot2['exposed_candidate_known'] == 0 and hot2['exposed_candidate_ms']['n'] == 0)


class TimelineStamp(unittest.TestCase):
    def test_own_bound_on_every_rank_and_outside_far_control(self):
        data, items, est, meta, doc = tl_run(outside=True, tl_drop={(r, 21) for r in range(4)})
        self.assertTrue(meta['timeline_mode'])
        for rank in range(4):
            self.assertEqual(item(items, rank, 11)['cls'], 'inside')          # rank 3 too: its own completion
            self.assertEqual(item(items, rank, 17)['cls'], 'far')
            self.assertEqual(item(items, rank, 17)['send_source'], 'transport-send')
            self.assertEqual(item(items, rank, 17)['queue'], 1 * MS)           # enter - engine_send_ns
            self.assertEqual(item(items, rank, 21)['why'], 'no_own_completion')
            self.assertEqual(item(items, rank, 11)['bracket'], 20 * MS + reader.QUANT_NS)   # D 15 + 5 ms
        self.assertEqual(meta['excluded']['no_own_completion'], 4)
        for rank in ('2', '3', 'stage0', 'stage1'):
            k = table(doc, rank, 'bank=split')['contrast']
            self.assertEqual(k['pairs'], 1)
            self.assertEqual(k['pairs_control_outside'], 1)
            self.assertEqual(k['pairs_same_uid_set'], dict(known=1, same=1))
            self.assertAlmostEqual(k['paired_D_diff_ms']['p50'], 5.0)
        body = reader.text(doc)
        self.assertIn('own: t_complete_query + 1 us', body)
        self.assertIn('timeline stamp: sessions native 8 outside 4', body)
        self.assertIn('pairs with an outside-sample control 1', body)

    def test_legacy_mode_without_the_stamp_keeps_the_borrowed_rule(self):
        data, items, est, meta, doc = run(standard(outside=True))
        self.assertFalse(meta['timeline_mode'])
        self.assertEqual(item(items, 3, 11)['cls'], 'unlocated')
        self.assertFalse(any(i['seq'] == 17 for i in items))                   # no engine send row, no stamp
        self.assertIn('borrowed: rank 2', reader.text(doc))

    def test_strata_from_actual_metadata(self):
        data, items, est, meta, doc = tl_run(outside=True, tl_drop={(r, 21) for r in range(4)},
                                             tl_meta={(r, 17): dict(num_tokens=16) for r in range(4)})
        self.assertNotEqual(item(items, 2, 11)['stratum'], item(items, 2, 17)['stratum'])
        k = table(doc, 2, 'bank=split')['contrast']
        self.assertEqual(k['pairs'], 0)
        self.assertEqual(k['insufficient'], 1)
        self.assertIn('INSUFFICIENT_SAME_ARM_SUPPORT', reader.text(doc))

    def test_unsupported_metadata_excluded(self):
        data, items, est, meta, doc = tl_run(tl_meta={(2, 11): dict(forward_calls=2), (1, 11): dict(history_tokens=None),
                                                      (0, 11): dict(handler_flags=4)})
        for rank in (0, 1, 2):
            self.assertEqual(item(items, rank, 11)['cls'], 'metadata_unsupported')
        self.assertEqual(item(items, 3, 11)['cls'], 'inside')
        self.assertEqual(meta['excluded']['metadata_unsupported'], 3)

    def test_d_row_must_match_its_handler_row(self):
        data, items, est, meta, doc = tl_run(tl_dev={(2, 11): 14 * MS})
        self.assertEqual(item(items, 2, 11)['cls'], 'inconsistent')
        self.assertEqual(meta['excluded']['timeline_mismatch_or_conflict'], 1)

    def test_conflicting_d_rows_inconsistent(self):
        rec = standard(timeline=True)
        rec.emit('decode-timeline', dict(side='worker', rank=2, base_ns=T, engine_epoch=1, txn=9, sample=9,
                                         anchor_sample=9, dropped=0, fields=TL_FIELDS,
                                         rows=dict(d=[[11, META, 1.0, 16.0, 15 * MS, 1, 1, us(300), us(303), us(390)]],
                                                   s=[[True, {}]])), W0 + 47, pid=12, stream=('tl', 9))
        data, items, est, meta, doc = run(rec)
        self.assertEqual(data['loss']['timeline_stamp']['conflicting_d_rows'], 1)
        self.assertEqual(item(items, 2, 11)['cls'], 'inconsistent')

    def test_stamp_loss_facts_printed(self):
        rec = standard(timeline=True)
        rec.lines = [l for l in rec.lines if not ('decode-timeline' in l and '"rank":0' in l and '"sample":1,' in l)]
        rec.emit('decode-timeline', dict(side='worker', rank=0, base_ns=T, engine_epoch=1, txn=3, sample=3,
                                         anchor_sample=3, dropped=0, fields=TL_FIELDS,
                                         rows=dict(s=[[False, {'pending': 2, 'log_budget_exhausted': 1}]])),
                 W0 + 48, pid=10, stream=('tl', 3))
        data, items, est, meta, doc = run(rec)
        s = data['loss']['timeline_stamp']
        self.assertEqual(s['sample_gaps'], 1)                  # native session 1 of rank 0 lost
        self.assertEqual(s['log_budget_exhausted_sessions'], 1)
        self.assertEqual(item(items, 0, 11)['why'], 'no_own_completion')
        body = reader.text(doc)
        self.assertIn('timeline missing_parts 0 conflicts 0 sample_gaps 1 incomplete 1 of 8 sessions', body)
        self.assertIn('budget_exhausted 1 pending 2', body)

    def test_exposed_wait_bins_cadence_and_baseline(self):
        data, items, est, meta, doc = tl_run(extra_busy=True, outside=True, qnative={0: 0.6, 1: 0.3, 2: 0.4, 3: 0.2})
        e = meta['exposed_wait']
        rows = {r['bin_ms']: r for r in e['table']}
        self.assertEqual(rows['0.5-1']['n'], 1)                    # seq 11
        self.assertEqual(rows['0.5-1']['exposed_candidate_ms']['p50'], 0.0)   # stream busy at launch: hidden
        self.assertEqual(rows['0']['n'], 2)                        # seq 15 and the far seq 21
        self.assertEqual(rows['0']['cadence_ms']['n'], 0)          # no next nonempty execute in their chunks
        self.assertIn('n3.g2-live-0 rank 2', e['outside_queue_baseline_ms'])
        body = reader.text(doc)
        self.assertIn('EXPOSED-WAIT', body)
        self.assertIn('never a sum of rank delays', body)

    def test_rules_hold(self):
        self.assertTrue(own_u_rule_holds())
        self.assertTrue(outside_arm_rule_holds())
        self.assertTrue(session_key_rule_holds())
        self.assertTrue(exposed_rule_holds())


class TimelineMutants(unittest.TestCase):
    def test_borrowed_upper_bound_caught(self):
        real = reader.own_upper_bound

        def borrowed(timeline, rank, seq, outs, sample):
            U, row, why = real(timeline, rank, seq, outs, sample)
            if why and sample is not None:
                U = reader.upper_bound(outs, sample, rank)[0]
                return U, timeline.get((2, seq)), None
            return U, row, why
        with mock.patch.object(reader, 'own_upper_bound', borrowed):
            self.assertFalse(own_u_rule_holds())

    def test_arm_by_wall_stamp_caught(self):
        def by_stamp(lo, hi, windows, owners):
            return 'n3.a'                    # what the outside line's log stamp (W0 + 12) falls in
        with mock.patch.object(reader, 'outside_arm', by_stamp):
            self.assertFalse(outside_arm_rule_holds())

    def test_line_only_key_caught(self):
        with mock.patch.object(reader, 'TIMELINE_KEY', ()):
            real = reader.assemble
            with mock.patch.object(reader, 'assemble', lambda r, extra=(), line_gaps=True: real(r, (), line_gaps)):
                self.assertFalse(session_key_rule_holds())

    def test_sum_over_peers_caught(self):
        with mock.patch.object(reader, 'stage_value', sum):
            self.assertFalse(exposed_rule_holds())

    def test_sum_over_stages_caught(self):
        with mock.patch.object(reader, 'over_stages', sum):
            self.assertFalse(exposed_rule_holds())

    def test_zero_filled_null_idle_caught(self):
        real = reader.num
        with mock.patch.object(reader, 'num', lambda v: 0 if v is None else real(v)):
            self.assertFalse(exposed_rule_holds())


if __name__ == '__main__':
    unittest.main()
