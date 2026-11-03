"""alpha-next-c1b queue: the six-arm numerics table (S1 S2 L1 L2 D S3), the arm's serial selection and numerics probe, the grade
(c1's grade over this table with the C1-aware open-rows rule, the C1-aware ring read and the numerics verdict), the derived texts
and the arm loop. CPU only; no pass here is box evidence.

Classes:
* Table: the declared rows, roles, selections and pairings; the views alpha-next and c1 read.
* ArmControl: the stage command per selection (serial sends local false), the numerics probe's request body and instrument, the
  arm record (OUT.base kept, numerics added).
* OpenRows / Ring: e3_c1_open_rows on its own (each check and its reason), then in the grade: the old rule stays on serial and
  C0-local arms, the new one replaces it on the C1-defer arm; check 4 grades INC ENGINE.
* NumericsRule: bit-equal mode, bound mode, outside (FAIL ENGINE), drift and missing evidence (INC).
* Grade: c1's synthetic run (test_alpha_next_c1_queue's C1Emit) over this table, serial arms on alpha-next's C0-off chunk: PASS
  with the sections; the c1 gates still bite.
* Derivation / Commands / Budget / ArmLoop: in test_alpha_next_c1b_queue's second half (the derived texts).
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import alpha_next_c1b_arms as T  # noqa: E402
import e3_alpha_next_c1_box as c1  # noqa: E402
import e3_alpha_next_c1b_box as box  # noqa: E402
import e3_c1_open_rows as rule  # noqa: E402
import test_alpha_next_c1_queue as tc1  # noqa: E402
import test_alpha_next_queue as tan  # noqa: E402
import test_alpha_perf_queue as tap  # noqa: E402

progress, ap, sb = box.progress, box.ap, c1.an.b
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
N = box.NUMERICS['generation_tokens']


def numerics(seed=0, *, change_at=None, nudge=0.0, membership=False, tokens=None):
    """A complete numerics record: token i is 100 + i; top-20 of position i are tokens i .. i+19 with fixed logprobs. ``change_at``
    changes the token from there on; ``nudge`` adds to every logprob; ``membership`` swaps one top-20 member at position 0."""
    toks = list(range(100, 100 + N)) if tokens is None else list(tokens)
    if change_at is not None:
        toks = toks[:change_at] + [7] * (N - change_at)
    top = []
    for i, t in enumerate(toks):
        row = [[t + j, -0.01 * (j + 1) - 0.001 * (i % 7) + nudge] for j in range(20)]
        if membership and i == 0:
            row[-1] = [999999, row[-1][1]]
        top.append(row)
    return dict(schema=box.NUMERICS_SCHEMA, prompt_sha256='a' * 64, prompt_chars=1, request={}, generated=N, tokens=toks, top=top,
                tokens_sha256='x', complete=True)


# --- table ---------------------------------------------------------------------------------------------------

class Table(unittest.TestCase):
    def test_six_arms_s1_s2_l1_l2_d_s3(self):
        rows = T.arms()
        self.assertEqual([r['role'] for r in rows], ['S1', 'S2', 'L1', 'L2', 'D', 'S3'])
        self.assertEqual([r['label'] for r in rows], ['c0off.g2-solo-0', 'c0off.g2-solo-1', 'c1off.g2-solo-0', 'c1off.g2-solo-1',
                                                       'c1defer.g2-solo-0', 'c0off.g2-solo-3'])
        self.assertEqual({(r['band'], r['cohort'], r['size'], r['part']) for r in rows}, {(1, 'solo', 2, 'core')})
        self.assertEqual([(r['stage_local'], r['overlap'], r['defer_bind'], r['c0']) for r in rows],
                         [(False, False, False, 'off')] * 2 + [(True, False, False, 'on')] * 2 + [(True, True, True, 'on'),
                                                                                               (False, False, False, 'off')])

    def test_pairings_and_views(self):
        d, l1, s1 = (T.by_role()[r] for r in ('D', 'L1', 'S1'))
        self.assertEqual(T.matched_off(d), l1['label'])
        self.assertIsNone(T.matched_off(l1))
        self.assertEqual(T.matched_serial(l1), s1['label'])
        view = box.table_view()
        self.assertEqual(view.C0, {'on': True, 'off': False})
        self.assertEqual(view.matched_off(l1), s1['label'])
        self.assertEqual(view.parse_selection('c0off.g2'), ('off', 2))
        self.assertEqual(view.parse_selection('c1defer.g2'), ('on', 2))
        one = box.table_view('serial')
        self.assertEqual(one.C0, {'off': False})
        self.assertEqual(one.label('off', 2, 'solo', 3), 'c0off.g2-solo-3')
        self.assertEqual(box.table_view('defer').C0, {'on': True})

    def test_the_table_refuses_a_malformed_order(self):
        with mock.patch.object(T, 'ARM_TABLE', T.ARM_TABLE[::-1]):
            with self.assertRaises(AssertionError):
                T.check()


# --- the arm --------------------------------------------------------------------------------------------------

class FakeStream:
    def __init__(self):
        self.token_count = 0


class ArmControl(unittest.TestCase):
    def test_each_selection_sends_local_overlap_and_defer_bind(self):
        for sel, local in (('serial', False), ('off', True), ('defer', True)):
            sent = []

            class Client:
                def call(self, action, **fields):
                    sent.append((action, fields))
                    return dict(fields, overlap=fields.get('overlap'), defer_bind=fields.get('defer_bind'),
                                previous_overlap=False, previous_defer_bind=False)
            names = box.arm_names(sel)
            release = names['release_alpha_next']

            def an_release(size, loc):
                def run(client, record, mode):
                    record['stage_switch'] = dict(local=loc)
                    reply = client.call(c1.STAGE_CONTROL, local=loc)
                    record['stage_switch'].update(local=reply['local'])
                return run
            with mock.patch.object(c1, 'AN_RELEASE', an_release):
                record = {}
                release(2, local)(Client(), record, 'burst')
            self.assertEqual(sent, [(c1.STAGE_CONTROL, dict(local=local, overlap=T.C1[sel], defer_bind=T.DEFER_BIND[sel]))])
            self.assertEqual(record['defer_bind']['requested'], T.DEFER_BIND[sel])
            self.assertEqual(names['T'].C0, {'on': True} if local else {'off': False})

    def test_the_probe_body_and_the_append_unchanged(self):
        captured, calls = {}, []
        cfg = dict(timeouts=dict(finish_s=60))

        def workload_view(cfg, **kw):
            return argparse.Namespace(**kw)

        def request(model, prompt, workload, salt):
            return dict(model=model, prompt=prompt, max_tokens=workload.generation_budget, min_tokens=workload.generation_budget,
                        temperature=workload.temperature, top_p=workload.top_p, seed=5, ignore_eos=True, cache_salt=salt)

        def start(url, model, bodies, cohort, workload, instrument=None):
            calls.append((bodies, cohort, instrument))
            return ['state']
        original = mock.Mock(return_value=('append-state', 'w'))
        with mock.patch.object(box.shared, '_workload_view', workload_view), mock.patch.object(box.shared, '_request', request), \
                mock.patch.object(box.correctness, '_start_streams', start), mock.patch.object(box.shared, '_send_prefill', original):
            send = box.numerics_send(captured)
            state, workload = send('u', 'm', cfg, 'lane', 0, 'PROMPT', 16, 3, 'salt')
            self.assertEqual(send('u', 'm', cfg, 'lane', 1, 'PROMPT2', 16, 3, 'salt'), ('append-state', 'w'))
        (bodies, cohort, instrument), = calls
        body = bodies[0]
        self.assertEqual((body['temperature'], body['top_p'], body['max_tokens'], body['min_tokens'], body['logprobs'],
                          body['return_tokens_as_token_ids'], body['cache_salt']), (0.0, 1.0, N, N, 20, True, 'salt'))
        self.assertIs(instrument, captured['instrument'])
        self.assertEqual(captured['request']['max_tokens'], N)
        self.assertNotIn('PROMPT', json.dumps({k: v for k, v in captured.items() if k != 'instrument'}))   # hashes and lengths only
        original.assert_called_once()

    def test_the_instrument_keeps_every_position_and_refuses_malformed_blocks(self):
        inst, state = box.Numerics(), FakeStream()
        top = {'token_id:%d' % k: -0.1 * k for k in range(20)}
        inst.consume(state, {'choices': [{'token_ids': [5, 6], 'logprobs': {'tokens': ['token_id:5', 'token_id:6'],
                                                                            'top_logprobs': [top, top]}}]})
        self.assertEqual(inst.tokens, [5, 6])
        self.assertEqual(len(inst.top[0]), 20)
        state.token_count = 2
        for bad in ({'tokens': ['token_id:9'], 'top_logprobs': [top]},                  # encoding differs
                    {'tokens': ['token_id:7'], 'top_logprobs': [dict(list(top.items())[:5])]},   # not top-20
                    {'tokens': ['token_id:7'], 'top_logprobs': [dict(top, **{'token_id:1': float('nan')})]}):
            with self.assertRaises(ValueError):
                box.Numerics.consume(copy.deepcopy(inst), state, {'choices': [{'token_ids': [7], 'logprobs': bad}]})
        record = box.numerics_record(dict(instrument=inst, prompt_sha256='p', prompt_chars=3, request={}))
        self.assertFalse(record['complete'])
        self.assertEqual(record['reason'], 'numerics-generated-2-of-%d' % N)
        self.assertEqual(box.numerics_record({})['reason'], 'numerics-probe-not-sent')

    def test_run_arm_keeps_alpha_nexts_record_and_adds_numerics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'arm.json'

            def an_run(args):
                self.assertEqual(args.stage_local, 'off')
                self.assertEqual(args.out, str(out) + '.base')
                self.assertIs(box.an.T.C0, box.an.T.C0)
                Path(args.out).write_text(json.dumps(dict(label='c0off.g2-solo-0', complete=True, reasons=[])))
                return 0
            a = argparse.Namespace(c1='serial', out=str(out))
            with mock.patch.object(box.an, 'run_arm', an_run):
                rc = box.run_arm(a)
            record = json.loads(out.read_text())
            self.assertEqual(rc, 2)
            self.assertEqual(record['role'], 'S1')
            self.assertEqual(record['reasons'], ['numerics:numerics-probe-not-sent'])
            self.assertTrue((Path(tmp) / 'arm.json.base').is_file())
            self.assertEqual(box.run_arm(a), 2)                     # never over an existing record


# --- (a) the rule on its own ---------------------------------------------------------------------------------

def flush(line, txn, open_rows, rows):
    return dict(line=line, identity=dict(txn=txn, open_rows=open_rows), rows=dict(n=rows))


def row(seq, submit, apply):
    return dict(seq=seq, t_submit=submit, t_apply=apply)


INDEX = {1: (frozenset({1, 2}), 2), 2: (frozenset({3, 4, 5}), 3), 3: (frozenset({6}), None)}


class OpenRows(unittest.TestCase):
    def good(self):
        return [flush(10, 1, 1, [row(1, 0, 10), row(2, 11, 20), row(3, 15, 18)]), flush(11, 2, 0, [row(4, 18, 30), row(5, 31, 40)]),
                flush(12, 3, 0, [row(6, 41, 50)])]

    def test_a_carried_successor_row_passes(self):
        out = rule.check_arm(self.good(), INDEX)
        self.assertEqual(out['reasons'], [])
        self.assertEqual(out['flagged'][0]['carried_seqs'], [4])

    def test_each_check_names_its_reason(self):
        last = self.good()[:1] + [flush(11, 2, 0, [row(4, 18, 30)]), flush(12, 3, 1, [row(6, 41, 50)])]
        self.assertEqual(rule.check_arm(last, INDEX)['reasons'], ['c1-open-rows-at-last-line'])
        skip = [self.good()[0], self.good()[2]]
        self.assertEqual(rule.check_arm(skip, INDEX)['reasons'], ['c1-open-rows-successor-mismatch'])
        named = {1: (INDEX[1][0], 5), 2: INDEX[2], 3: INDEX[3]}
        self.assertEqual(rule.check_arm(self.good(), named)['reasons'], ['c1-open-rows-successor-mismatch'])
        two = self.good()
        two[0]['identity']['open_rows'] = 2
        self.assertEqual(rule.check_arm(two, INDEX)['reasons'], ['c1-open-rows-not-carried'])
        own = self.good()
        own[1]['rows']['n'][0]['seq'] = 2                    # the carried row is the flushing txn's own
        self.assertEqual(rule.check_arm(own, INDEX)['reasons'], ['c1-open-row-not-successor'])
        bad = self.good()
        bad[0]['identity']['open_rows'] = None
        self.assertEqual(rule.check_arm(bad, INDEX)['reasons'], ['c1-open-rows-unreadable'])
        self.assertEqual(rule.ENGINE_REASONS, ('c1-open-row-not-successor',))

    def test_ring_increments(self):
        self.assertEqual(rule.ring_increments([0, 5, 5, 12, None, 3, 4]),
                         dict(per_chunk=[0, 5, 0, 7, 3, 1], total=16, max_chunk=7, resets=1))


# --- numerics ------------------------------------------------------------------------------------------------

def roles(**changes):
    out = {r: numerics() for r in T.ROLES}
    out.update(changes)
    return out


class NumericsRule(unittest.TestCase):
    def test_bit_equal_controls_and_candidates_are_within(self):
        out = box.numerics_verdict(roles())
        self.assertEqual(out['verdict'], 'WITHIN')
        self.assertEqual(out['bound'], dict(mode='bit-equal', common_tokens=N, max_abs_logprob=0.0, max_membership=0))
        self.assertTrue(all(c['within'] and c['bit_equal'] for c in out['comparisons']))
        self.assertEqual([(c['candidate'], c['reference']) for c in out['comparisons']],
                         [('L1', 'S1'), ('L2', 'S1'), ('D', 'L1'), ('D', 'L2'), ('S3', 'S1')])

    def test_bit_equal_controls_make_any_candidate_difference_outside(self):
        for cand in ('L1', 'L2', 'D'):
            out = box.numerics_verdict(roles(**{cand: numerics(nudge=1e-7)}))
            self.assertEqual(out['verdict'], 'OUTSIDE', cand)
        self.assertEqual(box.numerics_verdict(roles(D=numerics(change_at=200)))['verdict'], 'OUTSIDE')
        self.assertEqual(box.numerics_verdict(roles(L1=numerics(membership=True)))['verdict'], 'OUTSIDE')

    def test_unequal_controls_give_a_bound(self):
        base = roles(S2=numerics(nudge=0.002, change_at=120))
        out = box.numerics_verdict(base)
        self.assertEqual(out['bound']['mode'], 'bound')
        self.assertEqual(out['bound']['common_tokens'], 120)
        self.assertAlmostEqual(out['bound']['max_abs_logprob'], 0.002)
        self.assertEqual(out['verdict'], 'WITHIN')
        self.assertEqual(box.numerics_verdict(dict(base, D=numerics(nudge=0.001, change_at=150)))['verdict'], 'WITHIN')
        self.assertEqual(box.numerics_verdict(dict(base, D=numerics(nudge=0.003)))['verdict'], 'OUTSIDE')
        self.assertEqual(box.numerics_verdict(dict(base, D=numerics(change_at=100)))['verdict'], 'OUTSIDE')

    def test_drift_or_missing_evidence_is_inc(self):
        out = box.numerics_verdict(roles(S3=numerics(change_at=10)))
        self.assertEqual((out['verdict'], out['reasons']), ('INC', ['numerics-control-drift:S3-S1']))
        out = box.numerics_verdict(roles(L2=None))
        self.assertEqual((out['verdict'], out['reasons']), ('INC', ['numerics-evidence-missing:L2']))
        out = box.numerics_verdict(roles(S2=dict(numerics(), complete=False)))
        self.assertEqual(out['verdict'], 'INC')
        self.assertIsNone(out['bound'])
        out = box.numerics_verdict(roles(D=dict(numerics(), prompt_sha256='b' * 64)))
        self.assertIn('numerics-prompt-differs-between-arms', out['reasons'])
        self.assertEqual(out['verdict'], 'INC')


# --- the grade over a synthetic run ----------------------------------------------------------------------------

def synthetic_run(rows=None, *, numerics_by_role=None, edit=None):
    """c1's C1Emit over this table: the C1-defer arm with two chunks in flight, C0-local arms alpha-next's C0-on chunk, serial arms
    alpha-next's C0-off chunk (Step B's chunk, every command answered by all four ranks, no stage line)."""
    emit, arms = tc1.C1Emit(), []
    emit.spans = {}
    for r in (T.arms(T.GRADED) if rows is None else rows):
        name, band, size = r['label'], r['band'], r['size']
        cuts = tap.band_cuts(band)
        first_line = len(emit.lines)
        start = emit.tick(5_000_000)
        emit.tick(500_000)
        chunks = len(cuts) - 1
        lives = [8] + [0] * (chunks - 1)
        if r['overlap']:
            emit.overlap_arm(cuts, size=size, lives=lives, c1=r['c1'], band=band)
        else:
            for c, live in enumerate(lives):
                for rank in range(4):
                    emit.receipt_memory('off', band, rank, emit.txn + 1, cuts[c], cuts[c + 1])
                before = emit.seq
                emit.alpha_chunk(cuts[c], cuts[c + 1], size=size, live=live, local=r['stage_local'])
                emit.observe(emit.txn, cuts[c], cuts[c + 1], range(before + 1, emit.seq + 1))
        end = emit.tick(1_000_000)
        emit.spans[name] = (first_line, len(emit.lines))
        layer = tc1.LAYER_COMMANDS[size] * chunks
        fused = dict(selections=0, chunks=chunks, grants=12 * chunks, fused_commands=12 * chunks, run_ahead_polls=0,
                     ledger_checks=50 * chunks)
        groups = dict(selections=0, chunks=chunks, plan_commands=layer, group_commands=layer, group_layers=sb.LAYERS_PER_CHUNK * chunks)
        if r['stage_local']:
            stage = dict(selections=0, chunks=chunks, headroom_refused=chunks, local_releases=2 * chunks, both_stages_ns=2000000 * chunks)
        else:
            stage = dict(selections=0, chunks=0, headroom_refused=0, local_releases=0, both_stages_ns=0)
        stage.update(tc1.c1_counters(chunks, r['overlap']))
        stage.update(tc1.defer_counters(chunks, r['defer_bind']))
        arms.append(dict(label=name, selection=r['selection'], size=size, stage_local=r['stage_local'], c0=r['c0'], part=r['part'],
                         mode='burst', cohort=r['cohort'], index=r['index'], band=band, complete=True, reasons=[],
                         wall_start_s=emit.wall(start), wall_end_s=emit.wall(end),
                         progress_delta={'chunks': chunks, 'commands': layer + 5 * chunks}, fused_delta=fused,
                         group_delta=groups, group_max_after=size, stage_delta=stage,
                         ring_delta=dict(blocked_ns=0, episodes=0, blocked_max_ns_after=0, blocked_open_after=False,
                                         limiting={'2': 0}, limiting_sole={'2': 0}),
                         stop_delta=dict(partial_replays=0, bound_exhausted=0),
                         selection_switch=dict(fused=True, route='full'), group_switch=dict(size=size, route='full'),
                         stage_switch=dict(local=r['stage_local'], overlap=r['overlap'], defer_bind=r['defer_bind'], route='full'),
                         defer_bind=dict(requested=r['defer_bind'], acknowledged=r['defer_bind'], previous=False, replied=True),
                         outputs=tan.outputs(dict(r, c0=r['c0'])),
                         fresh={'prompt_tokens_observed': cuts[-1] + 1, 'ttft_s': float(chunks)},
                         decode_only={'aggregate_tokens_per_s': 400.0, 'stream_rates_tokens_per_s': [50.0] * 8},
                         client_basis={'prefill_tokens_per_s': 900.0, 'all_open_share_of_span': 0.1}, append={'ttft_s': 2.0},
                         metrics={}, c1=r['c1'], role=r['role'],
                         numerics=(numerics_by_role or {}).get(r['role'], numerics())))
    return emit, arms


_RUNS = {}


def run_of(key='core'):
    if key not in _RUNS:
        _RUNS[key] = synthetic_run()
    emit, arms = _RUNS[key]
    return emit, copy.deepcopy(arms)


class Grader:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()

    def __call__(self, emit, arms, lines=None, part='core'):
        q = Path(tempfile.mkdtemp(dir=self.tmp.name))
        tap.write_streams(q, emit.lines if lines is None else lines)
        logs = [str(q / 'engine-stream.log'), str(q / 'engine-stream.err')]
        return box.grade(arms, {a['label']: tap.PASS for a in arms}, tap.CHECKS, logs, {a['label']: tap.ALIVE for a in arms},
                         part=part)


class GradeBase(unittest.TestCase):
    def setUp(self):
        self.grade = Grader()
        self.addCleanup(self.grade.tmp.cleanup)


def set_open_rows(emit, name, nth, value):
    """The arm's nth engine progress flush line with open_rows = value (a copy of the lines)."""
    lines = list(emit.lines)
    lo, hi = emit.spans[name]
    hits = [i for i in range(lo, hi) if 'flash-s5 progress ' in lines[i][1] and '"open_rows":0' in lines[i][1]]
    i = hits[nth]
    lines[i] = (lines[i][0], lines[i][1].replace('"open_rows":0', '"open_rows":%d' % value, 1)) + tuple(lines[i][2:])
    return lines


class Grade(GradeBase):
    def test_the_six_arms_pass_with_the_numerics_and_rule_sections(self):
        emit, arms = run_of()
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['failures'], result['missing']), ('PASS', [], []), result['missing'])
        self.assertEqual(result['schema'], box.GRADE_SCHEMA)
        self.assertEqual(result['numerics_c1b']['verdict'], 'WITHIN')
        self.assertEqual(sorted(result['open_rows_rule']['arms']), ['c1defer.g2-solo-0'])
        for name, cell in result['table'].items():
            self.assertEqual(cell['stage_local'], T.by_label()[name]['stage_local'])
        text = box.report(result)
        self.assertIn('# q18a-alpha-next-c1b report', text)
        self.assertIn('## Numerics (the verdict input of this queue)', text)
        self.assertIn(rule.RULE, text)
        self.assertIs(c1.T.__name__, 'alpha_next_c1_arms')          # every seam restored
        self.assertIs(progress.progress_receipts, box.RAW_PROGRESS_RECEIPTS)
        self.assertIs(c1.an.ring_read, box.AN_RING_READ)

    def test_a_serial_arm_that_engaged_stage_local_is_inc(self):
        emit, arms = run_of()
        arms = [dict(a, stage_delta=dict(a['stage_delta'], chunks=3)) if a['label'] == 'c0off.g2-solo-0' else a for a in arms]
        result = self.grade(emit, arms)
        self.assertEqual(result['verdict'], 'INC')
        self.assertTrue(any('c0-off-arm-moved-a-stage-counter' in x for x in result['missing']), result['missing'])

    def test_numerics_outside_fails_engine_and_drift_is_inc(self):
        emit, arms = run_of()
        out = [dict(a, numerics=numerics(change_at=50)) if a['role'] == 'D' else a for a in arms]
        result = self.grade(emit, out)
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
        self.assertIn('numerics-outside:D-L1', result['failures'])
        drift = [dict(a, numerics=numerics(change_at=50)) if a['role'] == 'S3' else a for a in arms]
        result = self.grade(emit, drift)
        self.assertEqual((result['verdict'], result['classification']), ('INC', None))
        self.assertIn('numerics-control-drift:S3-S1', result['missing'])

    def test_open_rows_old_rule_on_serial_and_local_new_rule_on_defer(self):
        emit, arms = run_of()
        for name in ('c0off.g2-solo-0', 'c1off.g2-solo-1'):
            result = self.grade(emit, arms, lines=set_open_rows(emit, name, 0, 1))
            self.assertEqual(result['verdict'], 'INC')
            self.assertTrue(any(rule.SUPERSEDES in x for x in result['missing']), (name, result['missing']))
        result = self.grade(emit, arms, lines=set_open_rows(emit, 'c1defer.g2-solo-0', 0, 1))
        self.assertEqual(result['verdict'], 'INC')
        self.assertFalse(any(rule.SUPERSEDES in x for x in result['missing']), result['missing'])
        self.assertTrue(any('c1-open-rows-not-carried' in x for x in result['missing']), result['missing'])
        self.assertIsNone(result['classification'])

    def test_check_four_grades_inc_engine(self):
        emit, arms = run_of()
        facts = dict(rule=rule.RULE, reasons=['c1-open-row-not-successor'], flagged=[], checked=3)
        real = rule.check_arm
        with mock.patch.object(rule, 'check_arm', lambda flushes, overlap: facts if flushes else real(flushes, overlap)):
            result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['classification']), ('INC', ['ENGINE']))
        self.assertEqual(result['open_rows_rule']['engine_classified_arms'], ['c1defer.g2-solo-0'])

    def test_the_ring_read_takes_increments_on_the_defer_arm_only(self):
        lines = {'stage': [dict(txn=t, headroom_refused=0, headroom_blocked_ns=v) for t, v in ((1, 0), (2, 5), (3, 9))]}
        d = box.ring_read(dict(label='c1defer.g2-solo-0'), lines)
        self.assertEqual((d['headroom_blocked_ns_lines'], d['headroom_blocked_ns_max_chunk']), (9, 5))
        self.assertEqual((d['headroom_blocked_ns_lines_raw'], d['headroom_blocked_ns_max_chunk_raw']), (14, 9))
        s = box.ring_read(dict(label='c1off.g2-solo-0'), lines)
        self.assertEqual((s['headroom_blocked_ns_lines'], s['headroom_blocked_ns_max_chunk']), (14, 9))
        self.assertNotIn('headroom_blocked_ns_lines_raw', s)

    def test_c1_gates_still_bite_defer_bind_acknowledgement(self):
        emit, arms = run_of()
        bad = [dict(a, defer_bind=dict(a['defer_bind'], acknowledged=False)) if a['role'] == 'D' else a for a in arms]
        result = self.grade(emit, bad)
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))


if __name__ == '__main__':
    unittest.main()
