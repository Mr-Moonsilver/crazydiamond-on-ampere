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
* Budget / ArmLoop / Commands / QueueArgparse / Derivation / NoClock (part 2): the relative budget, the body under real bash over
  stub clients with the real grade, every python call parsed by its real parser, the texts against alpha-next-c1's, no clock.
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



# --- part 2: the budget, the derived texts, the commands and the arm loop in bash ------------------------------------------

import ast  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402

sys.path.insert(0, str(ROOT / 'tools'))
import alpha_next_c1b_gate as gate  # noqa: E402
import alpha_next_c1b_queue as queue  # noqa: E402
import alpha_next_c1_gate as c1_gate  # noqa: E402
import alpha_next_c1_queue as c1_queue  # noqa: E402

ap = c1.ap
CORE = T.labels('core')
ALL_LABELS = T.labels()
DRAFT = 'harness/runs/q18a_alpha_next_c1b.draft.sh'
SETTINGS = 'harness/runs/q18a_alpha_next_c1b_settings.sh'
DECLARE = 'harness/runs/q18a_alpha_next_c1b_declare.sh'
C1_DRAFT = 'harness/runs/q18a_alpha_next_c1.draft.sh'


class Budget(unittest.TestCase):
    def test_the_box_budget_and_the_entry_limits_are_relative_and_c1s_formulas(self):
        self.assertEqual((gate.BOX_BUDGET_S, gate.CLEANUP_S, gate.ACTIVE_S), (7800, 900, 6900))
        self.assertEqual((gate.ARM_RESERVE_S, gate.TAIL_RESERVE_S, gate.END_RESERVE_S),
                         (c1_gate.ARM_RESERVE_S, c1_gate.TAIL_RESERVE_S, c1_gate.END_RESERVE_S))
        self.assertEqual((gate.LAST_ARM_ENTRY_S - c1_gate.LAST_ARM_ENTRY_S, gate.LAST_TAIL_ENTRY_S - c1_gate.LAST_TAIL_ENTRY_S),
                         (gate.ACTIVE_S - c1_gate.ACTIVE_S,) * 2)
        self.assertEqual((gate.LAST_ARM_ENTRY_S, gate.LAST_TAIL_ENTRY_S), (2220, 3270))
        self.assertEqual((gate.ARMS, gate.CORE_ARMS, gate.TAIL_ARMS), (6, 6, 0))
        self.assertEqual(gate.PHASES['alpha_next_c1b_arms'], 6 * (1500 + 30))
        self.assertEqual({k: v for k, v in gate.PHASES.items() if k != 'alpha_next_c1b_arms' and k in c1_gate.PHASES
                          and k not in ('arm_entries', 'ingress')},
                         {k: v for k, v in c1_gate.PHASES.items() if k != 'alpha_next_c1_arms' and k not in ('arm_entries', 'ingress')})
        self.assertTrue(gate.arm_fits(2220) and not gate.arm_fits(2221))

    def test_six_arms_enter_at_the_walls_at_1_5x_and_at_2x(self):
        e = gate.EXPECTED_RUNTIME_S['nominal']
        self.assertEqual((e['arms_entering'], e['not_run']), (6, []))
        # first arm +480 s; the cold first 246 + 26, then 75 + 26 each; core FULL and grade 150, the end 600.
        self.assertEqual(e['last_arm_enters_s'], 480 + 272 + 4 * 101)
        self.assertEqual((e['last_arm_ends_s'], e['total_s'], e['total_min']), (1257, 2007, 33.5))
        self.assertEqual(e['min_margin_s'], 2220 - 1156)
        self.assertEqual(gate.EXPECTED_RUNTIME_S['slower']['not_run'], [])
        at2 = gate._estimate(lambda row, first: 2 * gate.arm_wall(row, first))
        self.assertEqual((at2['arms_entering'], at2['not_run']), (6, []))
        self.assertGreater(at2['min_margin_s'], 0)

    def test_the_tool_prints_the_table_with_its_limits(self):
        text = queue.table_text()
        for name, role in zip(ALL_LABELS, T.ROLES):
            self.assertRegex(text, r'\b%s\s+%s\b' % (role, re.escape(name)))
        self.assertIn('LAST_ARM_ENTRY_S 2220 s', text)
        self.assertIn('at 1.5x the walls: arms entering 6', text)


STUB_BOX = r"""#!/usr/bin/env python3
import json, os, shutil, sys, time
if sys.argv[1] in ('grade', 'read'):
    with open(os.environ['CALLS'], 'a') as f:
        f.write(json.dumps(dict(script='box', action=sys.argv[1])) + '\n')
    sys.path.insert(0, os.environ['REAL_HARNESS'])
    import e3_alpha_next_c1b_box as real
    raise SystemExit(real.main(sys.argv[1:]))
a = sys.argv
out = a[a.index('--out') + 1]
prefix = dict(serial='c0off', off='c1off', defer='c1defer')[a[a.index('--c1') + 1]]
label = '%s.g%s-%s-%s' % (prefix, a[a.index('--size') + 1], a[a.index('--cohort') + 1], a[a.index('--index') + 1])
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(dict(script='box', action='arm', label=label, c1=a[a.index('--c1') + 1],
                            workload=os.path.basename(a[a.index('--workload') + 1]))) + '\n')
if label == os.environ.get('DIE_IN'):
    with open(os.environ['Q'] + '/engine-stream.log', 'a') as f:
        f.write(os.environ['DEATH_LINE'])
    time.sleep(600)
shutil.copy(os.environ['ARMS'] + '/arm-%s.json' % label, out)
raise SystemExit(int(os.environ.get('RC_' + label.replace('.', '_').replace('-', '_'), '0')))
"""


class ArmLoop(unittest.TestCase):
    """queue.body() runs as written under bash; the arm and check clients are stubs, the watch, the grade and the read real."""

    def setUp(self):
        import shutil
        self.root = Path(tempfile.mkdtemp(prefix='alpha-next-c1b-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death.py', 'alpha_next_c1b_gate.py', 'alpha_next_c1b_arms.py', 'alpha_next_arms.py',
                     'alpha_next_stop_race.py', 'native39_trunk_gate.py', 'native40_box_gate.py', 'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_alpha_next_c1b_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def run_loop(self, *, jump_after=None, jump_to='LAST_ARM_ENTRY_S + 10', die_in=None, rcs=None, timeout=420, edit=None):
        import signal
        emit, arms = run_of()
        arms = [dict(a, **(edit or {}).get(a['label'], {})) for a in arms]
        tap.write_streams(self.q, emit.lines)
        for a in arms:
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        draft = queue.derive()[DRAFT]
        body = queue.body()
        self.assertIn(body, draft)
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; STOP_RACE=0; CONTAINER=qwen-flash-prod; SERVICE=vllm-flash; N0=0",
                  'Q=%s; H=%s; PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; _cid=c; IDLE_TIMEOUT_S=1' % (self.q, self.h),
                  'Q18A_READINESS=/r/ready; S5_OVERLAY=/o; S5_LAUNCHER_NAME=l.sh', 'health_ok() { return 0; }']
        script += [tap.le_fix_function(draft, name) for name in tap.LOOP_FUNCTIONS]
        if jump_after:
            for fn in ('record', 'read_step'):
                script += ['%s_hook() { [[ $1 == %s ]] && SECONDS=$((%s)); return 0; }' % (fn, jump_after, jump_to)]
            body = body.replace('read_step() {\n  local v\n', 'read_step() {\n  read_step_hook "$1"\n  local v\n')
            script += ['eval "$(declare -f record | sed \'1s/record/real_record/\')"',
                       'record() { real_record "$@"; local rc=$?; record_hook "$1"; return $rc; }']
        script += [body, 'echo "END FAILS=$FAILS INCS=$INCS CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"']
        path = self.root / 'loop.sh'
        path.write_text('\n'.join(script) + '\n')
        env = {k: v for k, v in os.environ.items() if k != 'HARNESS_DIR'}
        env.update(PATH='%s:%s' % (self.bin, os.environ['PATH']), Q=str(self.q), CALLS=str(self.root / 'calls.jsonl'),
                   ARMS=str(self.arms), REAL_HARNESS=str(HERE), DEATH_LINE=tap.DEATH_LINE, PYTHONDONTWRITEBYTECODE='1')
        for name, rc in (rcs or {}).items():
            env['RC_' + name.replace('.', '_').replace('-', '_')] = str(rc)
        if die_in:
            env['DIE_IN'] = die_in
        proc = subprocess.Popen(['bash', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        calls = [json.loads(x) for x in (self.root / 'calls.jsonl').read_text().splitlines()]
        steps = {m.group(1): m.group(2) for m in re.finditer(r'^STEP (\S+) (\S+)', out, flags=re.M)}
        return out, err, calls, steps

    def grade_file(self, name='alpha-next-c1b-grade.json'):
        return json.loads((self.q / name).read_text())

    def test_the_six_arms_in_order_on_band_1_then_the_grade_then_the_read(self):
        out, err, calls, steps = self.run_loop()
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-1500:], err[-800:]))
        self.assertEqual([c.get('label') or c['action'] for c in calls if c['script'] == 'box'], CORE + ['grade', 'read'])
        arm_calls = [c for c in calls if c['script'] == 'box' and c['action'] == 'arm']
        self.assertEqual([c['c1'] for c in arm_calls], ['serial', 'serial', 'off', 'off', 'defer', 'serial'])
        # every arm on band 1's workload, S3's round 3 included
        self.assertEqual({c['workload'] for c in arm_calls}, {ap.BAND_WORKLOAD[1] + '.json'})
        self.assertEqual([c['out'] for c in calls if c['script'] == 'checks' and c['action'] == 'full-idle'],
                         ['entry-%s.json' % x for x in CORE] + ['core-full-idle.json', 'end-full-idle.json'])
        for name in CORE:
            self.assertEqual(steps['arm-' + name], 'PASS', name)
        self.assertEqual((steps['alpha-next-c1b'], steps['alpha-next-c1b-read'], steps['tail-engine'], steps['end-full-idle']),
                         ('PASS', 'READ-PASS', 'PASS', 'PASS'))
        result = self.grade_file()
        self.assertEqual((result['schema'], result['verdict'], result['missing'], result['required_arms']),
                         (box.GRADE_SCHEMA, 'PASS', [], CORE))
        self.assertEqual(result['numerics_c1b']['verdict'], 'WITHIN')
        read = self.grade_file('alpha-next-c1b-read.json')
        self.assertEqual((read['verdict'], read['required_arms']), ('PASS', ALL_LABELS))
        self.assertTrue((self.q / 'alpha-next-c1b-report.md').read_text().startswith('# q18a-alpha-next-c1b report'))

    def test_past_the_entry_limit_the_arms_left_are_inc_and_the_numerics_rule_is_not_cut(self):
        out, err, calls, steps = self.run_loop(jump_after='ingress-' + CORE[2])
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box' and c['action'] == 'arm'], CORE[:3])
        for name in CORE[3:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        self.assertIn('END FAILS=0 INCS=%d CRASHED=0' % (len(CORE) - 3 + 1), out, out[-800:])
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('declared-arm-missing:' + CORE[3], result['missing'])

    def test_a_numerics_difference_outside_the_bound_fails_the_queue_engine(self):
        d = T.by_role()['D']['label']
        out, err, calls, steps = self.run_loop(edit={d: dict(numerics=numerics(change_at=50))})
        self.assertIn('END FAILS=1 INCS=0 CRASHED=0', out, out[-800:])
        self.assertEqual((steps['arm-' + d], steps['alpha-next-c1b']), ('PASS', 'FAIL'))
        result = self.grade_file()
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))

    def test_a_dead_engine_stops_the_run(self):
        out, err, calls, steps = self.run_loop(die_in=CORE[1])
        self.assertIn('CRASHED=1 SOURCE=engine-death', out, out[-800:])
        self.assertEqual(steps['arm-' + CORE[1]], 'FAIL')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box' and c['action'] == 'arm'], CORE[:2])
        self.assertEqual(steps['tail'], 'READ-INCONCLUSIVE')
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn(CORE[1] + ':engine-dead', result['failures'])


def body_commands(text):
    body = text[text.index(queue.body().splitlines()[0]):]
    body = body[:body.index('read_step alpha-next-c1b-read ')]
    joined = re.sub(r'\\\n\s*', ' ', body)
    rows = []
    for line in joined.splitlines():
        match = re.search(r'python3 "\$H/([a-z0-9_]+)\.py"(.*)$', line)
        if match:
            rows.append((match.group(1), re.split(r'\s+>\s*"', match.group(2))[0].strip()))
    return rows


class Commands(unittest.TestCase):
    def test_every_body_command_parses_with_its_module(self):
        import e3_native39_trunk_checks as checks
        modules = {'e3_alpha_next_c1b_box': box, 'e3_native39_trunk_checks': checks}
        rows = body_commands(queue.derive()[DRAFT])
        self.assertEqual(sorted({m for m, _ in rows}), sorted(modules))
        values = dict(tap.QUEUE_VALUES, _label='c0off.g2-solo-3', _sel='c0off.g2', _cohort='solo', _pair='3', _c1='serial',
                      _group='g2', **{'_group#g': '2'}, _workload='/h/workloads/q18a-native41-progress.json',
                      READ_ARGS=tap.QUEUE_VALUES['GRADE_ARGS'])
        actions = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, args=None, namespace=None):
            raise tan.held.Parsed(original(parser, args, namespace))
        for module, args in rows:
            argv = tan.held.expand(args, values)
            with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture):
                with self.assertRaises(tan.held.Parsed) as parsed, contextlib.redirect_stderr(io.StringIO()):
                    modules[module].main(argv)
            actions.append((module, argv[0], vars(parsed.exception.args[0])))
        self.assertEqual([a[1] for a in actions], ['full-idle', 'arm', 'ingress', 'instruments', 'full-idle', 'grade', 'instruments',
                                                   'full-idle', 'read'])
        arm = next(ns for m, a, ns in actions if a == 'arm')
        self.assertEqual((arm['c1'], arm['size'], arm['cohort'], arm['index'], arm['control_timeout_s']), ('serial', 2, 'solo', 3, 60.0))
        grade = next(ns for m, a, ns in actions if a == 'grade')
        self.assertEqual((grade['out'], grade['report']), ('/q/alpha-next-c1b-grade.json', '/q/alpha-next-c1b-report.md'))
        read = actions[-1][2]
        self.assertEqual((read['out'], read['report']), ('/q/alpha-next-c1b-read.json', '/q/alpha-next-c1b-read.md'))

    def test_the_loop_list_is_the_table_the_tail_empty_and_the_split_runs_in_bash(self):
        draft = queue.derive()[DRAFT]
        (core,) = re.findall(r'^\s*for _arm in ([a-z0-9 :.-]+); do$', draft, flags=re.M)
        self.assertEqual(re.findall(r'^\s*for _label in (.*); do$', draft, flags=re.M), [''])
        self.assertEqual([x.split(':') for x in core.split()], [[r['part'], r['label']] for r in T.arms(T.GRADED)])
        split = ('  IFS=- read -r _sel _cohort _pair <<< "$_label"\n  IFS=. read -r _c1 _group <<< "$_sel"\n'
                 '  case $_c1 in (c0off) _c1=serial;; (c1off) _c1=off;; (c1defer) _c1=defer;; (*) _c1=unknown;; esac\n'
                 '  if [[ $_pair == %s ]]; then' % T.band1_glob())
        self.assertIn(split, draft)
        script = ('for _arm in %s core:c1on.g2-solo-0 core:c0off.g2-solo-2; do _label=${_arm#*:}\n%s echo 1; else echo 2; fi\n'
                  'printf "%%s %%s %%s %%s\\n" "$_c1" "${_group#g}" "$_cohort" "$_pair"\ndone') % (core, split)
        done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
        got = done.stdout.split('\n')
        self.assertEqual([int(x) for x in got[0::2] if x], [r['band'] for r in T.arms()] + [1, 2])
        self.assertEqual([(c, int(s), co, int(i)) for c, s, co, i in (x.split() for x in got[1::2] if x)],
                         [(r['c1'], r['size'], r['cohort'], r['index']) for r in T.arms()]
                         + [('unknown', 2, 'solo', 0), ('serial', 2, 'solo', 2)])


ARGPARSE_VALUES = dict(tan.ARGPARSE_VALUES, QNAME='q18a-alpha-next-c1b', RUN_ID='q18a-alpha-next-c1b-fixture',
                       _label='c1defer.g2-solo-0', _sel='c1defer.g2', _c1='defer', _group='g2', _cohort='solo', _pair='0',
                       _workload='/h/workloads/q18a-native41-progress.json', _arm='core:c1defer.g2-solo-0')
ARGPARSE_VALUES.pop('_c0', None)


def bash_words(args, draft, values=None):
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    arrays = ('--arm "$Q/arm-c1defer.g2-solo-0.json" --ingress "c1defer.g2-solo-0=$Q/ingress-c1defer.g2-solo-0.json" '
              '--watch "c1defer.g2-solo-0=$Q/engine-watch-arm-c1defer.g2-solo-0.json"')
    lines += ['GRADE_ARGS=(%s)' % arrays, 'READ_ARGS=(%s)' % arrays,
              'source "$H/runs/q18a_alpha_next_c1b_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class QueueArgparse(unittest.TestCase):
    def test_every_queue_python_call_parses_with_its_real_parser(self):
        held = tan.held
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        parent = held.queue_python_calls(c1_queue.derive(), (HERE / 'qlib.sh').read_text())
        swap = lambda s: 'e3_alpha_next_c1b_box.py' if s == 'e3_alpha_next_c1_box.py' else s
        self.assertEqual(sorted(swap(c[1]) for c in parent), sorted(c[1] for c in calls))
        self.assertNotIn('e3_alpha_next_c1_box.py', {c[1] for c in calls})
        setcfg = next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file())
        rows = []
        for name, script, args, form in calls:
            with self.subTest(file=name.split('/')[-1], script=script, args=args[:60]):
                rc, words, err = bash_words(args, draft)
                self.assertEqual(rc, 0, err)
                if form != 'python3':
                    words = held.QLIB_CALLS[form][1](words)
                if script == 'SETCFG':
                    verdict, detail = held.setcfg_parse(setcfg, words)
                else:
                    verdict, detail = held.real_parse(HERE / script, words)
                self.assertEqual(verdict, 'parsed', (script, words[:4], detail))
                rows.append(dict(file=name.split('/')[-1], script=script, form=form,
                                 action=words[0] if words and not words[0].startswith('-') else None,
                                 words=len(words), verdict=verdict))
        self.assertEqual(len(rows), len(calls))
        ARGPARSE_RECEIPTS[:] = rows
        self.assertEqual(SLOT_RECEIPTS, [])

        def first(script, action):
            return next(a for n, s, a, f in calls if s == script and a.split()[0] == action)
        arm, grade = first('e3_alpha_next_c1b_box.py', 'arm'), first('e3_alpha_next_c1b_box.py', 'grade')
        for label, original, (old, new) in (('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec')),
                                            ('arm-undeclared-size', arm, ('--size "${_group#g}"', '--size 3')),
                                            ('arm-undeclared-c1', arm, ('--c1 "$_c1"', '--c1 maybe')),
                                            ('arm-unmapped-prefix', arm, ('--c1 "$_c1"', '--c1 unknown')),
                                            ('arm-c1-on-not-in-this-table', arm, ('--c1 "$_c1"', '--c1 on')),
                                            ('arm-band-2-round', arm, ('--index "$_pair"', '--index 2')),
                                            ('grade-without-out', grade, ('--out "$Q/alpha-next-c1b-grade.json"', ''))):
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1)
                rc, words, err = bash_words(original.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_alpha_next_c1b_box.py', words)[0], 'refused', label)


class Derivation(unittest.TestCase):
    def test_derived_texts_are_on_disk_and_check(self):
        for name, text in queue.derive().items():
            self.assertEqual((ROOT / name).read_text(), text, name)
            self.assertEqual(subprocess.run(['bash', '-n', str(ROOT / name)]).returncode, 0, name)
        queue.main(['--check'])

    def test_undoing_the_declared_deltas_gives_alpha_next_c1s_texts_back(self):
        texts = queue.derive()
        old = {s: (ROOT / 'harness/runs' / (queue.OLD + s)).read_text() for s in ('.draft.sh', '_settings.sh', '_declare.sh')}
        draft = texts[DRAFT].replace(queue.HEADER_NOTE, '', 1).replace(queue.body(), queue.c1_body(), 1)
        draft = draft.replace(queue.INTERFACE_NEW, queue.INTERFACE_OLD, 1)
        for a, b_, count in queue.DRAFT_NAMES:
            self.assertEqual(draft.count(b_), count, b_)
            draft = draft.replace(b_, a)
        self.assertEqual(draft, old['.draft.sh'])
        settings = texts[SETTINGS].replace(queue.SETTINGS_NOTE, '', 1)
        for a, b_, count in queue.SETTINGS_NAMES:
            settings = settings.replace(b_, a)
        self.assertEqual(settings, old['_settings.sh'])
        declare = texts[DECLARE].replace(queue.PINS_NEW, queue.PINS_OLD, 1)
        for a, b_, count in queue.DECLARE_NAMES:
            declare = declare.replace(b_, a)
        self.assertEqual(declare, old['_declare.sh'])

    def test_the_engine_settings_are_c1s_and_the_new_files_are_pinned(self):
        texts = queue.derive()
        parent = c1_queue.derive()
        settings = [line for line in texts[SETTINGS].splitlines() if not line.startswith('#')]
        self.assertEqual(settings, [line for line in parent['harness/runs/q18a_alpha_next_c1_settings.sh'].splitlines()
                                    if not line.startswith('#') and 'QNAME' not in line]
                         [:0] + [line.replace('q18a-alpha-next-c1}', 'q18a-alpha-next-c1b}') for line in
                                 parent['harness/runs/q18a_alpha_next_c1_settings.sh'].splitlines() if not line.startswith('#')])
        for name in c1_queue.SWITCH_NAMES:
            self.assertEqual(texts[DRAFT].count("assert env.get('%s') == '1'" % name), 1, name)
            self.assertEqual(texts[DECLARE].count("assert env.get('%s')=='1'" % name), 1, name)
        for name in queue.C1B_FILES.split():
            self.assertEqual(texts[DECLARE].count(name), 1, name)
            self.assertIn(name, texts[DRAFT])
        self.assertIn('e3_c1_open_rows.py; do', texts[DRAFT])
        self.assertIn('from alpha_next_c1b_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,LAST_TAIL_ENTRY_S,PHASES', texts[DRAFT])
        self.assertNotIn('${_c1#c1}', queue.body())

    def test_alpha_next_c1s_texts_still_derive_to_their_files(self):
        for name, text in c1_queue.derive().items():
            self.assertEqual((ROOT / name).read_text(), text, name)


NEW_FILES = ('harness/e3_alpha_next_c1b_box.py', 'harness/alpha_next_c1b_gate.py', 'harness/alpha_next_c1b_arms.py',
             'harness/e3_c1_open_rows.py', 'tools/alpha_next_c1b_queue.py')


class NoClock(unittest.TestCase):
    def test_no_absolute_time_or_window_in_the_new_files(self):
        for name in NEW_FILES:
            text = (ROOT / name).read_text()
            for line_no, line in enumerate(text.splitlines(), 1):
                for word in tan.FORBIDDEN:
                    self.assertNotIn(word, line, '%s:%d' % (name, line_no))
        draft = queue.derive()[DRAFT]
        self.assertEqual(draft.count('$(date'), 1)
        self.assertEqual(draft.count('time.time_ns()'), 1)
        self.assertIn('RUN_ID="q18a-alpha-next-c1b-$(date -u +%Y%m%dT%H%M%SZ)-$$"', draft)
        self.assertNotIn('HARD_END', draft)

    def test_the_gate_table_rule_and_box_import_no_clock(self):
        for name in NEW_FILES:
            tree = ast.parse((ROOT / name).read_text())
            imported = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            imported |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse(imported & {'time', 'datetime', 'calendar', 'zoneinfo'}, name)


if __name__ == '__main__':
    unittest.main()
