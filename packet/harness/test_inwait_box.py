#!/usr/bin/env python3
"""q18a-inwait box: the table, the arm's selection, the append rule, the gate both ways, the reads. CPU only; no pass here is box evidence. New logic only (the operator's discovery mode)."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import test_alpha_next_c1_queue as tq   # first: puts tools/ and the engine sources on the path
import test_attn_meta_box as tam
import inwait_arms as T
import e3_alpha_next_c1_box as c1box
import e3_attn_meta_box as am
import e3_inwait_box as box
from unittest import mock

an = c1box.an
tap = tq.tap
ROOT = Path(__file__).resolve().parents[1]
ORDER = ['sm.g2-live-0', 'ps1.g2-solo-0', 'cf.g2-solo-0', 'ps1.g2-live-0', 'cf.g2-live-0', 'cf.g2-live-1', 'ps1.g2-live-1', 'cf.g2-live-2', 'cf.g2-solo-2', 'cap.g2-live-0']
QUIET = dict(lazy_marked=0, merge_steps=0, merge_on=0, merge_off=0)
MOVED = dict(lazy_marked=4, merge_steps=4, merge_on=1, merge_off=1)
STEP_VIEW = dict(defer_bind_installed=True, defer_bind_selected=True, step_lazy_installed=True, step_lazy=False, step_merge_installed=True, step_merge='off', **{'step_' + k: 0 for k in QUIET})


_TABLE = mock.patch.object(T, 'ARM_TABLE', T.BUILD4C_TABLE)


def setUpModule():
    """These tests prove the box's rules over build-4c's table, whose off and cap rows carry them; the table this queue runs has its own tests in test_inwait_queue."""
    _TABLE.start()


def tearDownModule():
    _TABLE.stop()


class Table(unittest.TestCase):
    def test_the_arms_in_order(self):
        self.assertEqual(T.labels(), ORDER)
        self.assertEqual(T.labels('tail'), ['cap.g2-live-0'])
        self.assertEqual(T.COLD_ARM, ORDER[0])
        self.assertEqual([d['label'] for d in T.DECODE_ARMS], ['decode-am1'])
        self.assertEqual(T.REFERENCE_ARM, 'decode-am1')

    def test_each_selection_is_its_levers_and_c1_defer_group_2(self):
        for r in T.arms():
            self.assertEqual((r['attn'], r['lazy_recv'], r['decode_merge'], r['inwait']), (True, True, 'chunk', True), r['label'])
            self.assertIs(r['next_levers'], True, r['label'])
            self.assertIs(r['prepare_slices'], True, r['label'])
            self.assertIs(r['score_warm'], False, r['label'])
            self.assertIs(r['cow_fast'], r['c1'] in ('cf', 'sm'), r['label'])
            self.assertIs(r['reserved_cap'], r['c1'] == 'cap', r['label'])
            self.assertEqual((r['overlap'], r['defer_bind'], r['size']), (True, True, 2), r['label'])

    def test_the_view(self):
        view = box.table_view()
        self.assertEqual({r['c0'] for r in view.arms() if r['c1'] != 'ps1'}, {'on'})
        self.assertEqual({r['c0'] for r in view.arms() if r['c1'] == 'ps1'}, {'off'})
        self.assertEqual(view.label('on', 2, 'live', 1), 'cf.g2-live-1')
        self.assertEqual(view.label('off', 2, 'live', 1), 'ps1.g2-live-1')
        self.assertEqual(box.table_view('ps1').label('on', 2, 'live', 1), 'ps1.g2-live-1')

    def test_the_smoke_is_first_the_cap_last_and_band_two_has_its_workload_round(self):
        self.assertEqual(T.labels()[0], T.SMOKE_ARM)
        self.assertEqual(T.SMOKE_ARM, 'sm.g2-live-0')
        self.assertEqual(T.labels()[-1], 'cap.g2-live-0')
        self.assertEqual([r['label'] for r in T.arms() if r['band'] == 2], ['cf.g2-live-2', 'cf.g2-solo-2'])
        self.assertEqual(T.matched_off(T.by_label()['cf.g2-live-0']), 'ps1.g2-live-0')
        self.assertIsNone(T.matched_off(T.by_label()['cf.g2-live-2']), 'band 2 has no off arm')
        self.assertIsNone(T.matched_off(T.by_label()['sm.g2-live-0']))
        self.assertEqual(T.by_label()['cap.g2-live-0']['part'], 'tail', 'an OOM there changes no grade')


class Client(tq.FakeClient):
    def call(self, action, **fields):
        reply = super().call(action, **fields)
        if action == 'native_stage_local':
            reply.update(lazy_recv=fields.get('lazy_recv'), decode_merge=fields.get('decode_merge'), inwait=fields.get('inwait'), early_successor=fields.get('early_successor'))
        return reply


class ArmControl(unittest.TestCase):
    def release(self, sel, client):
        with mock.patch.object(T, 'ARM_TABLE', T.BUILD4C_TABLE + (('core', 'cfa', 'live', 0),)):
            row = next(r for r in T.arms() if r['c1'] == sel)
        order = []

        def select(_client, attn, record):
            order.append((attn, [a for a, _ in client.calls][-1]))
            record['attn_meta'] = dict(requested=attn, select={'0': {}})

        def slots(_client, attn, wanted, record):
            order.append(('levers', wanted, [a for a, _ in client.calls][-1]))
            for name, on in wanted.items():
                record[name] = dict(requested=on, select={'0': {}})
        with box._bound(an, box.arm_names(sel, row)), mock.patch.object(am, 'select_into', select), mock.patch.object(box, 'levers_into', slots):
            record = {}
            an.release_alpha_next(2, True)(client, record, 'burst')
        return record, order

    def test_every_selection_sends_its_fields_and_selects_inside_the_pause(self):
        for sel in T.SELECTIONS:
            client = Client()
            record, order = self.release(sel, client)
            self.assertEqual(dict(client.calls)['native_stage_local'], dict(local=True, overlap=True, defer_bind=True, lazy_recv=T.LAZY_RECV[sel], decode_merge=T.DECODE_MERGE[sel], inwait=T.INWAIT[sel], early_successor=T.NEXT[sel]))
            self.assertEqual(record['inwait'], dict(requested=T.INWAIT[sel], acknowledged=T.INWAIT[sel], early_requested=T.NEXT[sel], early_acknowledged=T.NEXT[sel]))
            self.assertEqual(dict(client.calls)['native_stage_local']['early_successor'], T.NEXT[sel])
            self.assertEqual(record['step'], dict(requested_lazy=T.LAZY_RECV[sel], requested_merge=T.DECODE_MERGE[sel], replied=True, acknowledged_lazy=T.LAZY_RECV[sel], acknowledged_merge=T.DECODE_MERGE[sel]))
            self.assertEqual(order, [('levers', dict(slots_fast=T.NEXT[sel], prepare_slices=T.SLICES[sel], score_warm=T.WARM[sel], cow_fast=T.COW_FAST[sel], reserved_cap=T.RESERVED_CAP[sel]), 'native_stage_local'), (T.ATTN[sel], 'native_stage_local')], sel)    # after the stage command, before release_auto_arm

    def test_the_actions_are_the_list_the_router_smoke_holds(self):
        client = Client()
        self.release('ps1', client)
        self.assertEqual([a for a, _ in client.calls], [a for a, _ in box.ACTIONS if a not in ('status', 'qualification_worker')])

    def test_no_select_is_an_error(self):
        row = next(r for r in T.arms() if r['c1'] == 'ps1')

        class Skips(Client):
            def __init__(self):
                super().__init__()

        with box._bound(an, box.arm_names('ps1', row)), mock.patch.object(am, '_Selecting', lambda client, attn, record: client), mock.patch.object(box, 'levers_into', lambda *a: None):
            with self.assertRaises(ValueError):
                an.release_alpha_next(2, True)(Skips(), {}, 'burst')


class AppendRule(unittest.TestCase):
    def record(self, **stage):
        return dict(reasons=[box.APPEND_REASON], complete=False, fused_delta_append=dict(a=0), group_delta_append=dict(b=0), stage_delta_append=dict(dict(c=0, step_lazy_marked=39), **stage))

    def test_only_step_keys_moved_is_exempt_and_the_delta_stays(self):
        record = self.record()
        self.assertIs(box.append_rule(record), True)
        self.assertEqual((record['reasons'], record['complete'], record['append_step_exempt'], record['stage_delta_append']['step_lazy_marked']), ([], True, dict(step_lazy_marked=39), 39))

    def test_any_other_counter_keeps_the_reason(self):
        for name, edit in (('stage', dict(stage_delta_append=dict(c=1, step_lazy_marked=39))), ('fused', dict(fused_delta_append=dict(a=1))), ('group', dict(group_delta_append=dict(b=2)))):
            record = dict(self.record(), **edit)
            self.assertIs(box.append_rule(record), False, name)
            self.assertEqual((record['reasons'], record['complete']), ([box.APPEND_REASON], False), name)
            self.assertNotIn('append_step_exempt', record)

    def test_another_reason_or_an_error_stays_incomplete(self):
        record = dict(self.record(), reasons=[box.APPEND_REASON, 'other'])
        box.append_rule(record)
        self.assertEqual((record['reasons'], record['complete']), (['other'], False))
        record = dict(self.record(), error=dict(type='x'))
        box.append_rule(record)
        self.assertIs(record['complete'], False)

    def test_the_real_arm_end_to_end_takes_rc_0(self):
        row = T.by_label()['ps1.g2-live-0']
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / 'arm.json')

            def inner(args):
                Path(args.out).write_text(json.dumps(self.record()))
                return 2
            args = mock.Mock(spec=['c1', 'size', 'cohort', 'index', 'out'], c1='ps1', size=2, cohort='live', index=0, out=out)
            with mock.patch.object(an, 'run_arm', inner), mock.patch.object(am, 'end_read', lambda a: dict(read=None)), mock.patch.object(box.argparse, 'Namespace', lambda **k: mock.Mock(out=k['out'])), \
                    mock.patch.object(box, 'vars', lambda _a: dict(out=out + '.inner.json'), create=True):
                self.assertEqual(box.run_arm(args), 0)
            record = json.loads(Path(out).read_text())
            self.assertIs(record['complete'], True)
            self.assertEqual(record['levers'], dict(attn=row['attn'], lazy_recv=True, decode_merge='chunk', inwait=True, next_levers=True, prepare_slices=row['prepare_slices'], score_warm=row['score_warm']))


def moved_of(row):
    return dict(QUIET, lazy_marked=4 if row['lazy_recv'] else 0, **({k: MOVED[k] for k in ('merge_steps', 'merge_on', 'merge_off')} if row['decode_merge'] != 'off' else {}))


class OpenRows(unittest.TestCase):
    """check_arm_inwait over the boundary the engine recorded (review section 7), not n's last apply."""

    OVERLAP = {5: (frozenset({1, 2}), 6), 6: (frozenset({3, 4, 9}), None)}
    COMMIT = 150000

    def flushes(self, carried, named=None, **identity):
        own = [dict(seq=1, t_submit=10000, t_apply=100000), dict(seq=2, t_submit=20000, t_apply=90000)]
        next_rows = carried + [dict(seq=9, t_submit=500000, t_apply=600000, origin='tick')]
        named = [r['seq'] for r in carried] if named is None else named
        return [dict(line=1, identity=dict(dict(txn=5, open_rows=len(named), commit_ns=self.COMMIT, open_seqs=named), **identity), rows=dict(n=own)),
                dict(line=2, identity=dict(txn=6, open_rows=0, commit_ns=700000, open_seqs=[]), rows=dict(n=next_rows))]

    def reasons(self, flushes):
        return box.check_arm_inwait(flushes, self.OVERLAP)['reasons']

    def test_rows_out_at_the_recorded_boundary_are_carried_whatever_the_last_apply(self):
        # Both sent after n's last apply (100000), before the commit: what 'last apply' called not carried.
        flushes = self.flushes([dict(seq=3, t_submit=120000, t_apply=200000, origin='wait'), dict(seq=4, t_submit=130000, t_apply=210000, origin='tick')])
        self.assertEqual(box.rule.check_arm(flushes, self.OVERLAP)['reasons'], ['c1-open-rows-not-carried'])
        out = box.check_arm_inwait(flushes, self.OVERLAP)
        self.assertEqual(out['reasons'], [])
        self.assertEqual((out['flagged'][0]['carried_seqs'], out['flagged'][0]['commit_ns']), ([3, 4], self.COMMIT))

    def test_no_record_is_named_not_inferred(self):
        flushes = self.flushes([dict(seq=3, t_submit=95000, t_apply=200000, origin='wait')])
        for key in ('commit_ns', 'open_seqs'):
            del flushes[0]['identity'][key]
        self.assertEqual(self.reasons(flushes), ['c1-commit-boundary-absent'])

    def test_a_named_row_missing_from_the_next_line_stays_inc(self):
        flushes = self.flushes([dict(seq=3, t_submit=95000, t_apply=200000, origin='wait')], named=[3, 4])
        self.assertIn('c1-open-rows-not-carried', self.reasons(flushes))

    def test_a_count_that_differs_from_the_names_stays_inc(self):
        flushes = self.flushes([dict(seq=3, t_submit=95000, t_apply=200000, origin='wait')], open_rows=3)
        self.assertEqual(self.reasons(flushes), ['c1-open-rows-not-carried'])

    def test_a_row_of_the_older_chunk_stays_inc(self):
        flushes = self.flushes([dict(seq=2, t_submit=95000, t_apply=200000, origin='wait')])
        self.assertEqual(self.reasons(flushes), ['c1-open-row-not-successor'])

    def test_a_row_sent_after_the_commit_stays_inc(self):
        flushes = self.flushes([dict(seq=3, t_submit=160000, t_apply=200000, origin='wait')])
        self.assertEqual(self.reasons(flushes), ['c1-open-row-sent-after-the-commit'])

    def test_a_row_applied_before_the_commit_stays_inc(self):
        flushes = self.flushes([dict(seq=3, t_submit=95000, t_apply=140000, origin='wait')])
        self.assertEqual(self.reasons(flushes), ['c1-open-row-applied-before-the-commit'])

    def test_a_row_out_at_the_commit_that_was_not_named_stays_inc(self):
        rows = [dict(seq=3, t_submit=95000, t_apply=200000, origin='wait'), dict(seq=4, t_submit=96000, t_apply=210000, origin='wait')]
        self.assertEqual(self.reasons(self.flushes(rows, named=[3])), ['c1-row-out-at-the-commit-not-recorded'])

    def test_a_row_at_the_boundary_within_the_lines_microsecond_is_carried(self):
        flushes = self.flushes([dict(seq=3, t_submit=self.COMMIT + 900, t_apply=self.COMMIT - 900, origin='wait')])
        self.assertEqual(self.reasons(flushes), [])

    def test_a_clean_flush_is_the_rule_unchanged(self):
        flushes = self.flushes([])
        self.assertEqual(box.check_arm_inwait(flushes, self.OVERLAP), box.rule.check_arm(flushes, self.OVERLAP))

    def test_the_read_counts_the_records(self):
        flushes = self.flushes([dict(seq=3, t_submit=95000, t_apply=200000, origin='wait')])
        self.assertEqual(box.commit_boundary_read(dict(a=flushes)), dict(a=dict(flushes=2, recorded=2, open_rows=1, max_open=1)))


class Smoke(unittest.TestCase):
    RANKS = ('0', '1', '2', '3')

    def good(self):
        each = dict.fromkeys(self.RANKS, 3)
        return dict(label=T.SMOKE_ARM, complete=True, reasons=[], outputs=dict(fresh=[1], append=[2], standing={'0': {}}), generation_requests_expected=10,
                    inwait=dict(acknowledged=True, early_acknowledged=True), attn_meta=dict(select={'0': {}}), stage_delta=dict(step_iw_applied=9, step_iw_early_begins=4),
                    progress_delta=dict(chunks=66, line_errors=0), slots_fast=dict(calls=dict(each)), prepare_slices=dict(counters=dict(sliced=dict(each))),
                    cow_fast=dict(counters=dict(calls=dict(each), rows=dict(each))))

    def test_an_engaged_arm_passes(self):
        self.assertEqual(box.smoke_check(self.good()), (True, []))

    def test_each_failure_has_its_name(self):
        for edit, reason in (
                (lambda r: r.update(complete=False), 'smoke-arm-not-complete'),
                (lambda r: r.update(reasons=['x']), 'smoke-arm-not-complete'),
                (lambda r: r['outputs'].pop('fresh'), 'smoke-request-shape-not-observed'),
                (lambda r: r.update(generation_requests_expected=9), 'smoke-request-shape-differs'),
                (lambda r: r['inwait'].update(acknowledged=False), 'smoke-control-not-acknowledged'),
                (lambda r: r['attn_meta'].update(select=None), 'smoke-select-not-sent'),
                (lambda r: r['progress_delta'].update(chunks=0), 'smoke-no-native-chunks'),
                (lambda r: r['stage_delta'].update(step_iw_applied=0), 'smoke-in-wait-not-engaged'),
                (lambda r: r['stage_delta'].update(step_iw_early_begins=0), 'smoke-early-successor-not-engaged'),
                (lambda r: r['slots_fast']['calls'].update({'2': 0}), 'smoke-slots-fast-not-engaged-on-every-rank'),
                (lambda r: r['prepare_slices']['counters']['sliced'].pop('1'), 'smoke-sliced-not-engaged-on-every-rank'),
                (lambda r: r['cow_fast']['counters']['calls'].update({'3': 0}), 'smoke-calls-not-engaged-on-every-rank'),
                (lambda r: r['cow_fast']['counters']['rows'].update({'0': 0}), 'smoke-rows-not-engaged-on-every-rank')):
            record = self.good()
            edit(record)
            ok, reasons = box.smoke_check(record)
            self.assertFalse(ok, reason)
            self.assertIn(reason, reasons)

    def test_an_unreadable_record_does_not_pass(self):
        self.assertEqual(box.smoke_check(None), (False, ['smoke-record-unreadable']))

    def test_the_cli_writes_its_record_and_says_so_in_the_rc(self):
        with tempfile.TemporaryDirectory() as root:
            arm, out = Path(root) / 'arm.json', Path(root) / 'smoke.json'
            arm.write_text(json.dumps(self.good()))
            self.assertEqual(box.main(['smoke', '--arm', str(arm), '--out', str(out)]), 0)
            self.assertIs(json.loads(out.read_text())['engaged'], True)
            arm.write_text(json.dumps(dict(self.good(), complete=False)))
            out2 = Path(root) / 'smoke2.json'
            self.assertEqual(box.main(['smoke', '--arm', str(arm), '--out', str(out2)]), 2)
            self.assertIs(json.loads(out2.read_text())['engaged'], False)


def synthetic(part):
    emit, arms = tq.synthetic_run(rows=[dict(r, c1='defer') for r in T.arms(part)])
    declared = T.by_label()
    for a in arms:
        row = declared[a['label']]
        lazy, merge = row['lazy_recv'], row['decode_merge']
        a['step'] = dict(requested_lazy=lazy, requested_merge=merge, replied=True, acknowledged_lazy=lazy, acknowledged_merge=merge)
        a['stage_switch'].update(lazy_recv=lazy, decode_merge=merge)
        a['stage_delta'].update({'step_' + k: v for k, v in moved_of(row).items()})
        on, nxt = row['inwait'], row['next_levers']
        a['inwait'] = dict(requested=on, acknowledged=on, early_requested=nxt, early_acknowledged=nxt)
        a['slots_fast'] = dict(requested=nxt, calls={str(r): 40 if nxt else 0 for r in range(4)})
        per = lambda v: {str(r): v for r in range(4)}
        cut, warm = row['prepare_slices'], row['score_warm']
        a['prepare_slices'] = dict(requested=cut, counters=dict(sliced=per(60 if cut else 0), yields=per(70 if cut else 0), peek_missing=per(0)))
        a['score_warm'] = dict(requested=warm, counters=dict(allocs=per(1 if warm else 0), flushes=per(0), rewarms=per(0), errors=per(0)))
        fast, cap = row['cow_fast'], row['reserved_cap']
        a['cow_fast'] = dict(requested=fast, counters=dict(calls=per(2 if fast else 0), pairs=per(2 if fast else 0), fallbacks=per(0), rows=per(9), row_errors=per(0)))
        a['reserved_cap'] = dict(requested=cap, counters=dict(flushes=per(1 if cap else 0)))
        a['stage_delta'].update({'step_iw_' + k: 0 for k in box.IW_DELTA})
        a['stage_delta'].update(step_iw_applied=30 if on else 0, step_iw_direct_opens=1 if on else 0, step_iw_direct_releases=1 if on else 0, step_iw_early_begins=30 if nxt else 0)
        a['progress'] = [dict(label=point, native_stage_local=dict(STEP_VIEW, step_lazy=lazy, step_merge=merge, step_iw_selected=on)) for point in ('before-fresh', 'after-fresh')]
        a['attn_meta'] = tam.block(row['attn'], [40] * 4 if row['attn'] else [0] * 4)
    return emit, arms


_RUN = {}


def run_of(part=T.GRADED):
    key = (part, T.ARM_TABLE)      # the fixture's table or the queue's own
    if key not in _RUN:
        _RUN[key] = synthetic(part)
    emit, arms = _RUN[key]
    return emit, copy.deepcopy(arms)


def decode_records(rates=(440.0,)):
    out = []
    for d, rate in zip(T.DECODE_ARMS, rates):
        record = tam.decode_record(d, rate)
        record['attn_meta'] = tam.block(d['attn'], [40] * 4 if d['attn'] else [0] * 4)
        record.update(wall_start_s=1000.0, wall_end_s=1060.0)
        out.append(record)
    return out


def full_run(part=T.GRADED):
    # The core grade runs before the tail: neither its record nor its lines exist yet. The read has every arm.
    emit, arms = run_of(part)
    return emit, decode_records() + arms


def grade_with(root, emit, arms, startup=tam.STARTUP, box_module=box, part='core'):
    q = Path(tempfile.mkdtemp(dir=root))
    tap.write_streams(q, emit.lines)
    extra = q / 'startup.log'
    extra.write_text(''.join('INFO x %s\n' % line for line in startup))
    logs = [str(q / 'engine-stream.log'), str(q / 'engine-stream.err'), str(extra)]
    return box_module.grade(arms, {a['label']: tap.PASS for a in arms}, tap.CHECKS, logs, {a['label']: tap.ALIVE for a in arms}, part=part)


class Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def grade(self, emit, arms, **kw):
        return grade_with(self.tmp.name, emit, arms, **kw)

    def edited(self, name, edit):
        emit, arms = full_run()
        edit(next(a for a in arms if a['label'] == name))
        return self.grade(emit, arms)

    def test_the_record_passes_with_every_section(self):
        result = self.grade(*full_run())
        self.assertEqual(result['verdict'], 'PASS', (result['failures'], result['missing']))
        self.assertEqual(result['schema'], 'inwait-grade/1')
        self.assertIs(result['numerics_covered'], False)
        self.assertEqual(sorted(result['step_gate']['arms']), sorted(ORDER[:-1]))
        self.assertEqual(sorted(result['attn_meta']), sorted(ORDER[:-1] + ['decode-am1']))
        self.assertIsNone(result['step_read']['error'])
        self.assertEqual(len(result['step_read']['pairs']), 3)     # cf live x2 and solo
        read = result['compose_read']
        self.assertEqual(sorted(read['arms']), sorted(ORDER))         # the table's every row, the cap's empty until the read
        cell = read['arms']['ps1.g2-live-0']
        self.assertIsNotNone(cell['prefill_tok_s'])
        self.assertIsNotNone(cell['decode_ratio_own_window'])
        self.assertEqual(cell['merge']['merge_steps'], 4)
        self.assertEqual(cell['attn_batched'], [40] * 4)
        during = result['table']['ps1.g2-live-0']['decode_during_tok_s']
        self.assertEqual(cell['decode_vs_fixed_reference'], round(during / T.FIXED_REFERENCE_TOK_S, 4))
        self.assertEqual(cell['decode_vs_this_run_am1'], round(during / 440.0, 4))
        self.assertEqual(cell['inwait']['applied'], 30)
        self.assertIsNotNone(read['pair']['live_on_over_off'])
        self.assertIn('ps1 band 1 live', read['cells'])
        text = box.report(result)
        self.assertTrue(text.startswith('# q18a-inwait'), text[:80])
        self.assertIn('## inwait reads', text)
        self.assertIn('Not covered: numerics', text)

    def test_the_read_takes_every_arm(self):
        emit, arms = full_run(None)
        read = grade_with(self.tmp.name, emit, arms, part='all')
        self.assertEqual(read['verdict'], 'PASS', (read['failures'], read['missing']))
        self.assertEqual(sorted(read['compose_read']['arms']), sorted(ORDER))
        self.assertIn('sm band 1 live', read['compose_read']['cells'])
        self.assertIsNotNone(read['compose_read']['pair']['live_cap_over_off'])

    def test_attn_leak_and_not_engaged(self):
        result = self.edited('ps1.g2-live-1', lambda a: a['attn_meta'].update(tam.block(True, [0, 40, 40, 40])))
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('ps1.g2-live-1:attn-meta-not-engaged', result['missing'])

    def test_step_gate_kept(self):
        result = self.edited('ps1.g2-live-0', lambda a: a['stage_delta'].update(step_merge_steps=0))
        self.assertIn('ps1.g2-live-0:step-selected-lever-never-moved:merge_steps', result['missing'])

    def test_inwait_check(self):
        result = self.edited('ps1.g2-live-0', lambda a: a['stage_delta'].update(step_iw_applied=0))
        self.assertIn('ps1.g2-live-0:inwait-selected-applied-nothing-in-the-wait', result['failures'])
        result = self.edited('ps1.g2-live-0', lambda a: a['cow_fast']['counters']['calls'].update({'0': 2}))
        self.assertIn('ps1.g2-live-0:cow-fast-off-moved-its-counter', result['failures'])
        result = self.edited('cf.g2-live-0', lambda a: a['cow_fast']['counters']['calls'].update({'1': 0}))
        self.assertIn('cf.g2-live-0:cow-fast-selected-not-engaged', result['missing'])
        result = self.edited('ps1.g2-live-0', lambda a: a['cow_fast']['counters']['rows'].update({'2': 0}))
        self.assertIn('ps1.g2-live-0:cow-receipt-rows-absent', result['missing'])
        result = self.edited('ps1.g2-live-0', lambda a: a['prepare_slices']['counters']['sliced'].update({'3': 0}))
        self.assertIn('ps1.g2-live-0:prepare-slices-selected-not-engaged', result['missing'])
        result = self.edited('ps1.g2-live-0', lambda a: a['prepare_slices'].pop('counters'))
        self.assertIn('ps1.g2-live-0:prepare-slices-counters-absent', result['missing'])
        result = self.edited('ps1.g2-live-0', lambda a: a['score_warm']['counters']['allocs'].update({'0': 1}))
        self.assertIn('ps1.g2-live-0:score-warm-off-moved-its-counter', result['failures'])
        result = self.edited('cf.g2-live-0', lambda a: a['score_warm']['counters']['errors'].update({'2': 1}))
        self.assertIn('cf.g2-live-0:score-warm-errors', result['missing'])
        result = self.edited('ps1.g2-live-0', lambda a: a['stage_delta'].update(step_iw_early_begins=0))
        self.assertIn('ps1.g2-live-0:early-successor-selected-not-engaged', result['missing'])
        result = self.edited('ps1.g2-live-1', lambda a: a['slots_fast'].update(calls={'0': 0, '1': 40, '2': 40, '3': 40}))
        self.assertIn('ps1.g2-live-1:slots-fast-selected-not-engaged', result['missing'])
        result = self.edited('ps1.g2-live-1', lambda a: a['inwait'].update(early_acknowledged=False))
        self.assertIn('ps1.g2-live-1:early-successor-selection-not-acknowledged', result['failures'])
        result = self.edited('ps1.g2-live-1', lambda a: a['inwait'].update(acknowledged=False))
        self.assertIn('ps1.g2-live-1:inwait-selection-not-acknowledged', result['failures'])
        result = self.edited('ps1.g2-solo-0', lambda a: a['stage_delta'].update(step_iw_applied=0))
        self.assertEqual(result['verdict'], 'PASS', 'a solo arm may apply nothing in the wait')

    def test_decode_arm_missing_or_a_leak(self):
        emit, arms = run_of()
        result = self.grade(emit, arms)
        self.assertIn('decode-am1:no-arm-record', result['missing'])

    def test_startup_line_missing_is_inc(self):
        result = self.grade(*full_run(), startup=tam.STARTUP[:2] + tam.STARTUP[3:])
        self.assertIn('attn-meta-startup-line-missing:rank-2', result['missing'])

    def test_a_kept_append_reason_stays_inc(self):
        result = self.edited('ps1.g2-live-1', lambda a: a.update(complete=False, reasons=[box.APPEND_REASON]))
        self.assertNotEqual(result['verdict'], 'PASS')


if __name__ == '__main__':
    unittest.main()
