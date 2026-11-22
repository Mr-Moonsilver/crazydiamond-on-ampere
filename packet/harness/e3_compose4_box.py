#!/usr/bin/env python3
"""q18a-compose-4 box arms and grade: every lever track on one engine, under the closed-loop agent replay (harness/compose4_arms.py). Discovery mode.

The live smoke is a1-trunk's arm with this table's row: its control, its returners, its receipts. The replay arm is the same entry and the same control, in the same order (pause, native_append_yield, resume at the idle FULL entry; inwait's
release, which selects the levers and hands the auto-arm back), then no standing cohort: compose4_replay's eight agents, a status every POLL_S, attn-meta's end read and the rank counters. The auto-arm stays released for the whole arm, so one
release serves every bulk. The one-stream arm is the same entry and release with one standing stream of the workload, then one decode window, no bulk.

The grade is small. FAIL: an engine death or signature, a startup failure or an attn-meta leak, a yield selected with the yield off. INC: an arm missing or incomplete, an ingress or check not PASS. The run's own rule is compose4_rule.py, on the hub, over the
fetched record: this grade only says the record is whole.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time

import a1_trunk_arms  # noqa: F401  (a1's box binds it)
import compose4_arms as T
import compose4_replay as replay
import e3_a1_trunk_box as a1
import e3_native_refusal_client as refusal

iw, am, ap, base, progress = a1.iw, a1.am, a1.ap, a1.base, a1.progress
shared, streams = a1.shared, a1.streams
exception_details = a1.exception_details
ARM_SCHEMA = 'compose4-arm/1'
GRADE_SCHEMA = 'compose4-grade/1'
ACTIONS = a1.ACTIONS
ACTION_NAMES = a1.ACTION_NAMES
POLL_S = 15.0
LANE = 'compose4-'
NOT_COVERED = ('numerics: no fixed-schedule check; prompts above 135,000 tokens and the replay default of rolling at 230,000 (the cap has no budget for a stock append there); real client traffic; connections 8 (a launch setting, both arms carry 32)')
SCOPE = 'the four levers at one joint point under the closed-loop replay: crash, OOM and stock prefill read by compose4_rule.py; not a release assessment and not a numerics result'


def status_row(tag, reply):
    return dict(base.status_view(tag, reply), append_yield=a1.yield_view(reply, False))


def small_status(reply):
    auto, row = reply.get('auto_arm') or {}, reply.get('append_yield') or {}
    return dict(route=reply.get('route'), route_generation=reply.get('route_generation'), requests=reply.get('scheduler_requests'), phase=auto.get('phase'), selection_seq=auto.get('selection_seq'), completed=auto.get('completed'),
                yield_counters=row.get('counters'), yield_refusals=row.get('refusals'), at_s=round(time.time(), 3))


class Poller(threading.Thread):
    """A read every POLL_S on its own client. A read never takes the lease (native_auto_arm.READS)."""

    def __init__(self, client, period=POLL_S):
        super().__init__(name='compose4-poll', daemon=True)
        self.client, self.period, self.rows, self.errors = client, period, [], 0
        self.done = threading.Event()

    def run(self):
        while not self.done.wait(self.period):
            try:
                self.rows.append(small_status(self.client.call('status')))
            except Exception:  # noqa: BLE001 - counted; the arm's own reads decide
                self.errors += 1


def entry_status(client, row, record):
    """The arm's entry: latched idle FULL, the launch bounds, no yield live; every arm the same."""
    reply = client.call('status')
    record['statuses'].append(status_row('entry', reply))
    auto = reply.get('auto_arm')
    if type(auto) is not dict:
        raise ValueError('auto-arm-policy-not-installed')
    if auto.get('manual_owned') is not True or auto.get('active') is not False or reply.get('route') != 'full' or reply.get('scheduler_requests') != 0:
        raise ValueError('arm-entry-not-latched-idle-full')
    config = (reply.get('append_yield') or {}).get('config') or {}
    if {k: config.get(k) for k in T.LAUNCH} != T.LAUNCH or (reply.get('append_yield') or {}).get('phase') is not None:
        raise ValueError('append-yield-launch-bounds-differ')


def select_and_release(client, row, record):
    a1.yield_control(client, row['yield_on'], record)
    iw.release_compose(row)(row['size'], row['stage_local'])(client, record, 'burst')
    for key, missing in (('attn_meta', 'attn-meta-select-not-sent'), ('inwait', 'inwait-field-not-sent')):
        if not record.get(key):
            raise ValueError(missing)


def base_record(a, row, client):
    return dict(schema=ARM_SCHEMA, kind=row['kind'], label=row['label'], cohort=row['cohort'], index=row['index'], part=row['part'], c1=row['c1'], size=row['size'], band=row['band'], selection=row['selection'], complete=False, reasons=[],
                statuses=[], progress=[], run_id=a.run_id, engine_epoch=client.epoch, yield_on=row['yield_on'], ple_private=row['ple_private'], wall_start_s=time.time())


def replay_arm(a, row, client, poll_client, *, run=None):
    run = run or replay.run
    record = base_record(a, row, client)
    mark = len(getattr(client, 'history', ()))
    poller = None
    sink = None
    try:
        if shared._package_gate(os.environ).get('ok') is not True:
            raise ValueError('reviewed-package-readiness-not-pass')
        entry_status(client, row, record)
        select_and_release(client, row, record)
        before = client.call('status')
        record['statuses'].append(status_row('before', before))
        record['progress'].append(dict(progress.progress_view(before) or {}, label='before', native_stage_local=iw.stage_view(before)))
        record['metrics'] = {'before': progress.preemption_metrics(a.url)}
        poller = Poller(poll_client)
        poller.start()
        record['replay_start_s'] = time.time()
        sink = replay.TurnSink(a.out + '.turns.jsonl', dict(label=row['label'], replay_start_s=record['replay_start_s'], replay_config=T.REPLAY))
        cfg, summary, turns, still_open = run(a.url, a.model, replay.key_from_env(), a.run_id, row['label'], sink=sink)
        record['replay_end_s'] = time.time()
        record.update(replay_config=cfg, replay_summary=summary, replay_still_open=still_open, replay_requests_sent=len(turns) + len(still_open))
        poller.done.set()
        poller.join(5.0)
        record['polled'] = dict(rows=poller.rows, errors=poller.errors)
        poller = None
        after = client.call('status')
        record['statuses'].append(status_row('after', after))
        record['progress'].append(dict(progress.progress_view(after) or {}, label='after', native_stage_local=iw.stage_view(after)))
        record['metrics']['after'] = progress.preemption_metrics(a.url)
        view = {s.get('label'): s for s in record['progress']}
        record['progress_delta'] = {k: progress.delta(view['after'], view['before'], k) for k in ('chunks', 'commands', 'direct_commands', 'stock_rpcs', 'waits', 'wait_sends', 'wait_capped', 'line_errors', 'trace_lines', 'trace_dropped')}
        if view['before'].get('native_stage_local') and view['after'].get('native_stage_local'):
            record['stage_delta'] = iw.stage_delta(view['after']['native_stage_local'], view['before']['native_stage_local'])
        if still_open:
            record['reasons'].append('replay-turns-still-open:%d' % len(still_open))
        if summary.get('errors'):
            record['reasons'].append('replay-turn-errors:%d' % summary['errors'])
        if summary.get('journal_write_failures'):
            record['reasons'].append('replay-journal-write-failures:%d' % summary['journal_write_failures'])
        if summary.get('agents_died'):
            record['reasons'].append('replay-agents-died:%d' % summary['agents_died'])
        fixed = summary.get('fixed_work')
        if fixed is not None and fixed.get('cap_hit') is not False:
            # brief 12: fixed work (compose4_replay, REPLAY_FIXED_WORK) whose agents did not all reach their target by the cap
            record['reasons'].append('replay-fixed-work-cap-hit:%d-of-%d-reached' % (
                sum(1 for r in fixed.get('reached_s') or () if r is not None), len(fixed.get('targets') or ())))
        over = [t['prompt_tokens'] for t in turns if type(t.get('prompt_tokens')) is int and t['prompt_tokens'] > T.MAX_PROMPT_TOKENS]
        if over:
            record['reasons'].append('replay-prompt-over-the-ceiling:%d' % len(over))
        record['control_actions'] = sorted({h.get('action') for h in list(getattr(client, 'history', ()))[mark:]})
        if not set(record['control_actions']) <= ACTION_NAMES:
            record['reasons'].append('arm-sent-a-non-receipt-action')
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        record['error'] = exception_details(exc, getattr(client, 'control', None))
        if getattr(exc, 'refusal', None) is not None:
            record['controller_refusal'] = exc.refusal
        record['reasons'].append('arm-error')
    finally:
        if poller is not None:
            poller.done.set()
            record['polled'] = dict(rows=poller.rows, errors=poller.errors)
        if sink is not None:
            sink.close()
        record['wall_end_s'] = time.time()
    record['complete'] = not record['reasons']
    return record


def one_stream_arm(a, row, cfg, budget, client, *, sleep=time.sleep):
    """One standing stream, decode only: the composed release, the workload's settle, one window. Same prompt and salt as stream 0 of the 8-stream baseline, so its cache serves it."""
    record = base_record(a, row, client)
    standing, sampler = [], None
    mark = len(getattr(client, 'history', ()))
    try:
        if shared._package_gate(os.environ).get('ok') is not True:
            raise ValueError('reviewed-package-readiness-not-pass')
        entry_status(client, row, record)
        workload = shared._workload_view(cfg, generation_budget=budget, streams=1)
        rows = streams.build_cohort(workload)
        if len(rows) != 1:
            raise ValueError('one-stream-cohort-not-one')
        body = shared._request(a.model, rows[0]['prompt'], workload, shared._cache_salt(a.run_id, progress.STANDING, 0))
        standing = streams._start_streams(a.url, a.model, [body], rows, workload)
        opened = streams._wait_open(standing, workload)
        record['open'] = {k: opened.get(k) for k in ('complete', 'reasons')}
        if opened.get('complete') is not True:
            raise ValueError('standing-stream-unopened')
        select_and_release(client, row, record)
        sampler = shared.CountSampler(standing)
        sampler.start()
        sleep(base.SETTLE_S)
        b0 = sampler.snapshot()
        sleep(T.ONE_STREAM_WINDOW_S)
        b1 = sampler.snapshot()
        record['decode_only'] = shared._stream_window(standing, sampler, b0, b1)
        record['control_actions'] = sorted({h.get('action') for h in list(getattr(client, 'history', ()))[mark:]})
        if not set(record['control_actions']) <= ACTION_NAMES:
            record['reasons'].append('arm-sent-a-non-receipt-action')
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        record['error'] = exception_details(exc, getattr(client, 'control', None))
        record['reasons'].append('arm-error')
    finally:
        if sampler:
            sampler.close()
        if standing:
            streams._stop_streams({}, standing, shared._workload_view(cfg, streams=1))
        record['wall_end_s'] = time.time()
    record['complete'] = not record['reasons']
    return record


def run_arm(a):
    name = T.label(a.c1, a.size, a.cohort, a.index)
    row = T.by_label().get(name)
    client = None
    try:
        if row is None:
            raise ValueError('arm-not-in-the-declared-table')
        client = refusal.RefusalKeepingClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        poll = refusal.RefusalKeepingClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        if row['kind'] == 'replay':
            record = replay_arm(a, row, client, poll)
        else:
            cfg, budget = a1.load_config(a, row['band'])
            record = a1.arm(a, row, cfg, budget, client, poll) if row['kind'] == 'live-arm' else one_stream_arm(a, row, cfg, budget, client)
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        record = dict(schema=ARM_SCHEMA, kind=(row or {}).get('kind'), label=name, complete=False, reasons=['arm-refused'], error=exception_details(exc))
    if row is not None and 'attn_meta' in record:
        block = record['attn_meta']
        block['end'] = am.end_read(a)
        read = (block['end'] or {}).get('read')
        block['deltas'] = am.deltas(block.get('select'), read)
        block['table'] = row['attn']
        record.setdefault('slots_fast', dict(requested=row['next_levers'], select=None))['calls'] = iw.slots_calls((record.get('slots_fast') or {}).get('select'), read)
        iw.rank_lever_counters(record, row, read)
    with open(a.out, 'x') as handle:
        json.dump(record, handle, sort_keys=True)
    print('COMPOSE4_ARM_RECORD %s %s' % (name, 'COMPLETE' if record.get('complete') is True else 'INCOMPLETE'))
    return 0 if record.get('complete') is True else 2


def grade(arms, ingress, checks, logs, watches=None, part='core'):
    known = [x for x in arms if x.get('label')]
    decode_records = {x['label']: x for x in known if x.get('kind') == 'decode-only'}
    prefill = [x for x in known if x.get('kind') != 'decode-only']
    by = {x['label']: x for x in prefill}
    want = T.arms()
    failures, missing = set(), set()
    for x in arms:
        if not x.get('label'):
            missing.add('arm-record-unreadable')
    afail, amiss, _classes, attn, start = iw.attn_gate(prefill, decode_records, ingress, watches, logs, want)
    failures |= afail
    missing |= amiss
    for name, watch in (watches or {}).items():
        if watch.get('verdict') == 'DEAD':
            failures.add('engine-death-watch:' + name)
    for name, hits in a1.engine_failures(known, logs).items():
        failures.add('engine-signature:%s:%s' % (name, ','.join(hits)))
    for name, check in (checks or {}).items():
        if check.get('verdict') != 'PASS':
            missing.add('check-%s-%s' % (name, check.get('verdict') or 'absent'))
    facts = {}
    for row in want:
        name = row['label']
        rec = by.get(name)
        if rec is None:
            missing.add(name + ':no-arm-record')
            continue
        if rec.get('complete') is not True or rec.get('reasons'):
            missing.add('%s:incomplete:%s' % (name, ','.join(rec.get('reasons') or ['no-reason'])))
        if (ingress or {}).get(name, {}).get('verdict') not in ('PASS', None):
            missing.add(name + ':ingress-not-the-arm')
        control = rec.get('yield_control') or {}
        if control.get('enabled') is not row['yield_on']:
            missing.add(name + ':yield-selection-not-acknowledged')
        ends = [s for s in rec.get('statuses') or () if s.get('label') in ('after', 'after-append')]
        counters = ((ends[-1].get('append_yield') or {}).get('counters') or {}) if ends else {}
        entries = [s for s in rec.get('statuses') or () if s.get('label') == 'entry']
        begun = ((entries[0].get('append_yield') or {}).get('counters') or {}) if entries else {}
        selected = (counters.get('selected') or 0) - (begun.get('selected') or 0)
        if not row['yield_on'] and selected:
            failures.add(name + ':yield-off-selected-a-yield')
        facts[name] = dict(yield_selected=selected, reserved_cap=(rec.get('reserved_cap') or {}).get('counters'), ple_scopes=((rec.get('ple_private') or {}).get('counters') or {}).get('scopes'), replay_summary=rec.get('replay_summary'))
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    return dict(schema=GRADE_SCHEMA, verdict=verdict, failures=sorted(failures), missing=sorted(missing), part=part, scope=SCOPE, not_covered=NOT_COVERED, is_budget_gate=False, arms=facts, attn_meta=attn, startup=start,
                decode_only={k: (v.get('decode_only') or {}).get('aggregate_tokens_per_s') for k, v in decode_records.items()}, fixed_reference_tok_s=T.FIXED_REFERENCE_TOK_S)


def report(result):
    lines = ['# q18a-compose-4 report', '', 'Verdict: %s (the record; the run\'s rule is compose4_rule.py on the hub)' % result['verdict'], 'Failures: %s' % (', '.join(result['failures']) or 'none'), 'Missing: %s' % (', '.join(result['missing']) or 'none'), '',
             'Not covered: ' + result['not_covered'], '', '| arm | yields | overdrafts per rank | replay |', '|---|---|---|---|']
    for name, row in result['arms'].items():
        overdrafts = (row.get('reserved_cap') or {}).get('overdrafts')
        summary = row.get('replay_summary') or {}
        lines.append('| %s | %s | %s | %s |' % (name, row.get('yield_selected'), overdrafts, {k: summary.get(k) for k in ('warm_turns', 'turns_per_hour', 'append_ttft_p50', 'fresh_ttft_p50', 'errors')} if summary else '-'))
    return '\n'.join(lines) + '\n'


def smoke_check(rec, row=None):
    """a1's smoke, over this table's row, and the levers the a1 arm doesn't carry: the private PLE buffer on the ranks that have the PLE, the warm block and the cap on every rank. Overdrafts here are a read."""
    row = row or T.by_label()[T.SMOKE_ARM]
    ok, reasons = a1.smoke_check(rec, row)
    if not isinstance(rec, dict):
        return ok, reasons
    reasons = list(reasons)
    scopes = ((rec.get('ple_private') or {}).get('counters') or {}).get('scopes') or {}
    if not (scopes.get('0') and scopes.get('1')):
        reasons.append('smoke-ple-private-not-engaged')
    allocs = ((rec.get('score_warm') or {}).get('counters') or {}).get('allocs')
    if not (type(allocs) is dict and len(allocs) == 4):
        reasons.append('smoke-score-warm-counters-absent')
    if (rec.get('reserved_cap') or {}).get('requested') is not True:
        reasons.append('smoke-reserved-cap-not-requested')
    if (rec.get('inwait') or {}).get('exact_acknowledged') is not True:
        reasons.append('smoke-exact-successor-not-acknowledged')
    return not reasons, sorted(set(reasons))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--c1', choices=T.SELECTIONS, required=True, help="this arm's selection (inherited flag name)")
    r.add_argument('--size', type=int, choices=T.SIZES, required=True)
    r.add_argument('--cohort', choices=progress.COHORTS, required=True)
    r.add_argument('--index', type=int, choices=T.INDEXES, required=True)
    r.add_argument('--workload', required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    r.add_argument('--out', required=True)
    k = sub.add_parser('smoke', help='the exact-path smoke: 0 engaged, 2 not (the queue aborts the table)')
    k.add_argument('--arm', required=True)
    k.add_argument('--out', required=True)
    d = sub.add_parser('decode-arm')
    for flag in ('--url', '--model', '--run-id', '--workload', '--out'):
        d.add_argument(flag, required=True)
    d.add_argument('--epoch', type=int, required=True)
    d.add_argument('--label', choices=tuple(x['label'] for x in T.DECODE_ARMS), required=True)
    d.add_argument('--attn', choices=('on', 'off'), required=True)
    d.add_argument('--control-timeout-s', type=float, default=120.0)
    for action, text in (('grade', 'the grade: the queue verdict'), ('read', 'the same over every arm: a read only')):
        g = sub.add_parser(action, help=text)
        g.add_argument('--arm', action='append', required=True)
        g.add_argument('--ingress', action='append', default=[], help='label=path')
        g.add_argument('--watch', action='append', default=[], help='label=engine-watch record path')
        g.add_argument('--check', action='append', default=[], help='label=path')
        g.add_argument('--log', action='append', default=[], help='followed engine log (stdout/stderr)')
        g.add_argument('--report', help='write the report here')
        g.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if a.action == 'arm':
        return run_arm(a)
    if a.action == 'decode-arm':
        return am.run_decode_arm(a)
    if a.action == 'smoke':
        ok, reasons = smoke_check(base.load(a.arm))
        with open(a.out, 'x') as handle:
            json.dump(dict(schema='compose4-smoke/1', engaged=ok, reasons=reasons), handle, sort_keys=True)
        print('COMPOSE4_SMOKE %s %s' % ('ENGAGED' if ok else 'NOT_ENGAGED', ','.join(reasons)))
        return 0 if ok else 2
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]} for x in a.arm]
    result = grade(arms, ap.load_pairs(a.ingress), ap.load_pairs(a.check), a.log, ap.load_pairs(a.watch), part='core' if a.action == 'grade' else 'all')
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    if a.report:
        with open(a.report, 'x') as handle:
            handle.write(report(result))
    print('COMPOSE4_%s %s' % ('GRADE' if a.action == 'grade' else 'READ', result['verdict']))
    if a.action == 'read':
        return 0 if result['verdict'] == 'PASS' else 2      # a read: never 1, the queue does not count it
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
