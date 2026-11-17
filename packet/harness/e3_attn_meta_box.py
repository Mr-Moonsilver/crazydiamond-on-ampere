#!/usr/bin/env python3
"""q18a-attn-meta box arms and grade: the 2x2 at group size 1 (attn-meta x C1 defer) plus P2/MP2 at G2 and the two
standalone decode baselines (lead amendment A, decision A: the attn-meta selection is a per-arm typed control).

Derived from e3_alpha_next_c1_box (imported as ``c1box``, unchanged): every arm and grade step not defined here is c1's,
run with this package's names bound in its place for one call (c1box._bound); c1's own arm and grade in the same process
are untouched. The table is attn_meta_arms (one declared table, bound as c1box.T): an arm not in it is refused before any
traffic. The attn-meta selection is in the arm's round index (attn_meta_arms.ATTN), so c1's labels, salts and matched
C1-off partners (P -> R, MP -> M) hold unchanged.

The arm (``arm``): c1's arm (burst, fused, the group size, C0 local, the C1 selection and defer_bind) with one control added
inside the same pause, after native_stage_local and before release_auto_arm: the qualification worker action
attn_meta_select {on} through the reviewed worker adapter (native_qualification_frontend WORKER_PATH,
``qualification_worker``; the engine driver admits it only paused idle FULL). The reply must show, on all four ranks,
``after.selected`` equal to the arm's selection and ``order.ok`` (the Selector is the instance prepare_attn or wrapped by
another seam, i.e. handler anatomy): else the release raises and the arm is c1's arm-error (incomplete, never retried).
The arm's start counters are that reply's ``after``. After c1's arm returns (its record written as ``<out>.c1.json``,
kept), one end read: pause, attn_meta_read, resume; the record (``<out>``) is c1's plus ``attn_meta`` {requested, select,
end, deltas per rank}. A failed end read is recorded, never retried.

The decode-only arm (``decode-arm``, DECODE_ARMS): entry as c1's arm (latched idle FULL, burst installed), the 8 standing
streams opened, the same release (burst, fused, G1, C0 local, C1 off, defer_bind false, attn_meta_select {on}),
settle, one decode window of the workload's window_s (the same window c1's arms measure before their bulk), streams
stopped, the end read. No fresh prompt, no append.

The grade (``grade`` core, ``read`` every arm) is c1's grade over the c1-type arms with this table bound, plus:
  attn-meta   per arm (decode arms included): the requested selection equals the table's (else FAIL HARNESS); select and
              end read on all four ranks (else INC 'attn-meta-counters-missing'); ``order.ok`` on every row (else FAIL
              ENGINE 'attn-meta-order'); the selection unchanged at the end read (else FAIL ENGINE
              'attn-meta-selection-changed'). On an attn-meta-on arm the GdnBatch's ``batched`` delta > 0 on every rank,
              else INC 'attn-meta-not-engaged'. On an attn-meta-off arm the delta == 0 on every rank, else FAIL ENGINE
              'attn-meta-selection-leak'. Fallback deltas per reason per rank are listed (a read).
  startup     every rank 0-3 logged ``flash-attn-meta installed schema=attn-meta-select/1 rank=R selector_outer=1
              wrapped_by_other=0 selected=0`` (the Selector installed at initialize_kv_cache, before handler anatomy wraps
              prepare_attn at the first RPC): a line with selector_outer=0 or wrapped_by_other=1 is FAIL ENGINE
              'attn-meta-startup-order'; a rank without a line is INC 'attn-meta-startup-line-missing'. (Anatomy found
              first makes attn_meta_batch.install refuse and the engine never starts: no skip exists.)
  decode arms crash-free (watch not DEAD), ingress PASS, complete; else FAIL / INC as c1's rule for the same evidence.
Reads (never verdict inputs): per arm, live and solo prefill tok/s, TTFT, decode tok/s during the bulk and three
retentions: (1) c1's decode_ratio (the arm's own decode-only window), (2) against the decode-only baseline arm at the same
attn-meta selection (the build's own standalone run), (3) against the attn-meta-off baseline (the frozen reference: lead,
3 Nov 13:4x; the off path is the stock function, proven equal); for attn-meta-off cells (2) and (3) are the same arm
(``same_arm``). Batched / fallback / off_calls deltas, c1's per-stage memory and OOM reads, and the summary: (a) the best
live prefill at decode retention (3) >= 0.8, (b) the best live prefill overall with its decode. Cross-run, labelled: the
absolute decode rates beside the q18a-alpha-next-c1 report's C1-off band-1 decode-only rate (``--cross-run``, a pinned
file; absent -> not supplied).
"""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import time

import attn_meta_arms as T
import e3_alpha_next_c1_box as c1box
from e3_native_qualification_driver import exception_details
from e3_native_runtime_client import NativeRuntimeClient

an = c1box.an
b, sa, ap, progress, base = an.b, an.sa, an.ap, an.progress, an.base
ARM_SCHEMA = 'attn-meta-arm/1'
GRADE_SCHEMA = 'attn-meta-grade/1'
READ_SCHEMA = 'attn-meta-read/1'
WORKER_ACTION = 'qualification_worker'
SELECT, READ = 'attn_meta_select', 'attn_meta_read'
RANKS = (0, 1, 2, 3)
DECODE_FLOOR = 0.8                        # the operator's decode target (summary (a)), a read
STARTUP_LINE = re.compile(r'flash-attn-meta installed schema=attn-meta-select/1 rank=(\S+) selector_outer=(\d) '
                          r'wrapped_by_other=(\d) selected=(\d)')
C1_RELEASE = c1box.release_c1             # c1's, taken before any binding


# --- the worker action --------------------------------------------------------------------------------

class Worker:
    """attn_meta_select / attn_meta_read over the reviewed qualification worker adapter (its own client identity)."""

    def __init__(self, control, run_id, epoch):
        self.control, self.run_id, self.epoch = control, run_id, epoch
        self.client_id = 'attn-meta-' + secrets.token_hex(8)
        self.seq = 0

    def call(self, worker_action, payload):
        self.seq += 1
        command = dict(schema=1, action=WORKER_ACTION, run_id=self.run_id, command_seq=self.seq, client_id=self.client_id,
                       engine_epoch=self.epoch, worker_action=worker_action, payload=payload)
        return rank_rows(self.control.worker(command), worker_action, self.epoch)


def rank_rows(reply, worker_action, epoch):
    """{rank: the worker's qualification mapping} from the adapter's reply, exact coverage; raises when it does not read."""
    if type(reply) is not dict or reply.get('action') != WORKER_ACTION or reply.get('worker_action') != worker_action:
        raise ValueError('attn-meta-reply-identity')
    if reply.get('engine_epoch') != epoch:
        raise ValueError('attn-meta-reply-epoch')
    rows = (reply.get('worker_reply') or {}).get('qualification')
    if type(rows) is not list or len(rows) != len(RANKS) or any(type(r) is not dict for r in rows):
        raise ValueError('attn-meta-reply-ranks')
    out = {}
    for row in rows:
        q = row.get('qualification')
        rank = row.get('worker_rank')
        if type(q) is not dict or q.get('worker_action') != worker_action or q.get('engine_supplied_verdict') is not False:
            raise ValueError('attn-meta-reply-row')
        out[rank] = {k: q.get(k) for k in ('before', 'after', 'counters', 'order')}
    if set(out) != set(RANKS):
        raise ValueError('attn-meta-reply-ranks')
    return out


def _keyed(rows):
    return {str(k): v for k, v in sorted(rows.items())}


def select_into(client, attn, record):
    """Inside the arm's pause: select, record, prove (all ranks selected as asked, order ok)."""
    block = record.setdefault('attn_meta', {})
    block.update(requested=attn, select=None)
    worker = Worker(client.control, client.run_id, client.epoch)
    block['client_id'] = worker.client_id
    rows = worker.call(SELECT, {'on': attn})
    block['select'] = _keyed(rows)
    if any((r.get('after') or {}).get('selected') is not attn for r in rows.values()):
        raise ValueError('attn-meta-selection-unproved')
    if any((r.get('order') or {}).get('ok') is not True for r in rows.values()):
        raise ValueError('attn-meta-order-not-ok')


class _Selecting:
    """The arm's client with attn_meta_select sent once, just before release_auto_arm (inside the same pause)."""

    def __init__(self, client, attn, record):
        self._client, self._attn, self._record, self._sent = client, attn, record, False

    def __getattr__(self, name):
        return getattr(self._client, name)

    def call(self, action, **fields):
        if action == 'release_auto_arm' and not self._sent:
            self._sent = True
            select_into(self._client, self._attn, self._record)
        return self._client.call(action, **fields)


def release_am(attn):
    """c1's release_c1 with the attn-meta select inside it; the record must show the select ran."""
    def release_c1(overlap, defer_bind=False):
        inner = C1_RELEASE(overlap, defer_bind)

        def factory(size, local):
            run_inner = inner(size, local)

            def run(client, record, mode):
                run_inner(_Selecting(client, attn, record), record, mode)
                if (record.get('attn_meta') or {}).get('select') is None:
                    raise ValueError('attn-meta-select-not-sent')
            return run
        return factory
    return release_c1


def end_read(a):
    """Pause, attn_meta_read, resume; every step recorded, never retried."""
    out = dict(read=None, paused=False, resumed=None)
    try:
        client = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        client.pause()
        out['paused'] = True
        try:
            out['read'] = _keyed(Worker(client.control, a.run_id, client.epoch).call(READ, {}))
        finally:
            reply = client.resume()
            out['resumed'] = reply.get('resumed') is True or reply.get('already_running') is True
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        out['error'] = exception_details(exc)
    return out


def deltas(select, end):
    """Per rank: batched, off_calls and fallback per reason, end minus the select's after; None when either is missing."""
    if not select or not end:
        return None
    out = {}
    for rank in map(str, RANKS):
        s, e = (select.get(rank) or {}).get('after'), (end.get(rank) or {}).get('counters')
        if type(s) is not dict or type(e) is not dict:
            out[rank] = None
            continue
        fb_s, fb_e = s.get('fallback') or {}, e.get('fallback') or {}
        out[rank] = dict(batched=_sub(e.get('batched'), s.get('batched')), off_calls=_sub(e.get('off_calls'), s.get('off_calls')),
                         fallback={k: _sub(fb_e.get(k), fb_s.get(k)) for k in sorted(set(fb_s) | set(fb_e))},
                         selected_end=e.get('selected'), order_end=(end.get(rank) or {}).get('order'))
    return out


def _sub(x, y):
    return x - y if type(x) is int and type(y) is int else None


# --- arms -----------------------------------------------------------------------------------------------

def _bindings(attn):
    return dict(T=T, release_c1=release_am(attn))


def run_arm(a):
    """c1's arm with this table and the attn-meta select; then the end read; ``<out>`` = c1's record + attn_meta."""
    name = T.label(a.c1, a.size, a.cohort, a.index)
    row = T.by_label().get(name)
    attn = a.attn == 'on'
    inner_out = a.out + '.c1.json'
    if row is None or row['attn'] is not attn:
        record = dict(schema=ARM_SCHEMA, label=name, complete=False, reasons=['arm-not-in-the-declared-table'],
                      attn_meta=dict(requested=attn, table=None if row is None else row['attn']))
        rc = 2
    else:
        inner = argparse.Namespace(**{k: v for k, v in vars(a).items() if k != 'attn'})
        inner.out = inner_out
        with c1box._bound(c1box, _bindings(attn)):
            rc = c1box.run_arm(inner)
        record = base.load(inner_out) or dict(label=name, complete=False, reasons=['c1-arm-record-unreadable'])
        block = record.setdefault('attn_meta', dict(requested=attn, select=None))
        block['end'] = end_read(a)
        block['deltas'] = deltas(block.get('select'), (block['end'] or {}).get('read'))
        block['table'] = row['attn']
    record.update(attn_meta_schema=ARM_SCHEMA, c1_record=Path(inner_out).name, cell=(row or {}).get('cell'))
    with open(a.out, 'x') as handle:
        json.dump(record, handle, sort_keys=True)
    print('ATTN_META_ARM %s %s' % (name, 'COMPLETE' if record.get('complete') is True else 'INCOMPLETE'))
    return 0 if record.get('complete') is True and rc == 0 else 2


def decode_rows():
    return {d['label']: d for d in T.DECODE_ARMS}


def run_decode_arm(a, *, sleep=time.sleep):
    """The standalone decode baseline: 8 standing streams, no bulk, one window (see the module text)."""
    row = decode_rows().get(a.label)
    record = dict(schema=ARM_SCHEMA, kind='decode-only', label=a.label, complete=False, reasons=[], statuses=[],
                  progress=[], run_id=a.run_id, cohort='decode', part='decode', size=1, c1='off', defer_bind=False,
                  overlap=False, cell=(row or {}).get('cell'), attn_meta=dict(requested=a.attn == 'on', select=None))
    standing, sampler, outputs, workload = [], None, {}, None
    record['wall_start_s'] = time.time()
    try:
        if row is None or row['attn'] is not (a.attn == 'on'):
            raise ValueError('arm-not-in-the-declared-table')
        data = json.loads(Path(a.workload).read_text())
        budget = ap.standing_budget(data, 1)
        problems, cfg = progress.shared.validate_manifest(data)
        if problems or cfg['repeats'] != 1 or budget is None:
            raise ValueError('frozen-native-workload-invalid')
        record['manifest_sha256'] = cfg['manifest_sha256']
        client = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch, timeout_s=a.control_timeout_s)
        with c1box._bound(c1box, _bindings(row['attn'])), c1box._bound(an, c1box.arm_names('off')), \
                b.arm_seams(1), an.arm_seams('on', 1, outputs, a.run_id):
            def status(tag):
                reply = client.call('status')
                record['statuses'].append(base.status_view(tag, reply))
                record['progress'].append(dict(progress.progress_view(reply) or {}, label=tag))
                return reply
            if cfg['decode']['prompt_tokens'] + budget > progress.MODEL_LEN:
                raise ValueError('standing-request-exceeds-model-length')
            if progress.shared._package_gate(os.environ).get('ok') is not True:
                raise ValueError('reviewed-package-readiness-not-pass')
            entry = status('entry')
            auto = entry.get('auto_arm')
            if (type(auto) is not dict or auto.get('manual_owned') is not True or auto.get('active') is not False
                    or entry.get('route') != 'full' or entry.get('scheduler_requests') != 0):
                raise ValueError('arm-entry-not-latched-idle-full')
            workload = progress.shared._workload_view(cfg, generation_budget=budget)
            rows = progress.streams.build_cohort(workload)
            bodies = [progress.shared._request(a.model, r['prompt'], workload,
                                               progress.shared._cache_salt(a.run_id, progress.STANDING, i))
                      for i, r in enumerate(rows)]
            standing = progress.streams._start_streams(a.url, a.model, bodies, rows, workload)
            opened = progress.streams._wait_open(standing, workload)
            record['open'] = {k: opened.get(k) for k in ('complete', 'reasons')}
            if opened.get('complete') is not True:
                raise ValueError('w2-standing-cohort-unopened')
            progress.release(client, record, 'burst')
            sampler = progress.shared.CountSampler(standing)
            sampler.start()
            sleep(base.SETTLE_S)
            b0 = sampler.snapshot()
            sleep(cfg['window_s'])
            b1 = sampler.snapshot()
            record['decode_only'] = progress.shared._stream_window(standing, sampler, b0, b1)
            record['notes'] = ['decode-only:' + r for r in base.stream_reasons(record['decode_only'], standing, b1)]
            status('after-decode')
        record['complete'] = not record['reasons']
    except Exception as exc:  # noqa: BLE001 - retained, never retried
        record['error'] = exception_details(exc)
        record['reasons'].append('arm-error')
        record['complete'] = False
    finally:
        if sampler:
            sampler.close()
        if standing and workload is not None:
            progress.streams._stop_streams({}, standing, workload)
        record['wall_end_s'] = time.time()
    block = record['attn_meta']
    block['end'] = end_read(a)
    block['deltas'] = deltas(block.get('select'), (block['end'] or {}).get('read'))
    block['table'] = (row or {}).get('attn')
    with open(a.out, 'x') as handle:
        json.dump(record, handle, sort_keys=True)
    print('ATTN_META_DECODE_ARM %s %s' % (a.label, 'COMPLETE' if record['complete'] else 'INCOMPLETE'))
    return 0 if record['complete'] else 2


# --- grade ----------------------------------------------------------------------------------------------

def startup_lines(logs):
    """{rank: [(selector_outer, wrapped_by_other, selected)]} from the engine logs."""
    out = {}
    for path in logs or ():
        try:
            text = Path(path).read_text(errors='replace')
        except OSError:
            continue
        for m in STARTUP_LINE.finditer(text):
            out.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return out


def startup_check(logs):
    lines = startup_lines(logs)
    failures, missing = [], []
    for rank in map(str, RANKS):
        rows = lines.get(rank)
        if not rows:
            missing.append('attn-meta-startup-line-missing:rank-' + rank)
        elif any(r != (1, 0, 0) for r in rows):
            failures.append('attn-meta-startup-order:rank-' + rank)
    extra = sorted(set(lines) - set(map(str, RANKS)))
    if extra:
        missing.append('attn-meta-startup-line-rank-unknown:' + ','.join(extra))
    return dict(lines={k: [list(r) for r in v] for k, v in sorted(lines.items())}, failures=failures, missing=missing)


def attn_check(record, declared_attn):
    """(failures, missing, facts) of one arm's attn-meta block."""
    name = record.get('label')
    block = record.get('attn_meta') or {}
    failures, missing = [], []
    if block.get('requested') is not declared_attn:
        failures.append(name + ':attn-meta-requested-differs-from-table')
    select, end = block.get('select'), (block.get('end') or {}).get('read')
    d = block.get('deltas') or {}
    if not select or not end or any(d.get(str(r)) is None for r in RANKS):
        missing.append(name + ':attn-meta-counters-missing')
        return failures, missing, dict(engaged=None, leak=None)
    orders = [(select[str(r)] or {}).get('order') for r in RANKS] + [d[str(r)].get('order_end') for r in RANKS]
    if any((o or {}).get('ok') is not True for o in orders):
        failures.append(name + ':attn-meta-order')
    if any(d[str(r)].get('selected_end') is not declared_attn for r in RANKS):
        failures.append(name + ':attn-meta-selection-changed')
    batched = [d[str(r)].get('batched') for r in RANKS]
    engaged = leak = None
    if any(type(x) is not int for x in batched):
        missing.append(name + ':attn-meta-counters-missing')
    elif declared_attn:
        engaged = all(x > 0 for x in batched)
        if not engaged:
            missing.append(name + ':attn-meta-not-engaged')
    else:
        leak = any(x != 0 for x in batched)
        if leak:
            failures.append(name + ':attn-meta-selection-leak')
    return failures, missing, dict(engaged=engaged, leak=leak, batched=batched,
                                   off_calls=[d[str(r)].get('off_calls') for r in RANKS],
                                   fallback={str(r): d[str(r)].get('fallback') for r in RANKS},
                                   wrapped_by_other=[(d[str(r)].get('order_end') or {}).get('wrapped_by_other') for r in RANKS])


def decode_verdict(record, ingress, watch):
    failures, missing = [], []
    name = record.get('label')
    if (watch or {}).get('verdict') == 'DEAD':
        failures.append(name + ':engine-dead')
    elif watch is None:
        missing.append(name + ':engine-watch-record-missing')
    if (ingress or {}).get('verdict') == 'FAIL':
        failures.append(name + ':ingress-invalid')
    elif (ingress or {}).get('verdict') != 'PASS':
        missing.append(name + ':ingress-not-pass')
    if record.get('complete') is not True:
        missing.append(name + ':decode-arm-incomplete')
    return failures, missing


def _rate(record):
    return ((record or {}).get('decode_only') or {}).get('aggregate_tokens_per_s')


def _ratio(x, y):
    return round(x / y, 4) if type(x) in (int, float) and type(y) in (int, float) and y else None


def reads(result, decode_records, rows, cross_run=None):
    """Per arm: prefill, decode and the three retentions; the 2x2 medians; summary (a) and (b). Never verdict inputs."""
    table = result.get('table') or {}
    base_rate = {d['attn']: _rate(decode_records.get(d['label'])) for d in T.DECODE_ARMS}
    out = {}
    for row in rows:
        cell = table.get(row['label']) or {}
        during = cell.get('decode_during_tok_s')
        out[row['label']] = dict(
            cell=row['cell'], attn=row['attn'], c1=row['c1'], size=row['size'], cohort=row['cohort'], band=row['band'],
            part=row['part'], prefill_tok_s=cell.get('prefill_tok_s'), ttft_s=cell.get('ttft_s'),
            decode_during_tok_s=during, decode_ratio_own_window=cell.get('decode_ratio'),
            retention_vs_standalone_same_attn=_ratio(during, base_rate.get(row['attn'])),
            retention_vs_frozen_reference=_ratio(during, base_rate.get(False)),
            same_arm=not row['attn'], gap_p99_ms=cell.get('gap_p99_ms'), p10_stream_ratio=cell.get('p10_stream_ratio'))
    cells = {}
    for r in out.values():
        if r['cohort'] == 'live' and r['band'] == 1:
            cells.setdefault(r['cell'], []).append(r)
    medians = {c: dict(prefill_tok_s=progress.median([r['prefill_tok_s'] for r in v]),
                       retention_vs_frozen_reference=progress.median([r['retention_vs_frozen_reference'] for r in v]),
                       retention_vs_standalone_same_attn=progress.median([r['retention_vs_standalone_same_attn'] for r in v]),
                       decode_ratio_own_window=progress.median([r['decode_ratio_own_window'] for r in v]),
                       arms=[label for label, x in out.items() if x['cell'] == c and x['cohort'] == 'live' and x['band'] == 1])
               for c, v in sorted(cells.items())}
    ok = [(m['prefill_tok_s'], c) for c, m in medians.items() if type(m['prefill_tok_s']) in (int, float)
          and type(m['retention_vs_frozen_reference']) in (int, float) and m['retention_vs_frozen_reference'] >= DECODE_FLOOR]
    every = [(m['prefill_tok_s'], c) for c, m in medians.items() if type(m['prefill_tok_s']) in (int, float)]
    best_a = max(ok)[1] if ok else None
    best_b = max(every)[1] if every else None
    summary = dict(
        a_best_prefill_at_decode_floor=None if best_a is None else dict(cell=best_a, **{k: medians[best_a][k] for k in (
            'prefill_tok_s', 'retention_vs_frozen_reference', 'retention_vs_standalone_same_attn')}),
        b_best_prefill_overall=None if best_b is None else dict(cell=best_b, **{k: medians[best_b][k] for k in (
            'prefill_tok_s', 'retention_vs_frozen_reference', 'retention_vs_standalone_same_attn')}),
        decode_floor=DECODE_FLOOR, basis='band 1 live arms, median per cell; decode retention (3): during the bulk '
                                         'over the attn-meta-off decode-only baseline arm (the frozen reference)')
    decode = {d['label']: dict(attn=d['attn'], aggregate_tokens_per_s=_rate(decode_records.get(d['label'])),
                               stream_rates_tokens_per_s=((decode_records.get(d['label']) or {}).get('decode_only') or {})
                               .get('stream_rates_tokens_per_s'),
                               gap_ms_p99=((decode_records.get(d['label']) or {}).get('decode_only') or {}).get('gap_ms_p99'))
              for d in T.DECODE_ARMS}
    decode['m_over_r'] = _ratio(decode['decode-am1']['aggregate_tokens_per_s'], decode['decode-am0']['aggregate_tokens_per_s'])
    return dict(arms=out, cells=medians, summary=summary, decode_only=decode, cross_run=cross_run or dict(supplied=False),
                is_gate=False)


def grade(arms, ingress, checks, logs, watches=None, part='core', cross_run=None):
    known = [a for a in arms if a.get('label')]
    decode_records = {a['label']: a for a in known if a.get('kind') == 'decode-only'}
    c1_arms = [a for a in arms if a.get('kind') != 'decode-only']
    with c1box._bound(c1box, dict(T=T)):
        result = c1box.grade(c1_arms, ingress, checks, logs, watches, part=part)
    failures, missing = set(result['failures']), set(result['missing'])
    classes = set(result.get('classification') or ())
    want = T.arms(None if part == 'all' else T.GRADED)
    attn = {}
    records = {a.get('label'): a for a in known}
    for row in want:
        rec = records.get(row['label'])
        if rec is None:
            continue                       # c1's grade already counts the missing record ('no-arm-record')
        f, m, facts = attn_check(rec, row['attn'])
        attn[row['label']] = dict(facts, failures=f, missing=m)
        failures.update(f)
        missing.update(m)
        if f:
            classes.add('HARNESS' if any(x.endswith('requested-differs-from-table') for x in f) else 'ENGINE')
    for d in T.DECODE_ARMS:
        rec = decode_records.get(d['label'])
        if rec is None:
            missing.add(d['label'] + ':no-arm-record')
            continue
        f, m = decode_verdict(rec, (ingress or {}).get(d['label']), (watches or {}).get(d['label']))
        af, am, facts = attn_check(rec, d['attn'])
        attn[d['label']] = dict(facts, failures=af, missing=am)
        failures.update(f + af)
        missing.update(m + am)
        if f or af:
            classes.add('ENGINE')
    start = startup_check(logs)
    failures.update(start['failures'])
    missing.update(start['missing'])
    if start['failures']:
        classes.add('ENGINE')
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    if verdict != 'FAIL':
        classes = set()
    elif not classes:
        classes = {'HARNESS'}
    rows = [r for r in T.arms() if r['label'] in (result.get('table') or {})]
    result.update(schema=GRADE_SCHEMA if part == 'core' else READ_SCHEMA, c1_schema=result.get('schema'), verdict=verdict,
                  classification=sorted(classes) or None, failures=sorted(failures), missing=sorted(missing),
                  attn_meta=attn, attn_meta_is_gate=True, startup=start,
                  attn_meta_reads=reads(result, decode_records, rows, cross_run),
                  declared=dict(cells=T.CELLS, rounds={str(k): v for k, v in T.ROUNDS.items()}, band2_index=T.BAND2_INDEX,
                                decode_arms=[d['label'] for d in T.DECODE_ARMS]),
                  axis_attn='attn-meta selection in the round index: off 0/1, on 3/4 (band 1), band 2 index 2 (MP, on)')
    return result


def _cell(v, n=1):
    return '-' if v is None else ('%.*f' % (n, v) if type(v) is float else str(v))


def report(result):
    r = result['attn_meta_reads']
    lines = ['# q18a-attn-meta %s: %s' % ('grade' if result['schema'] == GRADE_SCHEMA else 'read', result['verdict']), '',
             'Failures: %s' % (', '.join(result['failures']) or 'none'), 'Missing: %s' % (', '.join(result['missing']) or 'none'),
             '', '## attn-meta per arm (batched delta per rank 0-3; off-arms must be 0, on-arms > 0)', '',
             '| arm | engaged | leak | batched | off_calls | fallback (non-zero) | anatomy over selector |', '|---|---|---|---|---|---|---|']
    for name, a in sorted(result['attn_meta'].items()):
        fb = {rk: {k: v for k, v in (f or {}).items() if v} for rk, f in (a.get('fallback') or {}).items()}
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (name, a.get('engaged'), a.get('leak'), a.get('batched'),
                                                               a.get('off_calls'), {k: v for k, v in fb.items() if v} or '-',
                                                               a.get('wrapped_by_other')))
    lines += ['', '## Reads (never verdict inputs)', '',
              'Decode retention: (1) own decode-only window (c1), (2) the decode-only baseline at the same attn-meta '
              'selection (the build\'s own standalone run), (3) the attn-meta-off decode-only baseline (the frozen '
              'reference); for attn-meta-off cells (2) and (3) are the same arm.', '',
              '| arm | cell | cohort | band | prefill tok/s | TTFT s | decode tok/s | (1) | (2) | (3) |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for name, a in r['arms'].items():
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s%s |' % (
            name, a['cell'], a['cohort'], a['band'], _cell(a['prefill_tok_s']), _cell(a['ttft_s'], 2),
            _cell(a['decode_during_tok_s']), _cell(a['decode_ratio_own_window'], 3),
            _cell(a['retention_vs_standalone_same_attn'], 3), _cell(a['retention_vs_frozen_reference'], 3),
            ' (same arm)' if a['same_arm'] else ''))
    lines += ['', 'Decode-only baselines: %s' % json.dumps({k: v.get('aggregate_tokens_per_s') if type(v) is dict else v
                                                              for k, v in r['decode_only'].items()}),
              '', 'Summary: (a) %s; (b) %s' % (json.dumps(r['summary']['a_best_prefill_at_decode_floor']),
                                              json.dumps(r['summary']['b_best_prefill_overall'])),
              '', 'Cross-run (read, another queue and day): %s' % json.dumps(r['cross_run']), '']
    return '\n'.join(lines) + '\n' + c1box.report(dict(result, schema=result.get('c1_schema')))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    for flag in ('--url', '--model', '--run-id', '--workload', '--out'):
        r.add_argument(flag, required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--attn', choices=('on', 'off'), required=True)
    r.add_argument('--c1', choices=tuple(T.C1), required=True)
    r.add_argument('--size', type=int, choices=T.SIZES, required=True)
    r.add_argument('--cohort', choices=T.COHORTS, required=True)
    r.add_argument('--index', type=int, choices=T.INDEXES, required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    d = sub.add_parser('decode-arm')
    for flag in ('--url', '--model', '--run-id', '--workload', '--out'):
        d.add_argument(flag, required=True)
    d.add_argument('--epoch', type=int, required=True)
    d.add_argument('--label', choices=tuple(decode_rows()), required=True)
    d.add_argument('--attn', choices=('on', 'off'), required=True)
    d.add_argument('--control-timeout-s', type=float, default=120.0)
    for action in ('grade', 'read'):
        g = sub.add_parser(action)
        g.add_argument('--arm', action='append', required=True)
        g.add_argument('--ingress', action='append', default=[])
        g.add_argument('--watch', action='append', default=[])
        g.add_argument('--check', action='append', default=[])
        g.add_argument('--log', action='append', default=[])
        g.add_argument('--cross-run', help='the pinned q18a-alpha-next-c1 cross-run numbers (JSON), a read')
        g.add_argument('--report')
        g.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    if a.action == 'arm':
        return run_arm(a)
    if a.action == 'decode-arm':
        return run_decode_arm(a)
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]}
            for x in a.arm]
    cross = json.loads(Path(a.cross_run).read_text()) if a.cross_run else None
    result = grade(arms, ap.load_pairs(a.ingress), ap.load_pairs(a.check), a.log, ap.load_pairs(a.watch),
                   part='core' if a.action == 'grade' else 'all', cross_run=cross)
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    if a.report:
        with open(a.report, 'x') as handle:
            handle.write(report(result))
    print('ATTN_META_%s %s' % ('GRADE' if a.action == 'grade' else 'READ', result['verdict']))
    if a.action == 'read':
        return 0 if result['verdict'] == 'PASS' else 2
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
