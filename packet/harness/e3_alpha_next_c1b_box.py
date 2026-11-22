#!/usr/bin/env python3
"""alpha-next-c1b box arms and grade: the fixed-schedule numerics queue for C0 and C1 defer, on alpha-next-c1's engine unchanged.

Supersedes e3_alpha_next_c1_box (imported as ``c1``, unchanged) by name; its arm and grade run with the names below bound in their
place for one call (c1's ``_bound``), so c1's own arm and grade in the same process are untouched. The arm table is
alpha_next_c1b_arms (one declared table: S1 S2 L1 L2 D S3, band 1, solo; an arm not in it is refused before any traffic).

The arm: c1's (alpha-next's arm with C1's stage control), with
  serial     a 'serial' row selects native_stage_local {local: false, overlap: false, defer_bind: false} (C0 off); alpha-next's
             arm records stage_local False and c0 'off' for it. 'off' rows {local: true, overlap: false, defer_bind: false},
             'defer' rows {local: true, overlap: true, defer_bind: true}, as c1.
  numerics   the fresh request (the band 1 fresh prompt, recorded by sha) is the numerics probe (NUMERICS): temperature 0,
             top_p 1, 256 generated tokens (min = max, ignore_eos), logprobs 20 with return_tokens_as_token_ids; every
             generated position's token id and its top-20 (token id -> logprob, as returned) are kept in the arm record
             (``numerics``). The append request is unchanged (c1's).

The grade is c1's grade over this table (every c1 gate kept: crash-free, TP commitment, group and C0 engagement both ways, C1
engagement both ways, defer_bind acknowledged, one command in flight per stage, OOM, torn lines, signatures), with:
  (a) open rows   ``progress-line-open-rows-carried`` (e3_c1_open_rows) supersedes ``progress-line-with-open-rows`` on overlap
                  arms (the D arm), by name; on serial and C0-local arms the old rule stays exactly as it is. Check 4
                  (``c1-open-row-not-successor``) grades INC ENGINE (lead, 3 Nov ~10:10 CET), the others INC.
  (a) ring read   the C0 ring reading on overlap arms takes per-chunk blocked time from the increments of the stage line's
                  running total (``headroom_blocked_ns_lines`` and ``..._max_chunk`` become the increments' sum and largest
                  rise; the raw sum and max are kept beside them as ``*_raw``); C1-off arms unchanged. A read, never gated.
  numerics        the verdict input of this queue, declared before the run (lead, 3 Nov ~10:10 CET; DESIGN-c1-next (c)):
                  S1 against S2 fixes the bound: P = their common generated-token prefix, B = the largest |logprob difference|
                  over the top-20 tokens both report at the positions before P, M = the largest top-20 membership difference
                  there. S1 = S2 bit for bit gives P = 256, B = 0, M = 0: every candidate must then be bit-equal. Each
                  comparison (L1-S1, L2-S1: C0; D-L1, D-L2: C1 defer) must reach at least P common tokens and stay within B
                  and M before P. Any comparison outside: FAIL ENGINE. S3 against S1 outside: INC (control drift). A role
                  without its numerics evidence: INC. No retry-until-equal: every arm runs once.
alpha-next's numerics envelope read (standing greedy streams) stays a read. Receipts hold counts, hashes, token ids, logprobs and
host clock stamps, never prompts or auth.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import alpha_next_c1b_arms as T
import e3_alpha_next_c1_box as c1
import e3_c1_open_rows as rule

an, progress, ap, base, sa = c1.an, c1.progress, c1.ap, c1.base, c1.sa
shared = progress.shared
correctness = shared.correctness
_bound = c1._bound
GRADE_SCHEMA = 'alpha-next-c1b-grade/1'
READ_SCHEMA = 'alpha-next-c1b-read/1'
NUMERICS = dict(temperature=0.0, top_p=1.0, generation_tokens=256, logprobs=20, return_tokens_as_token_ids=True)
NUMERICS_IS_GATE = True
NUMERICS_SCHEMA = 'alpha-next-c1b-numerics/1'
COMPARISONS = (('L1', 'S1', 'C0'), ('L2', 'S1', 'C0'), ('D', 'L1', 'C1 defer'), ('D', 'L2', 'C1 defer'))
DRIFT = ('S3', 'S1', 'control drift')
# Taken before any binding: the raw assemble (c1's grade binds a regrouping one) and alpha-next's ring read.
RAW_ASSEMBLE = progress.assemble
RAW_PROGRESS_RECEIPTS = progress.progress_receipts
AN_RING_READ = an.ring_read
C1_TABLE_NAMES = c1._TABLE_NAMES
# c1's overlap-observe roles plus the serial selection (a read: serial arms open no successor).
OBSERVE_ROLE = dict(c1.OBSERVE_ROLE, serial='no successor expected (C0 off: serial)')


# --- the table as alpha-next's and c1's code read it ------------------------------------------------------

def table_view(sel=None):
    """This table as alpha-next's code reads alpha_next_arms. The C0 axis is alpha-next's own: serial rows are c0 'off' (the
    controls), C0-local and C1-defer rows c0 'on'; ``matched_off`` is alpha-next's pairing (a C0-on row to its serial row).
    For one arm (``sel``): the label and selection carry the selection; C0 holds the arm's own stage-local value."""
    names = {k: getattr(T, k) for k in C1_TABLE_NAMES}

    def parse_selection(token):
        parsed = T.parse_selection(token)
        return None if parsed is None else ('off' if parsed[0] == 'serial' else 'on', parsed[1])
    names.update(arms=T.arms, by_label=T.by_label, C0={'on': True, 'off': False}, parse_selection=parse_selection,
                 matched_off=T.matched_serial)
    if sel is not None:
        names.update(selection=lambda _c0, size: T.selection(sel, size),
                     label=lambda _c0, size, cohort, index: T.label(sel, size, cohort, index),
                     C0={'on': True} if T.LOCAL[sel] else {'off': False})
    return SimpleNamespace(**names)


# --- the numerics probe -------------------------------------------------------------------------------------

class Numerics:
    """The fresh stream's instrument (e3_decode_correctness's hook, as e3_teacher_decode_client.Instrument): every generated
    position's token id and top-20. A malformed logprob block fails the stream (correctness: teacher_logprob_evidence_invalid)."""

    def __init__(self):
        self.tokens, self.top = [], []

    def consume(self, state, event):
        choices = event.get('choices', [])
        if not choices:
            return
        if len(choices) != 1:
            raise ValueError('numerics choice count')
        row = choices[0]
        tokens = row.get('token_ids') or []
        if not tokens:
            return
        lp = row.get('logprobs') or {}
        top, encoded = lp.get('top_logprobs'), lp.get('tokens')
        if type(top) is not list or type(encoded) is not list or len(top) != len(tokens) or len(encoded) != len(tokens):
            raise ValueError('numerics logprobs missing')
        for i, (token, values, name) in enumerate(zip(tokens, top, encoded)):
            position = state.token_count + i
            if type(token) is not int or position != len(self.tokens) or name != 'token_id:' + str(token):
                raise ValueError('numerics token order or encoding differs')
            if (type(values) is not dict or not NUMERICS['logprobs'] <= len(values) <= NUMERICS['logprobs'] + 1
                    or any(type(k) is not str or not k.startswith('token_id:') or not k[9:].isdigit()
                           or type(v) not in (int, float) or not math.isfinite(v) for k, v in values.items())):
                raise ValueError('numerics top-20 incomplete')
            self.tokens.append(token)
            self.top.append([[int(k[9:]), float(v)] for k, v in values.items()])


def numerics_send(captured):
    """shared._send_prefill with the fresh request (index 0) made the numerics probe; the append (index 1) unchanged."""
    original = shared._send_prefill

    def send(url, model, cfg, lane, index, prompt, generation_budget, prompt_seed, salt):
        if index != 0:
            return original(url, model, cfg, lane, index, prompt, generation_budget, prompt_seed, salt)
        workload = shared._workload_view(cfg, streams=1, generation_budget=NUMERICS['generation_tokens'], prompt_seed=prompt_seed,
                                         finish_timeout_s=cfg['timeouts']['finish_s'], temperature=NUMERICS['temperature'],
                                         top_p=NUMERICS['top_p'])
        body = shared._request(model, prompt, workload, salt)
        body = dict(body, logprobs=NUMERICS['logprobs'], return_tokens_as_token_ids=NUMERICS['return_tokens_as_token_ids'])
        instrument = Numerics()
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        captured.update(instrument=instrument, prompt_sha256=digest, prompt_chars=len(prompt), manifest_budget=generation_budget,
                        request=dict(temperature=body['temperature'], top_p=body['top_p'], max_tokens=body['max_tokens'],
                                     min_tokens=body['min_tokens'], ignore_eos=body['ignore_eos'], logprobs=body['logprobs'],
                                     seed=body.get('seed'), cache_salt_sha256=hashlib.sha256(str(salt).encode()).hexdigest()))
        states = correctness._start_streams(url, model, [body], [{'prompt_sha256': digest, 'prompt_chars': len(prompt)}],
                                            workload, instrument=instrument)
        return states[0], workload
    return send


def numerics_record(captured):
    instrument = captured.get('instrument')
    if instrument is None:
        return dict(schema=NUMERICS_SCHEMA, complete=False, reason='numerics-probe-not-sent')
    n = len(instrument.tokens)
    out = dict(schema=NUMERICS_SCHEMA, prompt_sha256=captured['prompt_sha256'], prompt_chars=captured['prompt_chars'],
               request=captured['request'], generated=n, tokens=list(instrument.tokens), top=list(instrument.top),
               tokens_sha256=hashlib.sha256(json.dumps(instrument.tokens).encode()).hexdigest())
    out['complete'] = n == NUMERICS['generation_tokens']
    if not out['complete']:
        out['reason'] = 'numerics-generated-%d-of-%d' % (n, NUMERICS['generation_tokens'])
    return out


# --- the arm ------------------------------------------------------------------------------------------------

def _arm_print(*args, **kwargs):
    text = ' '.join(str(x) for x in args)
    print(text.replace('ALPHA_NEXT_ARM ', 'ALPHA_NEXT_C1B_ARM ', 1), **kwargs)


def arm_names(sel):
    overlap = T.C1[sel]
    return dict(T=table_view(sel), release_alpha_next=c1.release_c1(overlap, T.DEFER_BIND[sel]),
                stage_reasons=c1.stage_reasons_c1(overlap), stage_view=c1.stage_view, stage_delta=c1.stage_delta, print=_arm_print)


def run_arm(a):
    """alpha-next's arm through c1's names, with this arm's stage-local value, C1 selection and defer_bind, and the numerics
    probe. alpha-next writes its record to OUT.base (kept); this arm's record (that record plus ``numerics``) goes to OUT."""
    out = Path(a.out)
    first = Path(str(out) + '.base')
    if out.exists() or first.exists():
        return 2
    args = argparse.Namespace(**dict(vars(a), stage_local='on' if T.LOCAL[a.c1] else 'off', out=str(first)))
    captured = {}
    with _bound(c1, dict(T=T)), _bound(an, arm_names(a.c1)), _bound(shared, dict(_send_prefill=numerics_send(captured))):
        an.run_arm(args)
    record = json.loads(first.read_text())
    record['numerics'] = numerics_record(captured)
    declared = T.by_label().get(record.get('label')) or {}
    record.update(c1=a.c1, role=declared.get('role'), defer_bind_table=declared.get('defer_bind'))
    if record['numerics']['complete'] is not True:
        record['reasons'] = list(record.get('reasons') or []) + ['numerics:' + record['numerics']['reason']]
        record['complete'] = False
    with open(out, 'x') as handle:
        json.dump(record, handle, sort_keys=True)
    _arm_print('ALPHA_NEXT_ARM %s %s' % (record.get('label'), 'COMPLETE' if record.get('complete') else 'INCOMPLETE'))
    return 0 if record.get('complete') is True else 2


# --- (a) the open-rows rule and the ring read ---------------------------------------------------------------

_C1B = {}


def _overlap(label):
    return (T.by_label().get(label) or {}).get('overlap') is True


def _parse_overlap(text):
    m = c1.OVERLAP_LINE.search(text)
    if not m:
        return None
    return int(m.group(1)), set(c1.seqs_of(m.group(2))), None if m.group(3) == 'none' else int(m.group(3))


def progress_receipts(arms, records, parse_errors, streams_count=8):
    """held-f's receipts (through c1's regroup), then on overlap arms ``progress-line-with-open-rows`` replaced by the C1 rule
    over the raw (not regrouped) engine flushes of the arm and its stage-overlap lines."""
    out = RAW_PROGRESS_RECEIPTS(arms, records, parse_errors, streams_count)
    raw, _ = RAW_ASSEMBLE(records)
    engine = [f for f in raw if f['side'] == 'engine']
    overlap_lines = list(c1._RUN.get('overlap') or ())
    for arm in arms:
        name = arm.get('label')
        row = (out.get('arms') or {}).get(name)
        if row is None or not _overlap(name):
            continue
        mine = [f for f in engine if f['stamp'] is not None and ap.in_span(f['stamp'], arm)]
        txns = {(f.get('identity') or {}).get('txn') for f in mine}
        index = rule.overlap_index([t for s, t in overlap_lines if ap.in_span(s, arm)], _parse_overlap)
        facts = rule.check_arm(mine, {k: v for k, v in index.items() if k in txns})
        had = rule.SUPERSEDES in row['reasons']
        row['reasons'] = [x for x in row['reasons'] if x != rule.SUPERSEDES] + facts['reasons']
        row['open_rows_c1'] = dict(facts, superseded_reason_present=had)
        _C1B.setdefault('open_rows', {})[name] = row['open_rows_c1']
    return out


def ring_read(arm, lines):
    out = AN_RING_READ(arm, lines)
    if not _overlap(arm.get('label')):
        return dict(out, lines_basis='per chunk (C1 off: C0 resets the value at each begin)')
    stage = list(lines['stage'])
    if stage and all(type(x.get('txn')) is int for x in stage):
        stage.sort(key=lambda x: x['txn'])
    inc = rule.ring_increments([x['headroom_blocked_ns'] for x in stage])
    return dict(out, headroom_blocked_ns_lines_raw=out['headroom_blocked_ns_lines'],
                headroom_blocked_ns_max_chunk_raw=out['headroom_blocked_ns_max_chunk'],
                headroom_blocked_ns_lines=inc['total'], headroom_blocked_ns_max_chunk=inc['max_chunk'], running_total_resets=inc['resets'],
                lines_basis='increments of a running total (C1 arm: _maybe_commit does not reset _ring_blocked_chunk_ns)')


# --- numerics verdict -----------------------------------------------------------------------------------------

def _prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _diff(x, y, upto):
    """Largest |logprob difference| over the shared top-20 tokens and the largest top-20 membership difference, positions < upto."""
    worst, members = 0.0, 0
    for i in range(upto):
        a, b = dict(map(tuple, x['top'][i])), dict(map(tuple, y['top'][i]))
        for k in set(a) & set(b):
            worst = max(worst, abs(a[k] - b[k]))
        members = max(members, len(set(a) ^ set(b)))
    return worst, members


def numerics_verdict(by_role):
    """The declared rule over the roles' numerics records (role -> record or None)."""
    usable = {r: x for r, x in by_role.items()
              if x and x.get('complete') is True and len(x.get('tokens') or ()) == NUMERICS['generation_tokens']}
    prompts = {x['prompt_sha256'] for x in usable.values()}
    out = dict(schema=NUMERICS_SCHEMA, is_gate=NUMERICS_IS_GATE, roles={r: (x or {}).get('tokens_sha256') for r, x in by_role.items()},
               prompt_sha256=sorted(prompts), comparisons=[], reasons=[])
    if len(prompts) > 1:
        out['reasons'].append('numerics-prompt-differs-between-arms')
    for role in T.ROLES:
        if role not in usable:
            out['reasons'].append('numerics-evidence-missing:' + role)
    if 'S1' not in usable or 'S2' not in usable or len(prompts) > 1:
        out.update(verdict='INC', bound=None)
        return out
    s1, s2 = usable['S1'], usable['S2']
    p = _prefix(s1['tokens'], s2['tokens'])
    b, m = _diff(s1, s2, p)
    exact = p == NUMERICS['generation_tokens'] and b == 0.0 and m == 0
    out['bound'] = dict(mode='bit-equal' if exact else 'bound', common_tokens=p, max_abs_logprob=b, max_membership=m)
    outside, drift = [], []
    for cand, ref, what in COMPARISONS + (DRIFT,):
        if cand not in usable or ref not in usable:
            continue
        x, y = usable[cand], usable[ref]
        common = _prefix(x['tokens'], y['tokens'])
        upto = min(p, common)
        worst, members = _diff(x, y, upto)
        within = common >= p and worst <= b and members <= m
        out['comparisons'].append(dict(candidate=cand, reference=ref, axis=what, common_tokens=common, max_abs_logprob=worst,
                                       max_membership=members, within=within,
                                       bit_equal=x['tokens'] == y['tokens'] and x['top'] == y['top']))
        if not within:
            (drift if (cand, ref) == DRIFT[:2] else outside).append('%s-%s' % (cand, ref))
    if outside:
        out['reasons'] += ['numerics-outside:' + x for x in outside]
        out['verdict'] = 'OUTSIDE'
    elif drift or any(r.startswith('numerics-evidence-missing') for r in out['reasons']):
        out['reasons'] += ['numerics-control-drift:' + x for x in drift]
        out['verdict'] = 'INC'
    else:
        out['verdict'] = 'WITHIN'
    return out


# --- the grade --------------------------------------------------------------------------------------------

def grade(arms, ingress, checks, logs, watches=None, part='core'):
    """c1's grade over this table with (a) bound; then the numerics verdict and the record fields."""
    _C1B.clear()
    try:
        with _bound(c1, dict(T=T, table_view=table_view, OBSERVE_ROLE=OBSERVE_ROLE)), \
                _bound(progress, dict(progress_receipts=progress_receipts)), _bound(an, dict(ring_read=ring_read)):
            result = c1.grade(arms, ingress, checks, logs, watches, part=part)
        open_rows = dict(_C1B.get('open_rows') or {})
    finally:
        _C1B.clear()
    declared = T.by_label()
    for name, cell in result['table'].items():
        row = declared.get(name)
        if row:
            cell.update(c0=row['c0'], c1=row['c1'], overlap=row['overlap'], defer_bind=row['defer_bind'],
                        stage_local=row['stage_local'], role=row['role'])
    for row in result['arm_table']['arms']:
        want = declared.get(row.get('label'))
        if want:
            row.update(c0=want['c0'], role=want['role'])
    records = {a.get('label'): a for a in arms if a.get('label')}
    by_role = {row['role']: (records.get(row['label']) or {}).get('numerics') for row in T.arms()}
    numerics = numerics_verdict(by_role)
    failures, missing = set(result['failures']), set(result['missing'])
    classes = set(result['classification'] or ())
    if numerics['verdict'] == 'OUTSIDE' and NUMERICS_IS_GATE:
        failures.update(numerics['reasons'])
        classes.add('ENGINE')
    elif numerics['verdict'] == 'INC':
        missing.update(numerics['reasons'] or ['numerics-inconclusive'])
    engine_open_rows = sorted(name for name, f in open_rows.items() if set(f['reasons']) & set(rule.ENGINE_REASONS))
    verdict = 'FAIL' if failures else 'INC' if missing else 'PASS'
    if verdict == 'FAIL':
        classes = classes or {'HARNESS'}
    elif verdict == 'INC' and engine_open_rows:
        classes = {'ENGINE'}           # check 4 of the open-rows rule: INC ENGINE, not FAIL (lead)
    else:
        classes = set()
    result.update(schema=GRADE_SCHEMA if part == 'core' else READ_SCHEMA, verdict=verdict, classification=sorted(classes) or None,
                  failures=sorted(failures), missing=sorted(missing), numerics_c1b=numerics, numerics_c1b_is_gate=NUMERICS_IS_GATE,
                  open_rows_rule=dict(rule=rule.RULE, supersedes=rule.SUPERSEDES, arms=open_rows,
                                      engine_classified_arms=engine_open_rows),
                  axis=('this queue\'s C0 axis is alpha-next\'s own: c0 off = serial (c0off, the numerics controls S1 S2 S3), '
                        'c0 on = C0 local (c1off, L1 L2) and C1 defer (c1defer, D); c1\'s C1-off/C1-on names in the inherited '
                        'tables read: C1-off = serial, C1-on = C0 local or C1 defer'),
                  scope='fixed-schedule numerics of C0 local and C1 defer against serial on alpha-next-c1\'s engine (band 1, '
                        'solo, temperature 0, 256 tokens, top-20 logprobs); the mechanism gates of c1 kept; not a release '
                        'assessment')
    return result


# --- the report ---------------------------------------------------------------------------------------------

def report(result):
    with _bound(c1, dict(T=T, table_view=table_view)):
        text = c1.report(result)
    text = text.replace('# q18a-alpha-next-c1 report', '# q18a-alpha-next-c1b report', 1)
    text = text.replace('# q18a-alpha-next-c1 read', '# q18a-alpha-next-c1b read', 1)
    n = result.get('numerics_c1b') or {}
    lines = ['', '## Numerics (the verdict input of this queue)', '', 'Verdict: **%s**. Bound from S1 vs S2: %s.' % (
        n.get('verdict'), json.dumps(n.get('bound'), sort_keys=True)), '',
        '| candidate | reference | axis | common tokens | max abs logprob | max membership | within | bit-equal |',
        '|---|---|---|---|---|---|---|---|']
    for c in n.get('comparisons') or ():
        lines.append('| %s | %s | %s | %d | %.6g | %d | %s | %s |' % (c['candidate'], c['reference'], c['axis'], c['common_tokens'],
                                                                     c['max_abs_logprob'], c['max_membership'], c['within'],
                                                                     c['bit_equal']))
    lines += ['', 'Reasons: %s' % (', '.join(n.get('reasons') or ()) or '-'), '', 'Axis: ' + result['axis'] + '.', '',
              '### Open rows on overlap arms (%s, supersedes %s)' % (rule.RULE, rule.SUPERSEDES), '',
              '| arm | flushes checked | flagged | reasons |', '|---|---|---|---|']
    for name, f in sorted((result.get('open_rows_rule') or {}).get('arms', {}).items()):
        lines.append('| %s | %d | %d | %s |' % (name, f['checked'], len(f['flagged']), ', '.join(f['reasons']) or '-'))
    head, sep, tail = text.partition('\n## Mechanism\n')
    return head + '\n'.join(lines) + '\n' + sep + tail if sep else text + '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('arm')
    r.add_argument('--url', required=True)
    r.add_argument('--model', required=True)
    r.add_argument('--run-id', required=True)
    r.add_argument('--epoch', type=int, required=True)
    r.add_argument('--c1', choices=tuple(T.C1), required=True,
                   help='this arm\'s selection: serial (C0 off), off (C0 local, C1 off) or defer (C1 defer)')
    r.add_argument('--size', type=int, choices=T.SIZES, required=True)
    r.add_argument('--cohort', choices=progress.COHORTS, required=True)
    r.add_argument('--index', type=int, choices=T.INDEXES, required=True)
    r.add_argument('--workload', required=True)
    r.add_argument('--control-timeout-s', type=float, default=120.0)
    r.add_argument('--out', required=True)
    for action, text in (('grade', 'the core grade: the queue verdict'), ('read', 'every arm: a read only')):
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
    arms = [base.load(x) or {'label': None, 'complete': False, 'reasons': ['arm-artifact-unreadable:' + Path(x).name]}
            for x in a.arm]
    result = grade(arms, ap.load_pairs(a.ingress), ap.load_pairs(a.check), a.log, ap.load_pairs(a.watch),
                   part='core' if a.action == 'grade' else 'all')
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    if a.report:
        with open(a.report, 'x') as handle:
            handle.write(report(result))
    print('ALPHA_NEXT_C1B_%s %s' % ('GRADE' if a.action == 'grade' else 'READ', result['verdict']))
    if a.action == 'read':
        return 0 if result['verdict'] == 'PASS' else 2
    return {'PASS': 0, 'FAIL': 1}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
