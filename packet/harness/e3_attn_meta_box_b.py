#!/usr/bin/env python3
"""q18a-attn-meta box on the c1b base: e3_attn_meta_box (imported as ``am``, never edited: chain a ran it) with c1b's C1-aware rules.

The lead's reason for the rebase: q18a-alpha-next-c1's 99 flagged flushes were successor commands, not faults, so the old
``progress-line-with-open-rows`` would make every attn-meta C1 arm (P, MP, P2, MP2) INC. c1b's rule supersedes it by name on overlap
arms; on C1-off arms (R, M, P's and MP's partners, the decode baselines) the old rule stays exactly as it is. Bound for one grade
call (c1box._bound), nothing mutated outside it:
  open rows   e3_alpha_next_c1b_box.progress_receipts over this table (``progress-line-open-rows-carried``, e3_c1_open_rows);
              check 4 ``c1-open-row-not-successor`` keeps c1b's class: INC ENGINE, never a FAIL;
  ring read   e3_alpha_next_c1b_box.ring_read (the stage line's running total read as per-chunk increments on overlap arms): a read.
c1b's numerics probe and its six-arm table are NOT taken: this queue's table, arms, gates and reads are am's (amendment A).
Every other action (arm, decode-arm, read's arguments) is am's main, with this grade bound. Receipts: counts and hashes only.
"""
import sys

import e3_attn_meta_box as am
import e3_alpha_next_c1b_box as c1b

c1box, T = am.c1box, am.T
rule = c1b.rule
progress, an = c1b.progress, c1b.an
_bound = c1box._bound


def grade(arms, ingress, checks, logs, watches=None, part='core', cross_run=None):
    c1b._C1B.clear()
    try:
        with _bound(c1b, dict(T=T)), _bound(progress, dict(progress_receipts=c1b.progress_receipts)), \
                _bound(an, dict(ring_read=c1b.ring_read)):
            result = AM_GRADE(arms, ingress, checks, logs, watches, part=part, cross_run=cross_run)
        open_rows = dict(c1b._C1B.get('open_rows') or {})
    finally:
        c1b._C1B.clear()
    engine = sorted(name for name, f in open_rows.items() if set(f['reasons']) & set(rule.ENGINE_REASONS))
    classification = result.get('classification')
    if result['verdict'] == 'INC' and engine:
        classification = ['ENGINE']        # check 4 of the open-rows rule: INC ENGINE, not FAIL (lead, c1b)
    result.update(classification=classification, open_rows_rule=dict(rule=rule.RULE, supersedes=rule.SUPERSEDES, arms=open_rows,
                                                                     engine_classified_arms=engine))
    return result


AM_GRADE = am.grade
AM_REPORT = am.report


def report(result):
    text = AM_REPORT(result)
    rows = (result.get('open_rows_rule') or {}).get('arms') or {}
    if rows:
        text += '\nOpen rows (%s, supersedes %s on overlap arms):\n' % (rule.RULE, rule.SUPERSEDES)
        for name, f in sorted(rows.items()):
            text += '  %s: flagged %d of %s flushes checked, reasons %s\n' % (name, len(f.get('flagged') or ()), f.get('checked'), f.get('reasons'))
    return text


def main(argv=None):
    with _bound(am, dict(grade=grade, report=report)):
        return am.main(argv)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
