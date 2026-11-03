#!/usr/bin/env python3
"""q18a-attn-meta client-clock gate on the c1b base: tools/attn_meta_clock_scan.py (chain a ran it; never edited) with the arm root
e3_attn_meta_box_b (it imports e3_attn_meta_box and e3_alpha_next_c1b_box) in front of its ARM_MODULES, and the one declared unset-safe
death-scan form moved to native43_engine_death_h.py (tools/attn_meta_queue_b.py); c1b-h's tools/alpha_next_c1b_clock_scan_h.py rule. A
control adds: the old form back in the draft is not consumed. Every other rule and control is step_a's. No GPU, no box action.
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attn_meta_clock_scan as am  # noqa: E402  (configures step_a for attn-meta)

step_a = am.step_a
ARM_MODULES = ('e3_attn_meta_box_b', 'e3_alpha_next_c1b_box', 'alpha_next_c1b_arms', 'e3_c1_open_rows') + tuple(am.ARM_MODULES)
TEXTS = am.TEXTS
step_a.held_f.ARM = ARM_MODULES[0]
step_a.held_f.ARM_MODULES = ARM_MODULES
am.ARM_MODULES = ARM_MODULES
ENTRY = 'native43_engine_death_h'
step_a.DEATH_FORM = 'python3 "${H:-}/%s.py"' % ENTRY
_scan_core = step_a._held_f_scan


def death_resolution(texts):
    """(line, consumed problem) of the one declared "${H:-}/" form, or None when it is not exactly as declared."""
    rows = [(name, i) for name, text in texts.items() for i, line in enumerate(text.split('\n'), 1)
            if re.search(r'\bpython3\s+"?\$\{H:-\}/', line)]
    if len(rows) != 1 or rows[0][0] != TEXTS[0]:
        return None
    line = rows[0][1]
    text = texts[TEXTS[0]].split('\n')[line - 1]
    if step_a.DEATH_FORM not in text or step_a.base.functions(texts[TEXTS[0]]).get(line) != step_a.DEATH_FUNCTION:
        return None
    return line, 'unclassified-python-invocation:%s:%d:${H:-}/%s.py' % (TEXTS[0], line, ENTRY)


def scan(sources, texts, runner, setcfg):
    resolution = death_resolution(texts)
    if resolution:
        texts = dict(texts, **{step_a.DEATH_TEXT: 'python3 "$H/%s.py"\n' % ENTRY})
    result = _scan_core(sources, texts, runner, setcfg)
    consumed = bool(resolution) and resolution[1] in result['problems']
    if consumed:
        result['problems'] = [p for p in result['problems'] if p != resolution[1]]
    result['death_scan_resolution'] = dict(at='%s:%d' % (TEXTS[0], resolution[0]) if resolution else None,
                                           function=step_a.DEATH_FUNCTION, form=step_a.DEATH_FORM, consumed=consumed,
                                           entry_in_closure=ENTRY in result['closure'])
    if resolution and not result['death_scan_resolution']['entry_in_closure']:
        result['problems'].append('death-scan-entry-not-scanned')
    return result


step_a.death_resolution = death_resolution
step_a.scan = scan
step_a.held_f.scan = scan


def main(a):
    rc = am.main(a)
    record = json.loads(a.out.read_text())
    sources, texts = step_a.base.load(a.bundle.resolve() if a.bundle else None, a.tree.resolve() if a.tree else None)
    runner = a.runner.read_text() if a.runner else None
    setcfg = a.setcfg_reference.read_text() if a.setcfg_reference else None
    old = {TEXTS[0]: texts[TEXTS[0]].replace(step_a.DEATH_FORM, 'python3 "${H:-}/native43_engine_death.py"')}
    old_form = scan(sources, dict(texts, **old), runner, setcfg)['problems']
    record['negative_controls']['old-death-form-not-consumed'] = old_form
    resolved = record['scan']['death_scan_resolution']
    ok = rc == 0 and record['verdict'] == 'PASS' and bool(old_form) and resolved['consumed'] and resolved['entry_in_closure']
    record.update(schema='attn-meta-clock-scan-b/1', derived_from='tools/attn_meta_clock_scan.py (the _h death form, the _b arm root)',
                  verdict='PASS' if ok else 'FAIL', classification=None if ok else 'HARNESS')
    a.out.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], death_scan_resolution=resolved, old_form_problems=len(old_form))))
    return 0 if ok else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--bundle', type=Path)
    g.add_argument('--tree', type=Path)
    p.add_argument('--runner', type=Path)
    p.add_argument('--setcfg-reference', type=Path)
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
