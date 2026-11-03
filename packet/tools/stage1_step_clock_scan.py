#!/usr/bin/env python3
"""q18a-stage1-step client-clock gate: attn-meta's configuration by name (claude/attn-meta-c1b: tools/attn_meta_clock_scan.py and _b, flattened here, never edited there), Step A's scan
and controls, unchanged, over this queue.
  * shell texts: q18a_stage1_step{.draft,_declare,_settings}.sh (plus qlib.sh);
  * the arm root is e3_stage1_step_box, which imports e3_alpha_next_c1_box and e3_alpha_next_c1b_box; ARM_MODULES is that root, its table and reads, c1b's, then c1's root and alpha-next's read from their
    texts, so a client clock in any inherited arm still refuses; each listed module must be in the closure (held-f's rule);
  * the one declared unset-safe death-scan form is native43_engine_death_h.py's; a control adds: the old form back in the draft is not consumed.
The record is step_a_clock_scan's with its own schema name and the verdict re-checked. No GPU, no box action.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_a_clock_scan as step_a  # noqa: E402


def _arm_modules(name):
    for node in ast.parse((Path(__file__).resolve().with_name(name)).read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'ARM_MODULES':
            value = node.value.left if isinstance(node.value, ast.BinOp) else node.value
            return ast.literal_eval(value)
    raise AssertionError(name + ': ARM_MODULES not found')


TEXTS = ('q18a_stage1_step.draft.sh', 'q18a_stage1_step_declare.sh', 'q18a_stage1_step_settings.sh')
ARM_MODULES = ('e3_stage1_step_box', 'stage1_step_arms', 'stage1_step_read', 'e3_alpha_next_c1b_box', 'alpha_next_c1b_arms',
               'e3_c1_open_rows') + _arm_modules('alpha_next_c1_clock_scan.py') + _arm_modules('alpha_next_clock_scan.py')
step_a.TEXTS = TEXTS
step_a.base.DRAFT = TEXTS[0]
step_a.base.SHELL = TEXTS
step_a.held_f.ARM = ARM_MODULES[0]
step_a.held_f.ARM_MODULES = ARM_MODULES
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
    rc = step_a.main(a)
    record = json.loads(a.out.read_text())
    ok = rc == 0 and record['verdict'] == 'PASS' and record['arm_modules'] == list(ARM_MODULES) \
        and all(m in record['scan']['closure'] for m in ARM_MODULES) and sorted(record['shell_shas']) >= sorted(TEXTS)
    sources, texts = step_a.base.load(a.bundle.resolve() if a.bundle else None, a.tree.resolve() if a.tree else None)
    runner = a.runner.read_text() if a.runner else None
    setcfg = a.setcfg_reference.read_text() if a.setcfg_reference else None
    old = {TEXTS[0]: texts[TEXTS[0]].replace(step_a.DEATH_FORM, 'python3 "${H:-}/native43_engine_death.py"')}
    old_form = scan(sources, dict(texts, **old), runner, setcfg)['problems']
    record['negative_controls']['old-death-form-not-consumed'] = old_form
    resolved = record['scan']['death_scan_resolution']
    ok = ok and bool(old_form) and resolved['consumed'] and resolved['entry_in_closure']
    record.update(schema='stage1-step-clock-scan/1', derived_from='tools/attn_meta_clock_scan{,_b}.py by name, over tools/step_a_clock_scan.py (held-f\'s scan)',
                  verdict='PASS' if ok else 'FAIL', classification=None if ok else 'HARNESS')
    a.out.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], arm_modules=record['arm_modules'], death_scan_resolution=resolved, old_form_problems=len(old_form))))
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
