#!/usr/bin/env python3
"""Native43 l-f client-clock gate: the lifecycle clock scan (tools/native43_lifecycle_clock_scan.py) over the l-f queue.

The scanner, the rules, the qualification_phase cli resolution and the inherited negative controls are
native43_lifecycle_clock_scan's, called unchanged (run_scan, main).  Declared configuration for l-f:
  * client texts: q18a_native43_lifecycle_f{.draft,_declare,_settings}.sh instead of the l-e texts (qlib.sh and,
    when given, the owner runner as before); the cli resolution therefore also yields row1's
    native43_row1_qualification.py (D8), scanned as an entry with its closure;
  * one more declared resolution, for the le-fix death scan in host_crash_detected: that function can run from the
    exit trap before H is assigned (set -u), so the le-fix delta calls python3 "${H:-}/native43_engine_death.py".
    Exactly that one line (in host_crash_detected, target native43_engine_death.py, the only "${H:-}/" python3
    form in the texts) becomes an entry through a synthetic text, and exactly its one unclassified-invocation
    problem is consumed; any other "${H:-}/" form stays unclassified;
  * two l-f negative controls beside the inherited five (each must be flagged): a second "${H:-}/" death-tool call
    outside host_crash_detected, and a "${H:-}/" call of e3_native40_unarmed_box.py.
"""
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native43_lifecycle_clock_scan as life  # noqa: E402

base = life.base
LF = ('q18a_native43_lifecycle_f.draft.sh', 'q18a_native43_lifecycle_f_declare.sh', 'q18a_native43_lifecycle_f_settings.sh')
life.LIFECYCLE = LF
base.SHELL = LF
DEATH_FORM = 'python3 "${H:-}/native43_engine_death.py"'
DEATH_FUNCTION = 'host_crash_detected'
DEATH_TEXT = 'host_crash_detected-death-scan-resolution'
_lifecycle_run_scan = life.run_scan


def death_resolution(texts):
    """(line, consumed problem) of the one declared "${H:-}/" form, or None when it is not exactly as declared."""
    rows = [(name, i) for name, text in texts.items() for i, line in enumerate(text.split('\n'), 1)
            if re.search(r'\bpython3\s+"?\$\{H:-\}/', line)]
    if len(rows) != 1 or rows[0][0] != LF[0]:
        return None
    line = rows[0][1]
    text = texts[LF[0]].split('\n')[line - 1]
    if DEATH_FORM not in text or base.functions(texts[LF[0]]).get(line) != DEATH_FUNCTION:
        return None
    return line, 'unclassified-python-invocation:%s:%d:${H:-}/native43_engine_death.py' % (LF[0], line)


def run_scan(sources, texts, runner, setcfg):
    resolution = death_resolution(texts)
    if resolution:
        texts = dict(texts, **{DEATH_TEXT: 'python3 "$H/native43_engine_death.py"\n'})
    result = _lifecycle_run_scan(sources, texts, runner, setcfg)
    consumed = bool(resolution) and resolution[1] in result['problems']
    if consumed:
        result['problems'] = [p for p in result['problems'] if p != resolution[1]]
    result['death_scan_resolution'] = dict(at='%s:%d' % (LF[0], resolution[0]) if resolution else None,
                                           function=DEATH_FUNCTION, form=DEATH_FORM, consumed=consumed,
                                           entry_in_closure='native43_engine_death' in result['closure'])
    if resolution and not result['death_scan_resolution']['entry_in_closure']:
        result['problems'].append('death-scan-entry-not-scanned')
    return result


life.run_scan = run_scan


def lf_negatives(a):
    """The l-f negative controls over the same sources and texts the record scanned."""
    sources, texts = base.load(a.bundle.resolve() if a.bundle else None, a.tree.resolve() if a.tree else None)
    runner = a.runner.read_text() if a.runner else None
    setcfg = a.setcfg_reference.read_text() if a.setcfg_reference else None
    draft = LF[0]
    controls = {
        'second-unset-safe-death-call-outside-host-crash-detected':
            dict(texts, **{draft: texts[draft] + '\npython3 "${H:-}/native43_engine_death.py" scan --out "$Q/x.json"\n'}),
        'unset-safe-form-runs-native40-unarmed-box':
            dict(texts, **{draft: texts[draft] + '\npython3 "${H:-}/e3_native40_unarmed_box.py" --out "$Q/x.json"\n'}),
    }
    return {label: run_scan(sources, t, runner, setcfg)['problems'] for label, t in controls.items()}


def main(a):
    rc = life.main(a)
    record = json.loads(a.out.read_text())
    extra = lf_negatives(a)
    record['negative_controls'].update(extra)
    record['configuration'].update(death_scan_resolution=dict(function=DEATH_FUNCTION, form=DEATH_FORM),
                                   lf_negative_controls=sorted(extra))
    ok = rc == 0 and record['verdict'] == 'PASS' and all(extra.values())
    record.update(schema='native43-lf-clock-scan/1', verdict='PASS' if ok else 'FAIL',
                  classification=None if ok else 'HARNESS')
    a.out.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], lf_negatives={k: bool(v) for k, v in extra.items()},
                          death_scan_resolution=record['scan']['death_scan_resolution'])))
    return 0 if ok else 1


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--bundle', type=Path)
    g.add_argument('--tree', type=Path)
    p.add_argument('--runner', type=Path)
    p.add_argument('--setcfg-reference', type=Path)
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
