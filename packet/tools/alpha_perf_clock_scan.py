#!/usr/bin/env python3
"""Alpha-perf client-clock gate: held-f's scan (tools/native41_progress_clock_scan.py) over the alpha-perf queue.

The scanner, rules, allowlists and negative controls are held-e's and held-f's, called unchanged. Declared configuration:
  * shell texts: q18a_alpha_perf{.draft,_declare,_settings}.sh (plus qlib.sh);
  * the arm root is e3_alpha_perf_box, which imports held-f's arm, which imports held-e's: the window/evidence rule is
    applied to all three, and the arm control sends overlap_evidence from this root;
  * the le-fix death scan resolution, as native43_lf_clock_scan declares it: exactly the one "${H:-}" line in
    host_crash_detected becomes an entry through a synthetic text and its one unclassified-invocation problem is consumed;
  * two controls beside the inherited ones: a second "${H:-}" death-tool call outside host_crash_detected, and a "${H:-}"
    call of e3_native40_unarmed_box.py, each must be flagged.
Static, no box action.
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native41_progress_clock_scan as held_f  # noqa: E402

base = held_f.base
TEXTS = ('q18a_alpha_perf.draft.sh', 'q18a_alpha_perf_declare.sh', 'q18a_alpha_perf_settings.sh')
base.DRAFT = TEXTS[0]
base.SHELL = TEXTS
held_f.ARM = 'e3_alpha_perf_box'
held_f.ARM_MODULES = ('e3_alpha_perf_box', 'e3_native41_progress_box', 'e3_native41_autoarm_box')
DEATH_FORM = 'python3 "${H:-}/native43_engine_death.py"'
DEATH_FUNCTION = 'host_crash_detected'
DEATH_TEXT = 'host_crash_detected-death-scan-resolution'
_held_f_scan = held_f.scan


def death_resolution(texts):
    """(line, consumed problem) of the one declared "${H:-}/" form, or None when it is not exactly as declared."""
    rows = [(name, i) for name, text in texts.items() for i, line in enumerate(text.split('\n'), 1)
            if re.search(r'\bpython3\s+"?\$\{H:-\}/', line)]
    if len(rows) != 1 or rows[0][0] != TEXTS[0]:
        return None
    line = rows[0][1]
    text = texts[TEXTS[0]].split('\n')[line - 1]
    if DEATH_FORM not in text or base.functions(texts[TEXTS[0]]).get(line) != DEATH_FUNCTION:
        return None
    return line, 'unclassified-python-invocation:%s:%d:${H:-}/native43_engine_death.py' % (TEXTS[0], line)


def scan(sources, texts, runner, setcfg):
    resolution = death_resolution(texts)
    if resolution:
        texts = dict(texts, **{DEATH_TEXT: 'python3 "$H/native43_engine_death.py"\n'})
    result = _held_f_scan(sources, texts, runner, setcfg)
    consumed = bool(resolution) and resolution[1] in result['problems']
    if consumed:
        result['problems'] = [p for p in result['problems'] if p != resolution[1]]
    result['death_scan_resolution'] = dict(at='%s:%d' % (TEXTS[0], resolution[0]) if resolution else None,
                                           function=DEATH_FUNCTION, form=DEATH_FORM, consumed=consumed,
                                           entry_in_closure='native43_engine_death' in result['closure'])
    if resolution and not result['death_scan_resolution']['entry_in_closure']:
        result['problems'].append('death-scan-entry-not-scanned')
    return result


held_f.scan = scan


def main(a):
    rc = held_f.main(a)
    record = json.loads(a.out.read_text())
    sources, texts = base.load(a.bundle.resolve() if a.bundle else None, a.tree.resolve() if a.tree else None)
    runner = a.runner.read_text() if a.runner else None
    setcfg = a.setcfg_reference.read_text() if a.setcfg_reference else None
    draft = TEXTS[0]
    controls = {
        'second-unset-safe-death-call-outside-host-crash-detected':
            dict(texts, **{draft: texts[draft] + '\npython3 "${H:-}/native43_engine_death.py" scan --out "$Q/x.json"\n'}),
        'unset-safe-form-runs-native40-unarmed-box':
            dict(texts, **{draft: texts[draft] + '\npython3 "${H:-}/e3_native40_unarmed_box.py" --out "$Q/x.json"\n'}),
    }
    extra = {label: scan(sources, t, runner, setcfg)['problems'] for label, t in controls.items()}
    record['negative_controls'].update(extra)
    ok = rc == 0 and record['verdict'] == 'PASS' and all(extra.values())
    record.update(schema='alpha-perf-clock-scan/1', derived_from='tools/native41_progress_clock_scan.py (held-f)',
                  verdict='PASS' if ok else 'FAIL', classification=None if ok else 'HARNESS')
    a.out.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], negatives={k: bool(v) for k, v in extra.items()},
                          death_scan_resolution=record['scan']['death_scan_resolution'])))
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
