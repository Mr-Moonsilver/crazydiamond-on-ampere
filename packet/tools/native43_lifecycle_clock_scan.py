#!/usr/bin/env python3
"""Native43 lifecycle client-clock gate: native43_clock_scan.scan over the lifecycle queue, no client-side identity equality.

The rules and the scanner are native43_clock_scan's, called unchanged (scan(), load(), heredocs(), functions()).
Declared configuration for the lifecycle queue:
  * client texts: q18a_native43_lifecycle{.draft,_declare,_settings}.sh, qlib.sh and, when given, the owner runner;
  * e3_native_performance is a legitimate client module here (the lifecycle CLI's reuse prefill), so it leaves
    FORBIDDEN_MODULES and NAMES; its identity calls stay refused by the unchanged CALLS rule, and the lifecycle
    requirement below replaces the pair's boundary-client rule: e3_native_performance must be in the closure,
    import native43_boundary_clock.prove and name no clock_identity at all;
  * qualification_phase's `python3 "$H/$cli"`: cli is assigned in that function only (row1's
    e3_q18a_qualification.py, row4's e3_native35_p_c_placement.py); both values become entries through a
    synthetic text, and exactly that one unclassified-invocation problem is consumed by the resolution.
Negative controls (each must be flagged): the base four on the lifecycle draft plus the lifecycle's own reuse
module with the pre-fix native_window (the bundle-source version at the n43-f build commit).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native43_clock_scan as base  # noqa: E402

LIFECYCLE = ('q18a_native43_lifecycle.draft.sh', 'q18a_native43_lifecycle_declare.sh', 'q18a_native43_lifecycle_settings.sh')
PRE_FIX_COMMIT = '2a4917b7df6e6e264b0eefe8f9287d3a4fb915d6'  # n43-f build source: native_window with identity equality
base.SHELL = LIFECYCLE
base.FORBIDDEN_MODULES = ('e3_native40_unarmed_box',)
base.NAMES = re.compile(r'prove_clock_domain|clock_identity|native_window\(|NativeControlClient|e3_native40_unarmed_box')
CLI_TEXT = 'qualification_phase-cli-resolution'


def cli_resolution(draft):
    lines = draft.split('\n')
    body = base.functions(draft)
    rows = [i for i, line in enumerate(lines, 1) if 'python3 "$H/$cli"' in line]
    assert len(rows) == 1 and body.get(rows[0]) == 'qualification_phase', rows
    values = sorted({m for i, line in enumerate(lines, 1) if body.get(i) == 'qualification_phase'
                     for m in re.findall(r'\bcli=([A-Za-z0-9_]+\.py)\b', line)})
    assigned = [i for i, line in enumerate(lines, 1) if re.search(r'\bcli=', line)]
    assert all(body.get(i) == 'qualification_phase' for i in assigned), assigned
    return rows[0], values


def run_scan(sources, texts, runner, setcfg):
    draft = texts[LIFECYCLE[0]]
    line, values = cli_resolution(draft)
    texts = dict(texts, **{CLI_TEXT: ''.join('python3 "$H/%s"\n' % v for v in values)})
    result = base.scan(sources, texts, runner, setcfg)
    consumed = 'unclassified-python-invocation:%s:%d:$H/$cli' % (LIFECYCLE[0], line)
    problems = [p for p in result['problems'] if p not in (consumed, 'boundary-client-bracket-proof-missing')]
    performance = sources.get('e3_native_performance', ('', ''))[1]
    bracket = ('e3_native_performance' in result['closure']
               and bool(re.search(r'^from native43_boundary_clock import prove\b', performance, re.M))
               and 'clock_identity' not in performance)
    if not bracket:
        problems.append('reuse-clock-bracket-proof-missing')
    result.update(problems=problems, cli_resolution=dict(at='%s:%d' % (LIFECYCLE[0], line), values=values,
                                                         consumed=consumed in result['problems']),
                  reuse_clock_uses_native43_prove=bracket)
    return result


def main(a):
    bundle = a.bundle.resolve() if a.bundle else None
    tree = a.tree.resolve() if a.tree else None
    sources, texts = base.load(bundle, tree)
    runner = a.runner.read_text() if a.runner else None
    setcfg = a.setcfg_reference.read_text() if a.setcfg_reference else None
    result = run_scan(sources, texts, runner, setcfg)
    draft = LIFECYCLE[0]
    root = Path(__file__).resolve().parents[1]
    pre_fix = subprocess.check_output(['git', 'show', PRE_FIX_COMMIT + ':harness/e3_native_performance.py'], cwd=root, text=True)
    controls = {
        'queue-runs-native40-unarmed-box': (dict(texts, **{draft: texts[draft] + '\npython3 "$H/e3_native40_unarmed_box.py" --out "$Q/x.json"\n'}),
                                            {'e3_native40_unarmed_box': 'from native_overlap_metrics import clock_identity\n'}),
        'entry-calls-prove-clock-domain': (dict(texts, **{draft: texts[draft] + '\npython3 "$H/native43_synthetic_client.py"\n'}),
                                           {'native43_synthetic_client': 'from native_overlap_metrics import prove_clock_domain, clock_identity\n'
                                                                         'prove_clock_domain({}, clock_identity(), 1, 2)\n'}),
        'entry-compares-boot-id-inline': (dict(texts, **{draft: texts[draft] + '\npython3 -B "$H/native43_synthetic_inline.py"\n'}),
                                          {'native43_synthetic_inline': "from pathlib import Path\nok = Path('/proc/sys/kernel/random/boot_id').read_text() == 'x'\n"}),
        'unclassified-python-form': (dict(texts, **{draft: texts[draft] + '\npython3 "$X_DIR/../e3_q18a_lifecycle.py"\n'}), {}),
        'reuse-module-pre-fix-native-window': (texts, {'e3_native_performance': pre_fix}),
    }
    negatives = {}
    for label, (t, extra) in controls.items():
        s2 = dict(sources, **{k: ('<synthetic>/' + k + '.py', v) for k, v in extra.items()})
        negatives[label] = run_scan(s2, t, runner, setcfg)['problems']
    ok = not result['problems'] and all(negatives.values())
    origin = dict(bundle=str(bundle), bundle_manifest=base.sha(bundle / 'SHA256SUMS')) if bundle else dict(tree=str(tree))
    record = dict(schema='native43-lifecycle-clock-scan/1', verdict='PASS' if ok else 'FAIL',
                  classification=None if ok else 'HARNESS', **origin,
                  shell_shas={q: hashlib.sha256(t.encode()).hexdigest() for q, t in sorted(texts.items())},
                  runner=dict(path=str(a.runner), sha256=base.sha(a.runner)) if a.runner else None,
                  setcfg_reference=dict(path=str(a.setcfg_reference), sha256=base.sha(a.setcfg_reference),
                                        note='box-resident /tmp/setcfg.py is not in the bundle; its repo reference is scanned')
                  if a.setcfg_reference else None,
                  configuration=dict(shell=list(LIFECYCLE), forbidden_modules=list(base.FORBIDDEN_MODULES),
                                     names=base.NAMES.pattern, pre_fix_commit=PRE_FIX_COMMIT),
                  scan=result, negative_controls=negatives, static_only=True, box_action=False)
    a.out.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], problems=result['problems'], units=len(result['client_units']),
                          roots=len(result['roots']), closure=len(result['closure']), invocations=len(result['invocations']),
                          negatives={k: bool(v) for k, v in negatives.items()})))
    return 0 if ok else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--bundle', type=Path)
    g.add_argument('--tree', type=Path)
    p.add_argument('--runner', type=Path)
    p.add_argument('--setcfg-reference', type=Path)
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
