#!/usr/bin/env python3
"""Held-f client-clock gate: held-e's native41 gate (tools/native41_clock_scan.py) over the held-f queue texts.

Only the shell texts change (the held-f draft, declare and settings plus qlib.sh). The held-f arm client
(e3_native41_progress_box) is a queue root and imports held-e's arm (e3_native41_autoarm_box), which is then no
longer a root; held-e's window/evidence rule reads roots only, so it is applied here to both arm modules, and held-e's
arm-sends-overlap-evidence control (on e3_native41_autoarm_box) must still be flagged. One control is added: the
held-f arm itself sending overlap_evidence. Every other held-e rule, allowlist and negative control applies
unchanged. Static, no box action.
"""
import argparse
import hashlib
import json
from pathlib import Path

import native41_clock_scan as base

base.DRAFT = 'q18a_native41_progress.draft.sh'
base.SHELL = (base.DRAFT, 'q18a_native41_progress_declare.sh', 'q18a_native41_progress_settings.sh')
ARM = 'e3_native41_progress_box'
ARM_MODULES = (ARM, 'e3_native41_autoarm_box')


def scan(sources, texts, runner, setcfg):
    """held-e's scan plus its window/evidence rule over both arm modules (root or imported)."""
    result = base.scan(sources, texts, runner, setcfg)
    for m in ARM_MODULES:
        if m in result['roots'] or m not in result['closure']:
            continue
        for i, line in enumerate(sources[m][1].splitlines(), 1):
            if base.NATIVE_WINDOW.search(line):
                result['window_or_evidence_references'].append('%s:%d: %s' % (sources[m][0], i, line.strip()[:140]))
    if result['window_or_evidence_references'] and 'client-sends-window-or-evidence-action' not in result['problems']:
        result['problems'].append('client-sends-window-or-evidence-action')
    return result


def main(a):
    bundle = a.bundle.resolve() if a.bundle else None
    tree = a.tree.resolve() if a.tree else None
    sources, texts = base.load(bundle, tree)
    runner = a.runner.read_text() if a.runner else None
    setcfg = a.setcfg_reference.read_text() if a.setcfg_reference else None
    result = scan(sources, texts, runner, setcfg)
    draft = base.DRAFT
    site_src = sources[base.CONTAINER_PROOF][1]
    controls = {
        'queue-runs-native40-unarmed-box': (dict(texts, **{draft: texts[draft] + '\npython3 "$H/e3_native40_unarmed_box.py" --out "$Q/x.json"\n'}),
                                            {'e3_native40_unarmed_box': 'from native_overlap_metrics import clock_identity\n'}),
        'entry-calls-prove-clock-domain': (dict(texts, **{draft: texts[draft] + '\npython3 "$H/native43_synthetic_client.py"\n'}),
                                           {'native43_synthetic_client': 'from native_overlap_metrics import prove_clock_domain, clock_identity\n'
                                                                         'prove_clock_domain({}, clock_identity(), 1, 2)\n'}),
        'entry-compares-boot-id-inline': (dict(texts, **{draft: texts[draft] + '\npython3 -B "$H/native43_synthetic_inline.py"\n'}),
                                          {'native43_synthetic_inline': "from pathlib import Path\nok = Path('/proc/sys/kernel/random/boot_id').read_text() == 'x'\n"}),
        'unclassified-python-form': (dict(texts, **{draft: texts[draft] + '\npython3 "$X_DIR/../e3_native41_unarmed_box.py"\n'}), {}),
        'unarmed-client-compares-client-identity': (texts, {base.CONTAINER_PROOF: site_src.replace(
            'domain = prove_clock_domain(engine, expected, send, recv)',
            'from native_overlap_metrics import clock_identity\n    domain = prove_clock_domain(engine, clock_identity(), send, recv)')}),
        'docker-exec-not-read-only': (texts, {base.CONTAINER_PROOF: site_src.replace(
            "['exec', container_id, 'cat', '/proc/sys/kernel/random/boot_id']",
            "['exec', container_id, 'sh', '-c', 'cat /proc/sys/kernel/random/boot_id']")}),
        'arm-sends-overlap-evidence': (texts, {'e3_native41_autoarm_box': sources['e3_native41_autoarm_box'][1]
                                               + "\n\ndef _x(client):\n    return client.call('overlap_evidence')\n"}),
        'progress-arm-sends-overlap-evidence': (texts, {ARM: sources[ARM][1]
                                                        + "\n\ndef _x(client):\n    return client.call('overlap_evidence')\n"}),
    }
    negatives = {}
    for label, (t, extra) in controls.items():
        s2 = dict(sources, **{k: ('<synthetic>/' + k + '.py', v) for k, v in extra.items()})
        negatives[label] = scan(s2, t, runner, setcfg)['problems']
    ok = (not result['problems'] and all(negatives.values()) and ARM in result['roots']
          and all(m in result['closure'] for m in ARM_MODULES))
    origin = dict(bundle=str(bundle), bundle_manifest=base.sha(bundle / 'SHA256SUMS')) if bundle else dict(tree=str(tree))
    record = dict(schema='native41-progress-clock-scan/1', derived_from='tools/native41_clock_scan.py (held-e)',
                  verdict='PASS' if ok else 'FAIL', classification=None if ok else 'HARNESS',
                  **origin, shell_shas={q: hashlib.sha256(t.encode()).hexdigest() for q, t in sorted(texts.items())},
                  runner=dict(path=str(a.runner), sha256=base.sha(a.runner)) if a.runner else None,
                  setcfg_reference=dict(path=str(a.setcfg_reference), sha256=base.sha(a.setcfg_reference),
                                        note='box-resident /tmp/setcfg.py is not in the bundle; its repo reference is scanned')
                  if a.setcfg_reference else None,
                  scan=result, negative_controls=negatives, arm_modules=list(ARM_MODULES),
                  static_only=True, box_action=False)
    with a.out.open('x') as f:
        f.write(json.dumps(record, sort_keys=True, indent=2) + '\n')
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
