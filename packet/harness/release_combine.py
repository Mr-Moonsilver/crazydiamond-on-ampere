#!/usr/bin/env python3
"""Combine the two release phases of one candidate into the release verdict.

Phase V (QSA_COW_VERIFY=1): verify-release.sh skip-bench; its `bytes` gate is the byte/invariant evidence.
Phase P (verify off, the configuration that ships): extra coverage runs + verify-release.sh with the benchmark;
its `bytes` gate is NOT_RUN by design.
PASS needs: every gate of V PASS except bench NOT_RUN; every gate of P PASS except bytes NOT_RUN; every extra step
(STEPS file, "name VERDICT" lines) PASS; both manifests show the same overlay file hashes, weights digest and harness
hashes, and settings that differ only in QSA_COW_VERIFY. Any FAIL -> FAIL; anything else missing -> INCONCLUSIVE.
Completeness (24 Oct review: a run with only `bytes PASS` and empty files was approved): each phase must contain
exactly REQUIRED_GATES once each and end with an `END <run id>` line (written by verify-release.sh after its last
gate), the two phases must have different run ids, and STEPS must contain exactly the profile's REQUIRED_STEPS once
each. Missing, extra or duplicate names -> INCONCLUSIVE (duplicates FAIL); absent evidence is never a PASS.
usage: release_combine.py V_DIR P_DIR STEPS OUT.txt [text|vision]   exit 0 PASS, 1 FAIL, 2 INCONCLUSIVE
"""
import re, sys

vdir, pdir, steps_f, out = sys.argv[1:5]
profile = sys.argv[5] if len(sys.argv) > 5 else 'text'
fails, inconc, lines = [], [], []
REQUIRED_GATES = {'harness', 'identity', 'cow-hook', 'inside', 'after', 'steer-agent-31', 'steer-agent-32',
                  'steer-agent-33', 'steer-std-41', 'steer-std-42', 'coverage', 'bytes', 'errors', 'bench'}
REQUIRED_STEPS = {'text': {'restart-verify', 'restart-async', 'inside-gen200-cancel', 'long-fullpool'}}
REQUIRED_STEPS['vision'] = REQUIRED_STEPS['text'] | {f'vision-{m}' for m in ('quality', 'bound', 'followup', 'isolation', 'uuid')}
run_ids = {}


def complete(where, pairs, required):
    names = [k for k, _ in pairs]
    dup = sorted({k for k in names if names.count(k) > 1})
    if dup:
        fails.append(f'{where}: duplicate {dup}')
    missing, extra = sorted(required - set(names)), sorted(set(names) - required)
    if missing:
        inconc.append(f'{where}: missing {missing}')
    if extra:
        inconc.append(f'{where}: unexpected {extra}')


def verdicts(d):
    try:
        rows = [l.split()[:2] for l in open(f'{d}/verdicts.txt') if l.strip()]
    except OSError:
        inconc.append(f'{d}/verdicts.txt missing')
        return {}
    ends = [v for k, v in rows if k == 'END']
    rows = [r for r in rows if r[0] != 'END']
    if len(ends) != 1:
        inconc.append(f'{d}: no single END marker (incomplete run)')
    else:
        run_ids[d] = ends[0]
    complete(d, rows, REQUIRED_GATES)
    return dict(rows)


def gate(phase, vs, allowed_not_run):
    for k, v in vs.items():
        lines.append(f'{phase} {k} {v}')
        if v == 'FAIL':
            fails.append(f'{phase} {k}')
        elif v != 'PASS' and not (v == 'NOT_RUN' and k == allowed_not_run):
            inconc.append(f'{phase} {k} {v}')


V, P = verdicts(vdir), verdicts(pdir)
gate('verify-phase', V, 'bench')
gate('async-phase', P, 'bytes')
if V and V.get('bytes') != 'PASS':
    (fails if V.get('bytes') == 'FAIL' else inconc).append(f"verify-phase bytes {V.get('bytes')}")
if len(run_ids) == 2 and len(set(run_ids.values())) != 2:
    fails.append('both phases carry the same run id')
step_rows = []
try:
    step_rows = [l.split()[:2] for l in open(steps_f) if l.strip()]
except OSError:
    inconc.append('steps file missing')
complete('steps', step_rows, REQUIRED_STEPS[profile])
for name, v in step_rows:
    lines.append(f'step {name} {v}')
    (fails if v == 'FAIL' else inconc if v != 'PASS' else []).append(f'step {name} {v}')


def section(text, title):
    m = re.search(rf'^## {re.escape(title)}\n(.*?)(?=^## |\Z)', text, re.M | re.S)
    return m.group(1).strip() if m else None


def ident(d):
    t = open(f'{d}/manifest-start.txt').read()
    overlay = sorted(l.split()[0] + ' ' + l.split()[-1] for l in (section(t, 'mounted overlay files') or '').splitlines() if l)
    settings = sorted(l for l in (section(t, 'settings') or '').splitlines() if not l.startswith('QSA_COW_VERIFY'))
    weights = re.search(r'^weights digest (\w+)', t, re.M)
    return {'overlay': overlay, 'settings': settings, 'weights': weights and weights.group(1), 'harness': section(t, 'harness')}


try:
    iv, ip = ident(vdir), ident(pdir)
    for k in iv:
        if not iv[k]:
            inconc.append(f'manifest lacks {k}')
        elif iv[k] != ip[k]:
            fails.append(f'phases ran different {k}')
    lines.append(f"identity: {len(iv['overlay'])} overlay files, weights {iv['weights'][:12] if iv['weights'] else None}, "
                 f"settings identical except QSA_COW_VERIFY: {iv['settings'] == ip['settings']}")
except OSError as e:
    inconc.append(f'manifest missing: {e}')

verdict = 'FAIL' if fails else ('INCONCLUSIVE' if inconc else 'PASS')
open(out, 'w').write('\n'.join(lines + [f'fails {fails}', f'inconclusive {inconc}', f'RELEASE {verdict}']) + '\n')
print(f'RELEASE {verdict} fails {fails} inconclusive {inconc}')
sys.exit({'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[verdict])
