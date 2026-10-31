#!/usr/bin/env python3
"""Native39 W2 length: choose, declare and prove a plannable fresh prompt length on the hub.

--write   derive harness/workloads/q18a-w2-overlap-native39.json from the P-c
          manifest (harness/workloads/q18a-w2-overlap.json, unchanged): only
          prefill.fresh.prompt_tokens, notes and a length_declaration block
          change.  Refuses to overwrite.
--check   the committed manifest equals that derivation (exit 1 otherwise).
--receipt OUT  the hub planner proof, written once:
  * the token model reproduces P-c run 1 (197315 / 201306) exactly;
  * the search: every declared fresh prompt_tokens in [199000, 200000]
    (e3_overlap_performance.MAX_PROMPT_TOKENS is the cap) with both counts,
    residues and the common-shift tolerance; the chosen value is the one with
    the largest tolerance, nearest 200000;
  * the real scheduler_hooks.staged_chunk_plan with the declared stock-split
    leaf (test_native37_robust.geometry_split; the pinned-image probe checks
    the band rule against the pinned scheduler) refuses P-c's fresh, plans
    P-c's append, plans the chosen fresh from 0 and the chosen append from 0
    and from every 176-aligned start up to the fresh prompt's last 176
    boundary (every possible prefix hit), and agrees with the band rule on
    every searched length.
No engine, no network, no box action.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
import e3_native39_w2_length as length  # noqa: E402
import e3_overlap_performance as shared  # noqa: E402

PARENT = ROOT / 'harness/workloads/q18a-w2-overlap.json'
PARENT_SHA = 'e6dd96cadf82c0c7786bca2f561f159d0d0590329b3630d21918dae8f429fbb1'  # as run by P-c (bundle fe194172a6c8)
MANIFEST = ROOT / 'harness/workloads/q18a-w2-overlap-native39.json'
SEARCH = (199000, 200000)
NOTES = ('Native39 trunk W2: the P-c manifest with only the fresh prompt_tokens changed (200000 -> {p}) so that '
         'both the fresh ({nf} tokens, residue {rf}) and the append ({na}, residue {ra}) have a native plan; P-c\'s '
         'fresh (197315, residue 19) had none. Corpus construction, seeds, append suffix, decode streams, timeouts, '
         'reserve, route plan and claims are P-c\'s. length_declaration is derived by '
         'e3_native39_w2_length.declaration and re-derived by the queue guard.')


def parent():
    data = PARENT.read_bytes()
    assert hashlib.sha256(data).hexdigest() == PARENT_SHA, 'P-c manifest changed'
    return json.loads(data), PARENT_SHA


def search():
    rows, seen = [], {}
    suffix = shared._prompt(length.P_C_SPEC['append']['seed'], 1, length.P_C_SPEC['append']['prompt_tokens'])
    tail = 1 + length.model_tokens(suffix)
    for p in range(SEARCH[0], SEARCH[1] + 1):
        words = int(p / 2.2)
        if words in seen:  # same text: agent_replay.text uses int(tokens / 2.2) words
            continue
        nf = length.model_tokens(shared._prompt(length.P_C_SPEC['fresh']['seed'], 0, p))
        seen[words] = p
        tol = length.tolerance(nf, nf + tail) if length.band(nf)['plans'] and length.band(nf + tail)['plans'] else None
        rows.append(dict(prompt_tokens=p, words=words, fresh=nf, append=nf + tail,
                         fresh_plans=length.band(nf)['plans'], append_plans=length.band(nf + tail)['plans'],
                         tolerance=tol))
    good = [r for r in rows if r['tolerance'] is not None]
    best = max(good, key=lambda r: (min(r['tolerance'].values()), r['prompt_tokens']))
    return rows, best


def derive(chosen):
    data, _ = parent()
    out = copy.deepcopy(data)
    out['prefill']['fresh']['prompt_tokens'] = chosen
    declaration = length.declaration(out)
    out['notes'] = NOTES.format(p=chosen, nf=declaration['fresh']['tokens'], rf=declaration['fresh']['residue'],
                                na=declaration['append']['tokens'], ra=declaration['append']['residue'])
    out['length_declaration'] = declaration
    assert not shared.validate_manifest(out)[0]
    return out


def text(data):
    return json.dumps(data, indent=2) + '\n'


def planner_proof(nf, na, rows):
    import scheduler_hooks as sh
    from test_native37_robust import geometry_split, CAP, MIN_ROWS
    started = time.monotonic()

    def plans(start, n):
        return bool(sh.staged_chunk_plan(start, n - 1, CAP, MIN_ROWS, geometry_split(n)))
    singles = {name: dict(tokens=n, plans=plans(0, n), rule=length.band(n))
               for name, n in (('p-c-fresh', length.P_C_OBSERVED['fresh']),
                               ('p-c-append', length.P_C_OBSERVED['append']),
                               ('native39-fresh', nf), ('native39-append', na))}
    starts = list(range(0, nf // length.HASH * length.HASH + 1, length.HASH))
    refused_starts = [s for s in starts if not plans(s, na)]
    mismatches = []
    for r in rows:
        for family in ('fresh', 'append'):
            n = r[family]
            if plans(0, n) != length.band(n)['plans']:
                mismatches.append(dict(prompt_tokens=r['prompt_tokens'], family=family, tokens=n))
    return dict(cap=CAP, minimum_rows=MIN_ROWS, split_leaf='test_native37_robust.geometry_split',
                singles=singles, append_starts=dict(count=len(starts), first=starts[0], last=starts[-1],
                                                    step=length.HASH, refused=refused_starts),
                rule_mismatches=mismatches, lengths_checked=2 * len(rows),
                wall_s=round(time.monotonic() - started, 3))


def receipt(out):
    rows, best = search()
    manifest = json.loads(MANIFEST.read_bytes())
    declaration = manifest['length_declaration']
    nf, na = declaration['fresh']['tokens'], declaration['append']['tokens']
    proof = planner_proof(nf, na, rows)
    _, parent_sha = parent()
    failures = []
    if not declaration['token_model']['validation']['exact']:
        failures.append('token-model-does-not-reproduce-p-c')
    if best['prompt_tokens'] != manifest['prefill']['fresh']['prompt_tokens']:
        failures.append('declared-length-is-not-the-search-choice')
    s = proof['singles']
    if s['p-c-fresh']['plans'] or not s['p-c-append']['plans']:
        failures.append('p-c-controls-differ')
    if not (s['native39-fresh']['plans'] and s['native39-append']['plans']):
        failures.append('native39-length-has-no-plan')
    if proof['append_starts']['refused']:
        failures.append('append-refused-from-a-prefix-hit-start')
    if proof['rule_mismatches']:
        failures.append('planner-differs-from-band-rule')
    if manifest != derive(best['prompt_tokens']):
        failures.append('manifest-differs-from-derivation')
    record = dict(schema='native39-w2-length-proof/1', verdict='FAIL' if failures else 'PASS', failures=failures,
                  manifest=str(MANIFEST.relative_to(ROOT)), manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  parent=str(PARENT.relative_to(ROOT)), parent_sha256=parent_sha,
                  chosen=best, search=dict(range=SEARCH, distinct_texts=len(rows),
                                           plannable=sum(r['tolerance'] is not None for r in rows),
                                           best_tolerance=min(best['tolerance'].values())),
                  declaration=declaration, planner=proof, gpu_execution=False, box_action=False)
    with Path(out).open('x') as f:
        json.dump(record, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({k: record[k] for k in ('verdict', 'failures', 'chosen', 'manifest_sha256')}, sort_keys=True))
    return 0 if not failures else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--write', action='store_true')
    g.add_argument('--check', action='store_true')
    g.add_argument('--receipt', type=Path)
    a = p.parse_args()
    if a.receipt:
        return receipt(a.receipt)
    _, best = search()
    data = text(derive(best['prompt_tokens']))
    if a.check:
        ok = MANIFEST.read_text() == data
        print('native39 W2 manifest ' + ('verified' if ok else 'DIFFERS'), best['prompt_tokens'])
        return 0 if ok else 1
    assert not MANIFEST.exists(), 'manifest exists; use --check'
    MANIFEST.write_text(data)
    MANIFEST.chmod(0o664)
    print('native39 W2 manifest written', best['prompt_tokens'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
