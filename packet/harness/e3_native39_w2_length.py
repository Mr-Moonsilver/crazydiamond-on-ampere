#!/usr/bin/env python3
"""Native39 W2 fresh length that has a native plan: declaration, band rule and the queue guard.

P-c's W2 fresh prompt (prompt_tokens 200000 -> 197315 tokens on the box) had no
native plan; its append (201306 tokens) planned.  The trunk W2 manifest
(workloads/q18a-w2-overlap-native39.json) keeps the corpus construction
(e3_overlap_performance._prefill_prompts: agent_replay.text words, seed 2 fresh,
seed 7 append suffix of prompt_tokens 4096, joined by one newline under the same
cache salt) and declares a fresh prompt_tokens for which both requests plan.

Band rule.  Block 4048 = 23 x 176 (prefix_match_unit 176), so the last stock
stop before the end of an n-token prompt is max(n - n % 4048, n // 176 * 176)
and the native lane ends at n - 1 (one-token handback).  A tail of 1..63 rows
after that stop is below QSA_STAGE_MIN_ROWS (64), so
    refused  <=>  2 <= n % 176 <= 64.
A prefix-hit start is a multiple of 176, hence either a 4048 boundary or at
least 176 from one, so the same final stop and the same rule hold for the
append from its prefix hit.  tools/native39_w2_length.py proves this with the
real staged_chunk_plan; the pinned-image probe (e3_native39_trunk_probe P)
admits both requests through the real scheduler, the append from its real
prefix hit.

Token model (validated on P-c run 1: 197315 and 201306 exactly): every word of
the 40-word vocabulary is one token and every digit of its optional number is
one token; the append adds one newline token and the suffix.

Guard (queue, after the startup grade, before row1):
  e3_native39_w2_length.py --url URL --model MODEL --workload MANIFEST --out OUT.json
POSTs the two prompt texts to the engine's own /tokenize, and passes only if
the manifest's length_declaration is the one this module derives, both counts
equal it, and both plan.  rc 0 PASS, 1 FAIL (declaration or counts differ, or a
request would be refused: a HARNESS defect, the W2 lengths would not be the
declared ones), 2 INCONCLUSIVE (tokenize unavailable).  The bearer comes from
VLLM_API_KEY and is sent only to a loopback URL; no prompt text, token list or
key is written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e3_overlap_performance as shared  # noqa: E402

SCHEMA = 'native39-w2-length/1'
BLOCK, HASH, MIN_ROWS = 4048, 176, 64
REFUSED_RESIDUES = (2, MIN_ROWS)  # inclusive: n % 176 in [2, 64] has no plan
# P-c run 1 (results/box/q18a-native35-p-c-run-1, native35-performance-grade.json
# rounds[0].raw.full_serving_solo): the engine's prompt_tokens for the P-c manifest.
P_C_OBSERVED = {'fresh': 197315, 'append': 201306}
P_C_SPEC = {'fresh': {'prompt_tokens': 200000, 'seed': 2}, 'append': {'prompt_tokens': 4096, 'seed': 7}}
P_C_SOURCE = ('fn-main/results/box/q18a-native35-p-c-run-1/box/n35pc/q18a-native35-p-c/'
              'native35-performance-grade.json rounds[0].raw.full_serving_solo.{fresh,append}.prompt_tokens_observed')


def model_tokens(text):
    """Words plus digits of one space-joined agent_replay.text block."""
    return sum(1 + sum(ch.isdigit() for ch in word) for word in text.split(' '))


def band(n):
    lcp = n - n % BLOCK
    tail = n // HASH * HASH
    stop = max(lcp, tail if lcp < tail < n else 0)
    gap = (n - 1) - stop
    refused = 1 <= gap <= MIN_ROWS - 1
    assert refused == (REFUSED_RESIDUES[0] <= n % HASH <= REFUSED_RESIDUES[1])
    return dict(tokens=n, residue=n % HASH, stop=stop, gap=gap, plans=not refused)


def tolerance(fresh, append):
    """Largest common shift (tokens) of both counts that keeps both plannable."""
    def ok(d):
        return band(fresh + d)['plans'] and band(append + d)['plans']
    down = 0
    while ok(-(down + 1)):
        down += 1
    up = 0
    while ok(up + 1):
        up += 1
    return dict(down=down, up=up)


def counts(fresh_spec, append_spec):
    """Model counts of the exact _prefill_prompts texts for these specs."""
    fresh = shared._prompt(fresh_spec['seed'], 0, fresh_spec['prompt_tokens'])
    suffix = shared._prompt(append_spec['seed'], 1, append_spec['prompt_tokens'])
    nf = model_tokens(fresh)
    return fresh, fresh + '\n' + suffix, nf, nf + 1 + model_tokens(suffix)


def declaration(data):
    """The length_declaration block for a W2 manifest (computed, never read)."""
    problems, cfg = shared.validate_manifest({k: v for k, v in data.items() if k != 'length_declaration'})
    if problems:
        raise ValueError('manifest invalid: %s' % problems)
    prompts = shared._prefill_prompts(cfg)
    fresh, append, nf, na = counts(cfg['prefill']['fresh'], cfg['prefill']['append'])
    assert (prompts['fresh'], prompts['append']) == (fresh, append), 'corpus construction differs'
    _, _, pf, pa = counts(P_C_SPEC['fresh'], P_C_SPEC['append'])
    return dict(
        schema=SCHEMA,
        rule=dict(block=BLOCK, prefix_match_unit=HASH, minimum_stage_rows=MIN_ROWS,
                  refused_when='%d <= tokens %% %d <= %d' % (REFUSED_RESIDUES[0], HASH, REFUSED_RESIDUES[1]),
                  basis='native lane ends at tokens-1; last stock stop max(tokens - tokens % 4048, tokens // 176 * 176); '
                        'a 1..63-row tail is below QSA_STAGE_MIN_ROWS; 4048 = 23 x 176, so a 176-aligned prefix-hit '
                        'start keeps the same final stop'),
        token_model=dict(rule='one token per word plus one per digit; append = fresh + newline (1) + suffix',
                         validation=dict(source=P_C_SOURCE, p_c_spec=P_C_SPEC, observed=P_C_OBSERVED,
                                         model={'fresh': pf, 'append': pa},
                                         exact=(pf, pa) == (P_C_OBSERVED['fresh'], P_C_OBSERVED['append'])),
                         p_c=dict(fresh=band(pf), append=band(pa))),
        fresh=dict(prompt_tokens_declared=cfg['prefill']['fresh']['prompt_tokens'],
                   prompt_sha256=hashlib.sha256(fresh.encode()).hexdigest(), **band(nf)),
        append=dict(prompt_tokens_declared=cfg['prefill']['append']['prompt_tokens'],
                    prompt_sha256=hashlib.sha256(append.encode()).hexdigest(), **band(na)),
        common_shift_tolerance=tolerance(nf, na),
        max_prompt_tokens=shared.MAX_PROMPT_TOKENS)


def _post(url, path, body, timeout):
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('VLLM_API_KEY') or ''
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(url.rstrip('/') + path, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read())


def guard(url, model, workload, timeout=120):
    data = json.loads(Path(workload).read_bytes())
    record = dict(schema=SCHEMA + '/guard', workload=str(workload),
                  workload_sha256=hashlib.sha256(Path(workload).read_bytes()).hexdigest(),
                  verdict='INCONCLUSIVE', rc=2, failures=[])
    expected = declaration(data)
    if data.get('length_declaration') != expected:
        record.update(verdict='FAIL', rc=1, failures=['manifest-length-declaration-differs-from-derivation'])
        return record
    record['declared'] = {k: {f: expected[k][f] for f in ('tokens', 'residue', 'gap', 'plans')}
                          for k in ('fresh', 'append')}
    cfg = shared.validate_manifest({k: v for k, v in data.items() if k != 'length_declaration'})[1]
    prompts = shared._prefill_prompts(cfg)
    observed = {}
    try:
        for family in ('fresh', 'append'):
            reply = _post(url, '/tokenize', {'model': model, 'prompt': prompts[family]}, timeout)
            tokens = reply.get('tokens')
            count = reply.get('count')
            if type(tokens) is not list or (count is not None and count != len(tokens)):
                raise ValueError('tokenize-reply-shape')
            observed[family] = band(len(tokens))
    except (OSError, ValueError) as exc:
        record['error'] = type(exc).__name__
        return record
    record['observed'] = observed
    for family in ('fresh', 'append'):
        if observed[family]['tokens'] != expected[family]['tokens']:
            record['failures'].append(family + '-count-differs-from-declaration')
        if not observed[family]['plans']:
            record['failures'].append(family + '-has-no-native-plan')
    record.update(verdict='FAIL' if record['failures'] else 'PASS', rc=1 if record['failures'] else 0)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--workload', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args(argv)
    if urllib.parse.urlsplit(a.url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        print('refused: the bearer is only sent to a loopback URL')
        return 2
    try:
        record = guard(a.url, a.model, a.workload)
    except Exception as exc:  # noqa: BLE001 - classified HARNESS, never retried
        record = dict(schema=SCHEMA + '/guard', verdict='FAIL', rc=1, failures=['guard-error'],
                      error=type(exc).__name__, classification='HARNESS')
    with a.out.open('x') as f:
        json.dump(record, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({k: record.get(k) for k in ('verdict', 'rc', 'failures', 'declared', 'observed')}, sort_keys=True))
    return record['rc']


if __name__ == '__main__':
    raise SystemExit(main())
