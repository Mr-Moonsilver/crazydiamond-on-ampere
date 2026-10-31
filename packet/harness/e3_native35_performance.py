#!/usr/bin/env python3
"""Native35 P: one fixed native W2 schedule per invocation, explicit round salt namespace.

The frozen native W2 collector runs unchanged: ``e3_native_performance.run``
(and through it ``prefill``, ``native_window`` and ``grade``) over the frozen
q18a W2 workload, which still declares ``repeats=1``. P runs that schedule
twice, as two separate invocations with ``--round 0`` and ``--round 1``.

The only per-round difference is the cache-salt namespace. ``run`` derives
every cache salt as ``_cache_salt(client.run_id, lane, index)``; two
invocations with the same authenticated RUN_ID would reuse round 0's salts and
turn round 1's "fresh" 200k prompt into a prefix-cache hit. ``RoundSaltClient``
therefore exposes ``run_id = salt_namespace(RUN_ID, round)`` to ``run`` and
forwards everything else to the real ``NativeRuntimeClient``, whose own bound
methods send every session command with the real RUN_ID, client id and
command sequence. Salts stay identical for fresh -> append inside one lane and
round (the append is the same conversation) and differ across lanes and
rounds. No retry, no picking: each round's artifact is retained as written.
"""
import argparse
import json
import os
from pathlib import Path

import e3_native_performance as perf
import e3_overlap_performance as shared
from e3_native_runtime_client import NativeRuntimeClient
from e3_native_qualification_driver import exception_details

ROUNDS = 2
SCHEMA = 'native35-p-w2-round/1'


def salt_namespace(run_id, round_index):
    if type(run_id) is not str or not run_id or type(round_index) is not int:
        raise ValueError('native35-round-identity-invalid')
    if not 0 <= round_index < ROUNDS:
        raise ValueError('native35-round-index-outside-declared-schedule')
    return '%s|native35-p-w2-round-%d' % (run_id, round_index)


class RoundSaltClient:
    """The real native client; only the salt namespace attribute differs."""

    def __init__(self, client, namespace):
        object.__setattr__(self, '_client', client)
        object.__setattr__(self, 'run_id', namespace)

    def __getattr__(self, name):
        return getattr(self._client, name)

    def __setattr__(self, name, value):
        raise AttributeError('round-salt-client-is-read-only')


def salt_audit(cfg, namespace):
    """Every cache salt ``e3_native_performance.run`` derives in one round."""
    prefill = {lane: shared._cache_salt(namespace, lane, 0) for lane in shared.LANES}
    standing = [shared._cache_salt(namespace, 'overlap', index)
                for index in range(cfg['decode']['streams'])]
    return {
        'namespace': namespace,
        'derivation': "e3_overlap_performance._cache_salt(namespace, lane, index)",
        'prefill_salt_by_lane': prefill,
        'fresh_and_append_share_the_lane_salt': True,
        'overlap_standing_salts': standing,
        # Inherited from the frozen run(): standing stream 0 and the overlap
        # prefill both use index 0 of the overlap lane. Their prompts come from
        # different seeds (decode seed vs prefill seed), so no full prefix
        # block is shared; the native useful-token count shows any reuse.
        'inherited_overlap_index0_coincidence': standing[0] == prefill['overlap'],
        'distinct_prefill_lane_salts': len(set(prefill.values())) == len(prefill),
    }


def prompt_identity(cfg):
    prompts = shared._prefill_prompts(cfg)
    return {family: {'prompt_sha256': prompts[family + '_sha256'],
                     'prompt_tokens_declared': prompts[family + '_tokens']}
            for family in ('fresh', 'append')} | {
        'shared_prefix_tokens': prompts['shared_prefix_tokens']}


def round_record(a, cfg, client):
    record = {'schema': SCHEMA, 'round': a.round, 'rounds_declared': ROUNDS,
              'session_run_id': a.run_id, 'engine_epoch_declared': a.epoch,
              'salt_namespace': salt_namespace(a.run_id, a.round),
              'retry_selected': False, 'picked': False}
    if cfg is not None:
        record['salts'] = salt_audit(cfg, record['salt_namespace'])
        record['prompts'] = prompt_identity(cfg)
        record['workload_manifest_sha256'] = cfg['manifest_sha256']
    if client is not None:
        record['session_client'] = {'run_id': client.run_id, 'client_id': client.client_id,
                                    'commands_sent': client.sequence,
                                    'replies_retained': len(client.history)}
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--url', required=True); p.add_argument('--model', required=True)
    p.add_argument('--run-id', required=True); p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--round', type=int, required=True, choices=range(ROUNDS))
    p.add_argument('--workload', required=True); p.add_argument('--w1', required=True)
    p.add_argument('--gates', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        return 2
    result, cfg, client = {}, None, None
    try:
        gates = json.loads(Path(a.gates).read_text())
        if not perf.gates_allow_run(gates, a.gates):
            raise ValueError('same-queue-qualification-gates-not-pass')
        package = shared._package_gate(os.environ)
        if package.get('ok') is not True:
            raise ValueError('reviewed-package-readiness-not-pass')
        data = json.loads(Path(a.workload).read_text())
        problems, cfg = shared.validate_manifest(data)
        if problems or cfg['repeats'] != 1:
            cfg = None
            raise ValueError('frozen-native-workload-invalid')
        effective_read_s = perf.first_advance_read_timeout(cfg)
        cfg = dict(cfg, timeouts=dict(cfg['timeouts'], read_s=effective_read_s))
        w1 = json.loads(Path(a.w1).read_text())
        client = NativeRuntimeClient(a.url, a.run_id, epoch=a.epoch)
        result = perf.run(cfg, RoundSaltClient(client, salt_namespace(a.run_id, a.round)),
                          a.url, a.model, w1)
        result['first_advance_read_timeout_s'] = effective_read_s
        # run() labelled its record with the salt namespace; restore the
        # authenticated session identity and keep the namespace beside it.
        result['run_id'] = a.run_id
    except Exception as exc:
        result = {'schema': 's5-native-performance/1',
                  'grade': {'verdict': 'INCONCLUSIVE', 'missing': ['run-refused'], 'failures': []},
                  'error': exception_details(exc)}
    result['native35_round'] = round_record(a, cfg, client)
    with open(a.out, 'x') as handle:
        json.dump(result, handle, sort_keys=True)
    verdict = result['grade']['verdict']
    print('NATIVE35_W2_ROUND %d %s' % (a.round, verdict))
    return {'PASS': 0, 'FAIL': 1}.get(verdict, 2)


if __name__ == '__main__':
    raise SystemExit(main())
