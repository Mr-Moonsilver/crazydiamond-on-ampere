#!/usr/bin/env python3
"""C0 bit-identity receipt, CPU: switch on against switch off over the real driver and bridges.

Runs each case on the frozen driver, on the staged driver and on the staged driver with the per-arm switch off, then compares:
  ledgers       each rank's device-affecting work in order (kind, transaction, stage, op_seq, layer, seq, PP identity, tokens);
  commands      each rank's device commands as pickled bytes, unwrapped from the stage record;
  state         persistent state: cache calls, KV rows, allocations, commits, frontier, sequences, leases, ticket fates;
  frozen stream the unselected run's ring stream, byte for byte against the frozen driver's.
A case is adversarial when ranks read the ring at random rates and one stage lags.

Scope: device operations are the leaves declared in harness/stage_fixture_leaves.json, so this proves the same work in the
same order with the same arguments on each rank. Tensor values come from the box arms' prefill and decode comparison, not from
here. usage: c0_bit_identity.py [--out FILE]. Exit 0 only on PASS. No GPU, no box action.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]
from stage_net import RingStack  # noqa: E402
from stage_wire import StageScoped  # noqa: E402
from wire_v1 import SegmentCommand  # noqa: E402

SHAPES = ((13, 4), (5, 4), (9, 4), (17, 4), (6, 3), (33, 4))
SEEDS = 16


def sha(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def commands(network):
    """Each rank's device commands from the ring, in ring order, as pickled bytes."""
    out = {rank: [] for rank in (0, 1, 2, 3)}
    for envelope in network.sent_envelopes:
        args = envelope[1]
        message = args[0] if args else None
        if type(message) is StageScoped:
            message = message.message
        if type(message) is SegmentCommand:
            # The frozen stream broadcasts to all ranks; the command itself names who executes it.
            for rank in message.ranks:
                out[rank].append(pickle.dumps(message))
    return out


def replayed_once(rows):
    """A retried command is the same command: compare distinct commands in first-seen order."""
    return list(dict.fromkeys(rows))


def adversarial(network, rng):
    network.complete_everything()
    behind = rng.choice(((), (0, 1), (2, 3), (1,), (3,)))
    for rank in (0, 1, 2, 3):
        if rank in behind and rng.random() < 0.8:
            continue
        for _ in range(rng.randint(1, 4)):
            network.step(rank)
    network.settle()
    return network.driver.advance()


def run(selected, tokens, chunk, seed):
    network = RingStack(staged=selected is not None, tokens=tokens, chunk=chunk)
    if selected is False:
        network.driver.stage_selected = False
    tick = None
    if seed is not None:
        rng = random.Random(seed)
        tick = lambda: adversarial(network, rng)  # noqa: E731
    try:
        network.run_request(tick=tick)
        if network.worker_errors:
            raise AssertionError(network.worker_errors)
        return dict(ledgers=network.ledgers, commands={r: replayed_once(c) for r, c in commands(network).items()},
                    state=network.persistent_state(), stream=pickle.dumps(network.sent_envelopes),
                    commits=network.controller.commits, chunks=getattr(network.driver, 'stage_counters', {}).get('chunks'))
    finally:
        network.close_network()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out')
    options = parser.parse_args()
    cases, faults = [], []
    for tokens, chunk in SHAPES:
        for seed in (None,) + tuple(range(SEEDS)):
            name = dict(tokens=tokens, chunk=chunk, seed=seed)
            off = run(None, tokens, chunk, None)
            on = run(True, tokens, chunk, seed)
            row = dict(name, commits=on['commits'], staged_chunks=on['chunks'],
                       ledger=sha(off['ledgers']), state=sha(off['state']))
            if on['chunks'] != on['commits'] or on['commits'] == 0:
                faults.append(('staged path not exercised', name))
            for key in ('ledgers', 'commands', 'state'):
                if off[key] != on[key]:
                    faults.append((key + ' differ', name))
            if seed is None:
                unselected = run(False, tokens, chunk, None)
                if unselected['stream'] != off['stream']:
                    faults.append(('unselected stream differs from the frozen stream', name))
                if unselected['ledgers'] != off['ledgers'] or unselected['state'] != off['state']:
                    faults.append(('unselected ledger or state differs', name))
                row['frozen_stream'] = hashlib.sha256(off['stream']).hexdigest()
            cases.append(row)
    receipt = dict(schema='c0-bit-identity/1', verdict='FAIL' if faults else 'PASS', faults=[list(f) for f in faults[:20]],
                   fault_count=len(faults), cases=len(cases), seeds=SEEDS, shapes=[list(s) for s in SHAPES], rows=cases,
                   scope='CPU: same device work, order and arguments per rank; tensor values are for the box arms')
    text = json.dumps(receipt, indent=1, sort_keys=True) + '\n'
    if options.out:
        Path(options.out).write_text(text)
    print(json.dumps({k: receipt[k] for k in ('verdict', 'cases', 'fault_count', 'faults')})[:1500])
    return 0 if not faults else 1


if __name__ == '__main__':
    sys.exit(main())
