#!/usr/bin/env python3
"""Mutants for the C1 rebind repair and its overlap observation.

Each mutant: a new directory under ``--work`` (must not exist; kept), holding copies of
``harness/`` and ``patches/s5_engine/``; one or more exact text mutations (each must match
exactly once); then ``python3 -m unittest <modules>`` in the copied harness.  CAUGHT when the
run fails (rc != 0).  A ``baseline`` copy with no mutation runs first and must pass.
Prints and writes a JSON receipt (``--work``/receipt.json).

usage: c1_rebind_mutants.py --work NEW_DIR [--only NAME ...]
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('test_c1_rebind_witness', 'test_c1_rebind_on')
SO = 'patches/s5_engine/stage_overlap.py'
GATE_OLD = "        if stage == 0 and frame.stage[0]['phase'] == 'bind':\n"
GATE_SKIP_DEFERRED = ("        if stage == 0 and frame.stage[0]['phase'] == 'bind'"
                      " and not getattr(frame, 'deferred_bind', False):\n")

MUTANTS = {
    'baseline': [],
    # The successor's PP_SEND does not wait for (and never makes) its own binding.
    'M1-no-pp-send-wait': [
        (SO, "        return (cursor['_retry'] is None and cursor['index'] < len(cursor['plan'])\n",
         "        return False and (cursor['_retry'] is None and cursor['index'] < len(cursor['plan'])\n")],
    # The successor rebinds before n's proof, on a channel without the live-rebind refusal, so n's
    # proof is validated against the current (new) binding.
    'M2-proof-under-current-binding': [
        (SO, GATE_OLD, GATE_SKIP_DEFERRED),
        ('harness/test_e3_r4_worker_bridge.py',
         "        if any(ticket.state != 'retired' for ticket in self.tickets):\n"
         "            raise AssertionError('live PP channel rebound')\n",
         "        if False:\n"
         "            raise AssertionError('live PP channel rebound')\n"),
        ('patches/s5_engine/lane_comm.py',
         "        if self.tickets:\n            raise LaneCommError('cannot rebind a live PP channel')\n",
         "        if False:\n            raise LaneCommError('cannot rebind a live PP channel')\n")],
    # The successor rebinds before n's proof; the channel's refusal is intact.
    'M3-rebind-before-old-proof': [(SO, GATE_OLD, GATE_SKIP_DEFERRED)],
    # The gate opens once any sender's receipt is acked, not every sender's.
    'M4-any-sender-acked': [
        (SO, "                    and set(chunk['_receipt_acked']) == set(self.pp_peers))\n",
         "                    and bool(chunk['_receipt_acked']))\n")],
    # Observation: a completion's upper bound stamped at submit, not at apply.
    'O1-hi-at-submit': [
        (SO, "            if row[3] is None:\n                row[3] = t_apply\n",
         "            if row[3] is None:\n                row[3] = t_submit\n")],
    # Read: compares against the finalize's upper bound instead of its lower bound.
    'O2-read-fin-upper-bound': [
        ('harness/c1_overlap_read.py', "    elif layer_hi < fin_lo:\n",
         "    elif layer_hi < (result['fin_hi_ns'] or 0):\n")],
    # Read: a layer counts once any stage-0 TP rank finished it, not every rank.
    'O3-read-one-tp-rank': [
        ('harness/c1_overlap_read.py',
         "    complete = [max(by_rank.values()) for by_rank in layers.values()\n"
         "                if set(by_rank) == set(stage0) and all(hi is not None for hi in by_rank.values())]\n",
         "    complete = [min(hi for hi in by_rank.values() if hi is not None) for by_rank in layers.values()\n"
         "                if any(hi is not None for hi in by_rank.values())]\n")],
}


def copy_tree(dest):
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(ROOT / 'harness', dest / 'harness', ignore=ignore)
    shutil.copytree(ROOT / 'patches/s5_engine', dest / 'patches/s5_engine', ignore=ignore)


def run(name, mutations, work):
    dest = work / name
    copy_tree(dest)
    for path, old, new in mutations:
        target = dest / path
        text = target.read_text()
        if text.count(old) != 1:
            raise SystemExit('%s: mutation does not match exactly once in %s' % (name, path))
        target.write_text(text.replace(old, new))
    started = time.monotonic()
    proc = subprocess.run([sys.executable, '-m', 'unittest', *MODULES], cwd=dest / 'harness',
                          capture_output=True, text=True, timeout=3600)
    (dest / 'unittest.out').write_text(proc.stdout + proc.stderr)
    failed = sorted(set(re.findall(r'^(?:FAIL|ERROR): (\S+ \([^)]*\))', proc.stderr, re.M)))
    tail = [line for line in proc.stderr.splitlines()[-3:] if line.strip()]
    return dict(name=name, mutations=[[p, len(o)] for p, o, _ in mutations], rc=proc.returncode,
                caught=proc.returncode != 0, failed=failed, summary=tail,
                seconds=round(time.monotonic() - started, 1))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', required=True)
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args(argv)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=False)
    names = args.only or list(MUTANTS)
    if 'baseline' not in names:
        names = ['baseline'] + names
    results = []
    for name in names:
        result = run(name, MUTANTS[name], work)
        results.append(result)
        print(json.dumps({k: result[k] for k in ('name', 'rc', 'caught', 'seconds')}), flush=True)
    base = results[0]
    mutants = results[1:]
    receipt = dict(schema='c1-rebind-mutants/1', modules=list(MODULES), baseline_rc=base['rc'],
                   caught=sum(1 for r in mutants if r['caught']), total=len(mutants),
                   passed=base['rc'] == 0 and all(r['caught'] for r in mutants), results=results)
    (work / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({k: receipt[k] for k in ('baseline_rc', 'caught', 'total', 'passed')}))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
