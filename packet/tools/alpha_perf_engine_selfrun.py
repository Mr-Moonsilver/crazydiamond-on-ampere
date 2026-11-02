#!/usr/bin/env python3
"""Alpha-perf engine self-run: both parents' exact connected self-runs over the one frozen bundle. CPU only, no network.

Two containers, the composed engine in each, all results kept under OUT:
  * progress/: tools/native41_progress_engine_selfrun.py run() unchanged (held-f): held-e's 18 connected auto-arm and
    receipt cases and the eight connected native progress cases, burst included, plus its suites;
  * dense/: tools/native43_selfrun.py run() unchanged (the dense line): probe39, probe37, the spec's suites (dense
    boundary, QSA, sparse history, death watch and fatal log, S1, M3), the real-scheduler tick probe, the clock split.
OUT/result.json is the merged record (schema alpha-perf-selfrun/1) the frozen runner reads; verdict PASS only when both
are PASS. A failure keeps both records and is classified HARNESS unless a reader reclassifies it after looking.

Declared limit: burst x dense on one native chunk has no CPU composition here. The progress cases run the real driver,
scheduler, controller and worker bridges with command-level leaves below the QSA layer, and the dense suites run the
generated attention state machine with CUDA leaves. The two meet at the worker's QSA layer call, which neither mode
changes the signature of. The VM run has real graphs and the dense route but no driver, so no burst either: the box is the
first place both are live, and the delivery says so.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native41_progress_engine_selfrun as progress  # noqa: E402
import native43_selfrun as dense  # noqa: E402

SCHEMA = 'alpha-perf-selfrun/1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(spec, tree, bundle, out, name):
    spec, bundle, out = Path(spec).resolve(), Path(bundle).resolve(), Path(out).resolve()
    out.mkdir()
    out.chmod(0o777)
    rc_progress = progress.run(bundle, out / 'progress', name + '-progress')
    rc_dense = dense.run(spec, tree, bundle, out / 'dense', name + '-dense')
    p = json.loads((out / 'progress/result.json').read_text())
    d = json.loads((out / 'dense/result.json').read_text())
    ok = rc_progress == 0 and rc_dense == 0 and p['verdict'] == d['verdict'] == 'PASS'
    pieces = p['suites'], d['suites']
    # The progress record counts in 'tests'; the dense one keeps unittest's own line in 'ran' ("Ran 671 tests in 91.577s").
    ran = re.fullmatch(r'Ran (\d+) tests? in [0-9.]+s', pieces[1].get('ran') or '')
    tests = [pieces[0].get('tests') or 0, int(ran.group(1)) if ran else 0]
    ok = ok and ran is not None
    skips = [(x.get('skips') if x.get('skips') is not None else x.get('skipped')) or 0 for x in pieces]
    connected = {k: v for k, v in p['cases'].items()}
    record = dict(schema=SCHEMA, verdict='PASS' if ok else 'FAIL', classification=None if ok else 'HARNESS',
                  negative_control=False, container_rc=max(p['container_rc'], d['container_rc']),
                  manifest=p['manifest'], bundle_manifest=p['manifest'], bundle=p['bundle'],
                  overlay_manifest=p['overlay_manifest'], image=p['image'], network=p['network'],
                  gpu_device_requests=max(p['gpu_device_requests'], d['clock_split'].get('gpu_device_requests') or 0),
                  gpu_execution=False, box_action=False, candidate=d.get('label') or spec.name,
                  tests=sum(tests) + len(connected), skips=sum(skips), connected_checks=sum(v['checks'] for v in connected.values()),
                  suites=dict(ok=pieces[0].get('ok') is True and pieces[1].get('ok') is True, skips=sum(skips),
                              tests=sum(tests), progress=pieces[0], dense=pieces[1]),
                  cases=connected, probe39=d['probe39'], probe37=d['probe37'], py_compile=d['py_compile'],
                  scheduler_tick_probe=d.get('scheduler_tick_probe'), clock_split=d['clock_split'],
                  progress_record_sha256=sha(out / 'progress/result.json'), dense_record_sha256=sha(out / 'dense/result.json'),
                  spec=str(spec), spec_sha256=sha(spec),
                  declared_limit='burst x dense on one native chunk has no CPU composition; see the tool docstring')
    (out / 'result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(verdict=record['verdict'], tests=record['tests'], skips=record['skips'], cases=len(connected),
                          result_sha256=sha(out / 'result.json')), sort_keys=True))
    return 0 if ok else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    a = p.parse_args()
    raise SystemExit(run(a.spec, a.tree, a.bundle, a.out, a.name))
