#!/usr/bin/env python3
"""Hub-side staging check for q18a-inwait: tools/step_b_staging_check.py's check, unchanged, over the inwait runner and records
(c1's tools/alpha_next_c1_staging_check.py with its own schema name). The frozen runner's own staging require blocks are taken from
its text by AST and evaluated over the exact records to be delivered, with the same two negative controls. No GPU, no box action,
nothing is written but --out.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_b_staging_check as step_b  # noqa: E402

SCHEMA = 'inwait-staging-check/1'


def check(runner, selfrun, engine):
    result = step_b.check(runner, selfrun, engine)
    assert result['schema'] == 'step-b-staging-check/1', result['schema']
    result.update(schema=SCHEMA, derived_from='tools/step_b_staging_check.py')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--selfrun', type=Path, required=True)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = check(a.runner, a.selfrun, a.engine)
    with a.out.open('x') as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=1) + '\n')
    print(json.dumps(dict(verdict=result['verdict'], failing={k: v['failing'] for k, v in result['requires'].items()},
                          controls=result['negative_controls_refused']), sort_keys=True))
    raise SystemExit(0 if result['verdict'] == 'PASS' else 1)
