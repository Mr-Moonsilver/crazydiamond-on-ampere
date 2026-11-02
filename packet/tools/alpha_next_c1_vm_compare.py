#!/usr/bin/env python3
"""alpha-next-c1 VM comparison: tools/alpha_c1_vm_compare.py's compare, unchanged, over the plain and stage records of
tools/alpha_next_c1_vm_input.py. Only the record's schema name differs (alpha-next-c1-vm-compare/1) and it carries derived_from.
PASS needs both runs PASS on the real device, the plain run on the plain path with both switches off, the stage run through the stage
wire with a skip proved on every scoped call and overlap selected by the overlay, and the prefill and cohort tokens equal. Records only.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_c1_vm_compare as base  # noqa: E402

SCHEMA = 'alpha-next-c1-vm-compare/1'


def compare(plain, stage):
    result = base.compare(plain, stage)
    assert result['schema'] == 'alpha-c1-vm-compare/1', result['schema']
    result.update(schema=SCHEMA, derived_from='tools/alpha_c1_vm_compare.py')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--plain', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.plain, a.stage)
    with a.out.open('x') as handle:
        json.dump(result, handle, indent=1, sort_keys=True)
        handle.write('\n')
    print(result['verdict'], result['reasons'])
    return result['rc']


if __name__ == '__main__':
    raise SystemExit(main())
