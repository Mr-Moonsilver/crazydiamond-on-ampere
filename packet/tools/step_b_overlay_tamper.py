#!/usr/bin/env python3
"""Negative control for the Step B freeze's overlay check: tools/step_a_overlay_tamper.py's cases, unchanged, over step_b_freeze.

A one-byte change in an overlay file must fail overlay_level. The Step A tool reads `step_a_freeze` for overlay_level, DECLARATION and
RUNTIME; here that name resolves to tools/step_b_freeze.py for this process only, so the cases are the same text against Step B's
check, candidate and parent.

Usage: step_b_overlay_tamper.py --parent PARENT/native30_overlay --candidate BUNDLE/native30_overlay --out JSON
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_b_freeze  # noqa: E402

real_step_a_freeze = sys.modules.get('step_a_freeze')
sys.modules['step_a_freeze'] = step_b_freeze
import step_a_overlay_tamper as tamper  # noqa: E402
assert tamper.freeze is step_b_freeze
sys.modules['step_a_freeze'] = real_step_a_freeze


def main():
    rc = tamper.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    record = json.loads(out.read_text())
    record.update(schema='step-b-overlay-tamper/1', derived_from='tools/step_a_overlay_tamper.py')
    out.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
