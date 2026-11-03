#!/usr/bin/env python3
"""Write tools/alpha_next_c1b_queue_pins.json from the gate chain's build output (OUT); --check recomputes from OUT and compares.

tools/alpha_next_c1_queue_pins_write.py for alpha-next-c1b, imported and re-bound, never copy-edited: its functions are exec'd from
their own source in a copy of its module globals with the file, schema and note names re-bound to c1b's, and the freeze record
swapped exactly once (alpha-next-c1-freeze.json / alpha-next-c1-freeze/1 -> alpha-next-c1b-freeze.json / alpha-next-c1b-freeze/1).
Every rule (cross-checks, outside-the-build proof, placeholder fill, --supersede with history, never git commit) is c1's.
usage: alpha_next_c1b_queue_pins_write.py --out OUT [--check] [--supersede REASON]
"""
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import alpha_next_c1_queue_pins_write as c1  # noqa: E402

PINS = ROOT / 'tools/alpha_next_c1b_queue_pins.json'
REL = 'tools/alpha_next_c1b_queue_pins.json'
SCHEMA = 'alpha-next-c1b-queue-pins/1'
NOTE = c1.NOTE.replace('tools/alpha_next_c1_queue_pins_write.py', 'tools/alpha_next_c1b_queue_pins_write.py') \
    .replace('tools/alpha_next_c1_delivery.py', 'the alpha-next-c1b delivery')
FREEZE_SWAPS = (("(out / 'alpha-next-c1-freeze.json')", "(out / 'alpha-next-c1b-freeze.json')"),
                ("'freeze-proof.json', 'alpha-next-c1-freeze.json')", "'freeze-proof.json', 'alpha-next-c1b-freeze.json')"),
                ("'alpha-next-c1-freeze/1'", "'alpha-next-c1b-freeze/1'"),
                ("'alpha-next-c1 freeze not PASS'", "'alpha-next-c1b freeze not PASS'"))
NS = dict(c1.__dict__, PINS=PINS, REL=REL, SCHEMA=SCHEMA, NOTE=NOTE, __name__='alpha_next_c1b_queue_pins_write')
for _name in ('from_out', 'outside_build', 'record', 'placeholders', 'check', 'main'):
    _text = inspect.getsource(getattr(c1, _name))
    if _name == 'from_out':
        for old, new in FREEZE_SWAPS:
            assert _text.count(old) == 1, ('c1 writer changed', old)
            _text = _text.replace(old, new)
    exec(compile(_text, 'alpha_next_c1b_queue_pins_write:' + _name, 'exec'), NS)
from_out, outside_build, record, placeholders, check, main = (NS[n] for n in ('from_out', 'outside_build', 'record',
                                                                             'placeholders', 'check', 'main'))
PLACEHOLDER, KEYS = c1.PLACEHOLDER, c1.KEYS

if __name__ == '__main__':
    raise SystemExit(main())
