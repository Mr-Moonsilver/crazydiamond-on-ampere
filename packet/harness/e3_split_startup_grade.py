#!/usr/bin/env python3
"""parity-n3s startup grade: e3_native_startup_grade.py's text (never edited) with its two literal 44s bound to this table's decode SM
count, compose5_arms.DECODE_SMS (the decode-phase lineage's e3_dp3_startup_grade.py, by the same swaps). The rule stays exact:
requested, measured and membership all equal the declared count on every rank, so a driver-rounded split still fails at startup.
The launch's S5_REQUESTED_DECODE_SMS (the runner's, exported to the queue) must be that count too, or this refuses before grading.
Every swap applies exactly once.
"""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compose5_arms as table  # noqa: E402

REQUESTED = int(table.legal_decode_sms(table.DECODE_SMS))
SWAPS = (("split.get('requested_decode_sms')==44 and split.get('decode_sms')==44",
          "split.get('requested_decode_sms')==REQUESTED and split.get('decode_sms')==REQUESTED"),
         ("membership.get('decode_seen')==44", "membership.get('decode_seen')==REQUESTED"))


def launch_matches(env=None):
    """The launch's declared count (S5_REQUESTED_DECODE_SMS, when the queue exports it) against this table's."""
    env = os.environ if env is None else env
    value = env.get('S5_REQUESTED_DECODE_SMS')
    return value is None or value == str(REQUESTED)


def namespace():
    text = HERE.joinpath('e3_native_startup_grade.py').read_text()
    for old, new in SWAPS:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    ns = dict(__name__='e3_split_startup_grade_inner', __file__=str(HERE / 'e3_native_startup_grade.py'), REQUESTED=REQUESTED)
    exec(compile(text, 'e3_split_startup_grade:e3_native_startup_grade', 'exec'), ns)
    return ns


def main(argv=None):
    if not launch_matches():
        print('startup grade refused: S5_REQUESTED_DECODE_SMS=%r, the table declares %d' % (os.environ.get('S5_REQUESTED_DECODE_SMS'), REQUESTED),
              file=sys.stderr)
        return 2
    return namespace()['main'](sys.argv[1:] if argv is None else argv)


if __name__ == '__main__':
    raise SystemExit(main())
