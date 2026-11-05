#!/usr/bin/env python3
"""q18a-sm-split-<N> startup grade, decode-phase build-3: e3_native_startup_grade.py's text (never edited) with its two literal 44s bound to the launch's declared decode SM count,
S5_REQUESTED_DECODE_SMS (exported by the queue settings, read once here). The rule stays exact: requested, measured and membership all equal the declared count, so a driver-rounded split
still fails at startup. Every swap applies exactly once.
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inwait_arms import DEFAULT_DECODE_SMS, legal_decode_sms  # noqa: E402

REQUESTED = int(legal_decode_sms(os.environ.get('S5_REQUESTED_DECODE_SMS', DEFAULT_DECODE_SMS)))
SWAPS = (("split.get('requested_decode_sms')==44 and split.get('decode_sms')==44",
          "split.get('requested_decode_sms')==REQUESTED and split.get('decode_sms')==REQUESTED"),
         ("membership.get('decode_seen')==44", "membership.get('decode_seen')==REQUESTED"))


def namespace():
    text = (Path(__file__).resolve().with_name('e3_native_startup_grade.py')).read_text()
    for old, new in SWAPS:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    ns = dict(__name__='e3_dp3_startup_grade_inner', REQUESTED=REQUESTED)
    exec(compile(text, 'e3_dp3_startup_grade:e3_native_startup_grade', 'exec'), ns)
    return ns


if __name__ == '__main__':
    raise SystemExit(namespace()['main'](sys.argv[1:]))
