#!/usr/bin/env python3
"""q18a-compose-5 box arms and grade: compose-4's box over this table (harness/compose5_arms.py). Discovery mode.

Every arm kind compose-5 runs, compose-4's box already runs: the live arm for sm, fon and foff, the replay arm for cmp and ny, the one-stream arm. Only the table and the names differ, so the table is bound in here
and nothing that ran is edited. The rule is compose5_rule.py on the fetched record; this grade only says the record is whole.
"""
from contextlib import contextmanager
from functools import wraps

import compose5_arms as T
import e3_compose4_box as _box

ARM_SCHEMA = 'compose5-arm/1'
GRADE_SCHEMA = 'compose5-grade/1'
SCOPE = 'the yield priced under the closed-loop replay and in a frozen pair, on one engine: crash, OOM, stock prefill and the price read by compose5_rule.py; not a release assessment and not a numerics result'
NOT_COVERED = _box.NOT_COVERED
replay = _box.replay
_report = _box.report


def report(result):
    return _report(result).replace('q18a-compose-4', 'q18a-compose-5', 1).replace('compose4_rule.py', 'compose5_rule.py')


def __getattr__(name):
    """compose-4's box by its names, for a caller that only reads them."""
    return getattr(_box, name)


@contextmanager
def _bound():
    """This table and these names in compose-4's box for one call, and back: a suite that runs both boxes in one process sees each with its own."""
    names = dict(T=T, ARM_SCHEMA=ARM_SCHEMA, GRADE_SCHEMA=GRADE_SCHEMA, SCOPE=SCOPE, report=report)
    saved = {name: getattr(_box, name) for name in names}
    for name, value in names.items():
        setattr(_box, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(_box, name, value)


def _under_this_table(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        with _bound():
            return fn(*args, **kwargs)
    return call


grade = _under_this_table(_box.grade)
smoke_check = _under_this_table(_box.smoke_check)
run_arm = _under_this_table(_box.run_arm)
replay_arm = _under_this_table(_box.replay_arm)
one_stream_arm = _under_this_table(_box.one_stream_arm)
main = _under_this_table(_box.main)

if __name__ == '__main__':
    raise SystemExit(main())
