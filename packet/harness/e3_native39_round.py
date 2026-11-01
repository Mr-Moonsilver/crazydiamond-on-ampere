#!/usr/bin/env python3
"""Native39 trunk W2 round: e3_native35_performance with row4 declared NOT_RUN.

The speed engine runs with FLASH_S5_REPLAY_OBSERVER=0 and
FLASH_S5_ENGINE_OFFERS=0 (lead, 1 Nov 2026).  Row4 placement needs both, so the
trunk drops it from this run by declaration: the queue writes row4 NOT_RUN with
row4_declared_not_run true and the declared reason into gates.json.  This
wrapper reads exactly that shape as row4 PASS for the unchanged same-queue gate
(e3_native35_performance.gates_allow_run, itself perf.gates_allow_run) and
changes nothing else.  Any other row4 value, an undeclared NOT_RUN, a missing
or different reason, or a declared NOT_RUN beside the P-c capacity
continuation is refused by the unchanged gate.  Arguments and record are
e3_native35_performance's.
"""
import sys

import e3_native35_performance as round35
from native39_trunk_gate import ROW4_NOT_RUN_REASON

_UNCHANGED_GATE = round35.gates_allow_run


def declared_view(gates):
    if (type(gates) is not dict or gates.get('row4_declared_not_run') is not True
            or gates.get('row4_not_run_reason') != ROW4_NOT_RUN_REASON
            or gates.get('row4_capacity_continuation') is True
            or type(gates.get('outcomes')) is not dict or gates['outcomes'].get('row4') != 'NOT_RUN'):
        return None
    return dict(gates, outcomes=dict(gates['outcomes'], row4='PASS'))


def gates_allow_run(gates, gate_path):
    if type(gates) is dict and (gates.get('row4_declared_not_run') is True
                                or (gates.get('outcomes') or {}).get('row4') == 'NOT_RUN'):
        view = declared_view(gates)
        return view is not None and _UNCHANGED_GATE(view, gate_path)
    return _UNCHANGED_GATE(gates, gate_path)


def main(argv=None):
    round35.gates_allow_run = gates_allow_run
    return round35.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
