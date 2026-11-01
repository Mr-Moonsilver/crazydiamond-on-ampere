#!/usr/bin/env python3
"""Pinned-image self-run seam: real engine status via a pipe, no network/GPU.

Packaging starts --serve in the engine container (stdin/stdout pipes). The
client runs OUTSIDE that container. Mount a declared alternate boot_id file
onto /proc/sys/kernel/random/boot_id in that container if the natural views
are equal. No Python clock/identity function is replaced. Both timens offsets
must be observed zero; retain both view() records and actual bracket proof.
The model/transport geometry leaves are the native43 connected CPU fixture.
"""
import json
import sys
import time
from native43_boundary_clock import view


def engine_packet():
    from test_e3_r4_worker_bridge import Stack
    stack = Stack()
    driver = stack.driver
    driver.engine.scheduler = stack.scheduler
    packet = driver._status_record(time.monotonic_ns())
    return dict(status=packet, engine_view=view(),
                fixture='real-driver-and-four-bridges; fake device/model/PP transport')


def main():
    if sys.argv[1:] != ['--serve']:
        raise SystemExit('usage: native43_clock_probe.py --serve')
    for line in sys.stdin:
        if line.strip() != 'status':
            raise SystemExit('unknown probe command')
        print(json.dumps(engine_packet(), sort_keys=True), flush=True)

if __name__ == '__main__':
    main()
