"""Diagnostic client/EngineCore monotonic bracket proof across proc views.

Boot-id and namespace strings remain provenance, never equality gates. No
conversion of device time or replacement of engine timestamps is allowed.
"""
import os
from pathlib import Path


def view():
    return dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                time_namespace=os.readlink('/proc/self/ns/time'),
                offsets=Path('/proc/self/timens_offsets').read_text().strip())


def prove(snapshot, send_ns, recv_ns):
    stamp = snapshot.get('observed_clock_ns')
    clock = snapshot.get('clock') or {}
    valid = (type(send_ns) is int and type(recv_ns) is int
             and 0 < send_ns <= recv_ns and recv_ns - send_ns <= 10_000_000_000
             and type(stamp) is int and send_ns <= stamp <= recv_ns
             and clock.get('basis') == 'clock-monotonic')
    return dict(proved=valid, basis='engine-sample-inside-client-request-bracket',
                send_ns=send_ns, recv_ns=recv_ns, engine_ns=stamp,
                uncertainty_ns=recv_ns-send_ns,
                engine_clock=clock, client_clock=view(), identity_equality_required=False)
