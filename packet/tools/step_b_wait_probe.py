#!/usr/bin/env python3
"""Does a tensor form wait for the stream when work is queued? Step B's hidden-wait sweep.

Runs inside a flashnext-* container on VM <vm-id>'s single 3090, under the shared GPU lock. For each form: queue kernels on
the current stream, call the form and time the host, then time torch.cuda.synchronize() for what was still queued.

A form that does not wait leaves nearly the whole queue to the synchronize. A form that waits eats it. Two controls bound
the reading: `torch.tensor(values, device=)` and a blocking pageable `copy_` are the forms the brief measured as waiting,
and a small pageable `copy_(non_blocking=True)` is the one it measured as not waiting. If the controls do not come out
that way the row is INC, whatever the forms under test showed.

Forms under test are the exact statements on paths reachable with queued work:
  lane_private.py _populate_uid_once: uid.fill_(-2), then uid[0].copy_(slots[req_index]), same device, no non_blocking
  lane_private.py _initialize_lock_once: private.zero_()
"""
import json
import sys
import time

import torch

DEVICE = 'cuda:0'
ROUNDS = 7
QUEUE_SIZE = 3072
QUEUE_KERNELS = 12
# Below this share of the queue left for synchronize, the form ate the queue.
WAITS_BELOW = 0.2
# At or above this share, the form left the queue alone.
FREE_ABOVE = 0.8


def queue(a, b):
    out = None
    for _ in range(QUEUE_KERNELS):
        out = torch.matmul(a, b)
    return out


def main():
    torch.cuda.set_per_process_memory_fraction(1.0 / 12.0, 0)  # about 2 GiB of the 24 GiB card
    a = torch.randn(QUEUE_SIZE, QUEUE_SIZE, device=DEVICE)
    b = torch.randn(QUEUE_SIZE, QUEUE_SIZE, device=DEVICE)
    torch.cuda.synchronize()

    uid = torch.zeros(8, dtype=torch.int32, device=DEVICE)
    slots = torch.arange(8, dtype=torch.int32, device=DEVICE)
    lock = torch.zeros(4096, dtype=torch.int32, device=DEVICE)
    small = torch.zeros(4, dtype=torch.int64, device=DEVICE)
    pageable = torch.arange(4, dtype=torch.int64)
    torch.cuda.synchronize()

    def fill():
        uid.fill_(-2)

    def seed():
        uid[0].copy_(slots[3])

    def both():
        uid.fill_(-2)
        uid[0].copy_(slots[3])

    def zero():
        lock.zero_()

    def tensor_on_device():
        return torch.tensor([1, 2, 3, 4], device=DEVICE)

    def blocking_copy():
        small.copy_(pageable)

    def nonblocking_copy():
        small.copy_(pageable, non_blocking=True)

    forms = [
        ('none', 'baseline', lambda: None),
        ('uid.fill_', 'under-test', fill),
        ('uid[0].copy_(slots[i])', 'under-test', seed),
        ('_populate_uid_once body', 'under-test', both),
        ('private.zero_', 'under-test', zero),
        ('torch.tensor(device=)', 'control-waits', tensor_on_device),
        ('blocking pageable copy_', 'control-waits', blocking_copy),
        ('pageable copy_(non_blocking=True)', 'control-free', nonblocking_copy),
    ]
    rows = []
    for name, role, form in forms:
        samples = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter_ns()
            queue(a, b)
            t1 = time.perf_counter_ns()
            form()
            t2 = time.perf_counter_ns()
            torch.cuda.synchronize()
            t3 = time.perf_counter_ns()
            samples.append(dict(queue_ms=(t1 - t0) / 1e6, form_ms=(t2 - t1) / 1e6, left_ms=(t3 - t2) / 1e6))
        samples.sort(key=lambda row: row['form_ms'])
        middle = samples[len(samples) // 2]
        rows.append(dict(form=name, role=role, form_ms_median=round(middle['form_ms'], 3),
                         form_ms_max=round(samples[-1]['form_ms'], 3), left_ms_at_median=round(middle['left_ms'], 3),
                         enqueue_ms_at_median=round(middle['queue_ms'], 3)))
    base = rows[0]['left_ms_at_median']
    for row in rows:
        share = row['left_ms_at_median'] / base if base > 0 else 0.0
        row['share_of_queue_left'] = round(share, 3)
        row['reading'] = 'waits' if share < WAITS_BELOW else 'free' if share >= FREE_ABOVE else 'partial'
    controls_hold = base > 20.0 and all(
        row['reading'] == {'control-waits': 'waits', 'control-free': 'free'}[row['role']]
        for row in rows if row['role'].startswith('control'))
    tested = [row for row in rows if row['role'] == 'under-test']
    verdict = 'INC' if not controls_hold else 'PASS' if all(row['reading'] == 'free' for row in tested) else 'FAIL'
    json.dump(dict(schema='step-b-wait-probe/1', verdict=verdict, controls_hold=controls_hold, device=torch.cuda.get_device_name(0),
                   torch=torch.__version__, rounds=ROUNDS, queue=dict(size=QUEUE_SIZE, kernels=QUEUE_KERNELS, left_ms=base),
                   thresholds=dict(waits_below=WAITS_BELOW, free_above=FREE_ABOVE), rows=rows), sys.stdout, indent=1)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
