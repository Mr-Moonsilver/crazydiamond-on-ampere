#!/usr/bin/env python3
"""Bounded completion wait for the lifetime phase (v3 gate): after the leak test has ended, wait until the pressure
progress file (PRESSURE_READY of pressure2.py) shows, on EVERY stream, completed_requests >= MIN_COMPLETED and a
last token time after LEAK_END, so the coverage rule "every stream completes a request" and the bracket rule "pressure
tokens after the leak interval" can be satisfied before the pressure is asked to stop. Writes a small JSON with the
outcome. Exit 0 satisfied, 2 bound reached (the gate then stops the pressure anyway and the lifetime grader reports
INCONCLUSIVE faithfully), 1 unreadable progress after the bound.
usage: pressure_wait.py READY.json LEAK_END_WALL BOUND_S OUT.json [MIN_COMPLETED=1]
"""
import json, sys, time


def satisfied(d, leak_end, min_completed=1):
    c, last = d.get('completed_requests', []), d.get('last_tokens', [])
    n = d.get('issued', [])
    if not c or len(c) != len(last) or len(c) != len(n):
        return False
    return all(isinstance(x, int) and x >= min_completed for x in c) and \
        all(isinstance(t, (int, float)) and t > leak_end for t in last)


def wait(path, leak_end, bound_s, min_completed=1, sleep=lambda s: time.sleep(s), clock=time.monotonic):
    t0 = clock(); end = t0 + bound_s; last = None; readable = False
    while True:
        try:
            last = json.load(open(path)); readable = True
            if satisfied(last, leak_end, min_completed):
                return 0, {'satisfied': True, 'waited_s': clock() - t0, 'completed_requests': last['completed_requests'],
                           'last_tokens_after_leak_end': [t - leak_end for t in last['last_tokens']], 'max_tokens': last.get('max_tokens')}
        except (OSError, ValueError, KeyError):
            pass
        if clock() >= end:
            break
        sleep(min(2.0, max(0.0, end - clock())))
    if not readable or last is None:
        return 1, {'satisfied': False, 'waited_s': clock() - t0, 'error': 'progress file unreadable'}
    return 2, {'satisfied': False, 'waited_s': clock() - t0, 'completed_requests': last.get('completed_requests'),
               'last_tokens': last.get('last_tokens'), 'leak_end': leak_end, 'max_tokens': last.get('max_tokens')}


if __name__ == '__main__':
    path, leak_end, bound, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    minc = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    rc, info = wait(path, leak_end, bound, minc)
    info['rc'] = rc
    json.dump(info, open(out, 'x'), indent=1)
    print(json.dumps(info))
    sys.exit(rc)
