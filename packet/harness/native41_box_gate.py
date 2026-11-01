"""Native41 auto-arm box queue: budget-derived active cap and the Monday hard end.

The queue is native40's startup, bind and unarmed check, then four plain W2
arms (stock, auto, stock, auto) with their idle/typed-FULL entries, instruments
and ingress receipts, then a typed FULL and the inherited owned cleanup. The
active cap is the sum of the declared phase caps below; the cleanup margin is
the trunk's unchanged CLEANUP_DERIVATION. ``window_ok(time.time())`` is the
single executable entry check used by the draft and the owner runner: the last
arm is HARD_END minus active plus cleanup.
"""
from native35p_gate import ACTIVE_DERIVATION as P_ACTIVE_DERIVATION, CLEANUP_DERIVATION, HARD_END, NOT_BEFORE
from native39_trunk_gate import TRUNK_DERIVATION

# Lead, 1 Nov 2026: last arm before Mon 06:00Z minus budget (HARD_END is native34's 2026-11-02T06:00Z).
ARMS = 4
ARM_TIMEOUT_S = 1500
ARM_KILL_AFTER_S = 30
_P = {name: (seconds, why) for name, seconds, why in P_ACTIVE_DERIVATION}
_T = {name: (seconds, why) for name, seconds, why in TRUNK_DERIVATION}
ACTIVE_DERIVATION = (
    ('restart-to-health',) + _P['restart-to-health'],
    ('speculation-and-sampler',) + _P['speculation-and-sampler'],
    ('identity-start',) + _P['identity-start'],
    ('startup-ledger-headroom-plan',) + _P['startup-ledger-headroom-plan'],
    ('bind-receipt',) + _T['bind-receipt'],
    ('w2-length-guard',) + _T['w2-length-guard'],
    ('unarmed-stock-prefill', 195, 'cap: native40 check timeout 180 s + 15 s kill-after'),
    ('instruments-x2', 210, 'cap: 2 x (pause + resume at --control-timeout-s 30, 4 x docker exec at 10 s, 5 s start)'),
    ('arm-entries', 1080, 'cap: 2 x idle --timeout-s 120 + 2 x typed full-idle 420 (prepare, status, pause, resume '
                          'at 60 s plus 120 s idle); the second full-idle re-takes the lease before stock-1'),
    ('w2-arms', ARMS * (ARM_TIMEOUT_S + ARM_KILL_AFTER_S),
     'cap: 4 arms x (1500 s timeout + 30 s kill-after); one arm nominal ~250-550 s: 8 x 153,537 open '
     '(cold 250-410 s, q18a P) + 5 s settle + 20 s window + fresh 70-140 s + close <= 60 s + append ~10 s'),
    ('ingress-x4', 240, 'budget: 4 x 60 s, one pass over the followed log each'),
    ('end-full-idle', 420, 'cap: typed all-rank FULL at the end (prepare, status, pause, resume at 60 s plus 120 s idle)'),
    ('grade', 5, 'pure CPU over retained artifacts'),
    ('identity-end',) + _P['identity-end'],
    ('normal-path-cleanup',) + _P['normal-path-cleanup'],
)
ACTIVE_S = 10860
CLEANUP_S = 900
assert sum(seconds for _, seconds, _ in ACTIVE_DERIVATION) <= ACTIVE_S
assert sum(seconds for _, seconds, _ in CLEANUP_DERIVATION) <= CLEANUP_S
EXPECTED_RUNTIME_S = {'nominal': 2400, 'basis': 'restart ~320 s, identities ~320 s, four arms ~1300 s, entries, '
                                                 'receipts and cleanup ~460 s'}


def latest_arm():
    return HARD_END - ACTIVE_S - CLEANUP_S


def window_ok(now):
    return NOT_BEFORE <= now and now + ACTIVE_S + CLEANUP_S <= HARD_END
