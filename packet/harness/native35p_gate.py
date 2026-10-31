"""Native35 P window: the unchanged weekend bounds with the speed arm's own declared budget.

NOT_BEFORE and HARD_END are native34_gate's (2026-10-30T22:01Z,
2026-11-02T06:00Z), unchanged. O's 2100 s active / 420 s cleanup does not
bound six 8x153,537-token W1 cohorts plus two complete W2 schedules, so P
declares its own budget, derived below from the fixed schedule and the
inherited request/controller limits (no W1/W2 duration has been measured on
this engine). ``speed_window_ok(time.time())`` is the single executable check
used by the P draft's slot and the frozen P owner runner.
"""
from native30_gate import SNAPSHOT_CAP, host_receipt
from native34_gate import NOT_BEFORE, HARD_END

# Active budget: everything inside the outer timeout on the normal path.
# Each row: (phase, seconds, basis). 'measured' rows are 2x the native35 O
# run of 31 Oct (same startup path, model and topology); 'cap' rows are the
# inherited declared limit of that phase; W2 rows are 2x a conservative
# estimate (FULL solo prefill 3,000 tok/s, split/overlap 1,500 tok/s).
ACTIVE_DERIVATION = (
    ('restart-to-health', 780, 'cap: systemctl restart 240 s + 36 health tries x (5 s curl + 10 s sleep); O measured 321 s'),
    ('speculation-and-sampler', 60, 'cap: docker logs 30 s + sampler start 30 s'),
    ('identity-start', 360, 'cap: Q18A_IDENTITY_TIMEOUT_S; O measured 137 s'),
    ('startup-ledger-headroom-plan', 30, 'measured ~2 s in O'),
    ('row1', 876, '2x O measured 438 s'),
    ('rule4_controls', 252, '2x O measured 126 s'),
    ('rule4_candidate', 28, '2x O measured 14 s'),
    ('row4-placement', 900, 'cap: --timeout-s 900; native33-form real placement not yet measured with native35'),
    ('receipt-tail', 30, 'fixed sleep'),
    ('w1-six-repeats', 6060, 'cap: 6 x (900 s cohort open_timeout + 20 s window + 60 s drain + 30 s controller); cold 8x153,537 open ~250-410 s'),
    ('w2-two-rounds', 3080, '2 rounds x 2x conservative 770 s: full 70 + split 140 + overlap open 410 + fresh 140 + append/stop 10'),
    ('memory-and-aggregate-grades', 30, 'pure CPU over retained artifacts'),
    ('identity-end', 360, 'cap: Q18A_IDENTITY_TIMEOUT_S; O measured 184 s'),
    ('normal-path-cleanup', 716, 'same worst case as the timeout cleanup below'),
)
# Cleanup margin: owned cleanup after TERM before KILL (on_exit order).
CLEANUP_DERIVATION = (
    ('stop-sampler', 1, 'kill + wait'),
    ('capture-logs', 120, 'docker logs of <= 2 GiB retained json log (512m x 4), copy, error grep'),
    ('engine-stop', 440, 'systemctl stop 180 s + 20 s kill-after, 24 x 5 s inactive wait, 24 x 5 s GPU idle wait'),
    ('log-graders', 150, 'native30 localize + native35 grade; 10 s per 137 MB measured on O logs, ~2 GiB projected'),
    ('restores-and-records', 5, 'install/mv/cmp of launcher and defaults, cleanup.json, queue-grade.json'),
)
SPEED_ACTIVE_S = 14400
SPEED_CLEANUP_MARGIN_S = 900
SPEED_ACTIVE_AND_CLEANUP = SPEED_ACTIVE_S + SPEED_CLEANUP_MARGIN_S
EXPECTED_RUNTIME_S = {'nominal': 3300, 'conservative': 7200,
                      'basis': 'nominal: FULL prefill ~5,000 tok/s, W1 repeats 2-6 prefix-cached; conservative: all W1 cohorts cold at 3,000 tok/s'}


def speed_window_ok(now):
    return NOT_BEFORE <= now and now + SPEED_ACTIVE_AND_CLEANUP <= HARD_END
