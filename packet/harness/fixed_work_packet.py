"""Fixed work and the seed order on a j2 four-arm packet (brief 12, 14 Nov 2026). CPU only; env-gated.

Two harness settings, exported by a cx draft before its first python (tools/build_q2024p_j2_cx.py), read here:
  REPLAY_SEED_ORDER   unset/plain: j2's seeds (index 0 = seed 1, index 1 = seed 2; sp0 n30 n31 sp1 runs seed 1
                      stock first and seed 2 engine first). swap: index 0 = seed 2, index 1 = seed 1 (seed 2 stock
                      first, seed 1 engine first). The arm order is unchanged; only the seed each pair carries moves.
  REPLAY_FIXED_WORK / REPLAY_FIXED_CAP_S   compose4_replay's fixed work (harness/compose4_replay.fixed_work_targets).
``wrap_env(configure)`` is the bundle's n3j2_packet.configure, then the seed order (j2's table otherwise unchanged).
The mechanism rule (bundle tools/n3j2_junction_rule.py, patched by the builder only when one of them is set) asks
``expected_seed``, ``send_limit_s`` and ``replay_window_ok`` instead of its 480 s literals and its 1 + index seed.
With neither set every answer is the parent rule's own.
"""
import math
import os

import compose4_replay as replay

ORDER_ENV = 'REPLAY_SEED_ORDER'
SEEDS = {'plain': {0: 1, 1: 2}, 'swap': {0: 2, 1: 1}}
WINDOW_S = 480                      # the parent rule's literal: j2's DURATION


def seed_order(environ=None):
    env = os.environ if environ is None else environ
    value = env.get(ORDER_ENV, '') or 'plain'
    if value not in SEEDS:
        raise ValueError('%s must be plain or swap, not %r' % (ORDER_ENV, value))
    return value


def wrap_env(configure, environ=None):
    """The bundle's configure, then the seeds of REPLAY_SEED_ORDER (index -> seed); plain leaves j2's dict alone."""
    def configured():
        table = configure()
        order = seed_order(environ)
        if order != 'plain':
            table.REPLAY_SEEDS = dict(SEEDS[order])
        assert table.REPLAY_SEEDS == SEEDS[order], (table.REPLAY_SEEDS, order)
        return table
    configured.seed_order_env = ORDER_ENV
    configured.j2_configure = configure          # the configure it wraps (n3dmp_packet's name; the bundle walks it)
    return configured


def expected_seed(label, environ=None):
    """The seed a replay label must carry: index = the label's last digit."""
    return SEEDS[seed_order(environ)][int(label[-1])]


def fixed(seed, environ=None):
    """None (fixed work off) or (targets for ``seed``, cap seconds), parsed as the driver parses them."""
    env = os.environ if environ is None else environ
    if not env.get(replay.FIXED_WORK_ENV) and not env.get(replay.FIXED_CAP_ENV):
        return None
    cfg = dict(AGENTS=8, SEED=seed, DURATION=float(WINDOW_S))
    return replay.fixed_work_targets(cfg, environ=env, reply_text=env.get(replay.REPLY_TEXT_ENV, '') or 'model')


def send_limit_s(environ=None):
    """The last send time a journal row may carry: the cap under fixed work, else 480 s."""
    env = os.environ if environ is None else environ
    cap = env.get(replay.FIXED_CAP_ENV)
    if not env.get(replay.FIXED_WORK_ENV):
        return WINDOW_S
    return float(cap)


def replay_window_ok(arm, environ=None):
    """Plain: the replay ran its 480 s (start + 480 <= end). Fixed work: the arm's summary declares every agent reached
    the target this setting gives for the arm's seed, without the cap, and the replay ended after that stop and
    inside the cap plus the driver's finish bound."""
    start, end = arm.get('replay_start_s'), arm.get('replay_end_s')
    seed = (arm.get('replay_config') or {}).get('SEED')
    work = fixed(seed, environ) if type(seed) is int else None
    if work is None:
        return start + WINDOW_S <= end
    targets, cap = work
    got = (arm.get('replay_summary') or {}).get('fixed_work')
    if type(got) is not dict:
        return False
    end_s = got.get('end_s')
    if got.get('targets') != targets or got.get('cap_s') != cap or got.get('all_reached') is not True or \
            got.get('cap_hit') is not False or type(end_s) not in (int, float) or not math.isfinite(end_s):
        return False
    reached = got.get('reached_s')
    if type(reached) is not list or len(reached) != len(targets) or any(type(r) not in (int, float) for r in reached) or \
            max(reached) != end_s:
        return False
    return 0 < end_s <= cap and start + end_s <= end <= start + cap + replay.FINISH_BOUND_S
