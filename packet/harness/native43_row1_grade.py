"""Native43 l-f row1 grade: e3_native_row1_grade.grade with the route coverage a function of the policy.

The base grade (unchanged, it ran on the box) requires every local layer to be covered by both dense_dma
and sparse_dma layer records (``rankN:layer_route_cartesian_coverage``).  That is the current policy's
coverage: below the arena the row1 prompt is dense, beyond it sparse.  Under FLASH_S5_NATIVE_QSA_POLICY=dense
the native route is dense_dma everywhere, so that requirement can never be met and row1 could only be
INCONCLUSIVE.  This grade keeps every base check and verdict input and replaces only that one coverage rule:

  policy  coverage required per rank and local layer
  current dense_dma and sparse_dma layer records (the base rule, unchanged)
  dense   a passing dense_dma layer record, and a passing dense_dma output comparison with pieces >= 2
          (the reference re-run on its own dense branch beyond the arena: two or more reference fills);
          any sparse_dma record under the dense policy is a FAIL (``rankN:dense_policy_sparse_route``)

Missing dense coverage is ``rankN:dense_policy_route_coverage`` / ``rankN:dense_policy_beyond_arena_coverage``
(INCONCLUSIVE).  Records are counted only when they carry the packet's own identity; the base grade already
FAILs any record that does not.  Pure CPU; no engine side effect.
"""
import e3_native_row1_grade as base

POLICIES = ('current', 'dense')
CARTESIAN = 'layer_route_cartesian_coverage'


def _rows(packet, key):
    raw = packet.get('raw') if type(packet) is dict else None
    rows = raw.get(key) if type(raw) is dict else None
    return [row for row in rows if type(row) is dict] if type(rows) in (list, tuple) else []


def _own(row, packet, engine_epoch, names):
    return (row.get('rank') == packet.get('rank') and type(row.get('rank')) is int
            and row.get('worker_incarnation') == packet.get('worker_incarnation')
            and row.get('engine_epoch') == engine_epoch and type(row.get('engine_epoch')) is int
            and row.get('route_generation') == packet.get('route_generation')
            and row.get('layer') in names)


def dense_coverage(evidence, *, engine_epoch):
    """Per-rank dense-policy coverage: (failures, missing, the coverage checks by name)."""
    failures, missing, census = [], [], {}
    workers = evidence.get('workers') if type(evidence) is dict else None
    for packet in workers if type(workers) in (list, tuple) else []:
        if type(packet) is not dict or type(packet.get('rank')) is not int or packet['rank'] not in base.RANKS:
            continue
        rank = packet['rank']
        prefix = 'rank%d:' % rank
        names = packet.get('local_layers')
        if type(names) not in (list, tuple) or not names:
            continue
        names = [name for name in names if type(name) is str]
        records = [r for r in _rows(packet, 'records') if _own(r, packet, engine_epoch, names)]
        layers = [r for r in _rows(packet, 'layer_records') if _own(r, packet, engine_epoch, names)]
        if any(r.get('route') == 'sparse_dma' for r in records + layers):
            failures.append(prefix + 'dense_policy_sparse_route')
        dense_layers = {r['layer'] for r in layers if r.get('route') == 'dense_dma'
                        and r.get('passed') is True and r.get('aborted') is None}
        beyond = {r['layer'] for r in records if r.get('operation') == base.OUTPUT
                  and r.get('route') == 'dense_dma' and r.get('equal') is True and r.get('finite') is True
                  and type(r.get('pieces')) is int and r['pieces'] >= 2}
        census[prefix + 'dense_policy_route_coverage'] = set(names) <= dense_layers
        census[prefix + 'dense_policy_beyond_arena_coverage'] = set(names) <= beyond
        if not set(names) <= dense_layers:
            missing.append(prefix + 'dense_policy_route_coverage')
        if not set(names) <= beyond:
            missing.append(prefix + 'dense_policy_beyond_arena_coverage')
    return failures, missing, census


def grade(evidence, *, run_id, engine_epoch, policy):
    if policy not in POLICIES:
        raise ValueError('row1 grade policy must be one of %s' % (POLICIES,))
    result = base.grade(evidence, run_id=run_id, engine_epoch=engine_epoch)
    if policy == 'current':
        return dict(result, policy=policy)
    failures = list(result['failures'])
    missing = [m for m in result['missing'] if not m.endswith(':' + CARTESIAN)]
    extra_failures, extra_missing, coverage = dense_coverage(evidence, engine_epoch=engine_epoch)
    checks = dict(result['checks'], **coverage)
    for reason in extra_failures:
        checks[reason] = False
        if reason not in failures:
            failures.append(reason)
    for reason in extra_missing:
        if reason not in missing:
            missing.append(reason)
    verdict = 'FAIL' if failures else ('INCONCLUSIVE' if missing else 'PASS')
    return dict(result, verdict=verdict, failures=failures, missing=missing, checks=checks, policy=policy,
                replaced=[m for m in result['missing'] if m.endswith(':' + CARTESIAN)])
