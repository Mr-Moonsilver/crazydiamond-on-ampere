"""Native row1 qualification: a serial byte oracle for the sparse history route.

The E2 :mod:`exactness` mechanism is the oracle: a candidate snapshot is taken,
the original reference rewrites the *same* arena/output storage, a second snapshot
is taken, and the two are compared byte-for-byte.  No digest proves bytes here;
the digest only binds an immutable record to its identity and counts.

Two checks are performed per sparse-helper layer, strictly serially and outside
shipping performance:

``bucket``  after the attention consumer event of a bucket is host-queried
            complete and before the next fill, the compact rows are gathered
            from the original pinned UVA view into the same arena slice with the
            same rows/pieces, and compared.
``output``  after the last consumer, the original ``flash_e2_arena_attention``
            is re-run on the same query/arena/output and the QSA output before
            MoE is compared.  That reference may use its bulk UVA gather; it is
            the original path and is only run for qualification.

Nothing here allocates a second history GPU buffer.  Snapshots are bounded CPU
copies, produced while no compact fill is outstanding.
"""
import hashlib
import json
import time
from dataclasses import dataclass, field

from e3_types import E3ContractError, E3Pending, LaneId, LaneRole, TransactionId

# The pinned reference's own route names; the actual operation names differ for
# the compact route, so the mapping is explicit rather than assumed.
_REFERENCE_ROUTES = ('dense_dma', 'sparse_uva', 'empty')
_ACTUAL_ROUTES = {'dense_dma': 'dense_dma', 'sparse_uva': 'sparse_dma',
                  'empty': 'empty'}
_BUCKET_OPERATIONS = ('dense_page_bucket', 'sparse_row_bucket')
_OUTPUT_OPERATION = 'qsa_output_before_moe'


class Row1OracleError(E3ContractError):
    """A native row1 byte/output comparison failed or could not be completed."""


def _digest(kind, payload):
    body = dict(payload, kind=kind)
    text = json.dumps(body, sort_keys=True, default=str, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def _bounded(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise Row1OracleError(f'row1 {name} must be an int >= {minimum}')
    return value


@dataclass(frozen=True)
class Row1Identity:
    """Every owner/epoch that makes one comparison attributable to a run."""
    rank: int
    worker_incarnation: int
    engine_epoch: int
    route_generation: int
    storage_epoch: int
    source_epoch: int
    state_epoch: int
    mapping_version: int
    layer: str
    operation: str

    def __post_init__(self):
        for name in ('rank', 'route_generation', 'storage_epoch',
                     'source_epoch', 'mapping_version'):
            _bounded(getattr(self, name), name)
        # The worker incarnation is the state's own opaque identity.
        if type(self.worker_incarnation) is int:
            _bounded(self.worker_incarnation, 'worker_incarnation')
        elif (not isinstance(self.worker_incarnation, str)
              or not self.worker_incarnation):
            raise Row1OracleError('row1 identity needs the worker incarnation')
        # An unbound engine epoch (0) or state epoch (0) is never evidence.
        _bounded(self.engine_epoch, 'engine_epoch', minimum=1)
        _bounded(self.state_epoch, 'state_epoch', minimum=1)
        if not isinstance(self.layer, str) or not self.layer:
            raise Row1OracleError('row1 identity needs the layer name')
        if not isinstance(self.operation, str) or not self.operation:
            raise Row1OracleError('row1 identity needs the operation name')

    def as_dict(self):
        return dict(rank=self.rank, worker_incarnation=self.worker_incarnation,
                    engine_epoch=self.engine_epoch,
                    route_generation=self.route_generation,
                    storage_epoch=self.storage_epoch,
                    source_epoch=self.source_epoch,
                    state_epoch=self.state_epoch,
                    mapping_version=self.mapping_version,
                    layer=self.layer, operation=self.operation)


@dataclass(frozen=True)
class Row1Bucket:
    """One compact/dense bucket's exact row arithmetic."""
    index: int
    route: str                       # 'sparse_dma' or 'dense_dma'
    rows: int
    pieces: int
    compact_start: object            # int for the compact route, None for dense
    nbytes: int
    segments: tuple = field(repr=False)   # (source_row, arena_row, rows)

    def __post_init__(self):
        _bounded(self.index, 'bucket index')
        if self.route not in ('sparse_dma', 'dense_dma'):
            raise Row1OracleError('row1 bucket route is not a history route')
        _bounded(self.rows, 'bucket rows', minimum=1)
        _bounded(self.pieces, 'bucket pieces', minimum=1)
        _bounded(self.nbytes, 'bucket nbytes')
        if self.compact_start is not None:
            _bounded(self.compact_start, 'compact start')
        if type(self.segments) is not tuple or not self.segments:
            raise Row1OracleError('row1 bucket needs its gather segments')
        covered = 0
        for segment in self.segments:
            if type(segment) is not tuple or len(segment) != 3:
                raise Row1OracleError('row1 bucket segment is malformed')
            source_row, arena_row, rows = segment
            _bounded(source_row, 'segment source row')
            _bounded(arena_row, 'segment arena row')
            _bounded(rows, 'segment rows', minimum=1)
            covered += rows
        if covered != self.rows:
            raise Row1OracleError('row1 bucket segments do not cover its rows')


@dataclass(frozen=True)
class Row1Comparison:
    """One raw, immutable comparison record (bytes, not a digest)."""
    identity: Row1Identity
    bucket: int
    route: str
    rows: int
    pieces: int
    compact_start: object
    compared_bytes: int
    different_bytes: object
    equal: bool
    finite: object
    coverage_rows: int
    coverage_bytes: int
    observed_ns: int
    digest: str

    def as_dict(self):
        payload = dict(self.identity.as_dict(), bucket=self.bucket,
                       route=self.route, rows=self.rows, pieces=self.pieces,
                       compact_start=self.compact_start,
                       compared_bytes=self.compared_bytes,
                       different_bytes=self.different_bytes,
                       equal=self.equal, finite=self.finite,
                       coverage_rows=self.coverage_rows,
                       coverage_bytes=self.coverage_bytes,
                       observed_ns=self.observed_ns, digest=self.digest)
        return payload


@dataclass(frozen=True)
class Row1LayerRecord:
    """A layer's completed qualification, including its hold release."""
    identity: Row1Identity
    passed: bool
    aborted: object
    route: str
    bucket_checks: int
    bucket_rows: int
    bucket_bytes: int
    output_checks: int
    output_rows: int
    output_bytes: int
    hold_released: bool
    hold_stats: tuple
    digest: str

    def as_dict(self):
        return dict(self.identity.as_dict(), passed=self.passed,
                    aborted=self.aborted, route=self.route,
                    bucket_checks=self.bucket_checks,
                    bucket_rows=self.bucket_rows,
                    bucket_bytes=self.bucket_bytes,
                    output_checks=self.output_checks,
                    output_rows=self.output_rows,
                    output_bytes=self.output_bytes,
                    hold_released=self.hold_released,
                    hold_stats=self.hold_stats, digest=self.digest)


@dataclass(frozen=True)
class Row1Failure:
    """The first immutable failure record for a comparison."""
    identity: Row1Identity
    kind: str
    bucket: object
    compared_bytes: int
    different_bytes: object
    hold_released: bool
    evidence: tuple
    digest: str

    def as_dict(self):
        return dict(self.identity.as_dict(), kind=self.kind,
                    bucket=self.bucket, compared_bytes=self.compared_bytes,
                    different_bytes=self.different_bytes,
                    hold_released=self.hold_released,
                    evidence=self.evidence, digest=self.digest)


class Row1RecordStore:
    """A bounded, immutable, queryable record of the actual comparisons.

    There is no ``clear``/``drop``: the records are the qualification evidence
    for the process lifetime.  Exceeding the bound raises instead of silently
    discarding an observation.
    """

    def __init__(self, capacity=4096):
        if type(capacity) is not int or capacity < 1:
            raise Row1OracleError('row1 record store needs a positive bound')
        self._capacity = capacity
        self._comparisons = []
        self._layers = []
        self._failures = []

    @property
    def capacity(self):
        return self._capacity

    def _append(self, collection, record, name):
        if len(collection) >= self._capacity:
            raise Row1OracleError('row1 ' + name + ' store is full')
        collection.append(record)

    def append_comparison(self, record):
        if type(record) is not Row1Comparison:
            raise Row1OracleError('store needs a row1 comparison record')
        self._append(self._comparisons, record, 'comparison')

    def append_layer(self, record):
        if type(record) is not Row1LayerRecord:
            raise Row1OracleError('store needs a row1 layer record')
        self._append(self._layers, record, 'layer')

    def append_failure(self, record):
        if type(record) is not Row1Failure:
            raise Row1OracleError('store needs a row1 failure record')
        self._append(self._failures, record, 'failure')

    @staticmethod
    def _bounded(items, *, layer=None, limit=None):
        if layer is not None:
            if not isinstance(layer, str) or not layer:
                raise Row1OracleError('row1 query layer must be a name')
            items = [item for item in items if item.identity.layer == layer]
        if limit is not None:
            if type(limit) is not int or limit < 0:
                raise Row1OracleError('row1 query limit must be non-negative')
            items = items[:limit]
        return tuple(items)

    def records(self, *, layer=None, limit=None):
        return self._bounded(self._comparisons, layer=layer, limit=limit)

    def layer_records(self, *, layer=None, limit=None):
        return self._bounded(self._layers, layer=layer, limit=limit)

    def failures(self, *, layer=None, limit=None):
        return self._bounded(self._failures, layer=layer, limit=limit)

    def summary(self):
        return dict(comparisons=len(self._comparisons),
                    layers=len(self._layers),
                    failures=len(self._failures),
                    capacity=self._capacity,
                    scope=('serial row1 qualification only; CPU snapshots are '
                           'not shipping performance evidence'))


class Row1ReferenceSession:
    """The ``e2`` session object the original ``flash_e2_arena_attention`` needs.

    ``fill`` is the original dense ``torch.index_select(kv_cache, ...)`` over the
    pinned UVA view into the same arena.  The sparse branch of the original
    function performs its own UVA gather; this session only observes its route.
    """

    def __init__(self, backend, run):
        if not callable(getattr(backend, 'gather_reference', None)):
            raise Row1OracleError('reference session needs the E2 gather backend')
        if run is None or getattr(run, 'uva', None) is None:
            raise Row1OracleError('reference session needs the pinned UVA view')
        self.backend = backend
        self.run = run
        self.route_name = None
        self.fills = 0
        self.fill_pages = 0
        self.consumes = 0

    def route(self, name, pages):
        if name not in _REFERENCE_ROUTES:
            raise Row1OracleError('reference declared an unpinned route')
        _bounded(pages, 'reference route pages')
        if self.route_name is not None and self.route_name != name:
            raise Row1OracleError('reference route changed within one layer')
        self.route_name = name

    def fill(self, staged, lo, hi, table):
        if self.route_name != 'dense_dma':
            raise Row1OracleError('reference fill outside the dense route')
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi:
            raise Row1OracleError('reference fill has invalid bucket bounds')
        self.backend.gather_reference(self.run.uva, table, lo, hi, staged)
        self.fills += 1
        self.fill_pages += hi - lo

    def consume_begin(self):
        self.consumes += 1

    def consume_end(self):
        pass


class Row1Qualifier:
    """Strict serial qualification of one transaction's actual sparse route."""

    def __init__(self, store, *, transaction, lane, backend, epochs,
                 page_rows=4048, row_width=512, clock=time.monotonic_ns):
        if type(store) is not Row1RecordStore:
            raise Row1OracleError('qualifier needs the bounded record store')
        if type(transaction) is not TransactionId:
            raise Row1OracleError('qualifier needs the transaction')
        if type(lane) is not LaneId or lane.role is not LaneRole.PREFILL:
            raise Row1OracleError('qualifier needs the prefill lane')
        if not callable(epochs):
            raise Row1OracleError('qualifier needs the epoch provider')
        for name in ('snapshot', 'equal', 'nbytes', 'differing_bytes', 'finite',
                     'gather_rows', 'gather_reference'):
            if not callable(getattr(backend, name, None)):
                raise Row1OracleError('qualifier backend lacks ' + name)
        if type(page_rows) is not int or page_rows < 1:
            raise Row1OracleError('qualifier needs the page row geometry')
        if type(row_width) is not int or row_width < 1:
            raise Row1OracleError('qualifier needs the row width')
        self.store = store
        self.transaction = transaction
        self.lane = lane
        self.backend = backend
        self.epochs = epochs
        self.page_rows = page_rows
        self.row_width = row_width
        self.clock = clock
        self._checks = {}
        self._outputs = {}
        self._aborted = {}
        self.stats = dict(buckets=0, bucket_rows=0, bucket_bytes=0,
                          outputs=0, output_bytes=0, layers=0, failures=0,
                          aborted=0)

    # -- identity ---------------------------------------------------------
    def identity(self, run, operation):
        values = self.epochs()
        if type(values) is not dict:
            raise Row1OracleError('epoch provider must return an immutable mapping')
        try:
            worker = values['worker_incarnation']
            engine = values['engine_epoch']
            route = values['route_generation']
        except KeyError as exc:
            raise Row1OracleError('epoch provider is incomplete') from exc
        if getattr(run, 'state_epoch', None) is None:
            raise Row1OracleError('row1 comparison has no admitted state epoch')
        return Row1Identity(
            rank=self.lane.worker_rank,
            worker_incarnation=worker,
            engine_epoch=engine,
            route_generation=route,
            storage_epoch=run.allocation_epoch,
            source_epoch=run.source_epoch,
            state_epoch=run.state_epoch,
            mapping_version=run.mapping_version,
            layer=run.layer_name,
            operation=operation)

    def reference_session(self, run):
        return Row1ReferenceSession(self.backend, run)

    # -- checks -----------------------------------------------------------
    def bucket(self, run, bucket, staged):
        """Compare the actual arena bytes with the original UVA gather."""
        if type(bucket) is not Row1Bucket:
            raise Row1OracleError('bucket comparison needs a row1 bucket')
        if getattr(run, 'uva', None) is None:
            raise Row1OracleError('row1 bucket has no pinned UVA reference')
        if run.row1_failed is not None:
            raise Row1OracleError('row1 layer already failed: ' + str(run.row1_failed))
        identity = self.identity(
            run, 'sparse_row_bucket' if bucket.route == 'sparse_dma'
            else 'dense_page_bucket')
        started = self.clock()
        candidate = self.backend.snapshot(staged)
        gathered = 0
        for source_row, arena_row, rows in bucket.segments:
            self.backend.gather_rows(run.uva, source_row, rows, staged, arena_row)
            gathered += rows
        if gathered != bucket.rows:
            raise Row1OracleError('row1 gather did not cover the bucket')
        baseline = self.backend.snapshot(staged)
        equal = bool(self.backend.equal(candidate, baseline))
        different = None if equal else self.backend.differing_bytes(candidate, baseline)
        compared = int(self.backend.nbytes(candidate))
        observed = int(self.clock() - started)
        payload = dict(identity=identity.as_dict(), bucket=bucket.index,
                       route=bucket.route, rows=bucket.rows,
                       pieces=bucket.pieces, compact_start=bucket.compact_start,
                       compared_bytes=compared, different_bytes=different,
                       equal=equal, observed_ns=observed)
        record = Row1Comparison(
            identity=identity, bucket=bucket.index, route=bucket.route,
            rows=bucket.rows, pieces=bucket.pieces,
            compact_start=bucket.compact_start, compared_bytes=compared,
            different_bytes=different, equal=equal, finite=None,
            coverage_rows=bucket.rows, coverage_bytes=compared,
            observed_ns=observed, digest=_digest('row1_bucket', payload))
        self.store.append_comparison(record)
        self._checks.setdefault(run.layer_name, []).append(record)
        self.stats['buckets'] += 1
        self.stats['bucket_rows'] += bucket.rows
        self.stats['bucket_bytes'] += compared
        if not equal:
            self.fail('arena_bytes', run, identity, bucket=bucket.index,
                      compared_bytes=compared, different_bytes=different,
                      evidence=(('bucket_rows', bucket.rows),
                                ('pieces', bucket.pieces),
                                ('route', bucket.route),
                                ('compact_start', bucket.compact_start)))
        return record

    def output(self, run, session, reference, tensor, *, expected_route,
               expected_fills):
        """Compare the actual QSA output before MoE with the original function.

        ``expected_route`` names the pinned reference's own route; the stored
        record keeps the operation's actual route through ``_ACTUAL_ROUTES``.
        """
        if not isinstance(session, Row1ReferenceSession) or session.run is not run:
            raise Row1OracleError('output comparison needs its own reference session')
        if not callable(reference):
            raise Row1OracleError('output comparison needs the bound reference')
        if expected_route not in _REFERENCE_ROUTES:
            raise Row1OracleError('output comparison has an invalid expected route')
        _bounded(expected_fills, 'expected reference fills')
        if run.row1_failed is not None:
            raise Row1OracleError('row1 layer already failed: ' + str(run.row1_failed))
        identity = self.identity(run, _OUTPUT_OPERATION)
        started = self.clock()
        candidate = self.backend.snapshot(tensor)
        try:
            reference()
        except Row1OracleError:
            raise
        except BaseException as exc:
            self.fail('oracle_or_candidate_exception', run, identity, bucket=None,
                      compared_bytes=0, different_bytes=None,
                      evidence=(('error_type', type(exc).__name__),))
        baseline = self.backend.snapshot(tensor)
        route = session.route_name
        if route is None:
            self.fail('missing_reference_route', run, identity, bucket=None,
                      compared_bytes=0, different_bytes=None,
                      evidence=(('expected_route', expected_route),))
        if route != expected_route:
            self.fail('route_mismatch', run, identity, bucket=None,
                      compared_bytes=0, different_bytes=None,
                      evidence=(('reference_route', route),
                                ('expected_route', expected_route)))
        if session.fills != expected_fills:
            self.fail('reference_bucket_coverage', run, identity, bucket=None,
                      compared_bytes=0, different_bytes=None,
                      evidence=(('reference_fills', session.fills),
                                ('expected_fills', expected_fills)))
        equal = bool(self.backend.equal(candidate, baseline))
        different = None if equal else self.backend.differing_bytes(candidate, baseline)
        finite = (bool(self.backend.finite(candidate))
                  and bool(self.backend.finite(baseline)))
        rows = int(tensor.shape[0]) if len(getattr(tensor, 'shape', ())) else 0
        compared = int(self.backend.nbytes(candidate))
        observed = int(self.clock() - started)
        actual_route = _ACTUAL_ROUTES[route]
        payload = dict(identity=identity.as_dict(), bucket=None,
                       route=actual_route, rows=rows, pieces=session.fills,
                       compared_bytes=compared, different_bytes=different,
                       equal=equal, finite=finite, observed_ns=observed)
        record = Row1Comparison(
            identity=identity, bucket=-1, route=actual_route, rows=rows,
            pieces=session.fills, compact_start=None, compared_bytes=compared,
            different_bytes=different, equal=equal, finite=finite,
            coverage_rows=rows, coverage_bytes=compared,
            observed_ns=observed, digest=_digest('row1_output', payload))
        self.store.append_comparison(record)
        self._checks.setdefault(run.layer_name, []).append(record)
        self._outputs[run.layer_name] = record
        self.stats['outputs'] += 1
        self.stats['output_bytes'] += compared
        if not equal:
            self.fail('qsa_output_bytes', run, identity, bucket=None,
                      compared_bytes=compared, different_bytes=different,
                      evidence=(('route', actual_route), ('rows', rows),
                                ('reference_fills', session.fills)))
        if not finite:
            self.fail('nonfinite_qsa_output', run, identity, bucket=None,
                      compared_bytes=compared, different_bytes=None,
                      evidence=(('route', actual_route), ('rows', rows)))
        return record

    # -- lifecycle --------------------------------------------------------
    def fail(self, kind, run, identity, *, bucket, compared_bytes,
             different_bytes, evidence):
        if not isinstance(kind, str) or not kind:
            raise Row1OracleError('row1 failure needs a kind')
        run.row1_failed = kind
        hold = getattr(run, 'sparse_hold', None)
        payload = dict(identity=identity.as_dict(), kind=kind, bucket=bucket,
                       compared_bytes=int(compared_bytes),
                       different_bytes=different_bytes,
                       hold_released=bool(hold is not None and hold.released),
                       evidence=tuple(evidence))
        record = Row1Failure(
            identity=identity, kind=kind, bucket=bucket,
            compared_bytes=int(compared_bytes), different_bytes=different_bytes,
            hold_released=bool(hold is not None and hold.released),
            evidence=tuple(sorted((str(key), value) for key, value in evidence)),
            digest=_digest('row1_failure', payload))
        self.store.append_failure(record)
        self.stats['failures'] += 1
        raise Row1OracleError('row1 qualification failed: ' + kind)

    def check_releasable(self, run, *, cancelled=False):
        """The output comparison must precede every source-hold release.

        A cancelled transaction can never complete the comparison; its release
        is allowed only as an explicitly recorded abort, never as a pass.
        """
        if cancelled:
            self.abort(run, 'transaction cancelled')
            return True
        if getattr(run, 'row1_reference', None) is None:
            if getattr(run, 'sparse_hold', None) is not None:
                raise Row1OracleError('sparse source layer has no row1 reference')
            return True
        if run.row1_failed is not None:
            raise Row1OracleError(run.row1_failed)
        if run.layer_name not in self._outputs:
            raise E3Pending('row1 output comparison has not completed')
        return True

    def abort(self, run, reason):
        """Record a coordinated abort; no release happened at this point."""
        if not isinstance(reason, str) or not reason:
            raise Row1OracleError('row1 abort needs a reason')
        layer = run.layer_name
        if layer in self._aborted:
            return self._aborted[layer]
        self._aborted[layer] = reason
        self.stats['aborted'] += 1
        return reason

    def finish_layer(self, run, *, hold_stats):
        """Publish the layer's immutable record after its hold was released."""
        checks = self._checks.pop(run.layer_name, [])
        aborted = self._aborted.pop(run.layer_name, None)
        output = self._outputs.pop(run.layer_name, None)
        identity = self.identity(run, 'layer_row1')
        bucket_checks = [c for c in checks if c.identity.operation in _BUCKET_OPERATIONS]
        output_checks = [c for c in checks if c.identity.operation == _OUTPUT_OPERATION]
        routes = sorted({c.route for c in bucket_checks})
        if len(routes) > 1:
            route = 'mixed'
        elif routes:
            route = routes[0]
        elif output is not None:
            route = output.route
        else:
            route = 'none'
        passed = (aborted is None and output is not None
                  and bool(output.equal) and bool(output.finite)
                  and all(bool(check.equal) for check in checks))
        if aborted is None and not passed:
            raise Row1OracleError('row1 layer finished without a passing comparison')
        hold = getattr(run, 'sparse_hold', None)
        hold_released = bool(hold is not None and hold.released)
        stats = tuple(sorted((str(key), int(value))
                             for key, value in dict(hold_stats or {}).items()))
        payload = dict(identity=identity.as_dict(), passed=passed,
                       aborted=aborted, route=route,
                       bucket_checks=len(bucket_checks),
                       bucket_rows=sum(c.rows for c in bucket_checks),
                       bucket_bytes=sum(c.compared_bytes for c in bucket_checks),
                       output_checks=len(output_checks),
                       output_rows=sum(c.rows for c in output_checks),
                       output_bytes=sum(c.compared_bytes for c in output_checks),
                       hold_released=hold_released, hold_stats=stats)
        record = Row1LayerRecord(
            identity=identity, passed=passed, aborted=aborted, route=route,
            bucket_checks=len(bucket_checks),
            bucket_rows=sum(c.rows for c in bucket_checks),
            bucket_bytes=sum(c.compared_bytes for c in bucket_checks),
            output_checks=len(output_checks),
            output_rows=sum(c.rows for c in output_checks),
            output_bytes=sum(c.compared_bytes for c in output_checks),
            hold_released=hold_released, hold_stats=stats,
            digest=_digest('row1_layer', payload))
        self.store.append_layer(record)
        self.stats['layers'] += 1
        return record


__all__ = [
    'Row1Bucket', 'Row1Comparison', 'Row1Failure', 'Row1Identity',
    'Row1LayerRecord', 'Row1OracleError', 'Row1Qualifier', 'Row1RecordStore',
    'Row1ReferenceSession',
]
