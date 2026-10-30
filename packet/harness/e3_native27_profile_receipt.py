"""Bounded profile-refusal evidence for qualification and queue receipts.

The engine's recent-event journal can drop the causal entry during a long
prefill. The engine status retains its first and latest refusal independently;
this module carries that exact bounded record into public phase and queue
receipts without changing any gate or refusal decision.
"""

import copy
import json
from pathlib import Path


SCHEMA = 's5-schedule-profile-refusal/1'
MAX_RECEIPT_BYTES = 65536


def bounded_receipt(value):
    if type(value) is not dict or value.get('schema') != SCHEMA:
        return None
    count = value.get('count')
    if type(count) is not int or count < 1:
        return None
    for name in ('first', 'latest'):
        row = value.get(name)
        if type(row) is not dict or row.get('kind') != 'schedule_profile_refused':
            return None
        peers = row.get('peer_profiles')
        if type(peers) is not list or not 1 <= len(peers) <= 8:
            return None
        if any(type(peer) is not dict or type(peer.get('rank')) is not int
               for peer in peers):
            return None
    if len(json.dumps(value, sort_keys=True).encode()) > MAX_RECEIPT_BYTES:
        return None
    return copy.deepcopy(value)


def from_cleanup(cleanup):
    """Read the real all-rank return census saved by QualificationRefused."""
    if type(cleanup) is not dict:
        return None
    steps = cleanup.get('steps')
    if type(steps) is not dict:
        return None
    leave = steps.get('leave_full')
    if type(leave) is not dict:
        return None
    census = leave.get('census_before_return')
    if type(census) is not dict:
        return None
    snapshot = census.get('snapshot')
    if type(snapshot) is not dict:
        return None
    return bounded_receipt(snapshot.get('schedule_profile_refusals'))


def queue_grade_receipts(run_dir):
    """Copy only validated, small phase refusal records into queue-grade."""
    run_dir = Path(run_dir)
    phases = {}
    unavailable = {}
    for phase in ('row1', 'row4'):
        path = run_dir / ('qual-' + phase + '.json')
        if not path.is_file():
            continue
        try:
            with path.open() as handle:
                artifact = json.load(handle)
        except (OSError, ValueError) as exc:
            unavailable[phase] = type(exc).__name__
            continue
        if type(artifact) is not dict:
            unavailable[phase] = 'not-object'
            continue
        receipt = bounded_receipt(artifact.get('schedule_profile_refusals'))
        if receipt is not None:
            phases[phase] = receipt
    return {'schema': 'q18a-schedule-profile-refusals/1',
            'phases': phases, 'unavailable': unavailable}
