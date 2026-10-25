"""Additive failure diagnostics. Primary correctness evidence is immutable."""
import hashlib
import json
from pathlib import Path
import traceback
from s5_segment import write_json


def preserve_primary(path, record):
    path = Path(path)
    try:
        write_json(path, record)
    except FileExistsError:
        # Only this rank and its watchdog write here. Never replace either.
        existing = json.loads(path.read_text())
        if existing.get('fatal_correctness') is not True:
            raise RuntimeError('existing primary artifact is not the expected correctness failure')
    return dict(name=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def diagnostic(path, record, run):
    """Persist start + result separately; even a diagnostic exception cannot clear FAIL."""
    path = Path(path)
    primary = preserve_primary(path, record)
    stem = path.name.removesuffix('.error.json')
    started = path.with_name(stem + '.source-trace-start.json')
    finished = path.with_name(stem + '.source-trace.json')
    write_json(started, dict(primary=primary, peer_or_local_failed=True, started=True))
    try:
        result = run()
    except Exception as exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        result = dict(completed=False, error_type=type(exc).__name__)
    write_json(finished, dict(primary=primary, peer_or_local_failed=True, diagnostic_trace=result))
    return result


def worker_failure(path, exc, correctness_type, partial_rank, close):
    """Always expose the exception stack; retain an earlier correctness verdict."""
    path = Path(path)
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    fatal = isinstance(exc, correctness_type)
    try:
        if path.exists():
            fatal = fatal or json.loads(path.read_text()).get('fatal_correctness') is True
            sidecar = path.with_name(path.name.removesuffix('.error.json') + '.worker-exception.json')
            write_json(sidecar, dict(error_type=type(exc).__name__, fatal_correctness=fatal,
                                    primary=path.name, primary_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        else:
            write_json(path, dict(fatal_correctness=fatal, error=str(exc),
                                 evidence=getattr(exc, 'evidence', {}), partial_rank=partial_rank))
    except BaseException as artifact_error:
        traceback.print_exception(type(artifact_error), artifact_error, artifact_error.__traceback__)
    finally:
        try:
            close()
        except BaseException as close_error:
            traceback.print_exception(type(close_error), close_error, close_error.__traceback__)
    return 1 if fatal else 2
