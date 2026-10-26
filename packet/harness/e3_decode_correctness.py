#!/usr/bin/env python3
"""q17b short live-cohort decode-correctness client (offline build).

Three runs of one protocol
  1. full-A: prepare on full, open one fixed cohort on full, wait until every
     request has actually advanced, pause (keep / clear_cache=False), resume,
     measure a bounded decode window, pause_end, leave_full (no-op), resume,
     let the SAME requests finish their declared output budget.
  2. split-B: the same protocol with pause_enter_split instead of pause_full;
     the same live requests cross the transition.
  3. full-C: identical to full-A with a fresh cohort namespace.

The client compares the generated *integer token IDs* of A, B and C.  It is an
equality check, not a performance test and not proof of full E3 correctness.

Source basis (pinned completion API)
  reference/PROVENANCE-completion-api.txt pins image
  sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b
  (vllm 0.29.1rc1.dev47+gdc36fcce9):
    completion/protocol.py
      0be20e3d7f21ab6cbe0ba77cb50068aa39a35fc0369ef38de068ac38ca9800aa
    completion/serving.py
      75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527

  CompletionRequest.return_token_ids (protocol.py:169-178) requests plain
  integer generated-token deltas.  In streaming mode the first chunk may also
  carry prompt_token_ids, and CompletionResponseStreamChoice.token_ids
  (protocol.py:693) contains the delta tokens for that chunk
  (serving.py:415-419).  stream_options.include_usage +
  continuous_usage_stats supply cumulative completion counts.  cache_salt
  (protocol.py:200-212) is a source-backed prefix-cache namespace.  The client
  does NOT use return_tokens_as_token_ids: that field only encodes logprob
  token strings and is not an integer-token channel.

Controller basis
  The client imports the reviewed W1 controller command transport, command
  builder, receipt validator and schema-1 receipt chain from
  e3_w1_decode_driver.  It never speaks a different endpoint or invents a
  worker ACK.  Tests inject the controller transport.

Evidence boundary
  Prompts, token arrays and generated text stay in memory only (bounded by the
  manifest).  The artifact carries prompt hashes/lengths, trajectory hashes and
  lengths, bounded/redacted controller receipts whose retained hash can be
  recomputed, mismatch counts/first positions and fixed verdict reasons.

usage: e3_decode_correctness.py URL MODEL OUT.json
env: WORKLOAD=<manifest.json> [RUN_ID=...] [CONTROLLER_CMD=...]
     [API_SOURCE_DIR=<dir containing entrypoints/openai/...>]
"""
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import socket
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import e3_w1_decode_driver as protocol

HERE = Path(__file__).resolve().parent
CLIENT_NAME = 'q17b-short-correctness'
CLIENT_SCHEMA = 1
ROUTE_PLAN = ('full', 'split', 'full')
RUN_LABELS = ('full-A', 'split-B', 'full-C')
ENDPOINT = '/v1/completions'
EXPECTED_RANKS = tuple(protocol.EXPECTED_RANKS)

# Hard caps keep this a short correctness run; every effective value still has
# to be declared by the owner manifest.  These are bounds, not defaults.
MAX_STREAMS = 32
MAX_PROMPT_TOKENS = 200000
MAX_GENERATION_BUDGET = 8192
MAX_TOKEN_ID = 2 ** 31 - 1
MAX_REASON_CHARS = 300
ALLOWED_FINISH_REASONS = ('stop', 'length', 'abort')
KNOWN_STATUS = ('ok', 'error', 'timeout', 'busy', 'unavailable')
KNOWN_TRANSPORT_KINDS = ('subprocess', 'injected', 'injected_error')
KNOWN_TRANSPORT_ERRORS = ('TimeoutExpired', 'OSError', 'RuntimeError', 'ValueError',
                          'URLError', 'ConnectionResetError', 'BrokenPipeError',
                          'IncompleteRead', 'ProtocolError')
ACK = 'flash_s5_transition'
BARRIER_ACK = 'flash_s5_probe_barrier'
PROBE_MODE = 'decode-only-probe'

API_IMAGE_SHA256 = 'ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
API_SOURCE_PINS = {
    'entrypoints/openai/completion/protocol.py':
        '0be20e3d7f21ab6cbe0ba77cb50068aa39a35fc0369ef38de068ac38ca9800aa',
    'entrypoints/openai/completion/serving.py':
        '75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527',
}
PROTOCOL_FACTS = {
    'endpoint': ENDPOINT,
    'request_field': 'return_token_ids',
    'request_field_source': 'completion/protocol.py:169-178',
    'stream_field': 'choices[].token_ids',
    'stream_field_source': 'completion/protocol.py:693; completion/serving.py:415-419',
    'prompt_ids_field': 'choices[].prompt_token_ids (first chunk only)',
    'usage_fields': 'usage.completion_tokens with stream_options.include_usage + '
                    'continuous_usage_stats',
    'cache_namespace_field': 'cache_salt',
    'encoded_token_parser': False,
    'note': ('plain list[int] generated-token deltas; return_tokens_as_token_ids '
             'is not used because it encodes logprob token strings, not an '
             'integer generated-token channel'),
}

COALESCING_NOTE = (
    'client-visible SSE arrivals; a chunk with multiple token_ids is one '
    'coalesced arrival and is never expanded into per-token timestamps')
CLOCK_NOTE = (
    'all client event times are time.monotonic_ns with one wall-clock anchor; '
    'engine receipt clocks are a different clock and no offset is assumed')
OPEN_BASIS = ('first actual generated token delta or positive cumulative usage '
              'advance per live request; content-only events do not open a stream')
WINDOW_BASIS = (
    'start is the accepted resume receipt; end is the earlier of all requests '
    'reaching the declared in-window token bound and the declared wall bound')
CORRECTNESS_NOTE = (
    'short-run equality check over generated integer token IDs. It is not a '
    'speed threshold, not proof of all model behavior and not full E3 proof.')


class ConfigError(Exception):
    def __init__(self, message, reasons=None):
        super().__init__(message)
        self.reasons = list(reasons or [message])


def _is_int(value):
    return type(value) is int


def _is_number(value):
    return type(value) in (int, float) and math.isfinite(float(value))


def _sha256_json(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _ms(ns):
    return round(ns / 1_000_000.0, 3)


def _load_agent_replay():
    spec = importlib.util.spec_from_file_location('agent_replay_for_e3_correctness',
                                                  HERE / 'agent_replay.py')
    if spec is None or spec.loader is None:
        raise ImportError('agent_replay.py not found next to the client')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AR = _load_agent_replay()


# --- manifest ---------------------------------------------------------------

@dataclass(frozen=True)
class Workload:
    name: str
    streams: int
    prompt_seed: int
    prompt_tokens: int
    generation_budget: int
    sampling_seed: int
    temperature: float
    top_p: float
    window_s: float
    window_tokens: int
    open_timeout_s: float
    pause_timeout_s: float
    read_timeout_s: float
    drain_timeout_s: float
    join_timeout_s: float
    finish_timeout_s: float
    controller_timeout_s: float
    expected_output_tokens: int
    expected_finish_reason: str
    fresh_namespace_per_run: bool
    manifest_sha256: str


def _field_int(data, key, minimum, maximum, problems, label=None):
    label = label or key
    if key not in data:
        problems.append('manifest_field_missing:' + label)
        return None
    value = data[key]
    if not _is_int(value) or value < minimum or value > maximum:
        problems.append('manifest_field_invalid:' + label)
        return None
    return value


def _field_float(data, key, problems, label=None, maximum=3600.0):
    label = label or key
    if key not in data:
        problems.append('manifest_field_missing:' + label)
        return None
    value = data[key]
    if not _is_number(value) or float(value) <= 0 or float(value) > maximum:
        problems.append('manifest_field_invalid:' + label)
        return None
    return float(value)


def validate_manifest(data):
    """Pure manifest validation -> (problems, Workload|None, sha256)."""
    if type(data) is not dict:
        return ['manifest_not_object'], None, None
    sha = _sha256_json(data)
    problems = []
    if not _is_int(data.get('schema')) or data.get('schema') != 1:
        problems.append('manifest_schema_must_be_1')
    name = data.get('workload')
    if not isinstance(name, str) or not name.strip():
        problems.append('manifest_field_missing:workload')
        name = ''
    streams = _field_int(data, 'streams', 1, MAX_STREAMS, problems)
    budget = _field_int(data, 'generation_budget', 1, MAX_GENERATION_BUDGET, problems)

    prompt = data.get('prompt')
    prompt_seed = prompt_tokens = None
    if type(prompt) is not dict:
        problems.append('manifest_field_missing:prompt')
    else:
        prompt_seed = _field_int(prompt, 'seed', 0, 2 ** 63 - 1, problems, 'prompt.seed')
        prompt_tokens = _field_int(prompt, 'tokens', 1, MAX_PROMPT_TOKENS, problems,
                                   'prompt.tokens')

    sampling = data.get('sampling')
    sampling_seed = temperature = top_p = None
    if type(sampling) is not dict:
        problems.append('manifest_field_missing:sampling')
    else:
        sampling_seed = _field_int(sampling, 'seed', 0, 2 ** 63 - 1, problems,
                                   'sampling.seed')
        if 'temperature' not in sampling or not _is_number(sampling.get('temperature')):
            problems.append('manifest_field_missing:sampling.temperature')
        else:
            temperature = float(sampling['temperature'])
            if not 0.0 <= temperature <= 2.0:
                problems.append('manifest_field_invalid:sampling.temperature')
        if 'top_p' not in sampling or not _is_number(sampling.get('top_p')):
            problems.append('manifest_field_missing:sampling.top_p')
        else:
            top_p = float(sampling['top_p'])
            if not 0.0 < top_p <= 1.0:
                problems.append('manifest_field_invalid:sampling.top_p')
        if sampling.get('ignore_eos') is not True:
            problems.append('manifest_sampling_ignore_eos_must_be_true')
        if sampling.get('min_tokens') != 'equal_max':
            problems.append('manifest_sampling_min_tokens_must_equal_max')

    expected = data.get('expected')
    expected_tokens = expected_reason = None
    if type(expected) is not dict:
        problems.append('manifest_field_missing:expected')
    else:
        expected_tokens = _field_int(expected, 'output_tokens', 1,
                                     MAX_GENERATION_BUDGET, problems,
                                     'expected.output_tokens')
        expected_reason = expected.get('finish_reason')
        if expected_reason not in ALLOWED_FINISH_REASONS:
            problems.append('manifest_field_invalid:expected.finish_reason')
        if (budget is not None and expected_tokens is not None
                and expected_tokens != budget):
            problems.append('manifest_expected_must_equal_generation_budget')
        if expected_reason != 'length':
            problems.append('manifest_expected_finish_reason_must_be_length')

    bounds = data.get('bounds')
    window_s = window_tokens = None
    if type(bounds) is not dict:
        problems.append('manifest_field_missing:bounds')
    else:
        window_s = _field_float(bounds, 'window_s', problems, 'bounds.window_s', 120.0)
        window_tokens = _field_int(bounds, 'window_tokens', 1,
                                   MAX_GENERATION_BUDGET, problems,
                                   'bounds.window_tokens')
        if (budget is not None and window_tokens is not None
                and window_tokens >= budget):
            problems.append('manifest_window_tokens_must_be_below_budget')

    timeouts = data.get('timeouts')
    timeout_values = {}
    names = (('open_s', 'open_timeout_s'), ('pause_s', 'pause_timeout_s'),
             ('read_s', 'read_timeout_s'), ('drain_s', 'drain_timeout_s'),
             ('join_s', 'join_timeout_s'), ('finish_s', 'finish_timeout_s'),
             ('controller_s', 'controller_timeout_s'))
    if type(timeouts) is not dict:
        problems.append('manifest_field_missing:timeouts')
    else:
        for key, attr in names:
            timeout_values[attr] = _field_float(timeouts, key, problems,
                                                'timeouts.' + key, 3600.0)

    cache = data.get('cache')
    fresh = None
    if type(cache) is not dict:
        problems.append('manifest_field_missing:cache')
    else:
        fresh = cache.get('fresh_namespace_per_run')
        if fresh is not True:
            problems.append('manifest_cache_fresh_namespace_must_be_true')

    route_plan = data.get('route_plan')
    if type(route_plan) is not list or tuple(route_plan) != ROUTE_PLAN:
        problems.append('manifest_route_plan_must_be_full_split_full')

    if problems:
        return problems, None, sha
    workload = Workload(
        name=name, streams=streams, prompt_seed=prompt_seed,
        prompt_tokens=prompt_tokens, generation_budget=budget,
        sampling_seed=sampling_seed, temperature=temperature, top_p=top_p,
        window_s=window_s, window_tokens=window_tokens,
        open_timeout_s=timeout_values['open_timeout_s'],
        pause_timeout_s=timeout_values['pause_timeout_s'],
        read_timeout_s=timeout_values['read_timeout_s'],
        drain_timeout_s=timeout_values['drain_timeout_s'],
        join_timeout_s=timeout_values['join_timeout_s'],
        finish_timeout_s=timeout_values['finish_timeout_s'],
        controller_timeout_s=timeout_values['controller_timeout_s'],
        expected_output_tokens=expected_tokens,
        expected_finish_reason=expected_reason,
        fresh_namespace_per_run=fresh, manifest_sha256=sha)
    return [], workload, sha


@dataclass(frozen=True)
class Config:
    workload: Workload
    run_id: str
    out_path: str
    controller_cmd: str = ''
    # Mirrors of the W1 Config fields consumed by the shared controller_call /
    # _timeout_for helpers.  No new defaults: they come from the manifest.
    controller_timeout_s: float = 0.0
    read_timeout_s: float = 0.0
    drain_timeout_s: float = 0.0
    out_dir: str = ''

    def __post_init__(self):
        if not self.controller_timeout_s:
            object.__setattr__(self, 'controller_timeout_s',
                               self.workload.controller_timeout_s)
        if not self.read_timeout_s:
            object.__setattr__(self, 'read_timeout_s',
                               self.workload.read_timeout_s)
        if not self.drain_timeout_s:
            object.__setattr__(self, 'drain_timeout_s',
                               self.workload.drain_timeout_s)
        if not self.out_dir:
            object.__setattr__(self, 'out_dir',
                               str(Path(self.out_path).parent))


def load_manifest(env):
    raw_path = env.get('WORKLOAD')
    if not raw_path:
        return None, ['WORKLOAD manifest path is required; refusing implicit defaults']
    try:
        with Path(raw_path).open() as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        return None, ['workload_manifest_unreadable:' + type(exc).__name__]
    problems, workload, _sha = validate_manifest(data)
    if problems:
        return None, problems
    return workload, []


def config_from_env(env, out_path):
    workload, problems = load_manifest(env)
    if problems:
        raise ConfigError('owner workload manifest is absent or incomplete', problems)
    run_id = env.get('RUN_ID') or ('q17b-correct-%d' % time.time_ns())
    return Config(workload=workload, run_id=run_id,
                  out_path=str(Path(out_path).resolve()),
                  controller_cmd=env.get('CONTROLLER_CMD', ''))


# --- input cohort / requests ------------------------------------------------

def build_cohort(workload):
    """Seeded prompts, built once per run; prompt text is never persisted.

    Mirrors the reviewed W1 prompt stream (agent_replay.text) so the standing
    W1 cohort can be exercised with the same seed/prompt-token policy.
    """
    rows = []
    for index in range(workload.streams):
        rng = _random_for(workload.prompt_seed, index)
        text = AR.text(rng, workload.prompt_tokens)
        rows.append({
            'stream': index,
            'prompt': text,
            'prompt_sha256': _sha256_text(text),
            'prompt_chars': len(text),
        })
    return rows


def _random_for(seed, index):
    import random
    return random.Random(seed * 1000 + index)


def build_cache_salt(run_id, run_index):
    raw = 'q17b|%s|%d' % (run_id, run_index)
    return 'q17b-' + hashlib.sha256(raw.encode()).hexdigest()[:40]


def request_body(model, prompt, workload, cache_salt):
    """Source-backed completion request; n=1 and integer token deltas."""
    return {
        'model': model,
        'prompt': prompt,
        'max_tokens': workload.generation_budget,
        'min_tokens': workload.generation_budget,
        'temperature': workload.temperature,
        'top_p': workload.top_p,
        'seed': workload.sampling_seed,
        'ignore_eos': True,
        'n': 1,
        'stream': True,
        'return_token_ids': True,
        'stream_options': {
            'include_usage': True,
            'continuous_usage_stats': True,
        },
        'cache_salt': cache_salt,
    }


def request_sha256(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


# --- source identity --------------------------------------------------------

def verify_sources(env):
    """Hash the pinned completion sources when the reference tree is present.

    A present-but-different file is a hard source mismatch (no requests are
    sent).  An absent tree is recorded as unavailable; the pinned facts above
    still describe the client.  Never returns file contents.
    """
    root = Path(env.get('API_SOURCE_DIR') or (HERE.parent / 'reference' / 'vllm-src'))
    rows = {}
    problems = []
    for relative, expected in API_SOURCE_PINS.items():
        path = root / relative
        row = {'path': relative, 'expected_sha256': expected,
               'available': False, 'sha256': None, 'matches': None}
        if path.is_file():
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                problems.append('api_source_unreadable:' + relative)
                rows[relative] = row
                continue
            row.update(available=True, sha256=actual, matches=(actual == expected))
            if actual != expected:
                problems.append('api_source_hash_mismatch:' + relative)
        rows[relative] = row
    return {
        'image_sha256': API_IMAGE_SHA256,
        'provenance': 'reference/PROVENANCE-completion-api.txt',
        'facts': dict(PROTOCOL_FACTS),
        'files': rows,
        'available': any(row['available'] for row in rows.values()),
        'verified': all(row['matches'] is True for row in rows.values()),
        'unavailable': [relative for relative, row in rows.items()
                        if not row['available']],
    }, problems


# --- stream reading ---------------------------------------------------------

class StreamRecord:
    """In-memory stream evidence; token arrays never leave this object."""

    def __init__(self, index, prompt_sha256, prompt_chars, request_sha256_value):
        self.index = index
        self.prompt_sha256 = prompt_sha256
        self.prompt_chars = prompt_chars
        self.request_sha256 = request_sha256_value
        self.lock = threading.RLock()
        self.io_lock = threading.Lock()
        self.tokens = []
        self.token_count = 0
        self.finish_reason = None
        self.finish_at_ns = None
        self.first_advance_ns = None
        self.send_at_ns = None
        self.error = None
        self.error_at_ns = None
        self.cancelled = False
        self.malformed = None
        self.done_marker = False
        self.usage_completion = None
        self.usage_seen = False
        self.prompt_tokens = None
        self.prompt_token_ids_len = None
        self.arrivals = []
        self.coalesced_arrivals = 0
        self.usage_only_chunks = 0
        self.duplicate_usage_chunks = 0
        self.content_only_chunks = 0
        self.event_count = 0
        self.response = None
        self.thread = None
        self.stop = threading.Event()
        self.progress = threading.Event()
        self.finished = threading.Event()
        self.thread_done = False

    def count(self):
        with self.lock:
            return self.token_count

    def has_finished(self):
        with self.lock:
            return self.finish_reason is not None

    def fail(self, code):
        with self.lock:
            if self.malformed is None and self.error is None:
                self.malformed = code
        self.progress.set()
        self.finished.set()

    def cancel(self, join_timeout_s):
        with self.io_lock:
            self.stop.set()
            response = self.response
            if response is not None:
                _shutdown_response(response)
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join(timeout=max(0.0, join_timeout_s))
        with self.lock:
            self.cancelled = True
        return self.thread is None or not self.thread.is_alive()


def _response_socket(response):
    fp = getattr(response, 'fp', None)
    raw = getattr(fp, 'raw', None)
    sock = getattr(raw, '_sock', None)
    if sock is not None:
        return sock
    return getattr(response, 'fp', None)


def _shutdown_response(response):
    try:
        sock = _response_socket(response)
        if sock is not None and hasattr(sock, 'shutdown'):
            sock.shutdown(socket.SHUT_RDWR)
    except (OSError, ValueError):
        pass
    try:
        response.close()
    except (OSError, ValueError):
        pass


def _safe_exception_code(exc):
    name = type(exc).__name__
    if name in ('timeout', 'TimeoutError'):
        return 'read_timeout'
    if name in ('URLError',):
        return 'network_error'
    if name in ('IncompleteRead',):
        return 'incomplete_read'
    if name in ('ValueError', 'OSError', 'ConnectionResetError', 'BrokenPipeError'):
        return 'connection_closed'
    return 'stream_error'


def _consume_payload(state, payload, now_ns, workload):
    try:
        event = json.loads(payload)
    except ValueError:
        state.fail('invalid_json')
        return False
    if type(event) is not dict:
        state.fail('event_not_object')
        return False
    with state.lock:
        state.event_count += 1

    choices = event.get('choices')
    if choices is None:
        state.fail('choices_missing')
        return False
    if type(choices) is not list:
        state.fail('choices_not_list')
        return False
    if len(choices) > 1:
        state.fail('choices_multiple_for_n1')
        return False

    chunk_tokens = []
    chunk_finish = None
    chunk_text_seen = False
    first_prompt_ids = None
    for choice in choices:
        if type(choice) is not dict:
            state.fail('choice_not_object')
            return False
        index = choice.get('index')
        if not _is_int(index) or index != 0:
            state.fail('choice_index_invalid')
            return False
        ids = choice.get('token_ids', _MISSING)
        if ids is _MISSING or ids is None:
            state.fail('token_ids_missing')
            return False
        if type(ids) is not list:
            state.fail('token_ids_not_list')
            return False
        for token in ids:
            if not _is_int(token) or token < 0 or token > MAX_TOKEN_ID:
                state.fail('token_id_invalid')
                return False
        chunk_tokens.extend(ids)
        prompt_ids = choice.get('prompt_token_ids')
        if prompt_ids is not None:
            if type(prompt_ids) is not list:
                state.fail('prompt_token_ids_not_list')
                return False
            if any(not _is_int(item) or item < 0 for item in prompt_ids):
                state.fail('prompt_token_ids_invalid')
                return False
            if first_prompt_ids is None:
                first_prompt_ids = len(prompt_ids)
        finish = choice.get('finish_reason')
        if finish is not None:
            if not isinstance(finish, str) or finish not in ALLOWED_FINISH_REASONS:
                state.fail('finish_reason_invalid')
                return False
            if chunk_finish is not None:
                state.fail('finish_duplicate')
                return False
            chunk_finish = finish
        text = choice.get('text')
        if isinstance(text, str) and text:
            chunk_text_seen = True

    usage = event.get('usage')
    usage_completion = None
    usage_advanced = False
    if usage is not None:
        if type(usage) is not dict:
            state.fail('usage_not_object')
            return False
        completion = usage.get('completion_tokens')
        prompt = usage.get('prompt_tokens')
        if completion is not None:
            if not _is_int(completion) or completion < 0 or completion > workload.generation_budget:
                state.fail('usage_completion_invalid')
                return False
            with state.lock:
                previous = state.usage_completion
                if previous is not None and completion < previous:
                    state.fail('usage_regressed')
                    return False
                usage_advanced = ((completion > 0) if previous is None
                                  else (completion > previous))
                state.usage_completion = completion
                state.usage_seen = True
            usage_completion = completion
        if prompt is not None:
            if not _is_int(prompt) or prompt < 0:
                state.fail('usage_prompt_invalid')
                return False
            with state.lock:
                if state.prompt_tokens is not None and state.prompt_tokens != prompt:
                    state.fail('usage_prompt_changed')
                    return False
                state.prompt_tokens = prompt

    with state.lock:
        if first_prompt_ids is not None:
            if (state.prompt_token_ids_len is not None
                    and state.prompt_token_ids_len != first_prompt_ids):
                state.fail('prompt_token_ids_changed')
                return False
            state.prompt_token_ids_len = first_prompt_ids
        if state.finish_reason is not None and chunk_tokens:
            state.fail('token_after_finish')
            return False
        if state.token_count + len(chunk_tokens) > workload.generation_budget:
            state.fail('token_budget_exceeded')
            return False
        if chunk_tokens:
            state.tokens.extend(chunk_tokens)
            state.token_count += len(chunk_tokens)
        if chunk_finish is not None:
            state.finish_reason = chunk_finish
            state.finish_at_ns = now_ns

    if chunk_tokens:
        state.arrivals.append(now_ns)
        if len(chunk_tokens) > 1:
            state.coalesced_arrivals += 1
        if state.first_advance_ns is None:
            state.first_advance_ns = now_ns
        state.progress.set()
    elif usage_advanced and not chunk_tokens:
        state.usage_only_chunks += 1
        if state.first_advance_ns is None:
            state.first_advance_ns = now_ns
        state.progress.set()
    elif usage_completion is not None and not usage_advanced:
        state.duplicate_usage_chunks += 1
    if chunk_text_seen and not chunk_tokens and not usage_advanced:
        state.content_only_chunks += 1
    if chunk_finish is not None:
        state.finished.set()
        state.progress.set()
    return True


_MISSING = object()


def stream_reader(state, url, model, body, workload):
    """Read one live completion stream; retain only hashes/counts/timestamps."""
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('VLLM_API_KEY')
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(
        url + ENDPOINT, json.dumps(body).encode(), headers)
    state.send_at_ns = time.monotonic_ns()
    try:
        response = urllib.request.urlopen(request, timeout=workload.read_timeout_s)
        with state.io_lock:
            if state.stop.is_set():
                _shutdown_response(response)
                return
            state.response = response
        with response:
            data_lines = []
            for raw_line in response:
                if state.stop.is_set():
                    return
                line = raw_line.decode('utf-8', 'replace').rstrip('\r\n')
                if line == '':
                    if data_lines:
                        payload = '\n'.join(data_lines)
                        data_lines = []
                        if payload.strip() == '[DONE]':
                            with state.lock:
                                state.done_marker = True
                            return
                        if not _consume_payload(state, payload,
                                                time.monotonic_ns(), workload):
                            return
                    continue
                if line.startswith(':'):
                    continue
                field, _sep, value = line.partition(':')
                if field == 'data':
                    if value.startswith(' '):
                        value = value[1:]
                    data_lines.append(value)
            # EOF: a complete finish_reason is sufficient; an unfinished stream
            # is explicitly incomplete.
            with state.lock:
                if state.finish_reason is None and state.malformed is None \
                        and state.error is None:
                    state.error = 'closed_without_finish'
                    state.error_at_ns = time.monotonic_ns()
    except Exception as exc:  # noqa: BLE001 - cancellations are not errors
        if not state.stop.is_set():
            with state.lock:
                state.error = _safe_exception_code(exc)
                state.error_at_ns = time.monotonic_ns()
    finally:
        with state.io_lock:
            state.response = None
        state.thread_done = True
        state.finished.set()
        state.progress.set()


def _start_streams(url, model, bodies, cohort, workload, started=None):
    states = []
    for index, body in enumerate(bodies):
        state = StreamRecord(index, cohort[index]['prompt_sha256'],
                             cohort[index]['prompt_chars'], request_sha256(body))
        states.append(state)
    for state, body in zip(states, bodies):
        if started is not None:
            started.append(state)
        state.thread = threading.Thread(
            target=stream_reader, args=(state, url, model, body, workload),
            daemon=True)
        state.thread.start()
    return states


def _wait_open(states, workload):
    deadline = time.monotonic() + workload.open_timeout_s
    while time.monotonic() < deadline:
        if all(state.first_advance_ns is not None
               or state.error is not None
               or state.malformed is not None
               or state.finish_reason is not None for state in states):
            break
        for state in states:
            state.progress.wait(0.01)
            state.progress.clear()
    sealed_at = time.monotonic_ns()
    reasons = []
    opened = []
    for state in states:
        if state.send_at_ns is None:
            reasons.append('stream_not_submitted:%d' % state.index)
        if state.first_advance_ns is None:
            if state.error is not None:
                reasons.append('stream_not_advanced_error:%d' % state.index)
            elif state.malformed is not None:
                reasons.append('stream_not_advanced_malformed:%d' % state.index)
            else:
                reasons.append('stream_not_advanced:%d' % state.index)
        else:
            opened.append(state)
        if (state.finish_at_ns is not None
                and state.finish_at_ns < sealed_at):
            reasons.append('early_finish_before_seal:%d' % state.index)
        if (state.error_at_ns is not None
                and state.error_at_ns < sealed_at):
            reasons.append('stream_error_before_seal:%d' % state.index)
    order = sorted(opened, key=lambda item: item.first_advance_ns)
    return {
        'complete': not reasons,
        'opened_streams': len(opened),
        'advance_order': [state.index for state in order],
        'ttft_ms': [{'stream': state.index,
                     'ttft_ms': _ms(state.first_advance_ns - state.send_at_ns)}
                    for state in order if state.send_at_ns is not None],
        'sealed_at_ns': sealed_at,
        'submissions_after_seal': sum(
            1 for state in states
            if state.send_at_ns is not None and state.send_at_ns > sealed_at),
        'unsubmitted_at_seal': [state.index for state in states
                                if state.send_at_ns is None],
        'reasons': reasons,
        'basis': OPEN_BASIS,
        'streams': [{'stream': state.index,
                     'send_at_ns': state.send_at_ns,
                     'first_advance_at_ns': state.first_advance_ns}
                    for state in states],
    }


def _wait_window(states, starts, workload):
    start_ns = time.monotonic_ns()
    target_ns = start_ns + int(round(workload.window_s * 1e9))
    reason = None
    while True:
        now = time.monotonic_ns()
        early_finish = [state.index for state in states
                        if state.finish_at_ns is not None and state.finish_at_ns < now]
        early_errors = [state.index for state in states
                        if state.error_at_ns is not None and state.error_at_ns < now]
        malformed = [state.index for state in states if state.malformed is not None]
        if early_finish:
            reason = 'early_finish'
        elif early_errors or malformed:
            reason = 'stream_error'
        elif all(state.count() >= starts[state.index] + workload.window_tokens
                 for state in states):
            reason = 'token_bound'
        elif now >= target_ns:
            reason = 'wall_bound'
        if reason is not None:
            break
        wait_s = min(0.01, max(0.0, (target_ns - now) / 1e9))
        for state in states:
            state.progress.wait(wait_s)
            state.progress.clear()
    end_ns = time.monotonic_ns()
    rows = []
    for state in states:
        end_count = state.count()
        rows.append({
            'stream': state.index,
            'start_completion': starts[state.index],
            'end_completion': end_count,
            'completion_delta': end_count - starts[state.index],
        })
    complete = all(row['completion_delta'] >= workload.window_tokens for row in rows)
    early_finish = [state.index for state in states
                    if state.finish_at_ns is not None and state.finish_at_ns < end_ns]
    early_errors = [state.index for state in states
                    if (state.error_at_ns is not None and state.error_at_ns < end_ns)
                    or state.malformed is not None]
    return {
        'measured': True,
        'start_ns': start_ns,
        'end_ns': end_ns,
        'target_end_ns': target_ns,
        'wall_s': round((end_ns - start_ns) / 1e9, 6),
        'reason': reason,
        'complete': bool(complete and not early_finish and not early_errors),
        'streams': rows,
        'early_finish': early_finish,
        'errors_before_end': early_errors,
        'client_visible': True,
        'basis': WINDOW_BASIS,
        'clock_uncertainty': {
            'driver_boundary': 'time.monotonic_ns',
            'engine_clock_offset_unknown': True,
            'note': CLOCK_NOTE,
        },
    }


def _wait_finish(states, workload):
    deadline = time.monotonic() + workload.finish_timeout_s
    while time.monotonic() < deadline:
        if all(state.has_finished() or state.error is not None
               or state.malformed is not None for state in states):
            break
        for state in states:
            state.progress.wait(0.02)
            state.progress.clear()
    timed_out = [state.index for state in states
                 if not state.has_finished() and state.error is None
                 and state.malformed is None]
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return timed_out


def _cancel_streams(states, workload):
    alive = []
    for state in states:
        if not state.cancel(workload.join_timeout_s):
            alive.append(state.index)
    return alive


def _stop_streams(record, states, workload):
    """Actively cancel every owned stream and record any reader that survived."""
    alive = _cancel_streams(states, workload)
    record['readers_alive_after_join'] = alive
    if alive:
        record['reader_join_timeouts'] = sorted(set(
            record.get('reader_join_timeouts', []) + alive))
        record['abort_run'] = True
        record['complete'] = False
        _mark(record, 'inconclusive:reader_alive_after_join')
    return alive


# --- receipt redaction / evidence -------------------------------------------

def _redact_text(value):
    if not isinstance(value, str) or not value:
        return {'present': False, 'sha256': None, 'chars': 0}
    return {'present': True, 'sha256': _sha256_text(value), 'chars': len(value)}


def _redact_receipt(receipt):
    """Bounded, payload-free copy of an already schema-sanitized receipt."""
    if type(receipt) is not dict:
        return receipt
    out = dict(receipt)
    if out.get('status') not in KNOWN_STATUS:
        out['status'] = 'invalid_status'
    for key in ('run_id', 'cohort_id'):
        value = out.get(key)
        if value is not None and not (isinstance(value, str)
                                      and re.fullmatch(r'[A-Za-z0-9_-]{1,80}',
                                                       value)):
            out[key] = 'invalid_label'
    for key in ('route_before', 'route_after'):
        if key in out and out[key] not in ('full', 'split'):
            out[key] = 'invalid'
    for key in ('route_generation_before', 'route_generation_after',
                'engine_epoch', 'storage_epoch', 'command_seq', 'repeat',
                'observed_clock_ns'):
        if key in out and not _is_int(out[key]):
            out[key] = None
    ranks = out.get('expected_rank_set')
    if type(ranks) is list:
        out['expected_rank_set'] = [value if _is_int(value) else None
                                    for value in ranks][:8]
    if 'clock_basis' in out:
        out['clock_basis_sha256'] = _redact_text(out.pop('clock_basis'))
    eligibility = out.get('decode_eligibility')
    if type(eligibility) is dict:
        clean = dict(eligibility)
        if 'reason' in clean:
            clean['reason_sha256'] = _redact_text(clean.pop('reason'))
        out['decode_eligibility'] = clean
    interval = out.get('interval')
    if type(interval) is dict:
        clean = dict(interval)
        if 'observer_basis' in clean:
            clean['observer_basis_sha256'] = _redact_text(clean.pop('observer_basis'))
        out['interval'] = clean
    ranks = out.get('ranks')
    if type(ranks) is list:
        clean_ranks = []
        for entry in ranks:
            if type(entry) is not dict:
                clean_ranks.append(entry)
                continue
            clean = dict(entry)
            if clean.get('status') not in ('ok',):
                clean['status'] = 'invalid'
            ack = clean.get('transition_ack')
            if type(ack) is dict:
                clean_ack = dict(ack)
                if clean_ack.get('ack') != ACK:
                    clean_ack['ack'] = 'invalid'
                if clean_ack.get('mode') != PROBE_MODE:
                    clean_ack['mode'] = 'invalid'
                if clean_ack.get('route') not in ('full', 'split'):
                    clean_ack['route'] = 'invalid'
                clean['transition_ack'] = clean_ack
            clean_ranks.append(clean)
        out['ranks'] = clean_ranks
    barriers = out.get('barrier_replies')
    if type(barriers) is list:
        clean_barriers = []
        for row in barriers:
            if type(row) is not dict:
                clean_barriers.append(row)
                continue
            clean = dict(row)
            if clean.get('ack') != BARRIER_ACK:
                clean['ack'] = 'invalid'
            if clean.get('mode') != PROBE_MODE:
                clean['mode'] = 'invalid'
            if clean.get('route') not in ('full', 'split'):
                clean['route'] = 'invalid'
            clean_barriers.append(clean)
        out['barrier_replies'] = clean_barriers
    pause = out.get('pause')
    if type(pause) is dict and pause.get('mode') != 'keep':
        clean = dict(pause)
        clean['mode'] = 'invalid'
        out['pause'] = clean
    return out


def _safe_reasons(reasons):
    out = []
    for reason in reasons or []:
        text = str(reason)
        if text.startswith('inconclusive:controller_status:'):
            text = 'inconclusive:controller_status_non_ok'
        elif text.startswith('inconclusive:controller_transport:'):
            text = 'inconclusive:controller_transport'
        elif text.startswith('inconclusive:controller_exception:'):
            text = 'inconclusive:controller_exception'
        text = text[:MAX_REASON_CHARS]
        if text not in out:
            out.append(text)
    return out


def _safe_transport(transport):
    if type(transport) is not dict:
        return None
    out = {}
    for key in ('rc', 'seconds', 'stdout_len', 'stdout_sha256', 'stderr_len',
                'timeout'):
        value = transport.get(key)
        if key == 'seconds':
            if _is_number(value):
                out[key] = round(float(value), 3)
        elif key == 'stdout_sha256':
            if isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value):
                out[key] = value
        elif key == 'rc':
            if _is_int(value) or value is None:
                out[key] = value
        elif key in ('stdout_len', 'stderr_len'):
            if _is_int(value) and value >= 0:
                out[key] = value
        else:
            if type(value) is bool:
                out[key] = value
    kind = transport.get('kind')
    out['kind'] = kind if kind in KNOWN_TRANSPORT_KINDS else 'other'
    error = transport.get('error')
    if error is not None:
        out['error'] = error if error in KNOWN_TRANSPORT_ERRORS else 'other'
    return out


def _controller_call(controller, cfg, ctx, action, target_route):
    record = protocol.controller_call(controller, cfg, ctx, action, target_route)
    record['reasons'] = _safe_reasons(record.get('reasons'))
    record['transport'] = _safe_transport(record.get('transport'))
    receipt = _redact_receipt(record.get('receipt'))
    record['receipt'] = receipt
    record['retained_receipt_sha256'] = (
        _sha256_json(receipt) if type(receipt) in (dict, list) else None)
    record['receipt_hash_basis'] = (
        'receipt_sha256 binds the separately retained receipt_core; '
        'retained_receipt_sha256 binds the redacted display receipt')
    return record


def _all_reasons(runs):
    out = []
    for run in runs:
        for reason in run.get('reasons') or []:
            if reason not in out:
                out.append(reason)
    return out


def _severity_of(reasons):
    if any(str(reason).startswith('contract:') for reason in reasons):
        return 'contract'
    return 'inconclusive'


def _measurement_proof(receipt, route, split_replays_required):
    """Actual per-rank decode/route/replay evidence at pause_end.

    Missing counters are INCONCLUSIVE. A contradictory route or zero/regressed
    split replay delta within this window is CONTRACT. Lifetime replay counts
    alone never qualify the split run.
    """
    reasons = []
    if type(receipt) is not dict:
        return ['inconclusive:measurement_receipt_missing']
    observations = receipt.get('worker_observations')
    if type(observations) is not list:
        return ['inconclusive:worker_observations_missing']
    if len(observations) != len(EXPECTED_RANKS):
        reasons.append('inconclusive:worker_observations_rank_coverage')
    seen = set()
    for row in observations:
        if type(row) is not dict:
            reasons.append('contract:worker_observation_not_object')
            continue
        rank = row.get('rank')
        if not _is_int(rank) or rank not in EXPECTED_RANKS or rank in seen:
            reasons.append('contract:worker_observation_rank_invalid')
            continue
        seen.add(rank)
        counters = row.get('counters')
        if type(counters) is not dict:
            reasons.append('inconclusive:worker_counters_missing:%d' % rank)
            continue
        observed_route = counters.get('route')
        if observed_route not in ('full', 'split'):
            reasons.append('inconclusive:worker_route_missing:%d' % rank)
        elif observed_route != route:
            reasons.append('contract:worker_route_mismatch:%d' % rank)
        batches = counters.get('batch_observations')
        if type(batches) is not dict or not _is_int(batches.get('decode_batches')):
            reasons.append('inconclusive:decode_observation_missing:%d' % rank)
    if seen != set(EXPECTED_RANKS):
        reasons.append('inconclusive:worker_observations_rank_coverage')
    interval = receipt.get('interval')
    if type(interval) is not dict:
        reasons.append('inconclusive:interval_missing')
    else:
        if split_replays_required:
            snapshots = []
            for key in ('counters_before', 'counters_after'):
                rows = interval.get(key)
                values = {}
                if type(rows) is list:
                    for row in rows:
                        if type(row) is not dict:
                            continue
                        rank = row.get('rank')
                        if not _is_int(rank) or rank not in EXPECTED_RANKS:
                            reasons.append('contract:split_replay_rank_invalid')
                            continue
                        if rank in values:
                            reasons.append('contract:split_replay_rank_duplicate')
                        counters = row.get('counters')
                        values[rank] = (counters.get('replays')
                                        if type(counters) is dict else None)
                snapshots.append(values)
            for rank in EXPECTED_RANKS:
                old, new = (values.get(rank) for values in snapshots)
                if type(old) is bool or type(new) is bool:
                    reasons.append('contract:split_replay_type:%d' % rank)
                elif not _is_int(old) or not _is_int(new):
                    reasons.append('inconclusive:split_replay_unproven:%d' % rank)
                elif old < 0 or new < old:
                    reasons.append('contract:split_replay_regressed:%d' % rank)
                elif new == old:
                    reasons.append('contract:split_replay_zero:%d' % rank)
        before = _counter_map(interval.get('counters_before'))
        after = _counter_map(interval.get('counters_after'))
        for rank in EXPECTED_RANKS:
            if rank not in before or rank not in after:
                reasons.append('inconclusive:interval_decode_progress_missing:%d' % rank)
            elif after[rank] <= before[rank]:
                reasons.append('inconclusive:interval_decode_progress_missing:%d' % rank)
        invalidation = interval.get('invalidation')
        if type(invalidation) is not dict:
            reasons.append('inconclusive:interval_invalidation_missing')
        else:
            for key in ('prefill', 'mixed', 'unsupported', 'preempted'):
                value = invalidation.get(key)
                if not _is_int(value) or value < 0:
                    reasons.append('inconclusive:interval_invalidation_missing:' + key)
                elif value > 0:
                    reasons.append('contract:interval_invalidated:' + key)
    return reasons


def _counter_map(rows):
    result = {}
    if type(rows) is not list:
        return result
    for row in rows:
        if type(row) is not dict:
            continue
        rank = row.get('rank')
        steps = row.get('decode_steps')
        if _is_int(rank) and rank in EXPECTED_RANKS and _is_int(steps) and steps >= 0:
            result[rank] = steps
    return result


# --- one run ----------------------------------------------------------------

def _new_run_record(run_index, label, route, cohort_id):
    return {
        'run': run_index,
        'label': label,
        'route': route,
        'cohort_id': cohort_id,
        'status': 'started',
        'calls': [],
        'open': None,
        'window': None,
        'interval': None,
        'cleanup': None,
        'requests': [],
        'streams': [],
        'performance': None,
        'reasons': [],
        'contract': False,
        'complete': False,
        'abort_run': False,
        'not_run': False,
        '_tokens': None,
    }


def _mark(record, reason):
    if reason not in record['reasons']:
        record['reasons'].append(reason)


def _finish_run(record):
    record['contract'] = any(
        str(reason).startswith('contract:') for reason in record['reasons'])
    record['complete'] = (not record['reasons'] and not record['abort_run']
                          and record['status'] == 'completed')
    return record


def _call(record, ctx, controller, cfg, action, target_route):
    call = _controller_call(controller, cfg, ctx, action, target_route)
    # Shared controller transport alone owns acceptance and chain advancement.
    record['calls'].append(call)
    return call


def _call_by_action(record, action, last=False):
    matches = [call for call in record['calls'] if call.get('action') == action]
    if not matches:
        return None
    return matches[-1] if last else matches[0]


def _trajectory_rows(states, window, workload):
    rows = []
    for state in states:
        with state.lock:
            tokens = list(state.tokens)
            finish_reason = state.finish_reason
            usage_completion = state.usage_completion
            prompt_tokens = state.prompt_tokens
            prompt_ids_len = state.prompt_token_ids_len
            first_advance = state.first_advance_ns
            finish_at = state.finish_at_ns
            error = state.error
            malformed = state.malformed
            done_marker = state.done_marker
            arrivals = len(state.arrivals)
            coalesced = state.coalesced_arrivals
            usage_only = state.usage_only_chunks
            duplicate_usage = state.duplicate_usage_chunks
            content_only = state.content_only_chunks
            event_count = state.event_count
            thread_done = state.thread_done
        window_row = next((row for row in (window or {}).get('streams') or []
                           if row.get('stream') == state.index), {})
        problems = []
        if malformed is not None:
            problems.append('malformed:' + malformed)
        if error is not None:
            problems.append('error:' + error)
        if finish_reason is None:
            problems.append('no_finish')
        elif finish_reason != workload.expected_finish_reason:
            problems.append('early_eos' if finish_reason == 'stop'
                            else 'finish_mismatch:' + finish_reason)
        if len(tokens) != workload.expected_output_tokens:
            problems.append('length_mismatch')
        if not state.usage_seen:
            problems.append('usage_missing')
        elif usage_completion != len(tokens):
            problems.append('usage_token_mismatch')
        rows.append({
            'stream': state.index,
            'prompt_sha256': state.prompt_sha256,
            'prompt_chars': state.prompt_chars,
            'request_sha256': state.request_sha256,
            'trajectory_sha256': _sha256_json(tokens),
            'length': len(tokens),
            'finish_reason': finish_reason,
            'done_marker': done_marker,
            'first_advance_at_ns': first_advance,
            'finish_at_ns': finish_at,
            'usage_completion': usage_completion,
            'prompt_tokens': prompt_tokens,
            'prompt_token_ids_len': prompt_ids_len,
            'arrivals': arrivals,
            'coalesced_arrivals': coalesced,
            'usage_only_chunks': usage_only,
            'duplicate_usage_chunks': duplicate_usage,
            'content_only_chunks': content_only,
            'event_count': event_count,
            'reader_joined': thread_done,
            'window': {
                'start_completion': window_row.get('start_completion'),
                'end_completion': window_row.get('end_completion'),
                'completion_delta': window_row.get('completion_delta'),
            },
            'complete': not problems,
            'problems': problems,
            '_tokens': tokens,
        })
    return rows


def measure_run(cfg, controller, ctx, run_index, label, route, cohort):
    """Run one route; an unexpected exception still cancels and keeps evidence."""
    record = _new_run_record(run_index, label, route, None)
    started = []
    try:
        return _measure_run_inner(record, cfg, controller, ctx, run_index, label,
                                  route, cohort, started)
    except Exception as exc:  # noqa: BLE001 - keep partial evidence, no raw body
        record['status'] = 'exception'
        _mark(record, 'inconclusive:run_exception:' + type(exc).__name__)
        record['abort_run'] = True
        if started:
            _stop_streams(record, started, cfg.workload)
            if not record.get('streams'):
                record['streams'] = _trajectory_rows(
                    started, record.get('window'), cfg.workload)
        return _finish_run(record)
    finally:
        _stop_streams(record, started, cfg.workload)


def _measure_run_inner(record, cfg, controller, ctx, run_index, label, route,
                       cohort, started):
    workload = cfg.workload
    record['started_at_ns'] = time.monotonic_ns()
    cohort_id = 'cohort-' + _sha256_json(
        {'run_id': cfg.run_id, 'run': run_index, 'label': label})[:24]
    record['cohort_id'] = cohort_id
    ctx.update({'repeat': run_index, 'cohort_id': cohort_id})

    salt = build_cache_salt(cfg.run_id, run_index)
    bodies = [request_body(ctx['model'], row['prompt'], workload, salt)
              for row in cohort]
    record['requests'] = [{
        'stream': index,
        'prompt_sha256': cohort[index]['prompt_sha256'],
        'prompt_chars': cohort[index]['prompt_chars'],
        'request_sha256': request_sha256(body),
        'cache_salt_sha256': _sha256_text(salt),
    } for index, body in enumerate(bodies)]
    record['request_set_sha256'] = _sha256_json(
        [row['request_sha256'] for row in record['requests']])
    record['cache_salt_sha256'] = _sha256_text(salt)

    # 1. prepare on full binds the engine incarnation before any request.
    prepare = _call(record, ctx, controller, cfg, 'prepare', 'full')
    if not prepare.get('ok'):
        _mark(record, prepare.get('severity', 'inconclusive') + ':prepare_receipt_invalid')
        record['status'] = 'prepare_failed'
        record['abort_run'] = True
        return _finish_run(record)

    states = _start_streams(ctx['url'], ctx['model'], bodies, cohort, workload,
                            started)
    open_info = _wait_open(states, workload)
    record['open'] = open_info
    if not open_info['complete']:
        for code in open_info['reasons']:
            if code.startswith('stream_not_advanced_malformed'):
                _mark(record, 'contract:open_malformed:' + code)
            elif code.startswith('early_finish_before_seal'):
                _mark(record, 'contract:' + code)
            else:
                _mark(record, 'inconclusive:open_incomplete:' + code)
        record['status'] = 'open_incomplete'
        record['abort_run'] = True
        _stop_streams(record, states, workload)
        record['streams'] = _trajectory_rows(states, None, workload)
        return _finish_run(record)
    if open_info['submissions_after_seal']:
        _mark(record, 'contract:request_after_seal')
    if open_info['unsubmitted_at_seal']:
        _mark(record, 'inconclusive:open_incomplete:unsubmitted_at_seal')
        record['status'] = 'open_incomplete'
        record['abort_run'] = True
        _stop_streams(record, states, workload)
        record['streams'] = _trajectory_rows(states, None, workload)
        return _finish_run(record)

    # 2. completed pause with actual cohort eligibility; no transition guess.
    pause_action = 'pause_enter_split' if route == 'split' else 'pause_full'
    pause = _call(record, ctx, controller, cfg, pause_action, route)
    if not pause.get('ok'):
        severity = pause.get('severity', 'inconclusive')
        _mark(record, severity + ':pause_receipt_invalid')
        record['status'] = 'pause_failed'
        record['abort_run'] = True
        record['cleanup'] = {
            'attempted': False,
            'restore_before_drain': False,
            'reason': 'pause receipt did not prove the route state; refusing to '
                      'resume or restore an unknown/partially transitioned route',
        }
        _stop_streams(record, states, workload)
        record['streams'] = _trajectory_rows(states, None, workload)
        return _finish_run(record)

    # 3. accepted resume marks the measurement start.
    resume = _call(record, ctx, controller, cfg, 'resume', route)
    if not resume.get('ok'):
        _mark(record, resume.get('severity', 'inconclusive') + ':resume_receipt_invalid')
        record['status'] = 'resume_failed'
        record['abort_run'] = True
        record['cleanup'] = {
            'attempted': False,
            'restore_before_drain': False,
            'reason': 'resume receipt did not prove the boundary; route state is '
                      'not safe to resume or restore from this client',
        }
        _stop_streams(record, states, workload)
        record['streams'] = _trajectory_rows(states, None, workload)
        return _finish_run(record)

    starts = {state.index: state.count() for state in states}
    window = _wait_window(states, starts, workload)
    record['window'] = window
    if not window['complete']:
        if window['reason'] == 'wall_bound':
            _mark(record, 'inconclusive:window_wall_bound_before_token_bound')
        for index in window['early_finish']:
            _mark(record, 'contract:early_finish:%d' % index)
        for index in window['errors_before_end']:
            state = next((item for item in states if item.index == index), None)
            if state is not None and state.malformed is not None:
                _mark(record, 'contract:stream_malformed:%d:%s'
                      % (index, state.malformed))
            else:
                _mark(record, 'inconclusive:stream_error_before_window_end:%d' % index)

    # 4. pause_end before any cleanup; interval evidence is separate.
    pause_end = _call(record, ctx, controller, cfg, 'pause_end', route)
    pause_end_receipt = pause_end.get('receipt')
    measurement_reasons = []
    if not pause_end.get('ok'):
        measurement_reasons.extend(pause_end.get('reasons') or [])
    measurement_reasons.extend(
        _measurement_proof(pause_end_receipt, route, route == 'split'))
    for reason in measurement_reasons:
        _mark(record, reason)
    if type(pause_end_receipt) is dict:
        interval = pause_end_receipt.get('interval') or {}
        record['interval'] = {
            'observer_basis_sha256': interval.get('observer_basis_sha256'),
            'counters_before': interval.get('counters_before'),
            'counters_after': interval.get('counters_after'),
            'invalidation': interval.get('invalidation'),
            'valid': not measurement_reasons,
            'reasons': list(measurement_reasons),
        }

    # 5. cleanup needs an independently proven completed pause, not interval
    #    validity.  Never leave/resume an unproven route.
    lifecycle_ok = pause_end.get('lifecycle_ok') is True
    lifecycle_reasons = pause_end.get('lifecycle_reasons') or [
        'inconclusive:shared_lifecycle_proof_unavailable']
    if not lifecycle_ok:
        for reason in lifecycle_reasons:
            _mark(record, reason)
        record['cleanup'] = {
            'attempted': False,
            'restore_before_drain': False,
            'reason': 'completed pause/route lifecycle was not proven; refusing '
                      'to leave or resume an unknown route',
        }
        record['status'] = 'pause_end_failed'
        record['abort_run'] = True
        _stop_streams(record, states, workload)
        record['streams'] = _trajectory_rows(states, window, workload)
        return _finish_run(record)

    leave = _call(record, ctx, controller, cfg, 'leave_full', 'full')
    cleanup_resume = None
    restore_ok = False
    if leave.get('lifecycle_ok') is True:
        cleanup_resume = _call(record, ctx, controller, cfg, 'resume', 'full')
        restore_ok = cleanup_resume.get('lifecycle_ok') is True
    record['cleanup'] = {
        'attempted': True,
        'restore_before_drain': bool(restore_ok),
        'leave_ok': leave.get('lifecycle_ok') is True,
        'resume_ok': bool(cleanup_resume and cleanup_resume.get('lifecycle_ok') is True),
        'reason': (None if restore_ok else
                   'leave_full/cleanup resume did not prove the full route'),
    }
    if not restore_ok:
        severity = (leave.get('severity') if not leave.get('ok')
                    else cleanup_resume.get('severity', 'inconclusive')
                    if cleanup_resume is not None else 'inconclusive')
        _mark(record, severity + ':cleanup_restore_invalid')
        record['status'] = 'cleanup_failed'
        record['abort_run'] = True
        _stop_streams(record, states, workload)
        record['streams'] = _trajectory_rows(states, window, workload)
        return _finish_run(record)

    # 6. full is restored and resumed while the SAME requests stay live.
    finish_timeout = []
    if not (window['early_finish'] or window['errors_before_end']):
        finish_timeout = _wait_finish(states, workload)
        for index in finish_timeout:
            _mark(record, 'inconclusive:finish_timeout:%d' % index)
    alive = _stop_streams(record, states, workload)
    if alive:
        _mark(record, 'inconclusive:reader_alive_after_join')
    record['readers_alive_after_join'] = alive
    record['streams'] = _trajectory_rows(states, window, workload)
    for row in record['streams']:
        for problem in row['problems']:
            code, _sep, detail = problem.partition(':')
            if code == 'early_eos':
                _mark(record, 'contract:early_eos:%d' % row['stream'])
            elif code == 'finish_mismatch':
                _mark(record, 'contract:finish_mismatch:%d:%s'
                      % (row['stream'], detail))
            elif code == 'length_mismatch':
                _mark(record, 'contract:length_mismatch:%d' % row['stream'])
            elif code == 'no_finish':
                _mark(record, 'inconclusive:no_finish:%d' % row['stream'])
            elif code == 'usage_missing':
                _mark(record, 'inconclusive:usage_missing:%d' % row['stream'])
            elif code == 'usage_token_mismatch':
                _mark(record, 'contract:usage_token_mismatch:%d' % row['stream'])
            elif code == 'error':
                _mark(record, 'inconclusive:stream_error:%d:%s'
                      % (row['stream'], detail))
            elif code == 'malformed':
                _mark(record, 'contract:stream_malformed:%d:%s'
                      % (row['stream'], detail))
    record['performance'] = _performance(record, states, workload)
    record['status'] = 'completed'
    return _finish_run(record)


def _performance(record, states, workload):
    open_info = record.get('open') or {}
    window = record.get('window') or {}
    ttfts = [row['ttft_ms'] for row in open_info.get('ttft_ms') or []
             if _is_number(row.get('ttft_ms'))]
    arrival_rows = []
    for state in states:
        with state.lock:
            arrivals = list(state.arrivals)
        gaps = [_ms(b - a) for a, b in zip(arrivals, arrivals[1:])]
        arrival_rows.append({
            'stream': state.index,
            'arrivals': len(arrivals),
            'gap_ms_p50': _pct(gaps, 0.5),
            'gap_ms_p95': _pct(gaps, 0.95),
            'gap_ms_p99': _pct(gaps, 0.99),
            'gap_ms_max': max(gaps) if gaps else None,
        })
    return {
        'reported_separately': True,
        'no_speed_threshold': True,
        'verdict_independent': True,
        'open_wall_s': round(
            ((open_info.get('sealed_at_ns') or 0) - (record.get('started_at_ns') or 0))
            / 1e9, 6) if open_info.get('sealed_at_ns') else None,
        'window_wall_s': window.get('wall_s'),
        'window_tokens_total': sum(
            row.get('completion_delta') or 0 for row in window.get('streams') or []),
        'ttft_ms_p50': _pct(ttfts, 0.5),
        'ttft_ms_p95': _pct(ttfts, 0.95),
        'streams': arrival_rows,
        'coalescing_note': COALESCING_NOTE,
    }


def _pct(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    return round(values[min(len(values) - 1, int(q * len(values)))], 3)


# --- comparison / verdict ---------------------------------------------------

def _compare_rows(left_run, right_run):
    """Per-stream mismatch counts and first positions only (no tokens)."""
    left = {row['stream']: row for row in left_run.get('streams') or []}
    right = {row['stream']: row for row in right_run.get('streams') or []}
    mismatches = []
    for index in sorted(set(left) | set(right)):
        a = left.get(index)
        b = right.get(index)
        if a is None or b is None:
            mismatches.append({'stream': index, 'a_present': a is not None,
                               'b_present': b is not None, 'differing_positions': None,
                               'first_position': None})
            continue
        ta = a.get('_tokens')
        tb = b.get('_tokens')
        if ta is None or tb is None:
            mismatches.append({'stream': index, 'a_present': True, 'b_present': True,
                               'differing_positions': None, 'first_position': None})
            continue
        prefix = min(len(ta), len(tb))
        positions = [position for position in range(prefix)
                     if ta[position] != tb[position]]
        differing = len(positions) + abs(len(ta) - len(tb))
        if differing == 0:
            continue
        first = positions[0] if positions else prefix
        mismatches.append({
            'stream': index,
            'a_length': len(ta),
            'b_length': len(tb),
            'a_sha256': a.get('trajectory_sha256'),
            'b_sha256': b.get('trajectory_sha256'),
            'differing_positions': differing,
            'first_position': first,
        })
    return mismatches


def compare_runs(runs):
    a, b, c = runs
    result = {
        'basis': 'exact integer token-ID equality; no tolerance',
        'controls': {'comparable': False},
        'split': {'comparable': False, 'used_for_verdict': False},
    }
    if not (a.get('complete') and c.get('complete')):
        result['controls']['reason'] = 'control_run_incomplete'
    else:
        mismatches = _compare_rows(a, c)
        result['controls'] = {
            'comparable': True,
            'equal': not mismatches,
            'mismatches': mismatches,
            'a_label': a.get('label'),
            'c_label': c.get('label'),
        }
    if result['controls'].get('equal') is True:
        if not b.get('complete'):
            result['split']['reason'] = 'split_run_incomplete'
        else:
            mismatches = _compare_rows(a, b)
            result['split'] = {
                'comparable': True,
                'used_for_verdict': True,
                'equal': not mismatches,
                'mismatches': mismatches,
                'a_label': a.get('label'),
                'b_label': b.get('label'),
            }
    elif result['controls'].get('comparable'):
        result['split']['reason'] = 'controls_differ_not_used'
    return result


def decide(runs, comparison):
    run_reasons = _all_reasons(runs)
    contract_reasons = [reason for reason in run_reasons
                        if str(reason).startswith('contract:')]
    if contract_reasons:
        return 'FAIL', contract_reasons
    if any(not run.get('complete') for run in runs):
        reasons = [reason for reason in run_reasons
                   if not str(reason).startswith('contract:')]
        return 'INCONCLUSIVE', reasons
    controls = comparison.get('controls') or {}
    if controls.get('equal') is not True:
        return 'INCONCLUSIVE', ['inconclusive:control_instability']
    split = comparison.get('split') or {}
    if split.get('equal') is True:
        return 'PASS', []
    if split.get('equal') is False:
        return 'FAIL', ['contract:split_trajectory_differs']
    return 'INCONCLUSIVE', ['inconclusive:split_not_comparable']


def exit_code(verdict):
    return {'PASS': 0, 'FAIL': 1}.get(verdict, 2)


def _public_run(run):
    """Drop in-memory token arrays before the artifact is written."""
    out = {key: value for key, value in run.items() if key != '_tokens'}
    streams = []
    for row in out.get('streams') or []:
        streams.append({key: value for key, value in row.items() if key != '_tokens'})
    out['streams'] = streams
    return out


# --- run / artifacts --------------------------------------------------------

def _not_run_record(run_index, label, route, reason):
    record = _new_run_record(run_index, label, route, None)
    record['not_run'] = True
    record['status'] = 'not_run'
    _mark(record, 'inconclusive:not_run:' + reason)
    return _finish_run(record)


def run(cfg, url, model, controller, source=None):
    t0_ns = time.monotonic_ns()
    wall_anchor_ns = time.time_ns()
    if source is not None:
        _set_source_artifact(source)
    clock = {
        'monotonic_anchor_ns': t0_ns,
        'wall_anchor_ns': wall_anchor_ns,
        'basis': CLOCK_NOTE,
    }
    cohort = build_cohort(cfg.workload)
    ctx = {
        'model': model,
        'url': url,
        'run_id': cfg.run_id,
        'streams': cfg.workload.streams,
        'repeat': -1,
        'cohort_id': None,
        'command_seq': 0,
        'prior_receipt': None,
        'engine_epoch': None,
        'storage_epoch': None,
        'route': 'full',
        'route_generation': None,
    }
    runs = []
    abort_reason = None
    for run_index, (label, route) in enumerate(zip(RUN_LABELS, ROUTE_PLAN)):
        if abort_reason is not None:
            runs.append(_not_run_record(run_index, label, route, abort_reason))
            continue
        try:
            record = measure_run(cfg, controller, ctx, run_index, label, route,
                                 cohort)
        except Exception as exc:  # noqa: BLE001 - preserve evidence, no raw body
            record = _new_run_record(run_index, label, route, None)
            record['started_at_ns'] = time.monotonic_ns()
            record['status'] = 'exception'
            _mark(record, 'inconclusive:run_exception:' + type(exc).__name__)
            record['abort_run'] = True
            record = _finish_run(record)
        runs.append(record)
        if record.get('abort_run'):
            abort_reason = 'run_%d_%s' % (run_index, record.get('status', 'aborted'))
    comparison = compare_runs(runs)
    verdict, verdict_reasons = decide(runs, comparison)
    source_artifact = _api_source_artifact()
    if source_artifact.get('verified') is not True and verdict != 'FAIL':
        verdict = 'INCONCLUSIVE'
        verdict_reasons = list(verdict_reasons) + [
            'inconclusive:api_source_unverified']
    data = {
        'schema': CLIENT_SCHEMA,
        'client': CLIENT_NAME,
        'api_source': source_artifact,
        'clock': clock,
        'workload': {
            'name': cfg.workload.name,
            'manifest_sha256': cfg.workload.manifest_sha256,
            'streams': cfg.workload.streams,
            'prompt_seed': cfg.workload.prompt_seed,
            'prompt_tokens_target': cfg.workload.prompt_tokens,
            'prompt_builder': 'agent_replay.text/v1',
            'generation_budget': cfg.workload.generation_budget,
            'expected_output_tokens': cfg.workload.expected_output_tokens,
            'expected_finish_reason': cfg.workload.expected_finish_reason,
            'sampling': {
                'seed': cfg.workload.sampling_seed,
                'temperature': cfg.workload.temperature,
                'top_p': cfg.workload.top_p,
                'ignore_eos': True,
                'min_tokens': 'equal_max',
                'n': 1,
                'return_token_ids': True,
            },
            'bounds': {
                'window_s': cfg.workload.window_s,
                'window_tokens': cfg.workload.window_tokens,
            },
            'timeouts': {
                'open_s': cfg.workload.open_timeout_s,
                'pause_s': cfg.workload.pause_timeout_s,
                'read_s': cfg.workload.read_timeout_s,
                'drain_s': cfg.workload.drain_timeout_s,
                'join_s': cfg.workload.join_timeout_s,
                'finish_s': cfg.workload.finish_timeout_s,
                'controller_s': cfg.workload.controller_timeout_s,
            },
            'fresh_namespace_per_run': cfg.workload.fresh_namespace_per_run,
            'prompts': [{
                'stream': row['stream'],
                'sha256': row['prompt_sha256'],
                'chars': row['prompt_chars'],
            } for row in cohort],
        },
        'route_plan': list(ROUTE_PLAN),
        'run_id': cfg.run_id,
        'model_sha256': _sha256_text(str(model)),
        'runs': runs,
        'comparison': comparison,
        'verdict': verdict,
        'verdict_reasons': verdict_reasons,
        'notes': {
            'correctness': CORRECTNESS_NOTE,
            'evidence_boundary': 'prompts, token arrays and generated text stay in '
                                 'memory only; only hashes, lengths, counts and fixed '
                                 'reason codes are persisted',
            'receipt_hash_basis': 'each call retains the exact canonical receipt_core '
                                  'and chain hash, plus a separate redacted display '
                                  'receipt and its recomputable hash',
        },
    }
    data['runs'] = [_public_run(record) for record in runs]
    return data


def _api_source_artifact():
    # verify_sources is called by main; tests may call run() directly.
    return dict(_LAST_SOURCE or {
        'image_sha256': API_IMAGE_SHA256,
        'provenance': 'reference/PROVENANCE-completion-api.txt',
        'facts': dict(PROTOCOL_FACTS),
        'files': {}, 'available': None, 'verified': None, 'unavailable': [],
    })


_LAST_SOURCE = None


def _set_source_artifact(source):
    global _LAST_SOURCE
    _LAST_SOURCE = source


def write_output(path, data):
    """The one exclusive writer; never overwrites an existing artifact."""
    with open(path, 'x') as handle:
        json.dump(data, handle, indent=1)


def _plan_only(reasons, source=None):
    return {
        'schema': CLIENT_SCHEMA,
        'client': CLIENT_NAME,
        'api_source': source or _api_source_artifact(),
        'clock': None,
        'workload': {'status': 'unavailable', 'reasons': list(reasons)},
        'route_plan': list(ROUTE_PLAN),
        'runs': [],
        'comparison': None,
        'verdict': 'INCONCLUSIVE',
        'verdict_reasons': ['inconclusive:' + str(reason) for reason in reasons],
        'notes': {'correctness': CORRECTNESS_NOTE,
                  'plan_only': 'no requests were issued and no settings were invented'},
    }


def _write_or_report(path, data):
    try:
        write_output(path, data)
        return True
    except FileExistsError:
        return False
    except OSError:
        return False


def main(argv, controller=None, env=None):
    env = os.environ if env is None else env
    if len(argv) != 3:
        print('usage: e3_decode_correctness.py URL MODEL OUT.json', file=sys.stderr)
        return 2
    url, model, out_path = argv
    if Path(out_path).exists() or Path(out_path).is_symlink():
        print('e3_decode_correctness refuses an existing output artifact',
              file=sys.stderr)
        return 2

    source, source_problems = verify_sources(env)
    _set_source_artifact(source)
    if source_problems:
        data = _plan_only(source_problems, source)
        _write_or_report(out_path, data)
        print('e3_decode_correctness ' + json.dumps(
            {'verdict': data['verdict'], 'reasons': data['verdict_reasons']},
            sort_keys=True))
        return 2

    try:
        cfg = config_from_env(env, out_path)
    except ConfigError as exc:
        data = _plan_only(exc.reasons, source)
        _write_or_report(out_path, data)
        print('e3_decode_correctness ' + json.dumps(
            {'verdict': data['verdict'], 'reasons': data['verdict_reasons']},
            sort_keys=True))
        return 2

    if controller is None:
        command = env.get('CONTROLLER_CMD', '')
        if not command:
            data = _plan_only(
                ['CONTROLLER_CMD is required: a schema-1 controller receipt is '
                 'the only route/decode evidence'], source)
            _write_or_report(out_path, data)
            return 2
        try:
            controller = protocol.CommandController(command)
        except protocol.ConfigError as exc:
            data = _plan_only(exc.reasons, source)
            _write_or_report(out_path, data)
            return 2

    try:
        data = run(cfg, url, model, controller, source=source)
    except Exception as exc:  # noqa: BLE001 - keep partial evidence
        data = _plan_only(['run_failed:' + type(exc).__name__], source)
        data['run_id'] = cfg.run_id
        data['verdict'] = 'INCONCLUSIVE'
        data['verdict_reasons'] = ['inconclusive:run_failed:' + type(exc).__name__]
        _write_or_report(out_path, data)
        print('e3_decode_correctness run failed: ' + type(exc).__name__,
              file=sys.stderr)
        return 2

    if not _write_or_report(out_path, data):
        print('e3_decode_correctness cannot write ' + out_path, file=sys.stderr)
        return 2
    print('e3_decode_correctness ' + json.dumps(
        {'verdict': data['verdict'], 'reasons': data['verdict_reasons']},
        sort_keys=True))
    return exit_code(data['verdict'])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
