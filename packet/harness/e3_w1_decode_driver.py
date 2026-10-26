#!/usr/bin/env python3
"""E3 W1 decode-only matched-repeat load driver (offline build; no engine control).

This is the q17b W1 client for *live-cohort* decode-only windows.  It opens one
fixed cohort on the full route, waits until every live request has advanced,
seals admissions, asks the reviewer's controller for an awaited keep /
clear_cache=False pause plus real decode eligibility and all-rank transition
ACKs, resumes on the target route, and only then measures a bounded
client-visible window on the same live requests.  At the end it pauses again,
takes final interval counters, restores full, and lets the client connections
finish or cancels them.  It never talks to the engine except through the chat
URL; route changes go only through the injected controller.

The workload is an explicit owner manifest (``WORKLOAD=<path.json>``).  There
are no implicit generation-budget, seed, request-shape or window defaults: when
the manifest is absent or incomplete the driver writes a PLAN-ONLY /
INCONCLUSIVE artifact, issues no requests, and exits 2.  The standing W1
manifest must be named ``w1-standing`` and declare 8 streams at 153537 prompt
tokens plus an owner-declared positive generation budget, seed, request shape,
repeats, route plan and window lengths.  A differently named manifest is
treated as an explicitly declared non-standing workload.  The declared seed is
bound deliberately to both prompt construction and the sampling ``seed`` field
of every request; there is no separate guessed sampling seed and no unseeded
stochastic matched comparison.

Protocol per repeat
  1. ``prepare`` controller command verifies the full route and binds the
     engine incarnation before any request is opened.
  2. Open the fixed cohort ONCE on the full route: submit exactly STREAMS
     streaming requests with identical bytes every repeat.  An optional cache
     preparation pass (owner manifest) runs on full and is not counted.
  3. Wait until every distinct live request has advanced; a stream that never
     starts, errors, or finishes before the seal makes the cohort incomplete.
  4. Seal admissions: submit nothing new for the rest of the repeat.
  5. ``pause_enter_split`` (split repeat) or ``pause_full`` (full control):
     require a schema-1 receipt with an awaited keep pause
     (mode=keep, clear_cache=false, completed=true, drained=true), real
     worker-observation decode eligibility, and all-rank transition ACKs for
     any route change.  A full repeat takes the same pause/resume boundaries
     but requests no fictitious transition and no generation increment.
  6. ``resume``: an accepted receipt marks the measurement-start boundary.
     It does not prove that later execution stays decode-only.
  7. Measure WINDOW_S from client-visible token arrivals.  EOS, stream error,
     timeout, disconnect or an incomplete stream before the end invalidates the
     repeat.  There are no replacements, appends, newcomers or retries.
  8. At the window end, ``pause_end``: completed keep pause plus final
     all-rank interval counters and invalidation counters.  Any prefill /
     mixed / unsupported / preempted observation invalidates the interval;
     missing counter proof is INCONCLUSIVE, never a default zero.
  9. ``leave_full`` (split->full when needed; an explicit full->full no-op with
     no generation bump), then an explicit cleanup ``resume``.  Only after that
     are the client streams allowed to finish or be cancelled.  If the route
     state is unproven, the driver refuses to resume or restore it and aborts.

Evidence / timing
  * Raw client events use absolute ``time.monotonic_ns`` plus one wall-clock
    anchor at run start.  Events carry cumulative completion counts from
    ``stream_options.continuous_usage_stats`` where present; a multi-token
    advance is recorded as a coalesced client-visible arrival, never as one
    token per SSE message.  No prompt text, messages, token arrays, output
    content, auth headers or secrets are stored: only hashes, lengths, counts
    and timestamps.
  * Token / gap statistics are client-visible.  They are not kernel completion
    times and are never combined with engine clocks without saying so.
  * Context-position drift is explicit: the owner declares a per-request
    completion-delta range and the common bounded-window policy.  Actual
    start/end cumulative counts and deltas are reported per stream; an
    out-of-range or unproven exposure invalidates the repeat instead of
    selecting a favourable window.
  * TTFT belongs to the preparation/open phase and is reported separately from
    the measured window.  Cached-token/recompute evidence is descriptive
    preparation evidence, never a no-prefill or decode-only proof.
  * Summaries over valid repeats are explicitly labelled, carry total and
    invalid counts, and never issue PASS.  Performance is kept separate from
    output correctness.  Exit 0 requires every required repeat valid; exit 2 is
    insufficient or invalid evidence (including this plan-only path); exit 1 is
    a proved correctness or contract failure.

Controller transport
  ``CONTROLLER_CMD`` is split with shlex and run without a shell; one explicit
  schema-1 JSON command is written to stdin and one JSON receipt is read from
  stdout.  Exit 0 is transport success only.  Tests inject a controller object
  with the same ``execute(command, timeout_s) -> (receipt, transport)``
  interface, so no runtime controller source is needed here.

vLLM recompute rule (source: pinned tree, this repo)
  The preparation pass re-sends the same prompts.  ``get_computed_blocks`` caps
  the prefix-cache lookup at ``request.num_tokens - 1`` so at least one token
  is recomputed (``patches/ppgrouping/overlay/core/kv_cache_manager.py:290-295``),
  and ``find_longest_cache_hit`` returns full-block boundaries unless
  fine-grained partial hits are enabled
  (``patches/evictfirst/image/single_type_kv_cache_manager.py:790-821``).  With
  block size B and prompt length L the full-block recompute is
  ``L - floor((L-1)/B)*B == ((L-1) % B) + 1`` in [1, B]; the evictfirst
  partial-tail overlay can serve more, so an observed uncached count in [0, B]
  is consistent.  This is DESCRIPTIVE only.

usage: e3_w1_decode_driver.py URL MODEL OUT.json | --resummarize OUT.json
env: WORKLOAD=<manifest.json> CONTROLLER_CMD=<executable + args> [RUN_ID=...]
     [CONTROLLER_TIMEOUT_S=900] [READ_TIMEOUT_S=120] [DRAIN_TIMEOUT_S=60]
"""
import hashlib
import http.client
import importlib.util
import json
import math
import os
import random
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALID_ROUTES = ('full', 'split')
CONTROLLER_ACTIONS = ('prepare', 'pause_enter_split', 'pause_full', 'resume',
                      'pause_end', 'leave_full')
EXPECTED_RANKS = (0, 1, 2, 3)
STANDING_NAME = 'w1-standing'
STANDING_STREAMS = 8
STANDING_PROMPT_TOKENS = 153537
COALESCING_NOTE = (
    'client-visible SSE arrival timestamps; a multi-token advance is one coalesced '
    'client-visible arrival and is never treated as one token per SSE message')
RECOMPUTE_RULE = (
    'vLLM 0.29.1rc1 pinned tree: get_computed_blocks caps the prefix-cache lookup at '
    'request.num_tokens - 1 (patches/ppgrouping/overlay/core/kv_cache_manager.py:290-295) '
    'and find_longest_cache_hit returns full-block boundaries unless fine-grained partial '
    'hits are enabled (patches/evictfirst/image/single_type_kv_cache_manager.py:790-821). '
    'With block size B and prompt length L the full-block recompute is '
    'L - floor((L-1)/B)*B == ((L-1) % B) + 1, always in [1, B]; the evictfirst '
    'partial-tail overlay can serve more, so an observed uncached count in [0, B] is '
    'consistent.  DESCRIPTIVE preparation evidence only, never a no-prefill proof.')
WINDOW_BASIS = (
    'client-visible bounded window: start is immediately after the accepted resume '
    'receipt; end is the client monotonic end boundary issued before pause_end. '
    'SSE arrival times are client-visible transport times, not kernel completion times.')
CORRECTNESS_NOTE = (
    'No reference outputs or deterministic correctness oracle are available to this '
    'offline driver, so output correctness is not scored.  Performance statistics never '
    'stand in for output correctness.')
SEED_PURPOSE = (
    'the declared manifest seed is used for BOTH synthetic prompt construction and the '
    'sampling seed sent in every request; there is no separate guessed sampling seed '
    'and no unseeded stochastic matched comparison')


def _load_agent_replay():
    """Import the reviewed W2 benchmark without modifying it (text/pct/stream_chat)."""
    spec = importlib.util.spec_from_file_location('agent_replay_for_e3_w1',
                                                  HERE / 'agent_replay.py')
    if spec is None or spec.loader is None:
        raise ImportError('agent_replay.py not found next to the driver')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AR = _load_agent_replay()


class ConfigError(Exception):
    """A workload manifest or transport setting is missing or malformed."""

    def __init__(self, message, reasons=None):
        super().__init__(message)
        self.reasons = list(reasons or [message])


class ControllerTransportError(Exception):
    """The controller could not be run or did not produce a JSON receipt."""

    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = dict(record or {})


def _is_int(value):
    return type(value) is int  # bool is not an int for this schema


def _is_number(value):
    return type(value) in (int, float) and math.isfinite(float(value))


def _sha256_json(obj):
    raw = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def _ms(ns):
    return round(ns / 1_000_000.0, 3)


def _pct(values, q):
    return AR.pct(values, q)


def _severity(reasons):
    if any(str(reason).startswith('contract:') for reason in reasons):
        return 'contract'
    if reasons:
        return 'inconclusive'
    return 'valid'


# --- owner workload manifest -------------------------------------------------

@dataclass(frozen=True)
class Workload:
    name: str
    streams: int
    prompt_tokens: int
    seed: int
    generation_budget: int
    repeats: int
    route_plan: tuple
    temperature: float
    min_tokens_mode: str
    include_usage: bool
    continuous_usage_stats: bool
    enable_thinking: bool
    window_s: float
    open_timeout_s: float
    pause_timeout_s: float
    cache_preparation: bool
    completion_min: int
    completion_max: int
    start_completion_min: int
    start_completion_max: int
    context_position_policy: str
    block_size: int
    manifest_sha256: str


def _manifest_int(data, key, minimum, problems, optional=False):
    if key not in data:
        if not optional:
            problems.append('manifest_field_missing:' + key)
        return None
    value = data[key]
    if not _is_int(value) or value < minimum:
        problems.append('manifest_field_invalid:%s' % key)
        return None
    return value


def _manifest_float(data, key, problems, positive=False):
    if key not in data:
        problems.append('manifest_field_missing:' + key)
        return None
    value = data[key]
    if not _is_number(value) or (positive and float(value) <= 0):
        problems.append('manifest_field_invalid:' + key)
        return None
    return float(value)


def validate_workload(data):
    """Pure manifest validation -> (problems, Workload|None, sha)."""
    if type(data) is not dict:
        return ['manifest_not_object'], None, None
    sha = _sha256_json(data)
    problems = []
    if data.get('schema') != 1 or not _is_int(data.get('schema')):
        problems.append('manifest_schema_must_be_1')
    name = data.get('workload')
    if not isinstance(name, str) or not name.strip():
        problems.append('manifest_field_missing:workload')
        name = ''
    streams = _manifest_int(data, 'streams', 1, problems)
    prompt_tokens = _manifest_int(data, 'prompt_tokens', 1, problems)
    seed = _manifest_int(data, 'seed', 0, problems)
    budget = _manifest_int(data, 'generation_budget', 1, problems)
    repeats = _manifest_int(data, 'repeats', 1, problems)
    completion_min = None
    completion_max = None
    start_min = start_max = None
    if 'completion_policy' not in data or type(data.get('completion_policy')) is not dict:
        problems.append('manifest_field_missing:completion_policy')
    else:
        policy = data['completion_policy']
        completion_min = _manifest_int(policy, 'per_request_delta_min', 1, problems)
        completion_max = _manifest_int(policy, 'per_request_delta_max', 1, problems)
        start_min = _manifest_int(policy, 'per_request_start_min', 0, problems)
        start_max = _manifest_int(policy, 'per_request_start_max', 0, problems)
        if start_min is not None and start_max is not None and start_max < start_min:
            problems.append('completion_policy_start_max_below_min')
        if (completion_min is not None and completion_max is not None
                and completion_max < completion_min):
            problems.append('completion_policy_max_below_min')
        if policy.get('matched_across_routes') is not True:
            problems.append('completion_policy_must_match_routes')
    if 'context_position_policy' not in data:
        problems.append('manifest_field_missing:context_position_policy')
    context_policy = data.get('context_position_policy')
    if not isinstance(context_policy, str) or not context_policy.strip():
        problems.append('manifest_field_invalid:context_position_policy')
        context_policy = ''

    shape = data.get('request_shape')
    temperature = None
    min_tokens_mode = None
    include_usage = None
    continuous = None
    enable_thinking = None
    if type(shape) is not dict:
        problems.append('manifest_field_missing:request_shape')
    else:
        if 'temperature' not in shape or not _is_number(shape.get('temperature')):
            problems.append('manifest_field_missing:request_shape.temperature')
        else:
            temperature = float(shape['temperature'])
            if not 0.0 <= temperature <= 2.0:
                problems.append('manifest_field_invalid:request_shape.temperature')
        min_tokens_mode = shape.get('min_tokens')
        if min_tokens_mode not in ('equal_max', 'none'):
            problems.append('manifest_field_invalid:request_shape.min_tokens')
        for key in ('include_usage', 'continuous_usage_stats', 'enable_thinking'):
            if type(shape.get(key)) is not bool:
                problems.append('manifest_field_invalid:request_shape.' + key)
        include_usage = shape.get('include_usage')
        continuous = shape.get('continuous_usage_stats')
        enable_thinking = shape.get('enable_thinking')

    window_s = _manifest_float(data, 'window_s', problems, positive=True)
    open_timeout_s = _manifest_float(data, 'open_timeout_s', problems, positive=True)
    pause_timeout_s = _manifest_float(data, 'pause_timeout_s', problems, positive=True)
    cache_preparation = data.get('cache_preparation')
    if type(cache_preparation) is not bool:
        problems.append('manifest_field_invalid:cache_preparation')

    raw_plan = data.get('route_plan')
    plan = ()
    if type(raw_plan) is not list or not raw_plan:
        problems.append('manifest_field_missing:route_plan')
    else:
        if any(label not in VALID_ROUTES for label in raw_plan):
            problems.append('route_plan_labels_must_be_full_or_split')
        plan = tuple(raw_plan)
        if repeats is not None and len(plan) != repeats:
            problems.append('route_plan_length_must_equal_repeats')

    block_size = data.get('block_size')
    if block_size is not None and (not _is_int(block_size) or block_size < 1):
        problems.append('manifest_field_invalid:block_size')

    if name == STANDING_NAME:
        if streams is not None and streams != STANDING_STREAMS:
            problems.append('standing_w1_requires_%d_streams' % STANDING_STREAMS)
        if prompt_tokens is not None and prompt_tokens != STANDING_PROMPT_TOKENS:
            problems.append('standing_w1_requires_%d_prompt_tokens'
                            % STANDING_PROMPT_TOKENS)
        if set(plan) != set(VALID_ROUTES):
            problems.append('standing_w1_requires_full_and_split_control')

    if problems:
        return problems, None, sha
    assert streams is not None and prompt_tokens is not None and seed is not None
    assert budget is not None and repeats is not None and temperature is not None
    assert min_tokens_mode in ('equal_max', 'none')
    assert include_usage is not None and continuous is not None
    assert enable_thinking is not None and window_s is not None
    assert open_timeout_s is not None and pause_timeout_s is not None
    assert cache_preparation is not None and completion_min is not None
    assert completion_max is not None
    workload = Workload(
        name=name, streams=streams, prompt_tokens=prompt_tokens, seed=seed,
        generation_budget=budget, repeats=repeats, route_plan=plan,
        temperature=temperature, min_tokens_mode=min_tokens_mode,
        include_usage=include_usage, continuous_usage_stats=continuous,
        enable_thinking=enable_thinking, window_s=window_s,
        open_timeout_s=open_timeout_s, pause_timeout_s=pause_timeout_s,
        cache_preparation=cache_preparation, completion_min=completion_min,
        completion_max=completion_max, context_position_policy=context_policy,
        start_completion_min=start_min, start_completion_max=start_max,
        block_size=block_size, manifest_sha256=sha)
    return [], workload, sha


@dataclass(frozen=True)
class Config:
    workload: Workload
    run_id: str
    controller_cmd: str
    controller_timeout_s: float
    read_timeout_s: float
    drain_timeout_s: float
    out_path: str
    out_dir: str


def _env_float(env, name, default):
    raw = env.get(name)
    if raw is None or raw == '':
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(name + ' must be a number')
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(name + ' must be a positive finite number')
    return value


def load_manifest(env):
    raw_path = env.get('WORKLOAD')
    if not raw_path:
        return None, None, ['WORKLOAD manifest path is required; refusing implicit defaults']
    path = Path(raw_path)
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        return None, None, ['workload_manifest_unreadable:' + type(exc).__name__]
    problems, workload, sha = validate_workload(data)
    return workload, sha, problems


def config_from_env(env, out_path):
    workload, _sha, problems = load_manifest(env)
    if problems:
        raise ConfigError('owner workload manifest is absent or incomplete', problems)
    run_id = env.get('RUN_ID') or ('w1-%s-%d' % (Path(out_path).stem, time.time_ns()))
    return Config(
        workload=workload, run_id=run_id,
        controller_cmd=env.get('CONTROLLER_CMD', ''),
        controller_timeout_s=_env_float(env, 'CONTROLLER_TIMEOUT_S', 900.0),
        read_timeout_s=_env_float(env, 'READ_TIMEOUT_S', 120.0),
        drain_timeout_s=_env_float(env, 'DRAIN_TIMEOUT_S', 60.0),
        out_path=str(Path(out_path).resolve()),
        out_dir=str(Path(out_path).resolve().parent))


# --- fixed request set ------------------------------------------------------

def build_requests(workload):
    """Seeded conversations, built once and reused byte-for-byte.

    Mirrors agent_replay's seeding (random.Random(SEED * 1000 + i)) and uses its
    reviewed text().  Returns messages plus metadata; prompt text is never written
    to an artifact.
    """
    requests, meta = [], []
    for i in range(workload.streams):
        rng = random.Random(workload.seed * 1000 + i)
        system = 'You are coding agent %d, session 1.' % i
        user = 'Repository context:\n' + AR.text(rng, workload.prompt_tokens)
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': user}]
        requests.append(messages)
        meta.append({
            'stream': i,
            'prompt_sha256': _sha256_json(messages),
            'system_chars': len(system),
            'user_chars': len(user),
            'user_words': len(user.split()),
        })
    return requests, meta


def request_body(model, messages, workload):
    shape = {
        'temperature': workload.temperature,
        'min_tokens': workload.min_tokens_mode,
        'include_usage': workload.include_usage,
        'continuous_usage_stats': workload.continuous_usage_stats,
        'enable_thinking': workload.enable_thinking,
    }
    body = {
        'model': model,
        'messages': messages,
        'max_tokens': workload.generation_budget,
        'temperature': shape['temperature'],
        # The one declared workload seed is deliberately bound to BOTH prompt
        # construction and sampling. There is no second guessed parameter: a
        # stochastic matched comparison is only valid when the sampling seed is
        # explicit, so the request bytes and their hash bind it.
        'seed': workload.seed,
        'stream': True,
        'stream_options': {
            'include_usage': shape['include_usage'],
            'continuous_usage_stats': shape['continuous_usage_stats'],
        },
        'chat_template_kwargs': {'enable_thinking': shape['enable_thinking']},
    }
    if shape['min_tokens'] == 'equal_max':
        body['min_tokens'] = workload.generation_budget
    return body


def build_request_set(model, requests, workload):
    bodies = [request_body(model, messages, workload) for messages in requests]
    hashes = [hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
              for body in bodies]
    return bodies, hashes, _sha256_json(bodies)


# --- controller interface ---------------------------------------------------

class Controller:
    """Injectable controller interface; tests provide an in-process fake."""

    def execute(self, command, timeout_s):
        raise NotImplementedError


class CommandController(Controller):
    """JSON-on-stdin controller transport.  No shell, no interpolated syntax."""

    def __init__(self, command):
        self.argv = shlex.split(command)
        if not self.argv:
            raise ConfigError('CONTROLLER_CMD is empty')

    def execute(self, command, timeout_s):
        transport = {'kind': 'subprocess', 'rc': None, 'seconds': None,
                     'stdout_len': 0, 'stdout_sha256': None, 'stderr_len': 0,
                     'timeout': False}
        payload = json.dumps(command, sort_keys=True)
        started = time.monotonic()
        try:
            proc = subprocess.run(self.argv, input=payload, capture_output=True,
                                  text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            transport.update({'seconds': round(time.monotonic() - started, 3),
                              'timeout': True})
            raise ControllerTransportError('controller_timeout', transport)
        except OSError as exc:
            transport.update({'seconds': round(time.monotonic() - started, 3),
                              'error': type(exc).__name__})
            raise ControllerTransportError('controller_spawn_failed', transport)
        transport.update({
            'seconds': round(time.monotonic() - started, 3),
            'rc': proc.returncode,
            'stdout_len': len(proc.stdout),
            'stdout_sha256': hashlib.sha256(proc.stdout.encode()).hexdigest(),
            'stderr_len': len(proc.stderr),
        })
        if proc.returncode != 0:
            raise ControllerTransportError('controller_rc:%d' % proc.returncode,
                                           transport)
        try:
            receipt = json.loads(proc.stdout.strip())
        except ValueError:
            raise ControllerTransportError('controller_receipt_not_json', transport)
        return receipt, transport


# --- controller receipt schema ---------------------------------------------

# The prior-receipt chain binds this bounded metadata core.  It is a subset of
# the retained receipt projection below, so every hashed byte is retained.
RECEIPT_CORE_KEYS = (
    'schema', 'run_id', 'repeat', 'command_seq', 'action', 'cohort_id',
    'status', 'engine_epoch', 'storage_epoch', 'route_before', 'route_after',
    'route_generation_before', 'route_generation_after', 'expected_rank_set',
    'observed_clock_ns', 'clock_basis', 'pause', 'resume', 'ingress',
    'barrier_replies', 'interval', 'decode_eligibility', 'lifecycle_proven', 'observer_sha256',
)
MAX_STRING = 300
MAX_LIST = 64


def _bounded(value, depth=0):
    """Bounded metadata copy; never copies arbitrary payload-bearing keys."""
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return value if -(2 ** 63) < value < 2 ** 63 else None
    if type(value) is float:
        return value if math.isfinite(value) else None
    if type(value) is str:
        return value[:MAX_STRING]
    if depth >= 4:
        return None
    if type(value) in (list, tuple):
        return [_bounded(item, depth + 1) for item in list(value)[:MAX_LIST]]
    if type(value) is dict:
        return {str(key)[:64]: _bounded(item, depth + 1)
                for key, item in list(value.items())[:MAX_LIST]}
    return None



_TEXT_FIELDS = (('clock_basis',), ('decode_eligibility', 'reason'),
                ('interval', 'observer_basis'), ('ingress', 'basis'),
                ('interval', 'invalidation_basis'))
_TEXT_BASES = {
    'frontend monotonic completion; offset to client clock unknown':
        'frontend_monotonic_client_offset_unknown_v1',
    'paused live request/queued-token counts and last actual per-worker batch':
        'paused_live_counts_actual_batch_v1',
}


def _bind_receipt_text(core, original):
    """Bind narrative bytes without persisting arbitrary text in the core.

    Validate the original wire receipt separately. Hash before truncation, and
    retain this exact projected core so its chain digest is reproducible even
    when a display receipt redacts narrative fields.
    """
    for path in _TEXT_FIELDS:
        source, target = original, core
        for key in path[:-1]:
            source = source.get(key) if type(source) is dict else None
            target = target.get(key) if type(target) is dict else None
        key = path[-1]
        if type(source) is not dict or key not in source or type(target) is not dict:
            continue
        value = source[key]
        if type(value) is str:
            raw = value.encode('utf-8')
            marker = {'kind': 'text', 'sha256': hashlib.sha256(raw).hexdigest(),
                      'utf8_bytes': len(raw), 'nonblank': bool(value.strip()),
                      'known_basis': _TEXT_BASES.get(value)}
        else:
            marker = {'kind': 'null' if value is None else 'invalid_type'}
        target[key] = marker
    return core

def _receipt_core(obj):
    """Separately retained canonical core for the prior-receipt chain.

    Structured metadata comes from the bounded projection. Narrative fields
    bind original bytes through digests before display truncation/redaction.
    The session mirrors this projection; hash the retained core itself, not
    a potentially redacted display receipt.
    """
    retained = _sanitize_receipt(obj)
    if retained is None:
        return None
    core = {key: retained[key] for key in RECEIPT_CORE_KEYS if key in retained}
    return _bind_receipt_text(core, obj)


def _sanitize_inventory(inventory):
    """Inventory needed to bind bank/helper/storage; no native addresses."""
    if type(inventory) is not dict:
        return None
    out = {key: _bounded(inventory.get(key)) for key in (
        'mode', 'helper_source_sha256', 'library_sha256', 'descriptors',
        'descriptors_captured', 'raw_graphs_retained', 'split_pool_bytes',
        'split_pool_basis', 'split_pool_label', 'split_context_label',
        'split_stream_label', 'identity_scope', 'capture_stream_checks',
        'capture_memory', 'storage_epoch', 'route', 'route_generation',
        'communicators', 'pp_channel_bytes', 'ple_scratch_bytes', 'excluded',
        'split') if key in inventory}
    binding = inventory.get('binding_identity')
    if type(binding) is dict:
        out['binding_identity'] = {key: _bounded(binding.get(key)) for key in
                                   ('library_sha256', 'abi', 'helper_source_sha256')
                                   if key in binding}
    return out


def _sanitize_memory(memory):
    """Memory clocks/unknown values/error codes and explicit scope."""
    if type(memory) is not dict:
        return None
    out = {key: _bounded(memory.get(key)) for key in (
        'schema', 'wall_s', 'monotonic_ns', 'sample_end_wall_s',
        'sample_end_monotonic_ns', 'device_free_bytes', 'device_total_bytes',
        'synchronized_by_snapshot', 'peak_scope', 'device_scope',
        'errors') if key in memory}
    allocator = memory.get('allocator')
    if type(allocator) is dict:
        out['allocator'] = {key: _bounded(allocator.get(key)) for key in (
            'allocated_current_bytes', 'reserved_current_bytes',
            'allocated_peak_bytes', 'reserved_peak_bytes',
            'inactive_split_current_bytes', 'inactive_split_peak_bytes')
            if key in allocator}
    if 'scope' not in out:
        out['scope'] = 'allocator lifetime peaks; device point samples, sequential reads'
    return out


def _sanitize_batch(last):
    if type(last) is not dict:
        return None
    # Always spell out the bound batch facts; a missing value stays None and is
    # never silently replaced by a zero.
    return {key: _bounded(last.get(key)) for key in (
        'sequence', 'route', 'route_generation', 'num_reqs', 'num_tokens',
        'decode_only', 'full_graph')}


def _sanitize_counter_observation(counters):
    if type(counters) is not dict:
        return None
    out = {key: _bounded(counters.get(key)) for key in (
        'mode', 'state', 'route', 'route_generation', 'storage_epoch',
        'replays', 'replay_streams_match_owner', 'transition_count',
        'transition_latencies_s', 'interval', 'interval_index',
        'interval_invalidations', 'invalidation_reasons', 'quarantined',
        'in_flight', 'sampling_pending', 'transition_pending',
        'warmup_replays') if key in counters}
    for key in ('replays_by_bank', 'replays_by_stream', 'replays_by_descriptor'):
        value = counters.get(key)
        if type(value) is dict:
            out[key] = {str(k)[:64]: _bounded(v) for k, v in list(value.items())[:MAX_LIST]}
    batches = counters.get('batch_observations')
    if type(batches) is dict:
        # Spell out the bound batch counters; missing values stay None.
        observed = {key: _bounded(batches.get(key)) for key in (
            'batches', 'decode_batches', 'non_decode_batches', 'empty_batches',
            'full_graph_dispatches', 'non_full_dispatches',
            'scheduler_observations', 'preempted_requests',
            'unknown_preemption_observations')}
        observed['last'] = _sanitize_batch(batches.get('last'))
        out['batch_observations'] = observed
    return out


def _sanitize_observation(row):
    if type(row) is not dict:
        return None
    out = {'rank': _bounded(row.get('rank'))}
    if 'inventory' in row:
        out['inventory'] = _sanitize_inventory(row.get('inventory'))
    if 'counters' in row:
        out['counters'] = _sanitize_counter_observation(row.get('counters'))
    if 'memory' in row:
        out['memory'] = _sanitize_memory(row.get('memory'))
    return out


def _sanitize_interval(interval):
    if type(interval) is not dict:
        return None
    out = {key: _bounded(interval.get(key)) for key in (
        'available', 'valid', 'reasons', 'unavailable', 'observer_basis',
        'invalidation', 'invalidation_complete', 'invalidation_lower_bounds',
        'invalidation_basis') if key in interval}
    for key in ('counters_before', 'counters_after'):
        rows = interval.get(key)
        if type(rows) is not list:
            continue
        clean = []
        for row in rows[:MAX_LIST]:
            if type(row) is not dict:
                continue
            entry = {'rank': _bounded(row.get('rank')),
                     'decode_steps': _bounded(row.get('decode_steps')),
                     'counters': _sanitize_counter_observation(row.get('counters'))}
            clean.append(entry)
        out[key] = clean
    return out


def _sanitize_receipt(obj):
    """Bounded metadata-only receipt retained in the artifact.

    This is a superset of :data:`RECEIPT_CORE_KEYS`; the retained-receipt hash is
    computed from this exact object, so no hash refers to discarded material.
    Arbitrary payload-bearing keys and exception messages are never copied.
    """
    if type(obj) is not dict:
        return None
    out = {key: _bounded(obj.get(key)) for key in (
        'schema', 'run_id', 'repeat', 'command_seq', 'action', 'cohort_id',
        'status', 'engine_epoch', 'storage_epoch', 'route_before', 'route_after',
        'route_generation_before', 'route_generation_after', 'expected_rank_set',
        'observed_clock_ns', 'clock_basis', 'lifecycle_proven', 'measurement',
        'lifecycle', 'observer_sha256') if key in obj}
    observer = obj.get('observer')
    if type(observer) is dict:
        try:
            encoded = json.dumps(observer, allow_nan=False)
            if len(encoded.encode()) <= 36 * 1024 * 1024:
                out['observer'] = json.loads(encoded)
            else:
                out['observer'] = {'schema': 1, 'complete': False,
                                   'error': 'observer-receipt-too-large'}
        except (ValueError, TypeError):
            out['observer'] = {'schema': 1, 'complete': False,
                               'error': 'observer-receipt-invalid'}
    pause = obj.get('pause')
    if type(pause) is dict:
        out['pause'] = {key: _bounded(pause.get(key)) for key in
                        ('mode', 'clear_cache', 'completed', 'drained') if key in pause}
    resume = obj.get('resume')
    if type(resume) is dict:
        out['resume'] = {'completed': _bounded(resume.get('completed'))}
    ingress = obj.get('ingress')
    if type(ingress) is dict:
        out['ingress'] = {key: _bounded(ingress.get(key)) for key in
                          ('sealed', 'pending', 'rejected', 'basis') if key in ingress}
    barriers = obj.get('barrier_replies')
    if type(barriers) is list:
        out['barrier_replies'] = [
            {key: _bounded(row.get(key)) for key in
             ('rank', 'ack', 'mode', 'route', 'route_generation', 'storage_epoch',
              'engine_epoch', 'drained_streams') if key in row}
            for row in barriers[:MAX_LIST] if type(row) is dict]
    observations = obj.get('worker_observations')
    if type(observations) is list:
        out['worker_observations'] = [_sanitize_observation(row)
                                      for row in observations[:MAX_LIST]]
    ranks = obj.get('ranks')
    if type(ranks) is list:
        clean_ranks = []
        for entry in ranks[:MAX_LIST]:
            if type(entry) is not dict:
                clean_ranks.append(None)
                continue
            clean = {'rank': _bounded(entry.get('rank')),
                     'status': _bounded(entry.get('status'))}
            ack = entry.get('transition_ack')
            if type(ack) is dict:
                clean['transition_ack'] = {key: _bounded(ack.get(key)) for key in (
                    'rank', 'ack', 'mode', 'route', 'route_generation',
                    'storage_epoch', 'engine_epoch', 'drained_streams') if key in ack}
            clean_ranks.append(clean)
        out['ranks'] = clean_ranks
    eligibility = obj.get('decode_eligibility')
    if type(eligibility) is dict:
        clean = {key: _bounded(eligibility.get(key)) for key in
                 ('cohort_id', 'cohort_cardinality', 'reason') if key in eligibility}
        obs = eligibility.get('ranks')
        if type(obs) is list:
            clean['ranks'] = [
                {key: _bounded(row.get(key)) for key in
                 ('rank', 'batch_reqs', 'scheduled_tokens', 'prefill', 'sequence',
                  'num_reqs', 'num_tokens', 'decode_only', 'full_graph', 'route',
                  'route_generation') if key in row}
                for row in obs[:MAX_LIST] if type(row) is dict]
        out['decode_eligibility'] = clean
    if 'interval' in obj:
        out['interval'] = _sanitize_interval(obj.get('interval'))
    return out


# --- shared strict receipt identity/evidence checks -------------------------
#
# Measurement validity and cleanup (lifecycle) validity differ only in the
# measurement evidence they require.  Identity, rank coverage, incarnation,
# storage, route generation, barrier/transition ownership and drain proof are
# the same checks in both validators, so they live here once.  Every integer
# identity is checked with ``_is_int`` (bool is not an int for this schema)
# before any equality comparison.

def _validate_identity(obj, ctx, missing, contract):
    """Strict engine/storage/rank-set/route/generation identity."""
    engine = obj.get('engine_epoch')
    if 'engine_epoch' not in obj:
        missing('engine_epoch_missing')
        engine = None
    elif not _is_int(engine) or engine < 1:
        contract('engine_epoch_invalid')
        engine = None
    elif ctx.get('engine_epoch') is not None and engine != ctx['engine_epoch']:
        contract('engine_incarnation_changed')
    storage = obj.get('storage_epoch')
    if 'storage_epoch' not in obj:
        missing('storage_epoch_missing')
        storage = None
    elif not _is_int(storage) or storage < 1:
        contract('storage_epoch_invalid')
        storage = None
    elif ctx.get('storage_epoch') is not None and storage != ctx['storage_epoch']:
        contract('storage_epoch_changed')

    expected_ranks = list(EXPECTED_RANKS)
    echoed = obj.get('expected_rank_set')
    if 'expected_rank_set' not in obj:
        missing('expected_rank_set_missing')
    elif (type(echoed) is not list or len(echoed) != len(expected_ranks)
          or any(not _is_int(rank) for rank in echoed)
          or echoed != expected_ranks):
        contract('expected_rank_set_mismatch')

    route_before = obj.get('route_before')
    route_after = obj.get('route_after')
    routes_ok = (route_before in VALID_ROUTES and route_after in VALID_ROUTES)
    if 'route_before' not in obj:
        missing('route_before_missing')
    elif route_before not in VALID_ROUTES:
        contract('route_before_invalid')
    if 'route_after' not in obj:
        missing('route_after_missing')
    elif route_after not in VALID_ROUTES:
        contract('route_after_invalid')
    generation_before = obj.get('route_generation_before')
    generation_after = obj.get('route_generation_after')
    if 'route_generation_before' not in obj:
        missing('route_generation_before_missing')
    elif not _is_int(generation_before) or generation_before < 0:
        contract('route_generation_before_invalid')
    if 'route_generation_after' not in obj:
        missing('route_generation_after_missing')
    elif not _is_int(generation_after) or generation_after < 0:
        contract('route_generation_after_invalid')
    if ctx.get('route') is not None and route_before in VALID_ROUTES \
            and route_before != ctx['route']:
        contract('route_before_mismatch')
    if (ctx.get('route_generation') is not None
            and _is_int(generation_before)
            and generation_before != ctx['route_generation']):
        contract('route_generation_before_mismatch')
    return {'engine': engine, 'storage': storage,
            'expected_ranks': expected_ranks, 'routes_ok': routes_ok,
            'route_before': route_before, 'route_after': route_after,
            'generation_before': generation_before,
            'generation_after': generation_after}


def _validate_route_transition(action, route_before, route_after,
                               generation_before, generation_after, contract):
    """Action-specific route/generation continuity (shared by both)."""
    if action == 'prepare':
        if route_before != 'full' or route_after != 'full':
            contract('prepare_must_verify_full')
        if _is_int(generation_before) and _is_int(generation_after) \
                and generation_after != generation_before:
            contract('prepare_changed_generation')
    elif action == 'pause_enter_split':
        if route_before != 'full' or route_after != 'split':
            contract('pause_enter_split_route_invalid')
        if _is_int(generation_before) and _is_int(generation_after) \
                and generation_after != generation_before + 1:
            contract('pause_enter_split_generation_not_advanced')
    elif action == 'pause_full':
        if route_before != 'full' or route_after != 'full':
            contract('pause_full_route_invalid')
        if _is_int(generation_before) and _is_int(generation_after) \
                and generation_after != generation_before:
            contract('pause_full_changed_generation')
    elif action in ('resume', 'pause_end'):
        if route_after != route_before:
            contract(action + '_changed_route')
        if _is_int(generation_before) and _is_int(generation_after) \
                and generation_after != generation_before:
            contract(action + '_changed_generation')
    elif action == 'leave_full':
        if route_before == 'split':
            if route_after != 'full':
                contract('leave_full_did_not_restore_full')
            if _is_int(generation_before) and _is_int(generation_after) \
                    and generation_after != generation_before + 1:
                contract('leave_full_generation_not_advanced')
        else:
            if route_after != 'full':
                contract('leave_full_full_route_changed')
            if _is_int(generation_before) and _is_int(generation_after) \
                    and generation_after != generation_before:
                contract('leave_full_full_changed_generation')


def _validate_rank_evidence(obj, expected_ranks, transition, route_after,
                            generation_after, engine, storage, missing, contract):
    """Strict rank coverage, status and typed transition ownership."""
    if 'ranks' not in obj:
        missing('ranks_missing')
        return
    ranks = obj.get('ranks')
    if type(ranks) is not list:
        contract('ranks_not_list')
        return
    if len(ranks) != len(expected_ranks):
        contract('rank_coverage')
    seen = set()
    for entry in ranks:
        if type(entry) is not dict:
            contract('rank_entry_not_object')
            continue
        rank = entry.get('rank')
        if not _is_int(rank) or rank not in expected_ranks:
            contract('rank_id_invalid_or_foreign')
            continue
        if rank in seen:
            contract('rank_duplicate:%d' % rank)
            continue
        seen.add(rank)
        if entry.get('status') != 'ok':
            contract('rank_status_not_ok:%d' % rank)
        ack = entry.get('transition_ack')
        if transition:
            if ack is None:
                missing('transition_ack_missing:%d' % rank)
            elif type(ack) is not dict:
                contract('transition_ack_not_object:%d' % rank)
            else:
                if any(not _is_int(ack.get(key)) for key in
                       ('rank', 'route_generation', 'storage_epoch', 'engine_epoch')):
                    contract('transition_ack_identity_type:%d' % rank)
                if not _is_int(ack.get('rank')) or ack.get('rank') != rank:
                    contract('transition_ack_rank_mismatch:%d' % rank)
                if ack.get('ack') != 'flash_s5_transition':
                    contract('transition_ack_synthetic_or_missing:%d' % rank)
                if ack.get('mode') != 'decode-only-probe':
                    contract('transition_ack_mode:%d' % rank)
                if ack.get('route') != route_after:
                    contract('transition_ack_route:%d' % rank)
                if (not _is_int(ack.get('route_generation'))
                        or ack.get('route_generation') != generation_after):
                    contract('transition_ack_generation:%d' % rank)
                if (engine is None or not _is_int(ack.get('engine_epoch'))
                        or ack.get('engine_epoch') != engine):
                    contract('transition_ack_engine:%d' % rank)
                if (storage is None or not _is_int(ack.get('storage_epoch'))
                        or ack.get('storage_epoch') != storage):
                    contract('transition_ack_storage:%d' % rank)
                if (not _is_int(ack.get('drained_streams'))
                        or ack.get('drained_streams') < 1):
                    contract('transition_ack_drained_streams:%d' % rank)
        elif ack is not None:
            contract('unexpected_transition_ack:%d' % rank)
    if seen != set(expected_ranks):
        contract('rank_coverage')


def _validate_barriers(obj, expected_ranks, route_before, generation_before,
                       engine, storage, missing, contract):
    """Strict four-rank typed drain barriers and their ownership identity."""
    barriers = obj.get('barrier_replies')
    if type(barriers) is not list or len(barriers) != len(expected_ranks):
        missing('barrier_rank_coverage')
        return
    seen = set()
    for row in barriers:
        if type(row) is not dict:
            contract('barrier_not_object')
            continue
        rank = row.get('rank')
        if not _is_int(rank) or rank not in expected_ranks or rank in seen:
            contract('barrier_rank_invalid')
            continue
        seen.add(rank)
        if (row.get('ack') != 'flash_s5_probe_barrier'
                or row.get('mode') != 'decode-only-probe'
                or row.get('route') != route_before
                or not _is_int(row.get('drained_streams'))
                or row['drained_streams'] < 1):
            contract('barrier_not_drained:%d' % rank)
        for key, value in (('engine_epoch', engine),
                           ('storage_epoch', storage),
                           ('route_generation', generation_before)):
            if value is None or not _is_int(row.get(key)) or row[key] != value:
                contract('barrier_identity_differs:%d' % rank)
    if seen != set(expected_ranks):
        contract('barrier_rank_coverage')


def _validate_pause_evidence(obj, missing, contract):
    pause = obj.get('pause')
    if type(pause) is not dict:
        missing('pause_missing')
        return
    if pause.get('mode') != 'keep':
        contract('pause_mode_not_keep')
    if pause.get('clear_cache') is not False:
        contract('pause_clear_cache_not_false')
    if pause.get('completed') is not True:
        missing('pause_not_completed')
    if pause.get('drained') is not True:
        missing('pause_pending_work_not_drained')


def _validate_resume_evidence(obj, missing):
    resume = obj.get('resume')
    if type(resume) is not dict:
        missing('resume_missing')
    elif resume.get('completed') is not True:
        missing('resume_not_completed')


def validate_receipt(obj, command, ctx):
    """Pure schema-1 receipt validator.

    Returns (ok, reasons, severity, accepted_state).  Missing proof is
    ``inconclusive:``; a present but contradictory identity, rank set,
    transition ACK, route/generation change, pause/resume field or interval
    counter is ``contract:``.  ``ok`` is true only when there are no reasons.
    """
    reasons = []

    def missing(message):
        reasons.append('inconclusive:' + message)

    def contract(message):
        reasons.append('contract:' + message)

    if type(obj) is not dict:
        return False, ['contract:receipt_not_object'], 'contract', None
    if 'schema' not in obj:
        missing('receipt_schema_missing')
    elif not _is_int(obj.get('schema')) or obj.get('schema') != 1:
        contract('receipt_schema_invalid')
    status = obj.get('status')
    if status != 'ok':
        text = str(status)[:80] if status is not None else 'missing'
        return False, ['inconclusive:controller_status:' + text], 'inconclusive', None

    for key, expected in (('run_id', ctx['run_id']), ('repeat', ctx['repeat']),
                          ('command_seq', ctx['command_seq']),
                          ('action', ctx['action']),
                          ('cohort_id', ctx['cohort_id'])):
        if key not in obj:
            missing('receipt_' + key + '_missing')
        elif key in ('repeat', 'command_seq'):
            if not _is_int(obj.get(key)) or obj.get(key) != expected:
                contract('receipt_' + key + '_mismatch')
        elif obj.get(key) != expected:
            contract('receipt_' + key + '_mismatch')

    observed = obj.get('observed_clock_ns')
    if 'observed_clock_ns' not in obj:
        missing('observed_clock_ns_missing')
    elif not _is_int(observed) or observed < 0:
        contract('observed_clock_ns_invalid')
    basis = obj.get('clock_basis')
    if not isinstance(basis, str) or not basis.strip():
        missing('clock_basis_missing')

    identity = _validate_identity(obj, ctx, missing, contract)
    engine = identity['engine']
    storage = identity['storage']
    expected_ranks = identity['expected_ranks']
    route_before = identity['route_before']
    route_after = identity['route_after']
    generation_before = identity['generation_before']
    generation_after = identity['generation_after']

    action = ctx['action']
    if identity['routes_ok']:
        _validate_route_transition(action, route_before, route_after,
                                   generation_before, generation_after, contract)

    transition = (action in ('pause_enter_split', 'leave_full')
                  and route_before in VALID_ROUTES
                  and route_after in VALID_ROUTES
                  and route_before != route_after)

    _validate_rank_evidence(obj, expected_ranks, transition, route_after,
                            generation_after, engine, storage, missing, contract)

    if action in ('pause_enter_split', 'pause_full', 'pause_end'):
        _validate_pause_evidence(obj, missing, contract)

    if action == 'resume':
        _validate_resume_evidence(obj, missing)

    if action in ('pause_enter_split', 'pause_full', 'pause_end', 'resume'):
        _validate_barriers(obj, expected_ranks, route_before, generation_before,
                           engine, storage, missing, contract)

    if action in ('pause_enter_split', 'pause_full'):
        eligibility = obj.get('decode_eligibility')
        if type(eligibility) is not dict:
            missing('decode_eligibility_missing')
        else:
            if eligibility.get('cohort_id') != ctx['cohort_id']:
                contract('decode_eligibility_cohort_mismatch')
            cardinality = eligibility.get('cohort_cardinality')
            if not _is_int(cardinality) or cardinality != ctx['streams']:
                contract('decode_eligibility_cardinality')
            observations = eligibility.get('ranks')
            if type(observations) is not list:
                missing('decode_eligibility_observations_missing')
            else:
                observed_ranks = set()
                for row in observations:
                    if type(row) is not dict:
                        contract('decode_eligibility_observation_not_object')
                        continue
                    rank = row.get('rank')
                    if not _is_int(rank) or rank not in expected_ranks:
                        contract('decode_eligibility_foreign_rank')
                        continue
                    if rank in observed_ranks:
                        contract('decode_eligibility_duplicate_rank')
                    observed_ranks.add(rank)
                    if not _is_int(row.get('batch_reqs')) or row.get('batch_reqs') != ctx['streams']:
                        contract('decode_eligibility_batch_reqs:%d' % rank)
                    if not _is_int(row.get('scheduled_tokens')) \
                            or row.get('scheduled_tokens') != ctx['streams']:
                        contract('decode_eligibility_scheduled_tokens:%d' % rank)
                    if row.get('prefill') is not False:
                        contract('decode_eligibility_prefill_observed:%d' % rank)
                if observed_ranks != set(expected_ranks):
                    contract('decode_eligibility_rank_coverage')
            reason = eligibility.get('reason')
            if not isinstance(reason, str) or not reason.strip():
                missing('decode_eligibility_reason_missing')

    if action == 'pause_end':
        interval = obj.get('interval')
        if type(interval) is not dict:
            missing('interval_missing')
        else:
            basis_text = interval.get('observer_basis')
            if not isinstance(basis_text, str) or not basis_text.strip():
                missing('interval_observer_basis_missing')
            if interval.get('available') is False:
                missing('interval_evidence_unavailable')
            for reason in interval.get('reasons') or []:
                text = str(reason)[:120]
                if text.startswith('interval-invalidated:'):
                    contract('interval_invalidated:' + text.split(':', 1)[1])
                elif text:
                    missing(text.replace(':', '_'))
            unavailable = interval.get('unavailable')
            if type(unavailable) is list and unavailable and interval.get('available') is not False:
                missing('interval_evidence_incomplete')
            before = interval.get('counters_before')
            after = interval.get('counters_after')
            baselines, finishes = {}, {}
            for label, rows, target in (('before', before, baselines),
                                        ('after', after, finishes)):
                if type(rows) is not list:
                    missing('interval_counters_%s_missing' % label)
                    continue
                for row in rows:
                    if type(row) is not dict:
                        contract('interval_counter_row_not_object')
                        continue
                    rank = row.get('rank')
                    if not _is_int(rank) or rank not in expected_ranks:
                        contract('interval_counter_rank_foreign')
                        continue
                    if rank in target:
                        contract('interval_counter_rank_duplicate')
                    steps = row.get('decode_steps')
                    if steps is None:
                        # Explicitly unavailable evidence, never an invented zero.
                        missing('interval_counter_unavailable:%s:%s' % (label, rank))
                        target[rank] = None
                        continue
                    if not _is_int(steps) or steps < 0:
                        contract('interval_decode_steps:%s:%s' % (label, rank))
                        continue
                    target[rank] = steps
                if set(target) != set(expected_ranks):
                    contract('interval_counter_rank_coverage:' + label)
            for rank in expected_ranks:
                if (rank in baselines and rank in finishes
                        and _is_int(baselines[rank]) and _is_int(finishes[rank])
                        and finishes[rank] < baselines[rank]):
                    contract('interval_counter_regressed:%d' % rank)
            invalidation = interval.get('invalidation')
            complete = interval.get('invalidation_complete')
            if type(invalidation) is not dict:
                missing('interval_invalidation_missing')
            else:
                for key in ('prefill', 'mixed', 'unsupported', 'preempted'):
                    value = invalidation.get(key)
                    if not _is_int(value) or value < 0:
                        missing('interval_invalidation_%s_missing' % key)
                        continue
                    # A known positive contribution fails the measurement even
                    # when a different counter made the total a lower bound.
                    if value > 0:
                        contract('interval_invalidated:' + key)
                    done = complete.get(key) if type(complete) is dict else None
                    if done is False:
                        missing('interval_invalidation_incomplete:' + key)
                    elif type(done) is not bool:
                        missing('interval_invalidation_complete_%s_missing' % key)
            lower = interval.get('invalidation_lower_bounds')
            if type(lower) is not list or any(
                    not isinstance(item, str) for item in lower):
                missing('interval_invalidation_lower_bounds_missing')
            basis = interval.get('invalidation_basis')
            if not isinstance(basis, str) or not basis.strip():
                missing('interval_invalidation_basis_missing')

    severity = _severity(reasons)
    if reasons:
        return False, reasons, severity, None
    return True, [], 'valid', {
        'engine_epoch': engine,
        'storage_epoch': storage,
        'route': route_after,
        'route_generation': generation_after,
    }


# --- lifecycle validity (independent of measurement validity) ---------------

def validate_lifecycle_receipt(obj, command, ctx):
    """Prove only that the route can be safely restored/resumed.

    This deliberately ignores interval counters, decode eligibility and
    invalidation.  It requires the completed keep/clear_cache=False pause, four
    typed drain barriers, identity/route/storage/generation continuity and no
    ambiguous pending transition.  It returns the accepted route state used by
    the next command.  Missing proof is ``inconclusive:``; a contradiction is
    ``contract:``.  Neither is inferred from severity alone.
    """
    reasons = []

    def missing(message):
        reasons.append('inconclusive:' + message)

    def contract(message):
        reasons.append('contract:' + message)

    if type(obj) is not dict:
        return False, ['contract:receipt_not_object'], 'contract', None
    if not _is_int(obj.get('schema')) or obj.get('schema') != 1:
        contract('receipt_schema_invalid')
    if obj.get('status') != 'ok':
        text = str(obj.get('status'))[:80] if obj.get('status') is not None else 'missing'
        return False, ['inconclusive:controller_status:' + text], 'inconclusive', None
    for key, expected in (('run_id', ctx['run_id']), ('repeat', ctx['repeat']),
                          ('command_seq', ctx['command_seq']),
                          ('action', ctx['action']), ('cohort_id', ctx['cohort_id'])):
        if key not in obj:
            missing('receipt_' + key + '_missing')
        elif key in ('repeat', 'command_seq'):
            if not _is_int(obj.get(key)) or obj.get(key) != expected:
                contract('receipt_' + key + '_mismatch')
        elif obj.get(key) != expected:
            contract('receipt_' + key + '_mismatch')
    identity = _validate_identity(obj, ctx, missing, contract)
    engine = identity['engine']
    storage = identity['storage']
    expected_ranks = identity['expected_ranks']
    route_before = identity['route_before']
    route_after = identity['route_after']
    generation_before = identity['generation_before']
    generation_after = identity['generation_after']
    if not identity['routes_ok']:
        return False, reasons, _severity(reasons), None
    action = ctx['action']
    _validate_route_transition(action, route_before, route_after,
                               generation_before, generation_after, contract)

    transition = (action in ('pause_enter_split', 'leave_full')
                  and route_before != route_after)
    _validate_rank_evidence(obj, expected_ranks, transition, route_after,
                            generation_after, engine, storage, missing, contract)

    if action in ('pause_enter_split', 'pause_full', 'pause_end'):
        _validate_pause_evidence(obj, missing, contract)
    if action == 'resume':
        _validate_resume_evidence(obj, missing)
    if action in ('pause_enter_split', 'pause_full', 'pause_end', 'resume'):
        _validate_barriers(obj, expected_ranks, route_before, generation_before,
                           engine, storage, missing, contract)

    # If actual worker observations are present, they must agree and must not
    # report an ambiguous pending transition. Missing observations are not
    # treated as success; the typed barriers and pause are the proof.
    observations = obj.get('worker_observations')
    if type(observations) is list:
        seen = set()
        for row in observations:
            if type(row) is not dict:
                contract('worker_observation_not_object')
                continue
            rank = row.get('rank')
            if not _is_int(rank) or rank not in expected_ranks or rank in seen:
                contract('worker_observation_rank_invalid')
                continue
            seen.add(rank)
            counters = row.get('counters')
            if type(counters) is not dict:
                continue
            if counters.get('route') != route_after:
                contract('worker_observation_route_differs:%d' % rank)
            if _is_int(generation_after) and counters.get('route_generation') != generation_after:
                contract('worker_observation_generation_differs:%d' % rank)
            if (counters.get('transition_pending') is not False
                    or counters.get('in_flight') not in (0, None)
                    or counters.get('sampling_pending') not in (False, None)):
                contract('worker_observation_still_active:%d' % rank)
        if seen != set(expected_ranks):
            contract('worker_observation_rank_coverage')

    severity = _severity(reasons)
    if reasons:
        return False, reasons, severity, None
    return True, [], 'valid', {
        'engine_epoch': engine, 'storage_epoch': storage,
        'route': route_after, 'route_generation': generation_after}



def _next_command(ctx, action, target_route):
    seq = ctx['command_seq'] + 1
    return {
        'schema': 1,
        'run_id': ctx['run_id'],
        'repeat': ctx['repeat'],
        'command_seq': seq,
        'action': action,
        'target_route': target_route,
        'cohort_id': ctx['cohort_id'],
        'cohort_size': ctx['streams'],
        'prior_receipt': ctx.get('prior_receipt'),
        'expected_rank_set': list(EXPECTED_RANKS),
    }


def _timeout_for(action, cfg):
    if action in ('pause_enter_split', 'pause_full', 'pause_end', 'resume'):
        return min(cfg.controller_timeout_s, cfg.workload.pause_timeout_s)
    return cfg.controller_timeout_s


def controller_call(controller, cfg, ctx, action, target_route):
    """Send one command, validate one receipt.  Exit 0 is not proof by itself."""
    command = _next_command(ctx, action, target_route)
    record = {
        'action': action,
        'target_route': target_route,
        'command_seq': command['command_seq'],
        'prior_receipt': command.get('prior_receipt'),
        'issued_at_ns': time.monotonic_ns(),
        'returned_at_ns': None,
        'ok': False,
        'severity': 'inconclusive',
        'reasons': [],
        'lifecycle_ok': False,
        'lifecycle_severity': 'inconclusive',
        'lifecycle_reasons': [],
        'transport': None,
        'receipt': None,
        'receipt_sha256': None,
        'retained_receipt_sha256': None,
    }
    try:
        receipt, transport = controller.execute(command, _timeout_for(action, cfg))
    except ControllerTransportError as exc:
        record['returned_at_ns'] = time.monotonic_ns()
        record['transport'] = exc.record
        record['reasons'] = ['inconclusive:controller_transport:' + str(exc)[:120]]
        ctx['command_seq'] = command['command_seq']
        return record
    except Exception as exc:  # noqa: BLE001 - injected transport failure
        record['returned_at_ns'] = time.monotonic_ns()
        record['transport'] = {'kind': 'injected_error',
                               'error': type(exc).__name__}
        record['reasons'] = ['inconclusive:controller_exception:' + type(exc).__name__]
        ctx['command_seq'] = command['command_seq']
        return record
    if type(receipt) is dict and ('observer' in receipt or 'observer_sha256' in receipt):
        try:
            bound = (type(receipt.get('observer')) is dict and
                     receipt.get('observer_sha256') == _sha256_json(receipt['observer']))
        except (ValueError, TypeError):
            bound = False
        if not bound:
            record['returned_at_ns'] = time.monotonic_ns()
            record['reasons'] = ['contract:observer_receipt_hash_differs']
            record['severity'] = 'contract'
            ctx['command_seq'] = command['command_seq']
            return record
    record['returned_at_ns'] = time.monotonic_ns()
    record['transport'] = transport
    record['receipt'] = _sanitize_receipt(receipt)
    # Retain the exact canonical chain core independently of the display copy.
    record['receipt_core'] = _receipt_core(receipt)
    record['receipt_sha256'] = (_sha256_json(record['receipt_core'])
                                if type(receipt) is dict else None)
    record['retained_receipt_sha256'] = (_sha256_json(record['receipt'])
                                         if record['receipt'] is not None else None)
    check_ctx = dict(ctx)
    check_ctx.update({'action': action, 'command_seq': command['command_seq']})
    ok, reasons, severity, accepted = validate_receipt(receipt, command, check_ctx)
    lifecycle_ok, lifecycle_reasons, lifecycle_severity, lifecycle_state = \
        validate_lifecycle_receipt(receipt, command, check_ctx)
    record['ok'] = ok
    record['reasons'] = reasons
    record['severity'] = severity
    record['lifecycle_ok'] = lifecycle_ok
    record['lifecycle_reasons'] = lifecycle_reasons
    record['lifecycle_severity'] = lifecycle_severity
    ctx['command_seq'] = command['command_seq']
    # Route/identity continuity is proven by the independent lifecycle
    # validator.  The next command must chain the prior receipt even when the
    # measured interval is invalid (measurement validity is kept separately).
    if lifecycle_ok and lifecycle_state is not None:
        ctx.update(lifecycle_state)
        record['prior_receipt'] = command.get('prior_receipt')
        ctx['prior_receipt'] = {
            'command_seq': command['command_seq'],
            'receipt_sha256': record['receipt_sha256'],
        }
    return record


# --- streaming client --------------------------------------------------------

def _sanitize_usage(usage):
    if type(usage) is not dict:
        return None
    out = {}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        value = usage.get(key)
        if _is_int(value) and value >= 0:
            out[key] = value
    details = usage.get('prompt_tokens_details')
    if type(details) is dict:
        cached = details.get('cached_tokens')
        if _is_int(cached) and cached >= 0:
            out['cached_tokens'] = cached
    return out or None


class StreamProtocolError(Exception):
    """Explicit, bounded SSE evidence failure (message is a stable code)."""


MAX_STREAM_EVENTS = 20000
MAX_COUNT_SAMPLES = 100000


class StreamState:
    def __init__(self, index, messages, meta, body):
        self.index = index
        self.messages = messages
        self.meta = meta
        self.body = body
        self.request_sha256 = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest()
        self.opened = threading.Event()
        self.any_event = threading.Event()
        self.send_at_ns = None
        self.first_at_ns = None
        self.ttft_ms = None
        self.events = []
        self.events_truncated = 0
        # Token-advance arrivals: one timestamp per real positive cumulative
        # completion-token increase, including usage-only/reasoning-only events.
        self.arrivals = []
        self.arrival_increments = []
        self.arrival_kinds = []
        self.arrival_baseline_inferred = 0
        self.cumulative_advances = 0
        self.coalesced_advances = 0
        self.repeated_counts = 0
        self.cumulative_updates = 0
        self.last_cumulative = None
        self.counts = []
        self.counts_truncated = 0
        self.usage_only_advances = 0
        self.reasoning_advances = 0
        self.content_observations = 0
        self.content_without_count = 0
        self.reasoning_observations = 0
        self.first_content_at_ns = None
        self.missing_count_events = 0
        self.last_usage = None
        self.finish = None
        self.finish_at_ns = None
        self.error = None
        self.error_at_ns = None
        self.done_marker = False
        self.done_at_ns = None
        self.thread_done = False
        self.thread = None
        self.connection = None
        self.response = None
        self.sock = None
        # http.client closes its own socket reference when the response says
        # Connection: close, even though the response's makefile keeps the fd
        # alive.  A dup of our own connection socket survives that and is the
        # cancellation handle.
        self.cancel_sock = None
        self.cancel_requested = False

    def event(self, row):
        if len(self.events) < MAX_STREAM_EVENTS:
            self.events.append(row)
        else:
            self.events_truncated += 1

    def count_sample(self, at_ns, cumulative):
        if len(self.counts) < MAX_COUNT_SAMPLES:
            self.counts.append((at_ns, cumulative))
        else:
            # Keep the first sample (needed for the start boundary) and drop the
            # oldest interior sample; the truncation is explicit.
            self.counts_truncated += 1
            self.counts[1:2] = []


def _open_stream(state, cfg, url, headers):
    """Open the SSE connection and retain handles for active cancellation."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise StreamProtocolError('invalid_stream_url')
    path = (parts.path or '') + '/v1/chat/completions'
    if parts.query:
        path += '?' + parts.query
    cls = (http.client.HTTPSConnection if parts.scheme == 'https'
           else http.client.HTTPConnection)
    connection = cls(parts.hostname, parts.port, timeout=cfg.read_timeout_s)
    state.connection = connection
    connection.request('POST', path, body=json.dumps(state.body).encode(),
                       headers=headers)
    state.sock = connection.sock
    try:
        # Duplicate before getresponse: a Connection: close reply makes
        # http.client close its reference while the response keeps reading.
        state.cancel_sock = connection.sock.dup()
    except (AttributeError, OSError):
        state.cancel_sock = None
    response = connection.getresponse()
    state.response = response
    return response


def _cancel_stream(state):
    """Actively close this driver's own blocked SSE connection.

    A stop flag cannot interrupt a thread blocked in ``readline``.  Shutdown the
    owned socket (and close response/connection) before the bounded joins.  This
    never talks to the engine/scheduler and never waits for a silent peer.
    """
    state.cancel_requested = True
    sock = state.cancel_sock or state.sock or getattr(state.connection, 'sock', None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
    if state.cancel_sock is not None:
        try:
            state.cancel_sock.close()
        except OSError:
            pass
    for owned in (state.response, state.connection):
        if owned is not None:
            try:
                owned.close()
            except Exception:  # noqa: BLE001 - best-effort cancellation
                pass


def stream_reader(state, cfg, url, stop):
    """Stream one live request, retaining timestamps and counts only (no content).

    Cumulative ``completion_tokens`` advances are the only token arrivals.  An
    increase greater than one is one coalesced client-visible arrival; a
    repeated count is not an arrival; content without a count is a separate
    content observation and never token-count evidence.  Usage-only and
    reasoning-only events count when they carry an advance.  Missing counts,
    regressions, malformed SSE and early EOF are explicit.
    """
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('VLLM_API_KEY')
    if key:
        headers['Authorization'] = 'Bearer ' + key
    state.send_at_ns = time.monotonic_ns()
    try:
        response = _open_stream(state, cfg, url, headers)
        while not stop.is_set():
            raw_line = response.readline()
            if not raw_line:
                break
            line = raw_line.decode('utf-8', 'replace').strip()
            if not line or line.startswith(':') or not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                state.done_marker = True
                break
            try:
                event = json.loads(payload)
            except ValueError:
                state.event({'t_ns': time.monotonic_ns(), 'kind': 'unparsed'})
                raise StreamProtocolError('malformed_sse_json') from None
            if type(event) is not dict:
                raise StreamProtocolError('malformed_sse_event')
            now = time.monotonic_ns()
            usage = _sanitize_usage(event.get('usage'))
            completion = usage.get('completion_tokens') if usage else None
            previous_cumulative = state.last_cumulative
            increment = None
            if usage is not None:
                merged = dict(state.last_usage or {})
                merged.update(usage)
                state.last_usage = merged
            if completion is not None:
                state.cumulative_updates += 1
                if previous_cumulative is None:
                    increment = completion if completion > 0 else 0
                    if increment:
                        state.arrival_baseline_inferred += 1
                elif completion < previous_cumulative:
                    raise StreamProtocolError('completion_counter_regressed')
                elif completion == previous_cumulative:
                    increment = 0
                    state.repeated_counts += 1
                else:
                    increment = completion - previous_cumulative
                state.last_cumulative = completion
                state.count_sample(now, completion)
            elif usage is None:
                state.missing_count_events += 1

            choices = event.get('choices')
            if choices is None:
                choices = []
            if type(choices) is not list:
                raise StreamProtocolError('malformed_sse_choices')
            content_seen = False
            reasoning_seen = False
            for choice in choices:
                if type(choice) is not dict:
                    raise StreamProtocolError('malformed_sse_choice')
                delta = choice.get('delta')
                if delta is None:
                    delta = {}
                if type(delta) is not dict:
                    raise StreamProtocolError('malformed_sse_delta')
                piece = delta.get('content')
                reasoning = delta.get('reasoning_content')
                if reasoning is None:
                    reasoning = delta.get('reasoning')
                if piece:
                    content_seen = True
                    state.content_observations += 1
                    if state.first_content_at_ns is None:
                        state.first_content_at_ns = now
                    if completion is None:
                        state.content_without_count += 1
                if reasoning:
                    reasoning_seen = True
                    state.reasoning_observations += 1
                    if state.first_content_at_ns is None:
                        state.first_content_at_ns = now
                    if completion is None:
                        state.content_without_count += 1
                finish = choice.get('finish_reason')
                if finish:
                    state.finish = str(finish)
                    state.finish_at_ns = now
                    state.event({'t_ns': now, 'kind': 'finish',
                                 'finish_reason': str(finish)})
                    state.any_event.set()

            if increment:
                coalesced = increment > 1
                if coalesced:
                    state.coalesced_advances += 1
                if not content_seen and not reasoning_seen:
                    state.usage_only_advances += 1
                elif reasoning_seen and not content_seen:
                    state.reasoning_advances += 1
                state.cumulative_advances += 1
                state.arrivals.append(now)
                state.arrival_increments.append(increment)
                state.arrival_kinds.append(
                    'usage_only' if not content_seen and not reasoning_seen
                    else ('reasoning' if reasoning_seen and not content_seen
                          else 'content_chunk'))
                state.event({
                    't_ns': now, 'kind': 'token_advance',
                    'cumulative_completion': state.last_cumulative,
                    'increment': increment,
                    'coalesced': coalesced,
                    'usage_only': not content_seen and not reasoning_seen,
                    'reasoning_only': reasoning_seen and not content_seen,
                })
                if state.first_at_ns is None:
                    state.first_at_ns = now
                    state.ttft_ms = _ms(now - state.send_at_ns)
                    state.opened.set()
                state.any_event.set()
            elif completion is not None:
                state.event({'t_ns': now, 'kind': 'repeated_count',
                             'cumulative_completion': completion})
            elif content_seen or reasoning_seen:
                # Content without any cumulative count: a content observation,
                # explicitly not a token arrival or readiness basis.
                state.event({'t_ns': now, 'kind': 'content_without_count'})
                state.any_event.set()
            if not choices and usage is not None:
                state.event({'t_ns': now, 'kind': 'usage',
                             'cumulative_completion': state.last_cumulative})
            state.any_event.set()
    except StreamProtocolError as exc:
        if not stop.is_set():
            state.error = str(exc)
            state.error_at_ns = time.monotonic_ns()
    except Exception as exc:  # noqa: BLE001 - cancellation after stop is not an error
        if not stop.is_set():
            state.error = type(exc).__name__
            state.error_at_ns = time.monotonic_ns()
    finally:
        if not stop.is_set() and state.finish_at_ns is None and state.error is None:
            state.error = 'stream_closed_without_finish'
            state.error_at_ns = time.monotonic_ns()
        if state.last_cumulative is None:
            state.error = state.error or 'completion_counts_missing'
        # Do not close here: the finalizer owns cancellation/close and may still
        # need the socket handle for a bounded shutdown.
        state.thread_done = True
        state.done_at_ns = time.monotonic_ns()
        state.opened.set()
        state.any_event.set()


def _cumulative_at(state, boundary_ns):
    value = None
    for at_ns, cumulative in state.counts:
        if at_ns > boundary_ns:
            break
        value = cumulative
    return value


def _gaps_ms(times):
    return [round((later - earlier) / 1_000_000.0, 3)
            for earlier, later in zip(times, times[1:])]


def _stream_window(state, window):
    if not window or window.get('measured') is not True:
        return {
            'start_completion': None, 'end_completion': None,
            'completion_delta': None, 'in_range': None,
            'arrivals': 0, 'arrival_offsets_ms': [], 'gaps_ms': [],
            'gap_ms_p50': None, 'gap_ms_p95': None, 'gap_ms_p99': None,
            'gap_ms_max': None, 'open_gap_ms': None,
            'rate_tokens_per_s': None, 'arrival_rate_per_s': None,
            'coalesced_arrivals': 0, 'client_visible': True,
            'token_count_basis': 'none',
            'readiness_basis': 'none',
            'content_observations': 0,
            'content_without_count': state.content_without_count,
            'repeated_counts': state.repeated_counts,
            'usage_only_arrivals': 0,
            'reasoning_only_arrivals': 0,
            'counts_truncated': state.counts_truncated,
            'coalescing_note': COALESCING_NOTE,
            'cadence_basis': ('token-count arrivals only; content-only and '
                              'repeated-count events are not cadence'),
        }
    start_ns = int(window['start_ns'])
    end_ns = int(window['end_ns'])
    span_s = max(0.0, (end_ns - start_ns) / 1e9)
    arrival_rows = [(stamp, state.arrival_increments[i], state.arrival_kinds[i])
                    for i, stamp in enumerate(state.arrivals)
                    if start_ns <= stamp < end_ns]
    arrivals = [row[0] for row in arrival_rows]
    gaps = _gaps_ms(arrivals)
    start_completion = _cumulative_at(state, start_ns)
    end_completion = _cumulative_at(state, end_ns)
    delta = (end_completion - start_completion
             if start_completion is not None and end_completion is not None else None)
    prior_arrivals = [stamp for stamp in state.arrivals if stamp <= end_ns]
    open_gap = (_ms(end_ns - prior_arrivals[-1]) if prior_arrivals
                else (_ms(end_ns - start_ns) if span_s > 0 else None))
    coalesced = sum(1 for _, increment, _ in arrival_rows if increment > 1)
    content_in_window = sum(
        1 for event in state.events
        if event.get('kind') == 'content_without_count'
        and start_ns <= event.get('t_ns', 0) < end_ns)
    if start_completion is not None and end_completion is not None:
        count_basis = 'continuous_usage_stats'
    elif state.last_cumulative is not None:
        count_basis = 'partial_cumulative_counts'
    else:
        count_basis = 'unavailable'
    if arrivals:
        readiness_basis = 'first_cumulative_token_advance'
    elif state.first_content_at_ns is not None:
        readiness_basis = 'content_without_counts_not_token_evidence'
    else:
        readiness_basis = 'no_client_visible_data'
    return {
        'start_completion': start_completion,
        'end_completion': end_completion,
        'completion_delta': delta,
        'in_range': None,
        'arrivals': len(arrivals),
        'arrival_offsets_ms': [round((stamp - start_ns) / 1_000_000.0, 3)
                               for stamp in arrivals],
        'arrival_increments': [increment for _, increment, _ in arrival_rows],
        'arrival_kinds': [kind for _, _, kind in arrival_rows],
        'gaps_ms': gaps,
        'gap_ms_p50': _pct(gaps, 0.5),
        'gap_ms_p95': _pct(gaps, 0.95),
        'gap_ms_p99': _pct(gaps, 0.99),
        'gap_ms_max': max(gaps) if gaps else None,
        'open_gap_ms': open_gap,
        'rate_tokens_per_s': (round(delta / span_s, 3)
                              if delta is not None and span_s > 0 else None),
        'arrival_rate_per_s': round(len(arrivals) / span_s, 3) if span_s > 0 else None,
        'coalesced_arrivals': coalesced,
        'usage_only_arrivals': sum(1 for _, _, kind in arrival_rows
                                   if kind == 'usage_only'),
        'reasoning_only_arrivals': sum(1 for _, _, kind in arrival_rows
                                       if kind == 'reasoning'),
        'content_observations': state.content_observations,
        'content_without_count': state.content_without_count,
        'content_without_count_in_window': content_in_window,
        'repeated_counts': state.repeated_counts,
        'cumulative_advances_total': state.cumulative_advances,
        'missing_count_events': state.missing_count_events,
        'counts_truncated': state.counts_truncated,
        'client_visible': True,
        'token_count_basis': count_basis,
        'readiness_basis': readiness_basis,
        'coalescing_note': COALESCING_NOTE,
        'cadence_basis': ('token-count arrivals only; a multi-token/final usage '
                          'batch is one coalesced arrival, repeated counts and '
                          'content-only chunks are not cadence'),
    }


def _finalize_streams(cfg, states, stop, window, t0_ns):
    stop.set()
    stop_at_ns = time.monotonic_ns()
    # A silent/blocked SSE read cannot observe the stop event.  Actively close
    # only this driver's own connections before the bounded joins.  The engine
    # is not restarted/resumed for socket cleanup.
    for state in states:
        _cancel_stream(state)
    deadline = time.monotonic() + cfg.drain_timeout_s
    for state in states:
        if state.thread is not None and state.thread.is_alive():
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    records = []
    for state in states:
        timed_out = state.thread is not None and state.thread.is_alive()
        window_end = int(window['end_ns']) if window and window.get('measured') else None
        if timed_out:
            disposition = 'drain_timeout'
        elif state.finish_at_ns is not None and window_end is not None \
                and state.finish_at_ns < window_end:
            disposition = 'finished_before_window'
        elif state.finish_at_ns is not None:
            disposition = 'finished_after_window'
        elif state.error is not None:
            disposition = 'error'
        else:
            disposition = 'cancelled'
        usage = state.last_usage or {}
        prompt = usage.get('prompt_tokens')
        cached = usage.get('cached_tokens')
        records.append({
            'stream': state.index,
            'request_sha256': state.request_sha256,
            'prompt_sha256': state.meta.get('prompt_sha256'),
            'prompt_chars': state.meta.get('system_chars', 0)
                            + state.meta.get('user_chars', 0),
            'prompt_words': state.meta.get('user_words'),
            'send_at_ns': state.send_at_ns,
            'first_arrival_at_ns': state.first_at_ns,
            'first_content_at_ns': state.first_content_at_ns,
            'ttft_ms': round(state.ttft_ms, 3) if state.ttft_ms is not None else None,
            'finish': state.finish,
            'finish_at_ns': state.finish_at_ns,
            'error': state.error,
            'error_at_ns': state.error_at_ns,
            'error_before_window_end': bool(
                state.error_at_ns is not None and window_end is not None
                and state.error_at_ns < window_end),
            'done_at_ns': state.done_at_ns,
            'done_marker': state.done_marker,
            'disposition': disposition,
            'drain_timeout': timed_out,
            'cancel_requested': state.cancel_requested,
            'reader_done': state.thread_done,
            'prompt_tokens': prompt,
            'cached_tokens': cached,
            'uncached_tokens': ((prompt - cached)
                                if _is_int(prompt) and _is_int(cached) else None),
            'completion_tokens_final': usage.get('completion_tokens'),
            'completion_counts_observed': state.last_cumulative is not None,
            'usage_source': 'live_continuous' if usage else None,
            'window': _stream_window(state, window),
            'events': state.events,
            'event_count': len(state.events),
            'events_truncated': state.events_truncated,
        })
    return records, stop_at_ns


def _exposure(streams, workload):
    rows = []
    for stream in streams:
        window = stream.get('window') or {}
        delta = window.get('completion_delta')
        proven = _is_int(delta) and delta >= 0
        start = window.get('start_completion')
        in_range = (proven and workload.completion_min <= delta <= workload.completion_max
                    and _is_int(start) and workload.start_completion_min <= start <= workload.start_completion_max)
        window['in_range'] = in_range if proven else None
        rows.append({
            'stream': stream.get('stream'),
            'start_completion': window.get('start_completion'),
            'end_completion': window.get('end_completion'),
            'completion_delta': delta,
            'in_range': in_range if proven else None,
        })
    reasons = []
    for row in rows:
        if row['completion_delta'] is None:
            reasons.append('completion_counters_unavailable:%s' % row['stream'])
        elif row['in_range'] is not True:
            reasons.append('completion_delta_out_of_range:%s:%s'
                           % (row['stream'], row['completion_delta']))
    return {
        'policy': {
            'per_request_delta_min': workload.completion_min,
            'per_request_delta_max': workload.completion_max,
            'per_request_start_min': workload.start_completion_min,
            'per_request_start_max': workload.start_completion_max,
            'matched_across_routes': True,
            'context_position_policy': workload.context_position_policy,
            'common_window_s': workload.window_s,
        },
        'proven': not reasons,
        'streams': rows,
        'reasons': reasons,
        'note': ('owner-declared per-request completion range; out-of-range or '
                 'unproven exposure invalidates the repeat instead of choosing a '
                 'different window'),
    }


def _repeat_metrics(streams, window):
    wall_s = float(window.get('wall_s') or 0.0) if window else 0.0
    deltas = [stream['window'].get('completion_delta') for stream in streams
              if _is_int(stream['window'].get('completion_delta'))]
    rates = [stream['window'].get('rate_tokens_per_s') for stream in streams
             if stream['window'].get('rate_tokens_per_s') is not None]
    gaps = [gap for stream in streams for gap in stream['window'].get('gaps_ms') or []]
    open_gaps = [stream['window'].get('open_gap_ms') for stream in streams
                 if stream['window'].get('open_gap_ms') is not None]
    total = sum(deltas) if len(deltas) == len(streams) else None
    return {
        'streams': len(streams),
        'wall_s': round(wall_s, 6),
        'total_completion_delta': total,
        'aggregate_tokens_per_s': (round(total / wall_s, 3)
                                   if total is not None and wall_s > 0 else None),
        'in_window_arrivals': sum(stream['window'].get('arrivals', 0)
                                  for stream in streams),
        'stream_rates_tokens_per_s': rates,
        'stream_rate_p50': _pct(rates, 0.5),
        'stream_rate_p5': _pct(rates, 0.05),
        'gap_ms_p50': _pct(gaps, 0.5),
        'gap_ms_p95': _pct(gaps, 0.95),
        'gap_ms_p99': _pct(gaps, 0.99),
        'gap_ms_max': max(gaps) if gaps else None,
        'open_gap_ms_p50': _pct(open_gaps, 0.5),
        'open_gap_ms_p95': _pct(open_gaps, 0.95),
        'open_gap_ms_max': max(open_gaps) if open_gaps else None,
        'errors_before_window_end': sum(
            1 for stream in streams if stream.get('error_before_window_end')),
    }


# --- one repeat -------------------------------------------------------------

def _new_repeat_record(cfg, repeat, route):
    return {
        'schema': 2,
        'repeat': repeat,
        'route': route,
        'run_id': cfg.run_id,
        'cohort_id': None,
        'not_run': False,
        'phase': 'started',
        'calls': {},
        'required_calls': [],
        'open': {'complete': False, 'opened_streams': 0, 'reasons': []},
        'admissions_sealed': False,
        'sealed_at_ns': None,
        'submissions_after_seal': 0,
        'submissions_after_seal_streams': [],
        'unsubmitted_at_seal': [],
        'request_bytes_changed': [],
        'window': None,
        'exposure': None,
        'interval': None,
        'cleanup': None,
        'streams': [],
        'metrics': None,
        'drain': None,
        'valid': False,
        'reasons': [],
        'severity': 'inconclusive',
        'abort_run': False,
    }


def measure_repeat(cfg, controller, run_ctx, repeat, route, requests, meta,
                   expected_hashes, preparation):
    t0_ns = run_ctx['t0_ns']
    workload = cfg.workload
    record = _new_repeat_record(cfg, repeat, route)
    repeat_start_ns = time.monotonic_ns()
    cohort_id = 'cohort-' + _sha256_json(
        {'run_id': cfg.run_id, 'repeat': repeat})[:24]
    record['cohort_id'] = cohort_id
    run_ctx.update({'repeat': repeat, 'cohort_id': cohort_id})

    def finish_early(reason, severity='inconclusive'):
        record['reasons'].append(severity + ':' + reason)
        record['severity'] = _severity(record['reasons'])
        record['abort_run'] = True
        record['phase'] = 'aborted'
        return record

    # 1: prepare verifies the full route and binds the engine incarnation.
    prepare = controller_call(controller, cfg, run_ctx, 'prepare', 'full')
    record['calls']['prepare'] = prepare
    record['required_calls'].append('prepare')
    if not prepare['ok']:
        detail = ','.join(prepare['reasons'])
        record['reasons'] = [prepare['severity'] + ':prepare_receipt_invalid:'
                             + detail]
        record['severity'] = prepare['severity']
        record['phase'] = 'prepare_failed'
        record['abort_run'] = True
        return record

    # Optional owner-declared cache preparation, on full, uncounted.
    if repeat == run_ctx.get('repeat_start', 0) and workload.cache_preparation and not preparation.get('done'):
        preparation.update(prepare_cache(cfg, run_ctx['url'], run_ctx['model'],
                                         requests, meta, expected_hashes))
        preparation['done'] = True

    states = []
    for index, messages in enumerate(requests):
        body = request_body(run_ctx['model'], messages, workload)
        state = StreamState(index, messages, meta[index], body)
        states.append(state)
    stop = threading.Event()
    for state in states:
        state.thread = threading.Thread(
            target=stream_reader, args=(state, cfg, run_ctx['url'], stop),
            daemon=True)
        state.thread.start()

    # 2-3: every stream must advance; then seal admissions before any route change.
    deadline = time.monotonic_ns() + int(workload.open_timeout_s * 1e9)
    for state in states:
        remaining = max(0.0, (deadline - time.monotonic_ns()) / 1e9)
        state.opened.wait(remaining)
    opened = [state for state in states if state.first_at_ns is not None]
    sealed_at_ns = time.monotonic_ns()
    late_submissions = [state.index for state in states
                        if state.send_at_ns is not None
                        and state.send_at_ns > sealed_at_ns]
    record['submissions_after_seal'] = len(late_submissions)
    record['submissions_after_seal_streams'] = late_submissions
    record['unsubmitted_at_seal'] = [state.index for state in states
                                     if state.send_at_ns is None]
    record['admissions_sealed'] = True
    record['sealed_at_ns'] = sealed_at_ns
    open_reasons = []
    for state in states:
        if state.send_at_ns is None:
            open_reasons.append('stream_not_submitted:%d' % state.index)
        if state.first_at_ns is None:
            if state.content_observations and state.last_cumulative is None:
                open_reasons.append('stream_content_without_cumulative_counts:%d'
                                    % state.index)
            elif state.error:
                open_reasons.append('stream_not_advanced_error:%d:%s'
                                    % (state.index, state.error))
            else:
                open_reasons.append('stream_not_advanced:%d' % state.index)
        elif state.finish_at_ns is not None and state.finish_at_ns < sealed_at_ns:
            open_reasons.append('early_finish_before_seal:%d' % state.index)
    order = sorted(opened, key=lambda item: item.first_at_ns)
    record['open'] = {
        'complete': not open_reasons,
        'opened_streams': len(opened),
        'advance_order': [state.index for state in order],
        'ttft_ms': [{'stream': state.index, 'ttft_ms': round(state.ttft_ms, 3)}
                    for state in order if state.ttft_ms is not None],
        'wall_s': round((sealed_at_ns - repeat_start_ns) / 1e9, 3),
        'reasons': open_reasons,
        'basis': ('first client-visible token per live request; TTFT belongs to '
                  'preparation, not the measured decode window'),
    }
    if open_reasons:
        streams, stop_at = _finalize_streams(cfg, states, stop, None, t0_ns)
        record['streams'] = streams
        alive = [state.index for state in states
                 if state.thread is not None and state.thread.is_alive()]
        record['drain'] = {'stop_at_ns': stop_at,
                           'timeout': any(item['drain_timeout'] for item in streams),
                           'readers_alive': alive}
        record['cleanup'] = {
            'attempted': False,
            'restore_before_drain': True,
            'leave_ok': True,
            'resume_ok': True,
            'note': 'cohort never opened; no route change was requested',
        }
        return finish_early('open_incomplete:' + ','.join(open_reasons))

    def keep_measurement_reasons(call):
        """Keep FAIL/INCONCLUSIVE visible without blocking safe cleanup."""
        for reason in call.get('reasons') or []:
            text = str(reason)
            if not text.startswith(('contract:', 'inconclusive:')):
                text = 'inconclusive:' + text
            if text not in record['reasons']:
                record['reasons'].append(text)
        record['severity'] = _severity(record['reasons'])

    # 5: pause on the target route after the awaited pause/eligibility checks.
    pause_action = 'pause_enter_split' if route == 'split' else 'pause_full'
    pause = controller_call(controller, cfg, run_ctx, pause_action, route)
    record['calls']['pause'] = pause
    record['required_calls'].append('pause')
    if not pause['lifecycle_ok']:
        streams, stop_at = _finalize_streams(cfg, states, stop, None, t0_ns)
        record['streams'] = streams
        alive = [state.index for state in states
                 if state.thread is not None and state.thread.is_alive()]
        record['drain'] = {'stop_at_ns': stop_at,
                           'timeout': any(item['drain_timeout'] for item in streams),
                           'readers_alive': alive}
        record['cleanup'] = {
            'attempted': False,
            'restore_before_drain': False,
            'leave_ok': False,
            'resume_ok': False,
            'severity': pause['lifecycle_severity'],
            'reason': ('pause receipt did not prove the route state; refusing to '
                       'resume or restore an unknown/partially transitioned route'),
        }
        return finish_early('pause_lifecycle_invalid:' + ','.join(pause['lifecycle_reasons']),
                            pause['lifecycle_severity'])
    if not pause['ok']:
        # This receipt is the cohort entry receipt.  Lifecycle proof only says
        # the route is safely paused; it deliberately ignores decode
        # eligibility and cohort evidence.  Starting a measurement on a failed
        # entry is not cleanup, so keep the measurement failure visible and do
        # not resume into a window.
        keep_measurement_reasons(pause)
        streams, stop_at = _finalize_streams(cfg, states, stop, None, t0_ns)
        record['streams'] = streams
        alive = [state.index for state in states
                 if state.thread is not None and state.thread.is_alive()]
        record['drain'] = {'stop_at_ns': stop_at,
                           'timeout': any(item['drain_timeout'] for item in streams),
                           'readers_alive': alive}
        record['cleanup'] = {
            'attempted': False,
            'restore_before_drain': False,
            'leave_ok': False,
            'resume_ok': False,
            'severity': pause['severity'],
            'reason': ('entry receipt did not prove the cohort/decode entry; the '
                       'only remaining session action from this proved paused '
                       'state would be a measurement resume, so recovery is left '
                       'to the queue owner'),
        }
        return finish_early('pause_measurement_invalid:'
                            + ','.join(pause['reasons'] or ['entry_evidence']),
                            pause['severity'])

    # 6: the accepted resume receipt is the measurement-start boundary.
    resume = controller_call(controller, cfg, run_ctx, 'resume', route)
    record['calls']['resume'] = resume
    record['required_calls'].append('resume')
    if not resume['lifecycle_ok']:
        streams, stop_at = _finalize_streams(cfg, states, stop, None, t0_ns)
        record['streams'] = streams
        alive = [state.index for state in states
                 if state.thread is not None and state.thread.is_alive()]
        record['drain'] = {'stop_at_ns': stop_at,
                           'timeout': any(item['drain_timeout'] for item in streams),
                           'readers_alive': alive}
        record['cleanup'] = {
            'attempted': False,
            'restore_before_drain': False,
            'leave_ok': False,
            'resume_ok': False,
            'severity': resume['lifecycle_severity'],
            'reason': ('resume receipt did not prove the boundary; route state is '
                       'not safe to resume or restore from this driver'),
        }
        return finish_early('resume_lifecycle_invalid:' + ','.join(resume['lifecycle_reasons']),
                            resume['lifecycle_severity'])
    measurement_started = bool(resume['ok'])
    if not measurement_started:
        # The first resume is the measurement boundary.  If its evidence is
        # invalid the window must not start, but the route still has to be
        # restored through the ordinary legal state machine (pause_end ->
        # leave_full -> cleanup resume), each step validated independently.
        keep_measurement_reasons(resume)
        record['window'] = {
            'measured': False,
            'reason': ('first_resume_receipt_invalid:'
                       + ','.join(resume['reasons'] or ['boundary_evidence'])),
            'client_visible': True,
            'basis': ('the first resume receipt did not prove the measurement '
                      'boundary; no client-visible window was opened and no '
                      'interval is used as a measurement'),
        }

    if measurement_started:
        window_start_ns = time.monotonic_ns()
        target_end_ns = window_start_ns + int(round(workload.window_s * 1e9))
        early = []
        errors_before = []
        while True:
            now = time.monotonic_ns()
            early = [state.index for state in states
                     if state.finish_at_ns is not None
                     and state.finish_at_ns < min(now, target_end_ns)]
            errors_before = [state.index for state in states
                             if state.error_at_ns is not None
                             and state.error_at_ns < min(now, target_end_ns)]
            if now >= target_end_ns or early or errors_before:
                break
            wait_s = min((target_end_ns - now) / 1e9, 0.02)
            for state in states:
                state.any_event.wait(wait_s)
            for state in states:
                state.any_event.clear()
        end_ns = time.monotonic_ns()
        early = [state.index for state in states
                 if state.finish_at_ns is not None and state.finish_at_ns < end_ns]
        errors_before = [state.index for state in states
                         if state.error_at_ns is not None and state.error_at_ns < end_ns]
        wall_s = max(0.0, (end_ns - window_start_ns) / 1e9)
        record['window'] = {
            'measured': True,
            'start_ns': window_start_ns,
            'end_ns': end_ns,
            'target_end_ns': target_end_ns,
            'seconds': workload.window_s,
            'wall_s': round(wall_s, 6),
            'ended_early': end_ns < target_end_ns,
            'full': not early and not errors_before,
            'early_finish': early,
            'errors_before_end': errors_before,
            'client_visible': True,
            'basis': WINDOW_BASIS,
            'clock_uncertainty': {
                'driver_boundary': 'time.monotonic_ns',
                'engine_clock_unknown_offset': True,
                'note': ('client-visible boundaries and engine interval counters are '
                         'separate clocks; no synchronisation is assumed'),
            },
        }
        record['phase'] = 'measured'

    # 8: end boundary is issued before pause_end; counters arrive in its receipt.
    pause_end = controller_call(controller, cfg, run_ctx, 'pause_end', route)
    record['calls']['pause_end'] = pause_end
    record['required_calls'].append('pause_end')
    pause_end_return_ns = time.monotonic_ns()
    record['window']['pause_end_issued_at_ns'] = pause_end['issued_at_ns']
    record['window']['pause_end_returned_at_ns'] = pause_end_return_ns
    record['window']['pause_end_latency_ms'] = _ms(
        pause_end_return_ns - pause_end['issued_at_ns'])
    if measurement_started:
        record['window']['end_boundary_before_pause_end'] = (
            pause_end['issued_at_ns'] >= end_ns)
    receipt = pause_end.get('receipt') or {}
    observed_clock = receipt.get('observed_clock_ns')
    if _is_int(observed_clock) and 'clock_uncertainty' in record['window']:
        record['window']['clock_uncertainty'].update({
            'pause_end_receipt_clock_ns': observed_clock,
            'receipt_clock_basis': receipt.get('clock_basis'),
            'observed_client_minus_receipt_ns': pause_end_return_ns - observed_clock,
        })
    interval = receipt.get('interval') or {}
    record['interval'] = {
        'observer_basis': interval.get('observer_basis'),
        'counters_before': interval.get('counters_before'),
        'counters_after': interval.get('counters_after'),
        'unavailable': interval.get('unavailable'),
        'invalidation': interval.get('invalidation'),
        'invalidation_complete': interval.get('invalidation_complete'),
        'invalidation_lower_bounds': interval.get('invalidation_lower_bounds'),
        'invalidation_basis': interval.get('invalidation_basis'),
        'available': interval.get('available'),
        'valid': bool(pause_end['ok']),
        'lifecycle_proven': bool(pause_end['lifecycle_ok']),
        'reasons': ([] if pause_end['ok']
                    else list(pause_end['reasons'])
                    or ['inconclusive:pause_end_receipt_invalid']),
    }
    if not pause_end['ok']:
        # The measured interval is invalid, but that is a measurement result,
        # not by itself a reason to abandon a proven-safe route restore.
        keep_measurement_reasons(pause_end)
    record['measurement'] = {
        'valid': bool(pause_end['ok']) and measurement_started,
        'lifecycle_ok': bool(pause_end['lifecycle_ok']),
        'window_measured': measurement_started,
        'reasons': list(pause_end['reasons']),
    }

    # 9: restore full and resume only if the route chain is proven.  Lifecycle
    # proof is independent of the measured interval: a missing pause, partial or
    # contradictory transition ACK, unknown timeout, changed incarnation or drain
    # failure still refuses mutation.  An unavailable/invalid interval alone does
    # not guess that the route is unsafe.
    route_proven = pause_end['lifecycle_ok']
    leave_ok = resume_ok = False
    if route_proven:
        leave = controller_call(controller, cfg, run_ctx, 'leave_full', 'full')
        record['calls']['leave'] = leave
        record['required_calls'].append('leave')
        leave_ok = leave['lifecycle_ok']
        if leave_ok:
            cleanup_resume = controller_call(controller, cfg, run_ctx, 'resume', 'full')
            record['calls']['cleanup_resume'] = cleanup_resume
            record['required_calls'].append('cleanup_resume')
            resume_ok = cleanup_resume['lifecycle_ok']
    record['cleanup'] = {
        'attempted': bool(route_proven),
        'restore_before_drain': bool(leave_ok and resume_ok),
        'leave_ok': leave_ok,
        'resume_ok': resume_ok,
        'severity': ('valid' if leave_ok and resume_ok
                     else (record['calls'].get('leave', {}).get('lifecycle_severity')
                           or record['calls'].get('cleanup_resume', {}).get('lifecycle_severity')
                           or pause_end['lifecycle_severity'])),
        'reason': (None if leave_ok and resume_ok else
                   ('end receipt contradicted the route chain; refusing to resume '
                    'an unproven route' if not route_proven else
                    'leave_full/cleanup resume did not prove the full route')),
    }
    if not (leave_ok and resume_ok):
        # Route state is unproven; do not start another cohort.  A measured
        # window that is merely short/invalid but safely restored full may be
        # followed by the remaining required repeats (their verdicts are kept).
        record['abort_run'] = True

    # 10: only now are the live client streams allowed to finish or be cancelled.
    streams, stop_at = _finalize_streams(cfg, states, stop, record['window'], t0_ns)
    record['streams'] = streams
    alive = [state.index for state in states
             if state.thread is not None and state.thread.is_alive()]
    record['drain'] = {
        'stop_at_ns': stop_at,
        'timeout': any(item['drain_timeout'] for item in streams),
        'readers_alive': alive,
        'note': ('client connections actively cancelled after the route restore; '
                 'the driver never waits for all requests to finish before ending '
                 'the window'),
    }
    # A reader that survived its bounded join means the cohort is still live and
    # another repeat must not start.  Cancellation is not treated as a success.
    if alive:
        record['abort_run'] = True
    if record['drain']['timeout']:
        record['abort_run'] = True
    record['exposure'] = _exposure(streams, workload)
    record['metrics'] = _repeat_metrics(streams, record['window'])
    record['phase'] = 'completed'
    return record


# --- preparation evidence ---------------------------------------------------

def prepare_cache(cfg, url, model, requests, meta, expected_hashes):
    """Optional uncounted max_tokens=1 pass on full; descriptive evidence only."""
    rows = []
    for index, messages in enumerate(requests):
        row = {
            'stream': index,
            'prompt_sha256': meta[index].get('prompt_sha256'),
            'measured_request_sha256': expected_hashes[index],
            'counted': False,
        }
        started = time.monotonic()
        try:
            result = AR.stream_chat(url, model, messages, 1)
            prompt = result.get('prompt_tokens')
            cached = result.get('cached_tokens')
            uncached = prompt - cached if _is_int(prompt) and _is_int(cached) else None
            minimum = None
            if cfg.workload.block_size and _is_int(prompt):
                minimum = ((prompt - 1) % cfg.workload.block_size) + 1
            row.update({
                'ok': True,
                'seconds': round(result.get('seconds', time.monotonic() - started), 3),
                'prompt_tokens': prompt,
                'cached_tokens': cached,
                'uncached_tokens': uncached,
                'recompute_minimum': minimum,
                'within_last_block': (uncached is not None
                                      and cfg.workload.block_size is not None
                                      and 0 <= uncached <= cfg.workload.block_size),
            })
        except Exception as exc:  # noqa: BLE001 - evidence, never counted
            row.update({'ok': False, 'seconds': round(time.monotonic() - started, 3),
                        'error': type(exc).__name__})
        rows.append(row)
    return {
        'passes': 1,
        'counted': False,
        'max_tokens': 1,
        'block_size': cfg.workload.block_size,
        'rule': RECOMPUTE_RULE,
        'requests': rows,
        'all_ok': all(row.get('ok') for row in rows),
        'all_within_last_block': (all(row.get('within_last_block') for row in rows)
                                  if cfg.workload.block_size else None),
        'descriptive_only': True,
        'note': ('cached-token/recompute equality is preparation evidence, not a '
                 'no-prefill or decode-only proof, and never a validity gate'),
    }


def _apply_preparation_usage(record, preparation):
    """Fill prompt/cached counts for a cancelled stream from the prep pass, labelled."""
    rows = {row.get('stream'): row for row in (preparation.get('requests') or [])}
    for stream in record.get('streams') or []:
        if stream.get('prompt_tokens') is not None:
            stream['usage_source'] = stream.get('usage_source') or 'live_continuous'
            continue
        row = rows.get(stream.get('stream'))
        if row and row.get('ok') and row.get('prompt_tokens') is not None:
            stream.update({
                'prompt_tokens': row.get('prompt_tokens'),
                'cached_tokens': row.get('cached_tokens'),
                'uncached_tokens': row.get('uncached_tokens'),
                'usage_source': 'prepare',
                'usage_note': ('live stream cancelled before final usage; values are '
                               'the uncounted preparation pass over the same prompt'),
            })


# --- pure interval validity -------------------------------------------------

def interval_valid(record, config):
    """Frozen pure validity rule for one repeat -> (valid, reasons, severity).

    VALID only when the manifest was explicit; every required controller call
    was made and accepted; every stream advanced before the seal; no request
    was submitted after sealing or reused more than once; the bounded window
    actually ran to its end with no early EOS/error; completion exposure was
    proven and inside the declared range; the final pause_end interval counters
    were present, monotone, and show no prefill/mixed/unsupported/preempted
    work; and full was restored and resumed before the client streams were
    stopped.  Missing proof is INCONCLUSIVE; contradictions are CONTRACT.
    """
    if type(record) is not dict:
        return False, ['inconclusive:repeat_record_missing'], 'inconclusive'
    if record.get('not_run'):
        reasons = list(record.get('reasons') or ['inconclusive:repeat_not_run'])
        return False, reasons, _severity(reasons)
    reasons = []
    if record.get('run_error'):
        reasons.append('inconclusive:run_error')
    calls = record.get('calls') or {}
    for key in record.get('required_calls') or []:
        call = calls.get(key)
        if type(call) is not dict:
            reasons.append('inconclusive:' + key + '_not_called')
            continue
        if call.get('ok') is not True:
            severity = 'contract' if call.get('severity') == 'contract' \
                else 'inconclusive'
            detail = ','.join(str(item) for item in call.get('reasons') or [])
            reasons.append(severity + ':' + key + '_receipt_invalid'
                           + ((':' + detail) if detail else ''))
    opened = record.get('open') or {}
    if opened.get('complete') is not True:
        reasons.append('inconclusive:open_incomplete')
        for item in opened.get('reasons') or []:
            text = str(item)
            if not text.startswith(('contract:', 'inconclusive:')):
                text = 'inconclusive:' + text
            reasons.append(text)
    if record.get('admissions_sealed') is not True:
        reasons.append('inconclusive:admissions_not_sealed')
    if record.get('submissions_after_seal') != 0:
        reasons.append('contract:request_after_seal')
    for index in record.get('request_bytes_changed') or []:
        reasons.append('contract:request_bytes_changed:%s' % index)
    window = record.get('window')
    measured = type(window) is dict and window.get('measured') is True
    if not measured:
        reasons.append('inconclusive:window_not_measured')
    else:
        if window.get('full') is not True:
            reasons.append('inconclusive:window_incomplete')
            for index in window.get('early_finish') or []:
                reasons.append('inconclusive:early_finish:%s' % index)
            for index in window.get('errors_before_end') or []:
                reasons.append('inconclusive:stream_error_before_window_end:%s'
                               % index)
        if window.get('end_boundary_before_pause_end') is not True:
            reasons.append('contract:end_boundary_after_pause_end')
    exposure = record.get('exposure') or {}
    if measured:
        if exposure.get('proven') is not True:
            reasons.append('inconclusive:exposure_unproven')
            for item in exposure.get('reasons') or []:
                text = str(item)
                if not text.startswith(('contract:', 'inconclusive:')):
                    text = 'inconclusive:' + text
                reasons.append(text)
        else:
            for row in exposure.get('streams') or []:
                if row.get('in_range') is not True:
                    reasons.append('inconclusive:exposure_out_of_range:%s'
                                   % row.get('stream'))
    interval = record.get('interval') or {}
    pause_end_call = calls.get('pause_end') or {}
    if (measured and interval.get('valid') is not True
            and pause_end_call.get('ok') is True):
        for item in interval.get('reasons') or ['inconclusive:interval_counters_invalid']:
            text = str(item)
            if not text.startswith(('contract:', 'inconclusive:')):
                text = 'inconclusive:' + text
            reasons.append(text)
    cleanup = record.get('cleanup') or {}
    if measured and cleanup.get('restore_before_drain') is not True:
        severity = 'contract' if cleanup.get('severity') == 'contract' \
            else 'inconclusive'
        reasons.append(severity + ':restore_not_before_drain')
    if (record.get('drain') or {}).get('timeout'):
        reasons.append('inconclusive:drain_timeout')
    for stream in record.get('streams') or []:
        if stream.get('error_before_window_end'):
            reasons.append('inconclusive:stream_error_before_window_end:%s'
                           % stream.get('stream'))
    deduped = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    valid = not deduped
    return valid, deduped, _severity(deduped)


# --- summary ----------------------------------------------------------------

def _max_or_none(values):
    return max(values) if values else None


def summarize(repeats, config):
    """Statistics over VALID repeats only, explicitly labelled; never PASS."""
    required = config.get('REPEATS')
    parsed = []
    for record in repeats:
        valid, reasons, severity = interval_valid(record, config)
        record['valid'] = valid
        record['reasons'] = reasons
        record['severity'] = severity
        parsed.append(record)
    valid = [record for record in parsed if record.get('valid')]
    invalid = [{'repeat': record.get('repeat'), 'route': record.get('route'),
                'severity': record.get('severity'), 'reasons': record.get('reasons')}
               for record in parsed if not record.get('valid')]
    contract_count = sum(1 for record in parsed
                         if record.get('severity') == 'contract')
    totals = {
        'required': required,
        'attempted': len(parsed),
        'valid': len(valid),
        'invalid': len(invalid),
        'contract': contract_count,
        'not_run': sum(1 for record in parsed if record.get('not_run')),
    }
    routes = {}
    for label in VALID_ROUTES:
        chosen = [record for record in valid if record.get('route') == label]
        aggregates = [record['metrics'].get('aggregate_tokens_per_s')
                      for record in chosen
                      if (record.get('metrics') or {}).get('aggregate_tokens_per_s')
                      is not None]
        rates = [rate for record in chosen
                 for rate in (record.get('metrics') or {}).get(
                     'stream_rates_tokens_per_s') or []]
        gaps = [gap for record in chosen
                for stream in record.get('streams') or []
                for gap in (stream.get('window') or {}).get('gaps_ms') or []]
        open_gaps = [stream['window'].get('open_gap_ms')
                     for record in chosen
                     for stream in record.get('streams') or []
                     if (stream.get('window') or {}).get('open_gap_ms') is not None]
        deltas = [row.get('completion_delta')
                  for record in chosen
                  for row in (record.get('exposure') or {}).get('streams') or []
                  if _is_int(row.get('completion_delta'))]
        routes[label] = {
            'valid_repeats': [record.get('repeat') for record in chosen],
            'aggregate_tokens_per_s_median': _pct(aggregates, 0.5),
            'aggregate_tokens_per_s_min': min(aggregates) if aggregates else None,
            'aggregate_tokens_per_s_max': max(aggregates) if aggregates else None,
            'stream_rate_p50': _pct(rates, 0.5),
            'stream_rate_p5': _pct(rates, 0.05),
            'gap_ms_p50': _pct(gaps, 0.5),
            'gap_ms_p95': _pct(gaps, 0.95),
            'gap_ms_p99': _pct(gaps, 0.99),
            'gap_ms_max': max(gaps) if gaps else None,
            'open_gap_ms_p50': _pct(open_gaps, 0.5),
            'open_gap_ms_p95': _pct(open_gaps, 0.95),
            'open_gap_ms_max': max(open_gaps) if open_gaps else None,
            'completion_delta_min': min(deltas) if deltas else None,
            'completion_delta_median': _pct(deltas, 0.5),
            'completion_delta_max': max(deltas) if deltas else None,
            'streams_measured': len(rates),
        }
    full = routes['full']['aggregate_tokens_per_s_median']
    split = routes['split']['aggregate_tokens_per_s_median']
    matched = {
        'basis': ('median per-repeat aggregate decode-token rate over VALID '
                  'repeats only; client-visible counts'),
        'full_valid': len(routes['full']['valid_repeats']),
        'split_valid': len(routes['split']['valid_repeats']),
        'split_over_full': (round(split / full, 4) if full and split else None),
        'exposure_compatible': bool(
            routes['full']['valid_repeats'] and routes['split']['valid_repeats']),
        'reason': (None if full and split else
                   'one route has no VALID repeat with a completion-count rate'),
    }
    preparation_ttft = []
    for record in parsed:
        opened = record.get('open') or {}
        for row in opened.get('ttft_ms') or []:
            preparation_ttft.append(row.get('ttft_ms'))
    verdict = ('FAIL' if contract_count else
               'COMPLETE' if (totals['valid'] == required
                              and totals['attempted'] == required) else
               'INCONCLUSIVE')
    return {
        'verdict': verdict,
        'descriptive_valid_only': True,
        'totals': totals,
        'note': ('Valid-only descriptive statistics; invalid repeats are retained '
                 'below with every reason. This is not a PASS and never substitutes '
                 'performance for output correctness.'),
        'invalid': invalid,
        'routes': routes,
        'matched': matched,
        'preparation_ttft': {
            'basis': ('first client-visible token in the open/preparation phase; '
                      'reported separately from the measured window'),
            'ttft_ms_p50': _pct(preparation_ttft, 0.5),
            'ttft_ms_p95': _pct(preparation_ttft, 0.95),
            'samples': len(preparation_ttft),
        },
        'output_correctness': {
            'checked': False,
            'reason': CORRECTNESS_NOTE,
        },
    }


# --- run / artifacts --------------------------------------------------------

def _not_run_record(cfg, repeat, route, reason):
    record = _new_repeat_record(cfg, repeat, route)
    record['not_run'] = True
    record['phase'] = 'not_run'
    record['reasons'] = ['inconclusive:run_aborted:' + reason]
    record['severity'] = 'inconclusive'
    return record


def _config_artifact(cfg, model, expected_digest, clock):
    workload = cfg.workload
    return {
        'schema': 2,
        'WORKLOAD': workload.name,
        'WORKLOAD_SHA256': workload.manifest_sha256,
        'STREAMS': workload.streams,
        'PROMPT_TOKENS_TARGET': workload.prompt_tokens,
        'SEED': workload.seed,
        'SAMPLING_SEED': workload.seed,
        'SEED_PURPOSE': SEED_PURPOSE,
        'GENERATION_BUDGET': workload.generation_budget,
        'REPEATS': workload.repeats,
        'ROUTE_PLAN': list(workload.route_plan),
        'WINDOW_S': workload.window_s,
        'OPEN_TIMEOUT_S': workload.open_timeout_s,
        'PAUSE_TIMEOUT_S': workload.pause_timeout_s,
        'CACHE_PREPARATION': workload.cache_preparation,
        'REQUEST_SHAPE': {
            'temperature': workload.temperature,
            'min_tokens': workload.min_tokens_mode,
            'include_usage': workload.include_usage,
            'continuous_usage_stats': workload.continuous_usage_stats,
            'enable_thinking': workload.enable_thinking,
        },
        'COMPLETION_POLICY': {
            'per_request_delta_min': workload.completion_min,
            'per_request_delta_max': workload.completion_max,
            'per_request_start_min': workload.start_completion_min,
            'per_request_start_max': workload.start_completion_max,
            'matched_across_routes': True,
            'context_position_policy': workload.context_position_policy,
        },
        'BLOCK_SIZE': workload.block_size,
        'RUN_ID': cfg.run_id,
        'MODEL_SHA256': hashlib.sha256(str(model).encode()).hexdigest(),
        'CONTROLLER_PRESENT': bool(cfg.controller_cmd),
        'CONTROLLER_CMD_SHA256': (hashlib.sha256(cfg.controller_cmd.encode()).hexdigest()
                                  if cfg.controller_cmd else None),
        'REQUEST_SHA256': expected_digest,
        'RECOMPUTE_RULE': RECOMPUTE_RULE,
        'CLOCK': clock,
    }


def run(cfg, url, model, controller, context=None):
    t0_ns = time.monotonic_ns()
    wall_anchor_ns = time.time_ns()
    clock = {
        'monotonic_anchor_ns': t0_ns,
        'wall_anchor_ns': wall_anchor_ns,
        'basis': ('all event times are absolute time.monotonic_ns from the client; '
                  'one wall-clock anchor is recorded here and never used for '
                  'intervals'),
        'engine_clock_offset_unknown': True,
    }
    requests, meta = build_requests(cfg.workload)
    _bodies, expected_hashes, expected_digest = build_request_set(
        model, requests, cfg.workload)
    preparation = {
        'done': False,
        'counted': False,
        'passes': 0,
        'note': ('owner manifest enabled the optional cache preparation pass'
                 if cfg.workload.cache_preparation else
                 'owner manifest disabled the cache preparation pass'),
    }
    run_ctx = {
        'run_id': cfg.run_id,
        'url': url,
        'model': model,
        't0_ns': t0_ns,
        'engine_epoch': None,
        'storage_epoch': None,
        'route': 'full',
        'route_generation': None,
        'command_seq': 0,
        'prior_receipt': None,
        'streams': cfg.workload.streams,
        'repeat': 0,
        'cohort_id': None,
    }
    repeat_start = 0
    if context is not None:
        if (context.get('run_id') != cfg.run_id or context.get('route') != 'full'
                or type(context.get('command_seq')) is not int
                or type(context.get('repeat')) is not int
                or context.get('prior_receipt') is None):
            raise ConfigError('invalid composed session continuation')
        repeat_start = context['repeat'] + 1
        saved = {k: context[k] for k in ('engine_epoch', 'storage_epoch', 'route',
                 'route_generation', 'command_seq', 'prior_receipt', 'repeat')}
        context.update(run_ctx)
        context.update(saved)
        run_ctx = context
    run_ctx['repeat_start'] = repeat_start
    config = _config_artifact(cfg, model, expected_digest, clock)
    repeats = []
    abort_reason = None
    for repeat, route in enumerate(cfg.workload.route_plan, start=repeat_start):
        if abort_reason is not None:
            repeats.append(_not_run_record(cfg, repeat, route, abort_reason))
            continue
        record = measure_repeat(cfg, controller, run_ctx, repeat, route, requests,
                                meta, expected_hashes, preparation)
        _apply_preparation_usage(record, preparation)
        changed = [stream['stream'] for stream in record.get('streams') or []
                   if stream.get('request_sha256') != expected_hashes[stream['stream']]]
        record['request_bytes_changed'] = changed
        valid, reasons, severity = interval_valid(record, config)
        record['valid'] = valid
        record['reasons'] = reasons
        record['severity'] = severity
        repeats.append(record)
        if record.get('abort_run'):
            abort_reason = 'repeat_%d_%s' % (repeat, severity)
    summary = summarize(repeats, config)
    data = {
        'schema': 2,
        'config': config,
        'workload': {
            'status': 'loaded',
            'name': cfg.workload.name,
            'manifest_sha256': cfg.workload.manifest_sha256,
            'standing_w1': cfg.workload.name == STANDING_NAME,
        },
        'clock': clock,
        'preparation_evidence': preparation,
        'summary': summary,
        'repeats': repeats,
    }
    write_output(cfg.out_path, data)
    return data


def write_output(path, data):
    with open(path, 'x') as handle:
        json.dump(data, handle, indent=1)


def plan_only(env, out_path, reasons):
    run_id = env.get('RUN_ID') or ('w1-plan-%d' % time.time_ns())
    return {
        'schema': 2,
        'config': {
            'RUN_ID': run_id,
            'WORKLOAD_SHA256': None,
            'CONTROLLER_PRESENT': bool(env.get('CONTROLLER_CMD')),
        },
        'workload': {'status': 'unavailable', 'reasons': list(reasons)},
        'clock': None,
        'preparation_evidence': None,
        'summary': {
            'verdict': 'INCONCLUSIVE',
            'descriptive_valid_only': True,
            'totals': {'required': None, 'attempted': 0, 'valid': 0,
                       'invalid': 0, 'contract': 0, 'not_run': 0},
            'invalid': [{'repeat': None, 'route': None,
                         'severity': 'inconclusive',
                         'reasons': ['inconclusive:' + str(item)
                                     for item in reasons]}],
            'note': ('PLAN-ONLY: the owner workload manifest, generation budget, '
                     'seed, request shape or window lengths are absent, or the '
                     'controller is not configured. No requests were issued and no '
                     'settings were invented.'),
        },
        'repeats': [],
    }


def resummarize(path):
    with open(path) as handle:
        data = json.load(handle)
    config = data.get('config')
    if type(config) is not dict or not _is_int(config.get('REPEATS')):
        raise ValueError('artifact has no usable config.REPEATS')
    data['summary'] = summarize(data.get('repeats') or [], config)
    return data


def exit_code(data):
    summary = data.get('summary') or {}
    totals = summary.get('totals') or {}
    if (totals.get('contract') or 0) > 0:
        return 1
    if summary.get('verdict') == 'COMPLETE':
        return 0
    return 2


def main(argv, controller=None, env=None):
    env = os.environ if env is None else env
    if argv and argv[0] == '--resummarize':
        if len(argv) != 2:
            print('usage: e3_w1_decode_driver.py --resummarize OUT.json',
                  file=sys.stderr)
            return 2
        try:
            data = resummarize(argv[1])
        except (OSError, ValueError, KeyError) as exc:
            print('e3_w1_decode_driver resummarize failed: ' + type(exc).__name__,
                  file=sys.stderr)
            return 2
        print('e3_w1_decode_driver resummarized '
              + json.dumps(data.get('summary'), sort_keys=True))
        return exit_code(data)
    if len(argv) != 3:
        print('usage: e3_w1_decode_driver.py URL MODEL OUT.json '
              '[--resummarize OUT.json]', file=sys.stderr)
        return 2
    url, model, out_path = argv
    if Path(out_path).exists() or Path(out_path).is_symlink():
        print('e3_w1_decode_driver refuses an existing output artifact', file=sys.stderr)
        return 2
    try:
        cfg = config_from_env(env, out_path)
    except ConfigError as exc:
        data = plan_only(env, out_path, exc.reasons)
        try:
            write_output(out_path, data)
        except OSError:
            print('e3_w1_decode_driver cannot write ' + out_path, file=sys.stderr)
        print('e3_w1_decode_driver ' + json.dumps(data['summary'], sort_keys=True))
        return 2
    if controller is None:
        if not cfg.controller_cmd:
            data = plan_only(env, out_path,
                             ['CONTROLLER_CMD is required: a schema-1 controller '
                              'receipt is the only route/decode evidence'])
            try:
                write_output(out_path, data)
            except OSError:
                print('e3_w1_decode_driver cannot write ' + out_path, file=sys.stderr)
            print('e3_w1_decode_driver ' + json.dumps(data['summary'], sort_keys=True))
            return 2
        try:
            controller = CommandController(cfg.controller_cmd)
        except ConfigError as exc:
            data = plan_only(env, out_path, exc.reasons)
            write_output(out_path, data)
            print('e3_w1_decode_driver ' + json.dumps(data['summary'], sort_keys=True))
            return 2
    try:
        data = run(cfg, url, model, controller)
    except Exception as exc:  # noqa: BLE001 - keep an artifact for the operator
        data = plan_only(env, out_path, ['run_failed:' + type(exc).__name__])
        data['summary']['verdict'] = 'INCONCLUSIVE'
        try:
            write_output(out_path, data)
        except OSError:
            pass
        print('e3_w1_decode_driver run failed: ' + type(exc).__name__,
              file=sys.stderr)
        return 2
    print('e3_w1_decode_driver ' + json.dumps(data['summary'], sort_keys=True))
    return exit_code(data)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
