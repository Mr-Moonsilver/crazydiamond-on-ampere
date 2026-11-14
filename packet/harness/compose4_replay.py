#!/usr/bin/env python3
"""The closed-loop agent replay for q18a-compose-4: harness/agent_replay.py's loop (STALL-PLAN 3b.6) and its summary, with four changes for this run, nothing else.

  auth    the arm's key from the environment, as every box client; never printed.
  salt    one cache salt per arm, agent and session, so two arms with the same draws do not hit each other's cache.
  draws   one seeded stream per quantity (session starts, appends, replies, pauses, text), so a small difference in a reply's length can't shift what comes next: both arms send the same sequence, timing aside.
  ceiling an append that would take the prompt over MAX_PROMPT_TOKENS rolls to a new session instead (READY: the cap's arithmetic).
Each turn keeps its client-visible arrival times (ms from the arm's start), which the rule reader needs for decode beside a bulk. The text is agent_replay.text, unchanged.
REQUEST_IDS in the replay config defaults to False; True adds per-turn request/SSE identity evidence only.

Reply text (brief 11 B.1, 14 Nov 2026): REPLAY_REPLY_TEXT in the environment, read once by run().
  model   (default; also unset or empty) the model's sampled reply text goes back into the history, as before: the
          request sequence, the turn rows and the summary are byte-for-byte those of the earlier driver.
  seeded  the history gets a seeded synthetic text of the turn's drawn reply length instead (agent_replay.text from a
          NEW 'reply_text' stream, drawn once per turn before the request, so its position depends on the turn index
          alone). Every prompt and every session roll then depends on the seed alone, on both engines, whatever the
          model samples and whenever it completes. The start/append/reply/pause/text streams and the forced decode
          length (min_tokens = max_tokens = reply) are unchanged. Turn rows carry reply_text='seeded'; the summary
          carries reply_text='seeded'. The replay config is not changed (the j2 mechanism grade compares it with the
          table). Any other value is refused before the first agent starts.

Fixed work (brief 12, 14 Nov 2026): REPLAY_FIXED_WORK and REPLAY_FIXED_CAP_S in the environment, read once by run().
  unset/empty (default) the arm sends while the clock is inside DURATION, as before: nothing else changes.
  SEED=N0,...,N7[/SEED=...]  per seed, each agent's target turn count. The arm then sends past DURATION until EVERY
          agent has completed its own target (a turn without error), so each agent's first N turns run with all eight
          agents live; agents that reach their target early keep sending. When the last one gets there no agent sends
          again; turns already in flight finish (as at DURATION today). REPLAY_FIXED_CAP_S (> DURATION, required) is
          the hard cap: past it nothing is sent and summary['fixed_work']['cap_hit'] is True (the arm grades INC,
          e3_compose4_box). Needs REPLAY_REPLY_TEXT=seeded: only then are the first N turns the same work on both
          engines. The summary's turn counts stay those of the DURATION window (summarize, unchanged); the per-agent
          completion times are in the turn rows; summary['fixed_work'] holds the targets, the cap, each agent's
          time to its target and the stop time. The stop and the cap are checked where DURATION is (top of the
          loop, before the turn's text is built). Latency percentiles in the summary cover every turn sent. The replay
          config is not changed. A bad setting is refused before the first agent starts.
"""
import json
import os
import random
import threading
import time
import urllib.request
import hashlib

import agent_replay as ar
import compose4_arms as T

READ_TIMEOUT_S = 420.0            # one read; the arm's own timeout is 1500 s, so a dead engine shows as an error inside the arm
FINISH_BOUND_S = 300.0            # how long after the end of the duration the turns still open may take; then the arm is called a hang
REPLY_TEXT_ENV = 'REPLAY_REPLY_TEXT'
FIXED_WORK_ENV = 'REPLAY_FIXED_WORK'
FIXED_CAP_ENV = 'REPLAY_FIXED_CAP_S'
REPLY_TEXT_MODES = ('model', 'seeded')


def reply_text_mode(environ=None):
    value = (os.environ if environ is None else environ).get(REPLY_TEXT_ENV, '') or 'model'
    if value not in REPLY_TEXT_MODES:
        raise ValueError('%s must be one of %s, not %r' % (REPLY_TEXT_ENV, '|'.join(REPLY_TEXT_MODES), value))
    return value


def fixed_work_targets(cfg, environ=None, reply_text='model'):
    """None (fixed work off) or (targets for cfg's seed, one per agent, cap seconds); ValueError on a bad setting."""
    env = os.environ if environ is None else environ
    spec, cap = env.get(FIXED_WORK_ENV, ''), env.get(FIXED_CAP_ENV, '')
    if not spec:
        if cap:
            raise ValueError('%s without %s' % (FIXED_CAP_ENV, FIXED_WORK_ENV))
        return None
    if reply_text != 'seeded':
        raise ValueError('%s needs %s=seeded (the first N turns are the same work only then)' % (FIXED_WORK_ENV, REPLY_TEXT_ENV))
    table = {}
    for part in spec.split('/'):
        seed, sep, values = part.partition('=')
        try:
            seed, targets = int(seed), [int(v) for v in values.split(',')]
        except ValueError:
            raise ValueError('%s must be SEED=N0,...,N7[/SEED=...], not %r' % (FIXED_WORK_ENV, spec)) from None
        if not sep or seed in table or len(targets) != cfg['AGENTS'] or any(n < 1 for n in targets):
            raise ValueError('%s must give %d positive targets once per seed, not %r' % (FIXED_WORK_ENV, cfg['AGENTS'], spec))
        table[seed] = targets
    if cfg['SEED'] not in table:
        raise ValueError('%s has no targets for seed %d' % (FIXED_WORK_ENV, cfg['SEED']))
    try:
        cap = float(cap)
    except ValueError:
        raise ValueError('%s needs %s in seconds' % (FIXED_WORK_ENV, FIXED_CAP_ENV)) from None
    if not cap > cfg['DURATION']:
        raise ValueError('%s must be above DURATION (%s s)' % (FIXED_CAP_ENV, cfg['DURATION']))
    return table[cfg['SEED']], cap


class FixedWork:
    """Each agent's completed turns against its target; the stop when the last agent reaches its target."""

    def __init__(self, targets, cap_s, clock, origin):
        self.targets, self.cap_s, self.clock, self.origin = list(targets), cap_s, clock, origin
        self.done = [0] * len(targets)
        self.reached = [None] * len(targets)
        self.end_s = None
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def completed(self, agent):
        with self.lock:
            self.done[agent] += 1
            if self.done[agent] == self.targets[agent]:
                self.reached[agent] = round(self.clock() - self.origin, 3)
                if all(r is not None for r in self.reached) and self.end_s is None:
                    self.end_s = self.reached[agent]
                    self.stop.set()

    def summary(self):
        with self.lock:
            return dict(targets=list(self.targets), cap_s=self.cap_s, completed=list(self.done), reached_s=list(self.reached),
                        all_reached=self.end_s is not None, end_s=self.end_s, cap_hit=self.end_s is None)


def salt_of(run_id, label, agent, session):
    return 'q18a-' + hashlib.sha256(('compose-4|%s|%s|%d|%d' % (run_id, label, agent, session)).encode()).hexdigest()[:40]


def request_body(model, messages, max_tokens, salt):
    return {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'min_tokens': max_tokens, 'temperature': 0.7, 'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'enable_thinking': False}, 'cache_salt': salt}


def request_id_of(run_id, label, agent, session, turn_index):
    identity = json.dumps([run_id, label, agent, session, turn_index], separators=(',', ':'))
    return 'q18a-' + hashlib.sha256(identity.encode()).hexdigest()[:32]


def stream_turn(url, body, key, origin, *, timeout=READ_TIMEOUT_S, clock=time.monotonic):
    """One streamed turn: usage, finish, the reply's text and its arrival times in ms from ``origin``."""
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(), headers)
    started = clock()
    ttft, out, usage, finish, arrivals = None, [], {}, None, []
    expected_id = 'chatcmpl-' + body['request_id'] if 'request_id' in body else None
    response_id = None
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        for raw in reply:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            event = json.loads(line[6:])
            if expected_id is not None:
                event_id = event.get('id')
                if type(event_id) is not str or not event_id or len(event_id) > 128:
                    raise ValueError('missing or invalid SSE response id')
                if event_id != expected_id:
                    raise ValueError('SSE response id mismatch')
                response_id = event_id
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', ()):
                finish = choice.get('finish_reason') or finish
                piece = (choice.get('delta') or {}).get('content')
                if piece:
                    now = clock()
                    if ttft is None:
                        ttft = now - started
                    out.append(piece)
                    arrivals.append(int((now - origin) * 1000))
    result = {'ttft': ttft, 'seconds': clock() - started, 'text': ''.join(out), 'finish': finish, 'arrivals_ms': arrivals, 'completion_tokens': usage.get('completion_tokens'), 'prompt_tokens': usage.get('prompt_tokens'),
              'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens')}
    if expected_id is not None:
        if response_id is None:
            raise ValueError('missing SSE response id')
        result['response_id'] = response_id
    return result


def draws(seed, agent):
    def stream(kind):
        return random.Random('%d|%d|%s' % (seed, agent, kind))
    return {name: stream(name) for name in ('start', 'append', 'reply', 'pause', 'text', 'reply_text')}


class TurnSink:
    """The arm's turns as they finish, one JSON line each after a header, so an arm the death watch stops keeps every finished turn. One open per arm, a flush per turn, no fsync: a killed process's flushed lines are in the file. Never overwrites."""

    def __init__(self, path, header):
        self.handle = open(path, 'x')
        self.write(dict(header, header=True))

    def write(self, row):
        self.handle.write(json.dumps(row) + '\n')
        self.handle.flush()

    def close(self):
        self.handle.close()


def agent_loop(agent, url, model, key, cfg, run_id, label, origin, t_end, records, lock, *, clock=time.monotonic, sleep=time.sleep, stream=stream_turn, sink=None, faults=None, reply_text='model', fixed=None):
    assert reply_text in REPLY_TEXT_MODES, reply_text
    rng = draws(cfg['SEED'], agent)
    session, history_tokens, messages, salt = 0, 0, [], None
    turn_index = 0
    append_draw = None     # a drawn append that rolled the session is kept for the next: the sequence of draws doesn't depend on the roll
    while clock() < t_end and (fixed is None or not fixed.stop.is_set()):
        if append_draw is None and messages:
            append_draw = rng['append'].randint(cfg['APPEND_MIN'], cfg['APPEND_MAX'])
        if not messages or history_tokens + append_draw > T.MAX_PROMPT_TOKENS - T.PROMPT_SLACK_TOKENS:
            session += 1
            start = rng['start'].randint(cfg['START_MIN'], cfg['START_MAX'])
            salt = salt_of(run_id, label, agent, session)
            messages = [{'role': 'system', 'content': 'You are coding agent %d, session %d.' % (agent, session)},
                        {'role': 'user', 'content': 'Repository context:\n' + ar.text(rng['text'], start)}]
            history_tokens, kind = start, 'fresh'
        else:
            messages.append({'role': 'user', 'content': 'Tool output:\n' + ar.text(rng['text'], append_draw)})
            history_tokens += append_draw
            append_draw, kind = None, 'append'
        reply = rng['reply'].randint(cfg['REPLY_MIN'], cfg['REPLY_MAX'])
        synthetic = ar.text(rng['reply_text'], reply) if reply_text == 'seeded' else None
        sent = clock() - origin
        record = {'agent': agent, 'session': session, 'kind': kind, 't': round(sent, 3), 'history_est': history_tokens, 'reply_target': reply}
        if synthetic is not None:
            record['reply_text'] = 'seeded'
        turn_index += 1
        if cfg.get('REQUEST_IDS', False):
            record.update(client_request_id=request_id_of(run_id, label, agent, session, turn_index), response_id=None)
        try:
            body = request_body(model, messages, reply, salt)
            if cfg.get('REQUEST_IDS', False):
                body['request_id'] = record['client_request_id']
            got = stream(url, body, key, origin)
            if cfg.get('REQUEST_IDS', False):
                if got.get('response_id') != 'chatcmpl-' + record['client_request_id']:
                    raise ValueError('missing or mismatched replay response id')
                record['response_id'] = got['response_id']
            record.update({k: got[k] for k in ('ttft', 'seconds', 'completion_tokens', 'prompt_tokens', 'cached_tokens', 'finish', 'arrivals_ms')})
            record['complete'] = ar.turn_complete(got, reply)
            messages.append({'role': 'assistant', 'content': got['text'] if synthetic is None else synthetic})
            history_tokens = (got['prompt_tokens'] or history_tokens) + (got['completion_tokens'] or reply)
        except Exception as exc:  # noqa: BLE001 - recorded; the agent starts a new session, as agent_replay
            record['error'] = type(exc).__name__ + ':' + str(exc)[:200]
            messages = []
        else:
            if fixed is not None:
                fixed.completed(agent)
        with lock:
            records.append(record)
            if sink is not None:
                try:
                    sink.write(record)
                except Exception as exc:  # noqa: BLE001 - a turn the journal doesn't hold fails the arm; the load goes on (Codex P1)
                    if faults is not None:
                        faults['journal_write_failures'] += 1
                        faults['kinds'].add(type(exc).__name__)
                    else:
                        raise
        pause = rng['pause'].uniform(cfg['TOOL_MIN'], cfg['TOOL_MAX'])
        sleep(max(0.0, min(pause, t_end - clock())))


def _guarded_agent(counts, guard, /, *args, **kwargs):
    """Counts the agents that end normally. `still_open` only sees threads still alive, so a dead one looks finished (Codex P1)."""
    try:
        agent_loop(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - run() reads it as a death, and the arm fails on it
        with guard:
            counts['kinds'].add(type(exc).__name__)
    else:
        with guard:
            counts['agents_finished'] += 1


def run(url, model, key, run_id, label, *, cfg=None, clock=time.monotonic, sleep=time.sleep, stream=stream_turn, finish_bound_s=FINISH_BOUND_S, sink=None):
    """The replay; (config, summary, turns, still_open). Same start spacing as agent_replay's."""
    cfg = dict(T.REPLAY if cfg is None else cfg)
    mode = reply_text_mode()
    work = fixed_work_targets(cfg, reply_text=mode)
    records, lock = [], threading.Lock()
    faults = dict(journal_write_failures=0, agents_finished=0, kinds=set())
    origin = clock()
    t_end = origin + (cfg['DURATION'] if work is None else work[1])
    fixed = None if work is None else FixedWork(work[0], work[1], clock, origin)
    extra = {} if fixed is None else dict(fixed=fixed)
    threads = [threading.Thread(target=_guarded_agent, args=(faults, lock, i, url, model, key, cfg, run_id, label, origin, t_end, records, lock), kwargs=dict(clock=clock, sleep=sleep, stream=stream, sink=sink, faults=faults, reply_text=mode, **extra), name='compose4-agent-%d' % i, daemon=True)
               for i in range(cfg['AGENTS'])]
    for thread in threads:
        thread.start()
        sleep(0.5 if cfg['DURATION'] > 5 else 0.0)
    deadline = t_end + finish_bound_s
    for thread in threads:
        thread.join(max(0.0, deadline - clock()))
    still_open = [t.name for t in threads if t.is_alive()]
    with lock:
        turns = sorted(records, key=lambda r: r['t'])
    summary = ar.summarize(turns, cfg['WARMUP'], cfg['DURATION'])
    with lock:
        summary.update(journal_write_failures=faults['journal_write_failures'], agents_died=len(threads) - faults['agents_finished'] - len(still_open), fault_kinds=sorted(faults['kinds']))
        if mode == 'seeded':
            summary['reply_text'] = mode
    if fixed is not None:
        summary['fixed_work'] = fixed.summary()
    return cfg, summary, turns, still_open


def key_from_env():
    return os.environ.get('VLLM_API_KEY') or None
