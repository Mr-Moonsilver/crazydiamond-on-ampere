#!/usr/bin/env python3
"""The closed-loop agent replay for q18a-compose-4: harness/agent_replay.py's loop (STALL-PLAN 3b.6) and its summary, with four changes for this run, nothing else.

  auth    the arm's key from the environment, as every box client; never printed.
  salt    one cache salt per arm, agent and session, so two arms with the same draws do not hit each other's cache.
  draws   one seeded stream per quantity (session starts, appends, replies, pauses, text), so a small difference in a reply's length can't shift what comes next: both arms send the same sequence, timing aside.
  ceiling an append that would take the prompt over MAX_PROMPT_TOKENS rolls to a new session instead (READY: the cap's arithmetic).
Each turn keeps its client-visible arrival times (ms from the arm's start), which the rule reader needs for decode beside a bulk. The text is agent_replay.text, unchanged.
REQUEST_IDS in the replay config defaults to False; True adds per-turn request/SSE identity evidence only.
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
    return {name: stream(name) for name in ('start', 'append', 'reply', 'pause', 'text')}


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


def agent_loop(agent, url, model, key, cfg, run_id, label, origin, t_end, records, lock, *, clock=time.monotonic, sleep=time.sleep, stream=stream_turn, sink=None, faults=None):
    rng = draws(cfg['SEED'], agent)
    session, history_tokens, messages, salt = 0, 0, [], None
    turn_index = 0
    append_draw = None     # a drawn append that rolled the session is kept for the next: the sequence of draws doesn't depend on the roll
    while clock() < t_end:
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
        sent = clock() - origin
        record = {'agent': agent, 'session': session, 'kind': kind, 't': round(sent, 3), 'history_est': history_tokens, 'reply_target': reply}
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
            messages.append({'role': 'assistant', 'content': got['text']})
            history_tokens = (got['prompt_tokens'] or history_tokens) + (got['completion_tokens'] or reply)
        except Exception as exc:  # noqa: BLE001 - recorded; the agent starts a new session, as agent_replay
            record['error'] = type(exc).__name__ + ':' + str(exc)[:200]
            messages = []
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
    records, lock = [], threading.Lock()
    faults = dict(journal_write_failures=0, agents_finished=0, kinds=set())
    origin = clock()
    t_end = origin + cfg['DURATION']
    threads = [threading.Thread(target=_guarded_agent, args=(faults, lock, i, url, model, key, cfg, run_id, label, origin, t_end, records, lock), kwargs=dict(clock=clock, sleep=sleep, stream=stream, sink=sink, faults=faults), name='compose4-agent-%d' % i, daemon=True)
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
    return cfg, summary, turns, still_open


def key_from_env():
    return os.environ.get('VLLM_API_KEY') or None
