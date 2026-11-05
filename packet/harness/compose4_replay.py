#!/usr/bin/env python3
"""The closed-loop agent replay for q18a-compose-4: harness/agent_replay.py's loop (STALL-PLAN 3b.6) and its summary, with four changes for this run, nothing else.

  auth    the arm's key from the environment, as every box client; never printed.
  salt    one cache salt per arm, agent and session, so two arms with the same draws do not hit each other's cache.
  draws   one seeded stream per quantity (session starts, appends, replies, pauses, text), so a small difference in a reply's length can't shift what comes next: both arms send the same sequence, timing aside.
  ceiling an append that would take the prompt over MAX_PROMPT_TOKENS rolls to a new session instead (READY: the cap's arithmetic).
Each turn keeps its client-visible arrival times (ms from the arm's start), which the rule reader needs for decode beside a bulk. The text is agent_replay.text, unchanged.
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


def stream_turn(url, body, key, origin, *, timeout=READ_TIMEOUT_S, clock=time.monotonic):
    """One streamed turn: usage, finish, the reply's text and its arrival times in ms from ``origin``."""
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(), headers)
    started = clock()
    ttft, out, usage, finish, arrivals = None, [], {}, None, []
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        for raw in reply:
            line = raw.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            event = json.loads(line[6:])
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
    return {'ttft': ttft, 'seconds': clock() - started, 'text': ''.join(out), 'finish': finish, 'arrivals_ms': arrivals, 'completion_tokens': usage.get('completion_tokens'), 'prompt_tokens': usage.get('prompt_tokens'),
            'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens')}


def draws(seed, agent):
    def stream(kind):
        return random.Random('%d|%d|%s' % (seed, agent, kind))
    return {name: stream(name) for name in ('start', 'append', 'reply', 'pause', 'text')}


def agent_loop(agent, url, model, key, cfg, run_id, label, origin, t_end, records, lock, *, clock=time.monotonic, sleep=time.sleep, stream=stream_turn):
    rng = draws(cfg['SEED'], agent)
    session, history_tokens, messages, salt = 0, 0, [], None
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
        try:
            got = stream(url, request_body(model, messages, reply, salt), key, origin)
            record.update({k: got[k] for k in ('ttft', 'seconds', 'completion_tokens', 'prompt_tokens', 'cached_tokens', 'finish', 'arrivals_ms')})
            record['complete'] = ar.turn_complete(got, reply)
            messages.append({'role': 'assistant', 'content': got['text']})
            history_tokens = (got['prompt_tokens'] or history_tokens) + (got['completion_tokens'] or reply)
        except Exception as exc:  # noqa: BLE001 - recorded; the agent starts a new session, as agent_replay
            record['error'] = type(exc).__name__ + ':' + str(exc)[:200]
            messages = []
        with lock:
            records.append(record)
        pause = rng['pause'].uniform(cfg['TOOL_MIN'], cfg['TOOL_MAX'])
        sleep(max(0.0, min(pause, t_end - clock())))


def run(url, model, key, run_id, label, *, cfg=None, clock=time.monotonic, sleep=time.sleep, stream=stream_turn, finish_bound_s=FINISH_BOUND_S):
    """The replay; (config, summary, turns, still_open). Same start spacing as agent_replay's."""
    cfg = dict(T.REPLAY if cfg is None else cfg)
    records, lock = [], threading.Lock()
    origin = clock()
    t_end = origin + cfg['DURATION']
    threads = [threading.Thread(target=agent_loop, args=(i, url, model, key, cfg, run_id, label, origin, t_end, records, lock), kwargs=dict(clock=clock, sleep=sleep, stream=stream), name='compose4-agent-%d' % i, daemon=True)
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
    return cfg, summary, turns, still_open


def key_from_env():
    return os.environ.get('VLLM_API_KEY') or None
