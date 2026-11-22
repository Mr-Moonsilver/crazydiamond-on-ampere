#!/usr/bin/env python3
"""Final-engine serve smoke (b15/b16's, arm-aware), inside CT <ct-id> against the local engine (stdlib only; lane via serve_lane.py).

Checks, each printed on its own line, then SERVE_MESSAGES PASS|FAIL and SERVE_SMOKE PASS|FAIL (FAIL names the failed checks).
Every check is hard unless marked INFO; /v1/messages is hard here (Claude Code reaches the serve on /v1/messages):
  health        GET /health 200 (no auth)
  models        GET /v1/models with the key: every served id incl. Qwen3.8-Flash-Next and the LiteLLM id
  noauth        POST /v1/chat/completions WITHOUT the key: 401 on BOTH arms (c packet, RULE-TEST-SERVE-KEYED-V2)
  chat          POST /v1/chat/completions (Qwen3.8-Flash-Next, thinking off, max_tokens 32) 200 + text
  messages      POST /v1/messages (Anthropic shape) -> 200 + a text block
  stream        POST /v1/messages stream=true -> 200, SSE with message_start .. message_stop and text deltas
  tools         POST /v1/messages with one tool and a prompt that needs it -> 200; INFO: tool_use block or text
  concurrency   MAX_SEQS parallel short chats -> every one 200 + text (the declared stream count is accepted)
  maxtok        a short chat with max_tokens = max_model_len: 200 (clamped) or 400; printed, informational
  overcap       a chat whose prompt is over max_model_len -> 400 clean, engine still healthy after
  lane          a ~64k-token fresh prompt; native_admission.admitted must rise (native prefill engaged), route back
                to full and the auto-arm released (manual_owned False) after it
Key: VLLM_API_KEY from the environment; never printed (length only).
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

WORDS = ('alpha bravo cedar delta ember fjord granite harbor island jasper kettle lantern meadow nickel orchid pepper '
         'quartz river saddle timber umber velvet willow xenon yarrow zephyr anchor beacon canyon dune falcon glacier '
         'hollow ivory juniper kelp lagoon marble nectar onyx prairie quill raven sierra tundra upland vapor walnut').split()



def _packet_error(exc):
    import sys, traceback
    frame=traceback.extract_tb(exc.__traceback__)[-1]
    print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {str(exc)[:256]} at {frame.filename}:{frame.lineno}', file=sys.stderr)

def http(method, url, body=None, key=None, timeout=60.0):
    data = None if body is None else json.dumps(body).encode()
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read(), e.code
    except Exception as e:  # noqa: BLE001 - connection errors are a result
        _packet_error(e)
        return 0, {'error': type(e).__name__ + ':' + str(e)[:160]}, time.monotonic() - t0
    try:
        payload = json.loads(raw.decode(errors='replace') or 'null')
    except ValueError:
        payload = {'raw': raw[:200].decode(errors='replace')}
    return code, payload, time.monotonic() - t0


def short(payload):
    s = json.dumps(payload, sort_keys=True)
    return s if len(s) <= 220 else s[:220] + '...'


def chat_body(model, content, max_tokens):
    return {'model': model, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens,
            'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}


def chat_text(payload):
    try:
        return payload['choices'][0]['message'].get('content') or ''
    except (KeyError, IndexError, TypeError, AttributeError):
        return ''


def lane(a, field):
    """One typed status through serve_lane.py (the bundle's client); None if unavailable."""
    try:
        out = subprocess.run([sys.executable, '-B', a.lane, 'status', '--h', a.h, '--url', a.url, '--run-id', a.run_id],
                             capture_output=True, text=True, timeout=90)
        row = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as _packet_exc:  # noqa: BLE001
        _packet_error(_packet_exc)
        return None
    return row if field is None else row.get(field)


def filler(n_words, seed):
    rng = random.Random(seed)
    return ' '.join(rng.choice(WORDS) for _ in range(n_words))


# messages / stream (b packet, 15 Nov; <review-group>'s rule): the model thinks by default, and a 32-token /v1/messages reply was all
# thinking on the box (text '' -> FAIL).  The body now disables thinking the way chat_body does (chat_template_kwargs
# enable_thinking False; whether vLLM's /v1/messages honours it is unverified offline, its protocol module is not in the hub
# reference) AND asks for 512 tokens.  The pass condition is unchanged: non-empty TEXT.  A thinking-only reply FAILS.
# Text and thinking lengths are printed.
MESSAGES_MAX_TOKENS = 512


def messages_body(model):
    return {'model': model, 'max_tokens': MESSAGES_MAX_TOKENS, 'messages': [{'role': 'user', 'content': 'Reply with the word OK.'}],
            'chat_template_kwargs': {'enable_thinking': False}}


def message_parts(payload):
    """(text, thinking, stop_reason) of a non-stream /v1/messages reply."""
    blocks = payload.get('content') if isinstance(payload, dict) else None
    blocks = [b for b in blocks if isinstance(b, dict)] if isinstance(blocks, list) else []
    text = ''.join(b.get('text', '') or '' for b in blocks if b.get('type', 'text') == 'text')
    thinking = ''.join(b.get('thinking', '') or '' for b in blocks if b.get('type') == 'thinking')
    return text, thinking, payload.get('stop_reason') if isinstance(payload, dict) else None


def parse_sse(raw):
    """(events, text, thinking, stop_reason) from an Anthropic SSE body."""
    events, text, thinking, stop = [], '', '', None
    for line in raw.splitlines():
        if line.startswith('event:'):
            events.append(line.split(':', 1)[1].strip())
        elif line.startswith('data:'):
            try:
                d = json.loads(line[5:].strip())
            except ValueError:
                continue
            if not isinstance(d, dict):
                continue
            delta = d.get('delta') if isinstance(d.get('delta'), dict) else {}
            if d.get('type') == 'content_block_delta':
                text += delta.get('text', '') or ''
                thinking += delta.get('thinking', '') or ''
            elif d.get('type') == 'message_delta' and delta.get('stop_reason'):
                stop = delta['stop_reason']
    return events, text, thinking, stop


def reply_ok(text):
    """<review-group>'s rule: only non-empty text passes (thinking never stands in for it)."""
    return bool(text.strip())


def stream_ok(code, events, text):
    return code == 200 and events[:1] == ['message_start'] and 'message_stop' in events and reply_ok(text)


def main(argv=None):
    p = argparse.ArgumentParser(description='final-engine serve smoke')
    p.add_argument('--arm', choices=('engine', 'stock'), required=True)
    p.add_argument('--max-seqs', type=int, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--model', default='Qwen3.8-Flash-Next')
    p.add_argument('--litellm-model', default='cyankiwi/Qwen3.8-27B-AWQ-INT4')
    p.add_argument('--max-model-len', type=int, required=True)
    p.add_argument('--lane-words', type=int, default=48000, help='fresh words for the lane check (~64k tokens)')
    p.add_argument('--h', default='', help='bundle dir (for the typed status; engine arm)')
    p.add_argument('--lane', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'serve_lane.py'))
    p.add_argument('--run-id', default='')
    p.add_argument('--no-lane', action='store_true', help='stock arm: no native lane exists')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    key = os.environ.get('VLLM_API_KEY', '')
    results, fails = {}, []
    print('key length %d' % len(key))

    def check(name, ok, detail):
        results[name] = dict(ok=ok, detail=detail)
        print('%-9s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
        if not ok:
            fails.append(name)

    def sse(url, body, timeout=180.0):
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}, method='POST')
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode(errors='replace')
                code = r.status
        except urllib.error.HTTPError as e:
            return e.code, '', time.monotonic() - t0
        except Exception as e:  # noqa: BLE001
            _packet_error(e)
            return 0, '', time.monotonic() - t0
        return code, raw, time.monotonic() - t0

    code, payload, dt = http('GET', a.url + '/health', timeout=10)
    check('health', code == 200, 'http %s %.2fs' % (code, dt))

    code, payload, dt = http('GET', a.url + '/v1/models', key=key, timeout=20)
    ids = [m.get('id') for m in (payload or {}).get('data', []) if isinstance(m, dict)] if isinstance(payload, dict) else []
    check('models', code == 200 and a.model in ids and a.litellm_model in ids, 'http %s ids=%s' % (code, ids))

    code, payload, dt = http('POST', a.url + '/v1/chat/completions', chat_body(a.model, 'Reply with OK.', 8), key=None, timeout=60)
    check('noauth', code == 401, 'http %s (401 expected on both arms: key required on /v1, RULE-TEST-SERVE-KEYED-V2)' % code)

    code, payload, dt = http('POST', a.url + '/v1/chat/completions', chat_body(a.model, 'Reply with the word OK.', 32), key=key, timeout=180)
    text = chat_text(payload)
    check('chat', code == 200 and bool(text.strip()), 'http %s %.2fs text=%r usage=%s' % (code, dt, text.strip()[:40], (payload or {}).get('usage') if isinstance(payload, dict) else None))

    body = messages_body(a.model)
    code, payload, dt = http('POST', a.url + '/v1/messages', body, key=key, timeout=180)
    mtext, mthink, mstop = message_parts(payload)
    check('messages', code == 200 and reply_ok(mtext), 'http %s %.2fs text_len=%d thinking_len=%d stop=%s text=%r body=%s' % (
        code, dt, len(mtext.strip()), len(mthink.strip()), mstop, mtext.strip()[:40], short(payload) if code != 200 else '-'))

    code, raw, dt = sse(a.url + '/v1/messages', dict(body, stream=True))
    events, stext, sthink, sstop = parse_sse(raw)
    check('stream', stream_ok(code, events, stext),
          'http %s %.2fs events=%d first=%s last=%s text_len=%d thinking_len=%d stop=%s text=%r' % (
              code, dt, len(events), events[:1], events[-1:], len(stext.strip()), len(sthink.strip()), sstop, stext.strip()[:40]))

    tool = {'name': 'get_time', 'description': 'Return the current UTC time.', 'input_schema': {'type': 'object', 'properties': {}}}
    code, payload, dt = http('POST', a.url + '/v1/messages', {'model': a.model, 'max_tokens': 256, 'tools': [tool],
                             'messages': [{'role': 'user', 'content': 'What is the UTC time now? Use the tool.'}]}, key=key, timeout=180)
    kinds = [b.get('type') for b in (payload or {}).get('content', []) if isinstance(b, dict)] if isinstance(payload, dict) else []
    check('tools', code == 200, 'http %s %.2fs blocks=%s stop=%s (INFO: tool_use expected but not required)' % (
        code, dt, kinds, (payload or {}).get('stop_reason') if isinstance(payload, dict) else None))

    import threading
    got = [None] * a.max_seqs

    def one(i):
        c, pl, _ = http('POST', a.url + '/v1/chat/completions', chat_body(a.model, 'Stream %d: reply with the word OK.' % i, 16), key=key, timeout=600)
        got[i] = (c, bool(chat_text(pl).strip()))
    threads = [threading.Thread(target=one, args=(i,)) for i in range(a.max_seqs)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(660)
    ok = all(g is not None and g[0] == 200 and g[1] for g in got)
    check('concurrency', ok, '%d parallel: codes=%s %.1fs' % (a.max_seqs, [g[0] if g else None for g in got], time.monotonic() - t0))

    code, payload, dt = http('POST', a.url + '/v1/chat/completions', chat_body(a.model, 'Reply with OK.', a.max_model_len), key=key, timeout=180)
    results['maxtok'] = dict(ok=True, detail='http %s' % code)
    print('maxtok    INFO http %s (200 = max_tokens clamped to the cap, 400 = refused) %s' % (code, short(payload) if code != 200 else ''))

    over = filler(a.max_model_len + 4096, 'overcap-%d' % time.time_ns())
    code, payload, dt = http('POST', a.url + '/v1/chat/completions', chat_body(a.model, over, 16), key=key, timeout=300)
    hcode, _, _ = http('GET', a.url + '/health', timeout=10)
    check('overcap', code == 400 and hcode == 200, 'http %s %.2fs then health %s %s' % (code, dt, hcode, short(payload)))

    if a.no_lane or a.arm == 'stock':
        print('lane      NONE (stock arm: no native lane)')
    else:
        before = lane(a, None)
        prompt = 'Summarise the list in one word.\n' + filler(a.lane_words, 'lane-%d' % time.time_ns())
        code, payload, dt = http('POST', a.url + '/v1/chat/completions', chat_body(a.model, prompt, 16), key=key, timeout=900)
        usage = (payload or {}).get('usage') if isinstance(payload, dict) else None
        after = None
        for _ in range(30):
            after = lane(a, None)
            if after and after.get('scheduler_requests') == 0:
                break
            time.sleep(2)
        b, e = (before or {}).get('native_admitted'), (after or {}).get('native_admitted')
        engaged = type(b) is int and type(e) is int and e > b
        held = bool(after) and after.get('route') == 'full' and after.get('auto_arm_manual_owned') is False
        check('lane', code == 200 and engaged and held,
              'http %s %.1fs prompt_tokens=%s native_admitted %s -> %s route=%s manual_owned=%s refused=%s' % (
                  code, dt, (usage or {}).get('prompt_tokens'), b, e, (after or {}).get('route'),
                  (after or {}).get('auto_arm_manual_owned'), (after or {}).get('native_refused')))
        results['lane']['before'], results['lane']['after'] = before, after
    hard = list(fails)
    verdict = 'PASS' if not hard else 'FAIL'
    with open(a.out, 'x') as handle:
        json.dump(dict(schema='serve-smoke/2', arm=a.arm, verdict=verdict, fails=fails, messages_ok=results['messages']['ok'],
                       results=results, t=time.time()), handle, sort_keys=True)
    print('SERVE_MESSAGES %s' % ('PASS' if results['messages']['ok'] else 'FAIL (/v1/messages does not work on this engine: see the messages line)'))
    print('SERVE_SMOKE %s%s' % (verdict, (' ' + ','.join(hard)) if hard else ''))
    return 0 if not hard else 2


if __name__ == '__main__':
    raise SystemExit(main())
