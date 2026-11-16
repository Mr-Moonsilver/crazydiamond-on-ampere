#!/usr/bin/env python3
"""Hub self-run stub of the vLLM container the launcher starts (never on the box).  Started by the docker shim with the
launcher's own argv and env; binds 0.0.0.0:$SERVE_PORT and writes a vLLM-shaped log (the lines the packet reads).

Endpoints (schema-valid fakes): /health, /metrics (vllm:* gauges/counters), /v1/models, /v1/chat/completions (auth when
VLLM_API_KEY is in the env; 400 over max_model_len like vLLM), /v1/messages (plain, stream=true SSE, tool_use), and
/v1/flash_s5_native_overlap when FLASH_S5_OVERLAP=1: a typed session seeded from the RECORDED cxfx42 b1 replies
(startup-grade.json prepare/identity/resume, run_id rewritten; engine_epoch kept), with pause/status/control actions
synthesized in the shapes the bundle's clients check.  Token counts are word counts (a stub, not a tokenizer).
Real defaults (RULE-STUB-REAL-DEFAULTS-V1): response/tool ids are generated per request (never fixed, never echoed); the model
thinks by default on BOTH /v1/chat/completions (reasoning_content; content None and finish_reason length under 64 max_tokens)
and /v1/messages, unless chat_template_kwargs.enable_thinking is False.
Every request is appended to <state>/requests.jsonl (action/path, never bodies).
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {}
LOCK = threading.Lock()
VALIDATE_WORKER = None


def log(line):
    with LOCK:
        with open(STATE['log'], 'a') as f:
            f.write('%sZ %s\n' % (time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()) + '.%09d' % (time.time_ns() % 10**9), line))


def note(kind, **kw):
    with LOCK:
        with open(os.path.join(STATE['dir'], 'requests.jsonl'), 'a') as f:
            f.write(json.dumps(dict(t=time.time(), kind=kind, **kw), sort_keys=True) + '\n')


def words(messages):
    n = 0
    for m in messages or []:
        c = m.get('content')
        if isinstance(c, str):
            n += len(c.split())
        elif isinstance(c, list):
            n += sum(len(b.get('text', '').split()) for b in c if isinstance(b, dict))
    return n


class Session:
    """The typed native session: recorded replies for prepare/identity/resume, synthesized shapes for the rest."""

    def __init__(self, recorded):
        self.rec = recorded
        self.epoch = recorded['engine_epoch']
        self.run_id = None
        self.route_generation = 2
        self.paused = False
        self.manual_owned = True
        self.auto_active = False
        self.selection_seq = 0
        self.yield_enabled = False
        self.admitted = 0
        self.chunks = 0
        self.fields = {}

    def workers(self):
        return [dict(worker_rank=r, engine_epoch=self.epoch, engine_epoch_ok=True, barrier_clean=True, drained_streams=1,
                     pending_native_replays=0, retained_rows=0, pending_prefill_streams=0, pending_output_copies=0, route='full',
                     route_generation=self.route_generation, live_decode_requests=0, prefill_requests=0, running_requests=0,
                     pid=1000 + r, worker_incarnation='stub-%d' % r) for r in range(4)]

    def status(self):
        launch = dict(handback_steps=16, max_new=8192, per_bulk=8, reentry_retries=4)
        return dict(route='full', route_generation=self.route_generation, paused=self.paused, scheduler_requests=STATE['running'],
                    transition_pending=False, control_pending=None, return_pending=None, observed_clock_ns=time.monotonic_ns(),
                    admission_intent=None, admitted=self.admitted, refused=0,
                    auto_arm=dict(active=self.auto_active, manual_owned=self.manual_owned, phase='idle', selection_seq=self.selection_seq),
                    append_yield=dict(config=dict(enabled=self.yield_enabled, **launch), phase=None, counters={}),
                    native_admission=dict(admitted=self.admitted, refused=0, alloc_failures=0, split_stock_deferrals=0,
                                          split_hold_timeouts=0, refusal_seq=0, last_refusal=None),
                    native_progress=dict(chunks=self.chunks, commands=self.chunks * 4, direct_commands=self.chunks * 3,
                                         stock_rpcs=0, mode='burst', recording=True, line_errors=0),
                    native_work={}, workers=self.workers())

    def reply(self, cmd):
        action = cmd.get('action')
        if self.run_id is None and action == 'prepare':
            self.run_id = cmd.get('run_id')
        if self.run_id is not None and cmd.get('run_id') != self.run_id:
            return 409, dict(schema=1, error='session-run-id-changed')
        base = dict(schema=1, action=action, run_id=self.run_id, engine_epoch=self.epoch, route='full',
                    route_generation=self.route_generation, observed_clock_ns=time.monotonic_ns())
        if action in ('prepare', 'identity', 'resume') and action in self.rec:
            out = json.loads(json.dumps(self.rec[action]))
            out['run_id'] = self.run_id
            if action == 'resume':
                self.paused = False
            if action == 'prepare':
                self.route_generation += 1
                out['route_generation'] = self.route_generation
            return 200, out
        if action == 'status':
            return 200, dict(base, **self.status())
        if action == 'pause':
            self.paused = True
            return 200, dict(base, pause=dict(paused=True, mode='keep', clear_cache=False, called=True), barrier=dict(workers=self.workers()))
        if action == 'release_auto_arm':
            self.manual_owned, self.auto_active = False, False
            self.selection_seq += 1
            return 200, dict(base, released=True, manual_owned=False)
        # every other typed control: an acknowledged echo with the previous value (the clients compare requested == acknowledged)
        fields = {k: v for k, v in cmd.items() if k not in ('schema', 'action', 'run_id', 'client_id', 'command_seq', 'engine_epoch')}
        prev = self.fields.get(action, {})
        out = dict(base, **fields)
        out.update({('previous' if len(fields) == 1 else 'previous_' + k): prev.get(k, v) for k, v in fields.items()})
        out['acknowledged'] = True
        out['replied'] = True
        if action == 'native_progress':
            out['installed'] = fields.get('mode')
        if action == 'native_append_yield':
            self.yield_enabled = bool(fields.get('enabled'))
            out.update(handback_steps=16, max_new=8192, per_bulk=8, reentry_retries=4)
        self.fields[action] = fields
        return 200, out


def _worker(self, cmd):
    """qualification_worker: the adapter path echoes command_seq; worker_reply carries the epoch and per-rank rows."""
    if cmd.get('run_id') != self.run_id:
        return 409, dict(schema=1, error='session-run-id-changed')
    wa = cmd.get('worker_action')
    if VALIDATE_WORKER:
        try:
            VALIDATE_WORKER(cmd)        # the overlay's own server-side validator (native_qualification_frontend.validate_worker)
        except Exception as exc:  # noqa: BLE001 - the real endpoint answers 400 with the code
            return 400, dict(detail=str(exc))
    rows = self.rec.get('worker', {}).get(wa)
    if rows is None:
        return 400, dict(detail='stub-has-no-recorded-reply:%s' % wa)
    payload = cmd.get('payload') or {}
    out_rows = []
    for rank, kept in enumerate(json.loads(json.dumps(rows))):
        # the real worker's row (native_overlap_worker._attn_meta): the selector's record, the levers this command set, no verdict
        after = kept.get('after') if isinstance(kept.get('after'), dict) else {}
        if 'on' in payload:
            after['selected'] = payload['on']
        for name, value in payload.items():
            if isinstance(after.get(name), dict) and type(value) is bool:
                after[name]['selected'] = value
        q = dict(kept, worker_action=wa, graded_by='attn_meta_select', engine_supplied_verdict=False)
        out_rows.append(dict(worker_rank=rank, engine_epoch=self.epoch, action='qualification', qualification=q))
    reply = dict(engine_epoch=self.epoch, qualification=out_rows)
    # the server's _result: schema, action, run_id, command_seq, client_id, engine_epoch, worker_action, worker_reply
    return 200, dict(schema=1, action=cmd.get('action'), run_id=self.run_id, command_seq=cmd.get('command_seq'), client_id=cmd.get('client_id'),
                     engine_epoch=cmd.get('engine_epoch'), worker_action=wa, worker_reply=reply)


Session.worker = _worker


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, format, *args):  # noqa: A002 - the base class's name
        log('(APIServer pid=53) INFO:     127.0.0.1:%d - "%s %s HTTP/1.1" %s' % (self.client_address[1], self.command, self.path, args[1] if len(args) > 1 else '-'))

    def send(self, code, payload, ctype='application/json'):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authed(self):
        key = STATE['key']
        return not key or self.headers.get('Authorization') == 'Bearer ' + key

    def body(self):
        n = int(self.headers.get('Content-Length') or 0)
        return json.loads(self.rfile.read(n) or b'null')

    def do_GET(self):
        if self.path == '/health':
            return self.send(200, b'', 'text/plain')
        if self.path == '/metrics':
            s = STATE
            text = ''.join('vllm:%s{model_name="Qwen3.8-Flash-Next",engine="0"} %s\n' % kv for kv in (
                ('num_requests_running', s['running']), ('num_requests_waiting', 0), ('request_success_total', s['ok']),
                ('prompt_tokens_total', s['prompt']), ('generation_tokens_total', s['gen']), ('prefix_cache_hits_total', 0),
                ('prefix_cache_queries_total', s['prompt']), ('num_preemptions_total', 0), ('kv_cache_usage_perc', 0.01)))
            return self.send(200, text.encode(), 'text/plain')
        if self.path == '/v1/models':
            if not self.authed():
                return self.send(401, dict(error='Unauthorized'))
            return self.send(200, dict(object='list', data=[dict(id=n, object='model', max_model_len=STATE['mml']) for n in STATE['names']]))
        return self.send(404, dict(error='not found'))

    def do_POST(self):
        path = self.path
        try:
            body = self.body()
        except ValueError:
            return self.send(400, dict(error='bad json'))
        if path == '/v1/flash_s5_native_overlap':
            if not STATE['session']:
                return self.send(404, dict(detail='Not Found'))
            if not self.authed():
                return self.send(401, dict(error='Unauthorized'))
            code, out = STATE['session'].reply(body if isinstance(body, dict) else {})
            note('control', action=(body or {}).get('action'), code=code, fields=sorted(body or {}))
            return self.send(code, out)
        if path == '/v1/flash_s5_native_qualification_worker':
            if not STATE['session']:
                return self.send(404, dict(detail='Not Found'))
            if not self.authed():
                return self.send(401, dict(error='Unauthorized'))
            body = body if isinstance(body, dict) else {}
            code, out = STATE['session'].worker(body)
            note('worker', action=body.get('action'), worker_action=body.get('worker_action'), code=code,
                 payload=sorted((body.get('payload') or {}).items()) if isinstance(body.get('payload'), dict) else None)
            return self.send(code, out)
        if path not in ('/v1/chat/completions', '/v1/messages'):
            return self.send(404, dict(error='not found'))
        if not self.authed():
            note('http', path=path, code=401)
            return self.send(401, dict(error='Unauthorized'))
        n = words(body.get('messages')) + 10
        mt = int(body.get('max_tokens') or 16)
        if n + mt > STATE['mml']:
            note('http', path=path, code=400, prompt=n)
            return self.send(400, dict(error=dict(code=400, message="This model's maximum context length is %d tokens. However, you requested %d output tokens and your prompt contains at least %d input tokens" % (STATE['mml'], mt, n))))
        with LOCK:
            STATE['running'] += 1
            STATE['prompt'] += n
        try:
            hold = float(((body.get('metadata') or {}) if isinstance(body.get('metadata'), dict) else {}).get('stub_hold_s', 0))
            time.sleep(0.2 + min(n, 100000) / 400000.0 + min(hold, 120.0))
            if STATE['session'] and n > 2048:
                STATE['session'].admitted += 1 + n // 2048
                STATE['session'].chunks += 1 + n // 2048
        finally:
            with LOCK:
                STATE['running'] -= 1
                STATE['ok'] += 1
                STATE['gen'] += 2
        note('http', path=path, code=200, prompt=n, stream=bool(body.get('stream')))
        # the model thinks by default (as on the box): unless chat_template_kwargs.enable_thinking is False a thinking block
        # comes first, and under 64 max_tokens the thinking uses the whole budget (no text: the b packet's smoke failure)
        thinks = ((body.get('chat_template_kwargs') or {}).get('enable_thinking', True) is not False)
        short = int(body.get('max_tokens') or 16) < 64
        if path == '/v1/chat/completions':
            msg = dict(role='assistant', content=None if (thinks and short) else 'OK')
            if thinks:
                msg['reasoning_content'] = 'The user wants the word OK.'
            return self.send(200, dict(id='chatcmpl-' + uuid.uuid4().hex, object='chat.completion', model=body.get('model'),
                                       choices=[dict(index=0, message=msg, finish_reason='length' if (thinks and short) else 'stop')],
                                       usage=dict(prompt_tokens=n, completion_tokens=2, total_tokens=n + 2, prompt_tokens_details=dict(cached_tokens=0))))
        tools = body.get('tools')
        msg_id = 'msg_' + uuid.uuid4().hex
        think = [dict(type='thinking', thinking='The user wants the word OK.', signature='')] if thinks else []
        if tools:
            content = think + [dict(type='tool_use', id='toolu_' + uuid.uuid4().hex[:24], name=tools[0]['name'], input={})]
        else:
            content = think + ([] if (thinks and short) else [dict(type='text', text='OK')])
        stop = 'max_tokens' if (thinks and short and not tools) else ('tool_use' if tools else 'end_turn')
        if not body.get('stream'):
            return self.send(200, dict(id=msg_id, type='message', role='assistant', model=body.get('model'), content=content,
                                       stop_reason=stop, usage=dict(input_tokens=n, output_tokens=2)))
        ev = [('message_start', dict(type='message_start', message=dict(id=msg_id, type='message', role='assistant', content=[])))]
        for i, blk in enumerate(content):
            if blk['type'] == 'thinking':
                ev += [('content_block_start', dict(type='content_block_start', index=i, content_block=dict(type='thinking', thinking=''))),
                       ('content_block_delta', dict(type='content_block_delta', index=i, delta=dict(type='thinking_delta', thinking=blk['thinking'])))]
            elif blk['type'] == 'tool_use':
                ev += [('content_block_start', dict(type='content_block_start', index=i, content_block=dict(type='tool_use', id=blk['id'], name=blk['name'], input={}))),
                       ('content_block_delta', dict(type='content_block_delta', index=i, delta=dict(type='input_json_delta', partial_json='{}')))]
            else:
                ev += [('content_block_start', dict(type='content_block_start', index=i, content_block=dict(type='text', text=''))),
                       ('content_block_delta', dict(type='content_block_delta', index=i, delta=dict(type='text_delta', text=blk.get('text', ''))))]
            ev.append(('content_block_stop', dict(type='content_block_stop', index=i)))
        ev += [('message_delta', dict(type='message_delta', delta=dict(stop_reason=stop), usage=dict(output_tokens=2))),
               ('message_stop', dict(type='message_stop'))]
        text = ''.join('event: %s\ndata: %s\n\n' % (k, json.dumps(v)) for k, v in ev)
        return self.send(200, text.encode(), 'text/event-stream')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('--recorded', default='')
    p.add_argument('--recorded-arm', default='', help='the recorded cxfx42 b1 n3 arm record (per-rank worker rows)')
    p.add_argument('--overlay', default='', help='the bundle native30_overlay: its server-side validators check the commands')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('argv', nargs=argparse.REMAINDER)
    a = p.parse_args()
    words_ = a.argv[1:] if a.argv[:1] == ['--'] else a.argv
    def last(flag, default=None):
        vals = [words_[i + 1] for i, w in enumerate(words_[:-1]) if w == flag]
        return vals[-1] if vals else default
    names = []
    if '--served-model-name' in words_:
        i = words_.index('--served-model-name') + 1
        while i < len(words_) and not words_[i].startswith('--'):
            names.append(words_[i])
            i += 1
    STATE.update(dir=a.state, log=a.log, key=os.environ.get('VLLM_API_KEY', ''), mml=int(last('--max-model-len', '262144')),
                 names=names, running=0, ok=0, prompt=0, gen=0, session=None)
    if os.environ.get('FLASH_S5_OVERLAP') == '1':
        rec = json.load(open(a.recorded))
        arm = json.load(open(a.recorded_arm)) if a.recorded_arm else {}
        # the recorded per-rank attn_meta_select rows of the cxfx42 b1 n3 arm (one qualification row per rank, rank order)
        worker = {}
        sel = (arm.get('attn_meta') or {}).get('select')
        if isinstance(sel, dict):
            worker['attn_meta_select'] = [sel[str(r)] for r in range(4)]
        STATE['session'] = Session(dict(engine_epoch=rec['engine_epoch'], prepare=rec['prepare'], identity=rec['identity'], resume=rec['resume'],
                                        worker=worker, arm=arm))
        if a.overlay:
            import importlib.util
            spec = importlib.util.spec_from_file_location('stub_overlay_nqf', os.path.join(a.overlay, 'e3/native_qualification_frontend.py'))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            global VALIDATE_WORKER
            VALIDATE_WORKER = mod.validate_worker
    host = last('--host', '0.0.0.0')
    log('(APIServer pid=53) INFO 00-00 00:00:00 [utils.py:1] non-default args: {\'model_tag\': \'/model\', \'host\': \'%s\', \'max_model_len\': %d, \'max_num_seqs\': %s, \'served_model_name\': %r}' % (
        host, STATE['mml'], last('--max-num-seqs'), names))
    log('(EngineCore pid=90) INFO Initializing a V1 LLM engine (stub) with config: model=\'/model\', speculative_config=%s, tokenizer=\'/model\'' % (
        'MTP' if '--speculative-config' in words_ else 'None'))
    time.sleep(float(os.environ.get('STUB_START_DELAY_S', '2')))
    srv = ThreadingHTTPServer(('0.0.0.0', a.port), H)
    log('(APIServer pid=53) INFO:     Uvicorn running on http://%s:%d (Press CTRL+C to quit)' % (host, a.port))
    try:
        srv.serve_forever()
    finally:
        srv.server_close()


if __name__ == '__main__':
    sys.exit(main())
