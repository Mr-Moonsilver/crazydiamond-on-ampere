#!/usr/bin/env python3
"""Fixed-work memory fit, not a speed/adoption test; owner schedules the packet."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
import random
import threading
import time
import urllib.parse

NAME = 'q18a-fit-8x200k-d'
ARM_CAP_S={'e6':2460,'e4096':2460,'stock':2880}
OUTER_CAP_S=6000
BOUNDARY_RESERVE_S=120
PREFIXES = (60000, 120000, 160000, 192000)
COLD_HISTORY = 190000
OUTPUT = 8000
OUTPUT_CAPS = {60000:2048,120000:2048,160000:2048,190000:8000,192000:8000}
AGENTS = 8
FLOOR = 40 * 2**20
W19 = (135200768, 17760256, 181338112, 63897600)
CONTROL = '/v1/flash_s5_fit_memory'
ROLES = ('e6', 'e4096', 'stock')


class Hold(RuntimeError):
    pass


def sha(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def plan():
    return dict(schema='fit-8x200k-plan/1', name=NAME, prefixes=list(PREFIXES),
                agents=AGENTS, output_cap=OUTPUT, max_model_len=200000,
                staged_output_caps=OUTPUT_CAPS,
                common_prefix_tokens=64, prompt_seed=20261115,
                order=list(ROLES), e4096_only_if_e6='HOLD', stock_always=True,
                per_arm_minutes={r:n/60 for r,n in ARM_CAP_S.items()}, outer_minutes=100,
                cold_prefix=COLD_HISTORY,maximum_completed_context=PREFIXES[-1]+OUTPUT,
                scope='memory fit with fixed token prompts; not production-workflow or speed evidence')


def initial(member, length, cold=False):
    # Common opening, private continuation: sharing between a conversation's turns
    # is much larger than sharing between eight independent conversation bodies.
    rng = random.Random(20261115 + member + (1000 if cold else 0))
    return [1000 + i for i in range(64)] + [rng.randrange(1100, 10000) for _ in range(length - 64)]


def grow(previous_prompt, reply, member, length):
    prefix = previous_prompt + reply
    if len(prefix) > length:
        raise ValueError('growth-would-truncate-the-decoded-reply')
    rng = random.Random(20261115 + member * 1000000 + length)
    return prefix + [rng.randrange(1100, 10000) for _ in range(length-len(prefix))]


def headers():
    result = {'Content-Type': 'application/json'}
    key = os.environ.get('VLLM_API_KEY')
    if key:
        result['Authorization'] = 'Bearer ' + key
    return result


def keyed_serve(url):
    """Prove the actual listener rejects a request without the private key."""
    parsed=urllib.parse.urlsplit(url)
    if not os.environ.get('VLLM_API_KEY'):raise Hold('keyless-test-serve')
    codes=[]
    for auth in (False,True):
        conn=http.client.HTTPConnection(parsed.hostname,parsed.port or 80,timeout=10)
        try:
            conn.request('GET','/v1/models',headers=headers() if auth else {})
            response=conn.getresponse();codes.append(response.status);response.read()
        finally:conn.close()
    if codes!=[401,200]:raise Hold('keyless-test-serve')
    return dict(auth='key',unauthenticated_status=codes[0],authenticated_status=codes[1])


def connection(url, timeout):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != 'http' or not parts.hostname or parts.username or parts.password:
        raise ValueError('fit-requires-plain-local-owner-endpoint')
    return http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout), parts.path.rstrip('/')


def post(url, path, body, timeout=60):
    conn, root = connection(url, timeout)
    try:
        conn.request('POST', root + path, json.dumps(body).encode(), headers())
        reply = conn.getresponse()
        if reply.status != 200:
            raise Hold('control-http-' + str(reply.status) + ':' + reply.read(1024).decode(errors='replace'))
        return json.load(reply)
    finally:
        conn.close()


class Stream:
    def __init__(self, uid, prompt):
        self.uid, self.prompt = uid, prompt
        self.output_cap = OUTPUT_CAPS[len(prompt)]
        self.first = threading.Event()
        self.done = threading.Event()
        self.tokens = []
        self.error = None
        self.open_ns = self.first_ns = self.end_ns = None
        self.cached_tokens = None
        self.usage = None
        self.finish_reason = None
        self.socket = None
        self.http_status = None

    def record(self):
        return dict(uid=self.uid, prompt_tokens=len(self.prompt), prompt_sha256=sha(self.prompt),
                    output_tokens=len(self.tokens), output_cap=self.output_cap, output_sha256=sha(self.tokens),
                    resume_ns=self.open_ns, first_ns=self.first_ns, end_ns=self.end_ns,
                    cached_tokens=self.cached_tokens, finish_reason=self.finish_reason,
                    usage=self.usage, http_status=self.http_status, error=self.error)


def generate(url, model, stream, timeout):
    conn, root = connection(url, timeout)
    try:
        conn.connect()
        stream.socket = conn.sock
        stream.open_ns = time.monotonic_ns()
        body = dict(model=model, prompt=stream.prompt, max_tokens=stream.output_cap, temperature=0,
                    ignore_eos=True, stream=True, stream_options={'include_usage': True},
                    return_token_ids=True, request_id=stream.uid)
        conn.request('POST', root + '/v1/completions', json.dumps(body).encode(), headers())
        response = conn.getresponse()
        stream.http_status = response.status
        if response.status != 200:
            raise Hold('completion-http-' + str(response.status) + ':' + response.read(1024).decode(errors='replace'))
        ended = False
        for line in response:
            if not line.startswith(b'data: '):
                continue
            if line.strip() == b'data: [DONE]':
                ended = True
                break
            row = json.loads(line[6:])
            if row.get('error'):
                raise Hold('completion-error:' + str(row['error'])[:256])
            if row.get('usage'):
                stream.usage = row['usage']
                stream.cached_tokens = (row['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
            for choice in row.get('choices', []):
                tokens = choice.get('token_ids')
                if tokens:
                    if type(tokens) is not list or any(type(t) is not int or t < 0 for t in tokens):
                        raise Hold('invalid-output-token-ids')
                    if stream.first_ns is None:
                        stream.first_ns = time.monotonic_ns()
                        stream.first.set()
                    stream.tokens.extend(tokens)
                if choice.get('finish_reason') is not None:
                    stream.finish_reason = choice['finish_reason']
        if not ended or len(stream.tokens) != stream.output_cap or stream.finish_reason != 'length':
            raise Hold('incomplete-fixed-output:' + str(len(stream.tokens)))
    except Exception as exc:
        from packet_client_contract import report_exception
        stream.error = report_exception(exc)
    finally:
        stream.end_ns = time.monotonic_ns()
        stream.first.set()
        stream.done.set()
        conn.close()


def startup_proven(startup):
    from startup_vram_budget import assess as budget_assess
    if type(startup) is not dict or startup.get('verdict')!='PASS':return False
    peers=startup.get('peers',{})
    if set(peers)!=set(map(str,range(4))):return False
    for rank in range(4):
        p=peers[str(rank)];receipts=p.get('receipts',[])
        expected=dict(total_delta_bytes=4*2**20,outside_delta_bytes=4*2**20,
                      outside_absolute_bytes=(1259,1371,1197,1309)[rank]*2**20)
        if (p.get('verdict')!='PASS' or p.get('limits')!=expected or len(receipts)!=2
                or receipts[0].get('rank')!=rank or receipts[1].get('rank')!=rank
                or receipts[0].get('phase')!='before' or receipts[1].get('phase')!='after'
                or budget_assess(receipts[1].get('before'),receipts[1].get('after'),expected).get('status')!='PASS'
                or receipts[0].get('snapshot')!=receipts[1].get('before')):return False
    return True


def check_status(status, role):
    if status.get('schema') != 'fit-memory/1' or status.get('role') != role:
        raise Hold('fit-status-identity')
    if status.get('hold'):
        raise Hold(str(status['hold']))
    if not startup_proven(status.get('startup')):
        raise Hold('four-rank-startup-budget-not-PASS')
    expected=dict(max_model_len=200000,max_num_seqs=8,kv_cache_memory_bytes=2150000000,
                  tensor_parallel_size=2,pipeline_parallel_size=2,rowcache_reqs='8',
                  rowcache_slots={'e6':'6144','e4096':'4096','stock':'10240'}[role])
    if status.get('configuration') != expected:
        raise Hold('effective-fit-settings-mismatch')


def select_native(args, epoch, record):
    """Reuse cxfx42's actual pause/select/release protocol, not launch flags alone."""
    import n3j2_packet
    from e3_native_runtime_client import NativeRuntimeClient
    import e3_compose5_box as box
    table=n3j2_packet.configure()
    row=next(r for r in table.arms() if r['label']=='n3.g2-live-0')
    record.update(role=args.role,selection=row,control=[])
    client=NativeRuntimeClient(args.url,'fit-'+args.role,epoch=epoch,timeout_s=60)
    try:
        record['prepare']=client.prepare('full')
        with box._bound():
            box.select_and_release(client,row,record)
    finally:
        record['control']=client.history
    return record


def run(args):
    record = dict(schema='fit-8x200k-client/1', role=args.role, plan=plan(), complete=False,
                  stages=[], cold=None, verdict='HOLD', reason='protocol-incomplete')
    current = []
    all_streams = []
    futures = []
    pool = ThreadPoolExecutor(max_workers=AGENTS)
    started = time.monotonic()

    def status(startup=False):
        value = post(args.url, CONTROL, {'action': 'status'})
        record['memory'] = value
        if startup and value.get('startup',{}).get('verdict') != 'PASS' and not value.get('hold'):
            return value
        check_status(value, args.role)
        return value

    def wait(streams, first=False):
        while True:
            if time.monotonic()-started >= args.timeout_s:
                raise Hold('declared-arm-budget-exhausted')
            status()
            if any(s.error for s in streams):
                raise Hold('stream-error')
            if all((s.first if first else s.done).is_set() for s in streams):
                return
            time.sleep(0.1)

    try:
        record['auth']=keyed_serve(args.url)
        while status(startup=True).get('startup',{}).get('verdict') != 'PASS':
            if time.monotonic()-started>120:raise Hold('startup-memory-proof-timeout')
            time.sleep(0.1)
        record['initial_native_coverage'] = record['memory'].get('native_coverage')
        if args.role!='stock':
            record['native_selection']={}
            select_native(args,record['memory'].get('engine_epoch'),record['native_selection'])
        previous = [([], []) for _ in range(AGENTS)]
        for length in PREFIXES:
            current = []
            before=status().get('native_admissions')
            stage = dict(prompt_tokens=length, conversations=[], complete=False,native_admissions_before=before)
            record['stages'].append(stage)
            for member, (old, reply) in enumerate(previous):
                prompt = initial(member, length) if not old else grow(old, reply, member, length)
                stream = Stream(f'fit-{args.role}-grow-{length}-{member}', prompt)
                current.append(stream)
                all_streams.append(stream)
                futures.append(pool.submit(generate, args.url, args.model, stream, args.timeout_s))
            wait(current)
            stage['conversations'] = [s.record() for s in current]
            stage['complete'] = True
            after=status().get('native_admissions')
            stage['native_admissions']=(after-before if type(before) is int and type(after) is int else None)
            if args.role!='stock' and not (type(stage['native_admissions']) is int and stage['native_admissions']>0):
                raise Hold('no-native-execution')
            previous = [(s.prompt, s.tokens) for s in current]
        # Reuse seven final prefixes for a full 8k decode bracket. The eighth
        # prompt has a new private body and arrives only after all seven emit.
        before=status().get('native_admissions')
        current = [Stream(f'fit-{args.role}-cold-companion-{m}', previous[m][0]) for m in range(7)]
        all_streams.extend(current)
        for stream in current:
            futures.append(pool.submit(generate, args.url, args.model, stream, args.timeout_s))
        wait(current, first=True)
        if any(s.done.is_set() for s in current):
            raise Hold('seven-decoders-not-live-at-cold-arrival')
        cold = Stream(f'fit-{args.role}-cold-probe', initial(7, COLD_HISTORY, cold=True))
        all_streams.append(cold)
        cold_open = time.monotonic_ns()
        futures.append(pool.submit(generate, args.url, args.model, cold, args.timeout_s))
        current.append(cold)
        wait(current)
        if any(s.end_ns <= cold.first_ns for s in current[:-1]):
            raise Hold('seven-decoders-did-not-span-cold-prefill')
        record['cold'] = dict(resume_ns=cold_open, live_at_arrival=7,
                              conversations=[s.record() for s in current])
        after=status().get('native_admissions')
        record['cold']['native_admissions']=(after-before if type(before) is int and type(after) is int else None)
        if args.role!='stock' and not (type(record['cold']['native_admissions']) is int and record['cold']['native_admissions']>0):
            raise Hold('no-native-execution')
        record['complete'] = True
        record['verdict'] = 'FIT'
        record['reason'] = 'fixed-growth-and-cold-under-seven-live-complete'
    except Exception as exc:
        from packet_client_contract import report_exception
        record['reason'] = report_exception(exc)
    finally:
        # The engine owns stopping GPU work. Clients cannot pretend a dropped
        # socket closes a native source lease or completes a PP send.
        try:
            settled = post(args.url, CONTROL, {'action': 'finish' if record['complete'] else 'stop'})
            until = time.monotonic()+120
            while settled.get('settled') is not True and time.monotonic()<until:
                time.sleep(0.1)
                settled = post(args.url, CONTROL, {'action': 'status'})
            record['memory'] = settled
            record['permit_latency']=settled.get('permit_latency')
            record['simultaneous_resident_token_peak']=settled.get('simultaneous_resident_token_peak')
            if settled.get('settled') is not True:
                record.update(verdict='HOLD', reason='all-rank-stop-unsettled')
        except Exception as exc:
            from packet_client_contract import report_exception
            record.update(verdict='HOLD', reason=report_exception(exc))
        for stream in current:
            if not stream.done.is_set() and stream.socket is not None:
                import socket
                try:
                    stream.socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass  # The server may already have closed an aborted request.
        pool.shutdown(wait=True, cancel_futures=False)
        if record['stages'] and not record['stages'][-1]['complete']:
            record['stages'][-1]['conversations']=[s.record() for s in current]
        record['errors400'] = sum(s.http_status == 400 for s in all_streams)
        record['elapsed_s'] = time.monotonic()-started
    return record


def assess(record):
    reasons = []
    if record.get('auth')!={'auth':'key','unauthenticated_status':401,'authenticated_status':200}:
        reasons.append('keyless-test-serve')
    if record.get('complete') is not True:
        reasons.append('incomplete-workload')
    memory = record.get('memory', {})
    if memory.get('role') != record.get('role') or memory.get('settled') is not True:
        reasons.append('unsettled-or-foreign-memory-proof')
    minimum = memory.get('min_free_bytes', {})
    if set(minimum) != set(map(str, range(4))) or any(type(x) is not int or x < FLOOR for x in minimum.values()):
        reasons.append('40MiB-floor-or-missing-peer')
    if not startup_proven(memory.get('startup')):
        reasons.append('startup-budget')
    if memory.get('preemptions') != 0:
        reasons.append('preemption-or-unknown')
    if (type(memory.get('ooms')) is int and memory['ooms']>0) or memory.get('fatal'):
        return dict(verdict='FAIL', reasons=['OOM-or-fatal'], raw=record)
    if type(memory.get('ooms')) is not int or memory['ooms']<0:
        reasons.append('OOM-counter-unknown')
    coverage = memory.get('coverage', {})
    if (not coverage.get('decode_rounds') or coverage.get('decode_checked') != coverage.get('decode_rounds')
            or coverage.get('chunk_checked') != coverage.get('chunks') or not coverage.get('checks')):
        reasons.append('boundary-coverage-incomplete')
    stages = record.get('stages', [])
    if [s.get('prompt_tokens') for s in stages] != list(PREFIXES):
        reasons.append('growth-plan')
    for stage in stages:
        members = stage.get('conversations', [])
        if stage.get('complete') is not True or len(members) != AGENTS:
            reasons.append('eight-members-not-complete')
        for member in members:
            if member.get('prompt_tokens') != stage.get('prompt_tokens') or member.get('output_tokens') != OUTPUT_CAPS[stage['prompt_tokens']] or member.get('error'):
                reasons.append('wrong-fixed-work')
    cold = record.get('cold') or {}
    if cold.get('live_at_arrival') != 7 or len(cold.get('conversations', [])) != AGENTS:
        reasons.append('cold-beside-seven-unproven')
    members=cold.get('conversations',[])
    if len(members)==8:
        probe=members[-1]
        if (probe.get('prompt_tokens')!=COLD_HISTORY or probe.get('output_tokens')!=OUTPUT
                or probe.get('error') or type(probe.get('first_ns')) is not int):
            reasons.append('cold-fixed-work-incomplete')
        for companion in members[:-1]:
            if (companion.get('prompt_tokens')!=PREFIXES[-1] or companion.get('output_tokens')!=OUTPUT
                    or companion.get('error') or type(companion.get('first_ns')) is not int
                    or type(companion.get('end_ns')) is not int
                    or type(probe.get('resume_ns')) is not int
                    or type(probe.get('first_ns')) is not int
                    or not companion['first_ns']<=probe['resume_ns']<probe['first_ns']<companion['end_ns']):
                reasons.append('seven-live-decoders-not-spanning-cold-prefill')
    if record.get('errors400')!=0:reasons.append('HTTP400-or-unknown')
    if memory.get('hold'):
        reasons.append('engine-floor-HOLD')
    native = None
    if record.get('role') in ('e6','e4096'):
        if (any(type(s.get('native_admissions')) is not int or s['native_admissions']<=0 for s in stages)
                or type(cold.get('native_admissions')) is not int or cold['native_admissions']<=0):
            reasons.append('no-native-execution')
        from e3_compose5_box import coverage_delta
        wrap = lambda raw:dict(append_yield=dict(declines=dict(coverage=raw)))
        native = coverage_delta(wrap(record.get('initial_native_coverage')),wrap(memory.get('native_coverage')))
        if native is None:
            reasons.append('native-coverage-unavailable')
        else:
            if native.get('live_native_share')!=1.0 or native.get('all_native_share')!=1.0:
                reasons.append('native-served-share-not-one')
            if native.get('all_native_share_strict') is None or native.get('live_native_share_strict') is None:
                reasons.append('strict-share-unknown')
            if native.get('errors') or native.get('open_native'):
                reasons.append('native-coverage-errors-or-open-owner')
            unexpected = {state:{key:value for key,value in native['rows'][state]['stock'].items()
                                  if key not in ('handback','geometry-tail') and value}
                          for state in ('live','idle')}
            if any(unexpected.values()):
                reasons.append('unexplained-stock-prefill')
            handback,geometry=native['handback'],native['geometry_tail']
            if (not handback['every_row_after_a_native_turn'] or handback['excess_rows']
                    or geometry['requests']>geometry['turns']
                    or geometry['rows']>64*geometry['turns']
                    or (geometry['rows'] and (geometry['max_r'] is None or geometry['max_r']>64
                                             or geometry['max_rows'] is None or geometry['max_rows']>64))):
                reasons.append('registered-exception-proof-missing')
    return dict(verdict='HOLD' if reasons else 'FIT', reasons=sorted(set(reasons)),
                minimum_mib={r:v/2**20 for r,v in minimum.items()},
                delta_used_vs_w19_mib={str(r):(W19[r]-minimum[str(r)])/2**20 for r in range(4) if str(r) in minimum},
                preemptions=memory.get('preemptions'), errors400=record.get('errors400'),
                permit_latency=memory.get('permit_latency'),
                simultaneous_resident_token_peak=memory.get('simultaneous_resident_token_peak'),
                native_coverage=native, raw=record)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    collect = sub.add_parser('collect')
    collect.add_argument('--url', required=True)
    collect.add_argument('--model', required=True)
    collect.add_argument('--role', choices=ROLES, required=True)
    collect.add_argument('--timeout-s', type=float, default=1560)
    collect.add_argument('--out', type=Path, required=True)
    reader = sub.add_parser('read')
    reader.add_argument('--record', type=Path, required=True)
    reader.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise ValueError('fresh-output-required')
    try:
        result = run(a) if a.action == 'collect' else assess(json.loads(a.record.read_text()))
    except Exception as exc:
        from packet_client_contract import report_exception
        result=dict(verdict='HOLD',reason=report_exception(exc))
    with a.out.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print('FIT_8X200K '+result['verdict']+' '+str(result.get('reason',result.get('reasons',[]))))
    if result['verdict'] != 'FIT':
        from packet_client_contract import report_exception
        report_exception(Hold(str(result.get('reason',result.get('reasons',[])))))
    return {'FIT':0, 'HOLD':2, 'FAIL':1}[result['verdict']]


if __name__ == '__main__':
    from packet_client_contract import report_exception
    try:
        raise SystemExit(main())
    except Exception as exc:
        report_exception(exc)
        raise SystemExit(2)
