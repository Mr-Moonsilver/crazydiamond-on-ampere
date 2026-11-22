#!/usr/bin/env python3
"""Final-engine serve recorder (b15's, plus the VRAM floor), inside CT <ct-id>: numbers only, one JSON line per tick (default
60 s) to <out>/recorder.jsonl.  Between ticks, every VRAM_SAMPLE_S (5 s) one nvidia-smi read checks the VRAM_FLOOR_MIB floor on
every GPU; the tick line carries min free per GPU since the last tick and the count of floor samples.  The first sample below
the floor (or a GPU missing) writes <out>/floor-breach.json once (exclusive create) and a 'floor' line; FLOOR_ACTION=drain
also starts serve-drain.sh <out> once.  Any breach makes the A/B arm invalid (reader rule); the engine is not signalled.

Per tick: /metrics (vllm:* counters and gauges; TTFT, ITL/TPOT, e2e, queue and prefill histograms as cumulative
buckets), the lane (one typed status through serve_lane.py: native_admission admitted/refused, native_progress chunks,
route, auto-arm phase/manual_owned), the native prefill share (approx: native chunks x 2048 rows vs the uncached prompt
tokens since the first tick), per-rank VRAM (nvidia-smi), the service (active, NRestarts, container running), root
free GB, and error counts from the followed engine log (OOM, Traceback, ERROR lines, 5xx in the access log).  No prompt
text, no key, no request bodies.  Runs until SIGTERM or the stop file <out>/recorder.stop exists.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_vram  # noqa: E402

KEEP = re.compile(r'^vllm:(num_requests_running|num_requests_waiting|request_success_total|prompt_tokens_total|'
                  r'generation_tokens_total|num_preemptions_total|kv_cache_usage_perc|prefix_cache_queries_total|'
                  r'prefix_cache_hits_total|request_prompt_tokens|request_generation_tokens|'
                  r'time_to_first_token_seconds|inter_token_latency_seconds|time_per_output_token_seconds|'
                  r'e2e_request_latency_seconds|request_queue_time_seconds|request_prefill_time_seconds|'
                  r'request_decode_time_seconds|iteration_tokens_total|corrupted_requests_total)'
                  r'(_bucket|_sum|_count|_total)?(\{[^}]*\})? (\S+)$')
DROP_LABELS = re.compile(r'(model_name|engine)="[^"]*",?')
ERR = dict(oom=re.compile(r'OutOfMemory|out of memory|CUDA error', re.I), traceback=re.compile(r'Traceback \(most recent'),
           error=re.compile(r'\bERROR\b'), http5xx=re.compile(r'" 5\d\d\b'), http4xx=re.compile(r'" 4\d\d\b'))
CHUNK_ROWS = 2048                      # FLASH_S5_CHUNK_TOKENS in cxfx42's EXTRA_ENV
STOP = False



def _packet_error(exc):
    import sys, traceback
    frame=traceback.extract_tb(exc.__traceback__)[-1]
    print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {str(exc)[:256]} at {frame.filename}:{frame.lineno}', file=sys.stderr)

def metrics(url):
    try:
        text = urllib.request.urlopen(url.rstrip('/') + '/metrics', timeout=10).read().decode(errors='replace')
    except Exception as exc:  # noqa: BLE001
        _packet_error(exc)
        return {'error': type(exc).__name__}
    out = {}
    for line in text.splitlines():
        if line.startswith('#'):
            continue
        m = KEEP.match(line)
        if not m:
            continue
        labels = DROP_LABELS.sub('', m.group(3) or '').replace(',}', '}').replace('{}', '')
        try:
            value = float(m.group(4))
        except ValueError:
            continue
        name = m.group(1) + (m.group(2) or '') + labels
        out[name] = out.get(name, 0.0) + value
    return out


def total(m, prefix):
    return sum(v for k, v in m.items() if k == prefix or k.startswith(prefix + '{')) if isinstance(m, dict) else None


def vram():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
        return {r.split(',')[0].strip(): [int(x) for x in r.split(',')[1:]] for r in out.strip().splitlines()}
    except Exception as exc:  # noqa: BLE001
        _packet_error(exc)
        return {'error': type(exc).__name__}


def service(unit, container):
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # noqa: BLE001
            _packet_error(exc)
            return 'error:' + type(exc).__name__
    return dict(active=sh('systemctl', 'is-active', unit), nrestarts=sh('systemctl', 'show', unit, '-p', 'NRestarts', '--value'),
                container=sh('docker', 'inspect', '--format', '{{.State.Status}}', container) or 'absent')


class LogCounter:
    """Incremental error counts over the followed engine log (bytes since the last tick)."""

    def __init__(self, path):
        self.path, self.pos, self.counts = path, 0, {k: 0 for k in ERR}

    def tick(self):
        try:
            with open(self.path, 'rb') as handle:
                handle.seek(self.pos)
                data = handle.read()
                self.pos = handle.tell()
        except OSError:
            return dict(self.counts, log='unreadable')
        cut = data.rfind(b'\n') + 1
        self.pos -= len(data) - cut
        for line in data[:cut].decode(errors='replace').splitlines():
            for k, rx in ERR.items():
                if rx.search(line):
                    self.counts[k] += 1
        return dict(self.counts, log_bytes=self.pos)


class Lane:
    """One typed status per tick through the bundle's own client, imported once (no per-tick interpreter start)."""

    def __init__(self, a):
        self.a, self.client, self.mod = a, None, None

    def __call__(self):
        if self.a.no_lane:
            return None
        try:
            if self.client is None:
                sys.path.insert(0, os.path.dirname(os.path.abspath(self.a.lane)))
                import serve_lane
                self.mod = serve_lane
                c5 = serve_lane.load(self.a.h)
                self.client = serve_lane.client_for(c5, argparse.Namespace(url=self.a.url, run_id=self.a.run_id, control_timeout_s=60.0))
            status = self.client.call('status')
            del self.client.history[:]
            return self.mod.compact(status)
        except Exception as exc:  # noqa: BLE001 - a failed read is a number of the tick, the next tick retries
            _packet_error(exc)
            self.client = None
            return {'error': type(exc).__name__ + ':' + str(exc)[:120]}


class Floor:
    """5 s VRAM floor samples between ticks; min free per GPU since the last tick; one latched breach record."""

    def __init__(self, a, p):
        self.a, self.floor, self.action = a, int(p['VRAM_FLOOR_MIB']), p.get('FLOOR_ACTION', 'record')
        self.every = float(p.get('VRAM_SAMPLE_S', '5'))
        self.reset()
        self.breached = os.path.exists(os.path.join(a.out, 'floor-breach.json'))
        self.last = 0.0

    def reset(self):
        self.min_free, self.samples, self.holds, self.lows = {}, 0, 0, 0

    def poll(self, now):
        if now - self.last < self.every:
            return
        self.last = now
        rows = serve_vram.sample()
        status, low = serve_vram.floor_check(rows, self.floor)
        self.samples += 1
        if status == 'HOLD':
            self.holds += 1
        else:
            for g, (used, total) in rows.items():
                self.min_free[g] = min(self.min_free.get(g, 1 << 30), total - used)
        if status != 'PASS':
            self.lows += 1
            if not self.breached:
                self.breached = True
                rec = dict(schema='serve-floor-breach/1', t=now, status=status, low=low, floor_mib=self.floor, action=self.action,
                           rows={g: list(v) for g, v in rows.items()} if 'error' not in rows else rows)
                try:
                    with open(os.path.join(self.a.out, 'floor-breach.json'), 'x') as handle:
                        json.dump(rec, handle, sort_keys=True)
                except FileExistsError:
                    pass
                with open(os.path.join(self.a.out, 'recorder.jsonl'), 'a') as handle:
                    row = dict(rec, breach_schema=rec['schema'], schema='serve-recorder/2', kind='floor')
                    handle.write(json.dumps(row, sort_keys=True) + '\n')
                if self.action == 'drain':
                    subprocess.Popen(['setsid', 'bash', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'serve-drain.sh'), self.a.out],
                                     stdin=subprocess.DEVNULL, stdout=open(os.path.join(self.a.out, 'floor-drain.log'), 'a'), stderr=subprocess.STDOUT)

    def take(self):
        out = dict(min_free_mib=dict(self.min_free), samples=self.samples, holds=self.holds, below_floor_samples=self.lows,
                   floor_mib=self.floor, breached=self.breached, errors=getattr(self, 'errors', 0), last_error=getattr(self, 'last_error', None))
        self.reset()
        return out


def guarded_poll(floor, now):
    """A floor sample that raises is counted (vram_floor.errors) and never ends the recorder."""
    try:
        floor.poll(now)
    except Exception as exc:  # noqa: BLE001 - a recorder that dies mid-serve records nothing more
        _packet_error(exc)
        floor.errors = getattr(floor, 'errors', 0) + 1
        floor.last_error = type(exc).__name__


def _stop(*_):
    global STOP
    STOP = True


def main(argv=None):
    p = argparse.ArgumentParser(description='final-engine serve recorder')
    p.add_argument('--params', required=True, help='serve-params.env (floor, sample interval, action)')
    p.add_argument('--arm', choices=('engine', 'stock'), required=True)
    p.add_argument('--root-path', default='/', help='the filesystem whose free space is the root guard')
    p.add_argument('--out', required=True, help='an existing directory')
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--interval-s', type=float, default=60.0)
    p.add_argument('--engine-log', required=True, help='the followed docker log')
    p.add_argument('--h', default='', help='bundle dir (typed status)')
    p.add_argument('--lane', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'serve_lane.py'))
    p.add_argument('--run-id', default='')
    p.add_argument('--no-lane', action='store_true')
    p.add_argument('--unit', default='vllm-flash')
    p.add_argument('--container', default='qwen-flash-prod')
    p.add_argument('--root-min-gb', type=float, default=28.0)
    p.add_argument('--ticks', type=int, default=0, help='stop after N ticks (0 = until stopped)')
    a = p.parse_args(argv)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    path = os.path.join(a.out, 'recorder.jsonl')
    logc = LogCounter(a.engine_log)
    lane = Lane(a)
    floor = Floor(a, serve_vram.params(a.params))
    first = None
    n = 0
    while not STOP and not os.path.exists(os.path.join(a.out, 'recorder.stop')):
        t0 = time.time()
        m = metrics(a.url)
        st = lane()
        root_gb = round(shutil.disk_usage(a.root_path).free / 2**30, 1)
        guarded_poll(floor, time.time())
        row = dict(schema='serve-recorder/2', arm=a.arm, t=round(t0, 3), tick=n, metrics=m, lane=st, vram_mib=vram(), vram_floor=floor.take(),
                   service=service(a.unit, a.container), root_free_gb=root_gb, root_ok=root_gb >= a.root_min_gb,
                   log_counts=logc.tick())
        prompt, hits = total(m, 'prompt_tokens_total'), total(m, 'prefix_cache_hits_total')
        chunks = (st or {}).get('progress_chunks') if isinstance(st, dict) else None
        if first is None and prompt is not None and type(chunks) is int:
            first = dict(prompt=prompt, hits=hits or 0.0, chunks=chunks)
        if first is not None and prompt is not None and type(chunks) is int:
            native = (chunks - first['chunks']) * CHUNK_ROWS
            uncached = (prompt - first['prompt']) - ((hits or 0.0) - first['hits'])
            row['share'] = dict(native_rows_approx=native, uncached_prompt_tokens=uncached,
                                native_share_approx=round(min(1.0, native / uncached), 4) if uncached > 0 else None,
                                requests_ok=total(m, 'request_success_total'), basis='since first tick; chunks x 2048 is an upper bound')
        with open(path, 'a') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
        n += 1
        if a.ticks and n >= a.ticks:
            break
        while not STOP and time.time() - t0 < a.interval_s and not os.path.exists(os.path.join(a.out, 'recorder.stop')):
            guarded_poll(floor, time.time())
            time.sleep(1)
    print('SERVE_RECORDER stopped after %d ticks' % n)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
