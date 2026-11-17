#!/usr/bin/env python3
"""Infra-cleared VM<vm-id> self-run/service owner. Controls only its own encoder container."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_encoder_service import error_info
from vision_encoder_backend import MODEL_MANIFEST_SHA
from vision_speech_guard import (ASR, FIXTURE, HEALTH, GPU_LIMIT_MIB, Refused, SpeechMismatch, SpeechLatency,
    check_asr, check_health, check_memory, validate_clearance)

IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'


def verify_model(model):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
    from vision_encoder_bundle import verify
    if verify(model) != MODEL_MANIFEST_SHA:
        raise Refused('vision model manifest differs')


def command(argv, timeout=20):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise Refused('owned command failed: ' + Path(argv[0]).name + ' rc=' + str(result.returncode))
    return result.stdout.strip()


def retry_delay(health_url):
    until = time.monotonic() + 60
    while time.monotonic() < until:
        check_health(health_url)
        time.sleep(min(1, max(0, until - time.monotonic())))


def physical(gpu_uuid, pid=None):
    free = command(['nvidia-smi', '-i', gpu_uuid, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], timeout=3)
    rows = command(['nvidia-smi', '--query-compute-apps=pid,used_memory,gpu_uuid', '--format=csv,noheader,nounits'], timeout=3)
    owned = 0
    for row in rows.splitlines():
        fields = [v.strip() for v in row.split(',')]
        if len(fields) != 3:
            raise Refused('unreadable GPU process memory')
        if pid is not None and fields[0] == str(pid) and fields[2] == gpu_uuid:
            owned += int(fields[1])
    return dict(owned_mib=owned, free_mib=int(free))


def request(url, path, key=None, body=None):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    req = urllib.request.Request(url + path, headers=headers,
        data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=40) as reply:
            return json.loads(reply.read(9 * 2**20))
    except urllib.error.HTTPError as exc:
        value = json.loads(exc.read(4096))
        if path == '/health' and exc.code == 503:
            return value
        raise Refused('encoder-http-' + str(exc.code) + ' ' + str(value.get('error', 'missing error'))) from exc


def probe(url, key, images):
    result = []
    for path in images:
        response = request(url, '/v1/encode', key,
            dict(image_base64=base64.b64encode(path.read_bytes()).decode()))
        if (response.get('schema') != 'flash-next-image-embeds/1'
                or response.get('manifest_sha256') != MODEL_MANIFEST_SHA
                or response.get('dtype') != 'bfloat16'
                or len(response.get('shape', [])) != 2 or response['shape'][1] != 2560
                or not 1 <= response['shape'][0] <= 1024
                or not re.fullmatch('[0-9a-f]{64}', response.get('embedding_sha256', ''))):
            raise Refused('encoder response shape/identity')
        # Never retain the image/embedding wire payload in a packet record.
        result.append(dict(image=path.name, shape=response['shape'],
            manifest_sha256=response['manifest_sha256'], embedding_sha256=response['embedding_sha256'],
            metrics=response['metrics']))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--clearance', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--expected-asr', type=Path, required=True)
    p.add_argument('--asr-url', required=True)
    p.add_argument('--asr-max-seconds', type=float, required=True)
    p.add_argument('--asr-text-field', default='text')
    p.add_argument('--speech-health', default=HEALTH)
    p.add_argument('--fixture', type=Path, default=Path(FIXTURE))
    p.add_argument('--images', type=Path, nargs='+', required=True)
    p.add_argument('--ceiling',action='store_true',help='one-shot infra c serial N-image sweep; never GPU batching')
    p.add_argument('--service', action='store_true', help='stay up after probe; owner must stop for handback')
    args = p.parse_args(argv)
    if not re.fullmatch('[a-z0-9-]{1,64}', args.run_id) or args.asr_max_seconds <= 0:
        p.error('bounded run id and positive ASR latency limit required')
    clear = validate_clearance(args.clearance, args.run_id)
    if args.ceiling:
        if args.service:raise Refused('ceiling-is-one-shot-not-service')
        import vision_encoder_ceiling as ceiling
        ceiling.preflight(clear)
    verify_model(args.model)
    if (args.key_file.is_symlink() or not args.key_file.is_file()
            or args.key_file.stat().st_mode & 0o777 != 0o600):
        raise Refused('private encoder key file required before GPU start')
    key_fields = dict(line.split('=', 1) for line in args.key_file.read_text().splitlines() if '=' in line)
    key = key_fields.get('VISION_ENCODER_API_KEY', '')
    if not re.fullmatch('[0-9a-f]{64}', key):
        raise Refused('encoder key missing')
    if any(not path.is_file() or path.stat().st_size > 8 * 2**20 for path in args.images):
        raise Refused('probe image missing/oversized')
    expected = args.expected_asr.read_text()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / 'owner.json').write_text(json.dumps(dict(pid=os.getpid(), run_id=args.run_id,
        run_dir=str(args.run_dir.resolve()), source=Path(__file__).name)) + '\n')
    import fcntl
    lock = open('/opt/flashnext/.job.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    name = 'flashnext-vision-' + args.run_id
    docker = ['sudo', '-n', 'docker']
    if command(docker + ['image', 'inspect', '--format', '{{.Id}}', IMAGE]) != IMAGE:
        raise Refused('pinned VM image absent')
    check_memory(**physical(clear['gpu_uuid']), started=False)
    record = dict(schema='vision-encoder-job/1', run_id=args.run_id, verdict='HOLD',
        reason='not-complete', image=IMAGE, port=9041, gpu_limit_mib=GPU_LIMIT_MIB,
        model_manifest_sha256=MODEL_MANIFEST_SHA,
        gpu_uuid=clear['gpu_uuid'], physical_peak_source='nvidia-smi owned-container-PID samples',
        physical_peak_mib=0, memory_samples=0, speech={}, images=[], owned_gpu_released=False)
    def save():
        pending = args.run_dir / 'result.next.json'
        pending.write_text(json.dumps(record, indent=2) + '\n')
        pending.replace(args.run_dir / 'result.json')
    def sample_memory():
        sample = check_memory(**physical(clear['gpu_uuid'], pid))
        record['physical_peak_mib'] = max(record['physical_peak_mib'], sample['owned_mib'])
        record['memory_samples'] += 1
        with (args.run_dir / 'memory.jsonl').open('a') as log:
            log.write(json.dumps(dict(monotonic=time.monotonic(), **sample)) + '\n')
        return sample
    def asr():
        record['speech']['asr_calls']=record['speech'].get('asr_calls',0)+1
        result=check_asr(args.asr_url, args.fixture, expected, args.asr_max_seconds, args.asr_text_field)
        return dict(result,observed_wall_ns=time.time_ns())
    save()
    try:
        record['speech']['pre_health'] = check_health(args.speech_health)
        try:
            record['speech']['pre_asr'] = asr()
        except SpeechLatency as exc:
            if args.ceiling:raise
            record['speech']['pre_asr_first_latency'] = error_info(exc)
            save()
            retry_delay(args.speech_health)
            record['speech']['pre_asr'] = asr()
            record['speech']['pre_asr_attempts'] = 2
    except (SpeechMismatch, SpeechLatency) as exc:
        record.update(verdict='FAIL', reason=error_info(exc), owned_gpu_released=True)
        print('PACKET_CLIENT_ERROR ' + json.dumps(record['reason']), file=sys.stderr, flush=True)
        save()
        lock.close()
        print('VISION_ENCODER_JOB FAIL speech-precheck')
        return 1
    save()
    root = Path(__file__).resolve().parent.parent
    argv = docker + ['run', '--pull', 'never', '--name', name, '--label', 'flashnext.vision.run=' + args.run_id,
        '--log-driver', 'local', '--log-opt', 'max-size=16m', '--log-opt', 'max-file=2',
        '--gpus', 'device=' + clear['gpu_uuid'], '--network', 'host', '--read-only', '--cpus', '2',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=256m',
        '--memory', '4g', '--memory-swap', '4g', '--user', str(os.getuid()) + ':' + str(os.getgid()),
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '-v', str(root) + ':/packet:ro', '-v', str(args.model.resolve()) + ':/model:ro',
        '-v', str(args.run_dir.resolve()) + ':/artifacts', '-v', str(args.key_file.resolve()) + ':/encoder.env:ro',
        '-e', 'HF_HUB_OFFLINE=1', '-e', 'HF_HUB_DISABLE_TELEMETRY=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
        '-e', 'TMPDIR=/artifacts', '-e', 'VLLM_CACHE_ROOT=/artifacts/cache', '-e', 'TRITON_CACHE_DIR=/artifacts/triton',
        '-e', 'FLASHINFER_WORKSPACE_BASE=/artifacts/flashinfer', '-e', 'HF_HOME=/artifacts/hf',
        '-e', 'XDG_CACHE_HOME=/artifacts/xdg', '-e', 'TORCH_EXTENSIONS_DIR=/artifacts/torch-extensions',
        '--entrypoint', 'python3', IMAGE, '-B', '/packet/harness/vision_encoder_service.py',
        '--model', '/model', '--run-dir', '/artifacts', '--key-file', '/encoder.env']
    (args.run_dir / 'argv.json').write_text(json.dumps(argv) + '\n')
    stop_requested = False
    retry_after_stop = False
    proc = None
    pid = None
    def signal_stop(*_):
        nonlocal stop_requested
        stop_requested = True
        if args.ceiling:ceiling.cancel()
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        with (args.run_dir / 'service.stdout.log').open('xb') as stdout, (args.run_dir / 'service.stderr.log').open('xb') as stderr:
            proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr)
            deadline = time.monotonic() + 600
            url = 'http://<encoder-host>:9041'
            phase = 'startup'
            future = None
            load_future = None
            asr_started = None
            next_asr = None
            next_log_budget = 0
            while not stop_requested:
                record['speech']['last_health']=dict(check_health(args.speech_health),observed_wall_ns=time.time_ns())
                if proc.poll() is not None:
                    raise Refused('encoder exited before owned stop')
                if pid is None:
                    try:
                        found = command(docker + ['inspect', '--format', '{{.State.Pid}} {{index .Config.Labels "flashnext.vision.run"}}', name])
                    except Refused:
                        if time.monotonic() >= deadline:
                            raise
                    else:
                        value, label = found.split()
                        if label != args.run_id:
                            raise Refused('owned container identity differs')
                        pid = int(value) or None
                        record['gpu_pid'] = pid
                if args.service and time.monotonic()>=next_log_budget:
                    retained=sum(p.stat().st_size for p in args.run_dir.rglob('*') if p.is_file())
                    if retained>256*2**20:raise Refused('service-retained-output-budget; stop before1GiB')
                    next_log_budget=time.monotonic()+15
                if args.ceiling:
                    ceiling.card(clear,pid,command);record['ceiling']=ceiling.snapshot()
                if pid:
                    sample = check_memory(**physical(clear['gpu_uuid'], pid))
                    new_peak=sample['owned_mib']>record['physical_peak_mib']
                    record['physical_peak_mib'] = max(record['physical_peak_mib'], sample['owned_mib'])
                    record['memory_samples'] += 1
                    if not args.service or new_peak or time.monotonic()>=record.get('next_memory_log',0):
                        with (args.run_dir / 'memory.jsonl').open('a') as log:
                            log.write(json.dumps(dict(monotonic=time.monotonic(), **sample)) + '\n')
                        record['next_memory_log']=time.monotonic()+15
                if phase == 'startup':
                    try:
                        health = request(url, '/health')
                    except (urllib.error.URLError, TimeoutError, ConnectionError):
                        health = None
                    if health and health.get('ready'):
                        record['loaded'] = health['weights']
                        future = (pool.submit(ceiling.sweep,url,key,args.images,record['speech']['pre_asr'],asr,command,measure=sample_memory) if args.ceiling else pool.submit(probe, url, key, args.images))
                        phase = 'probe'
                    elif health and health.get('error'):
                        raise Refused('encoder-startup-error ' + json.dumps(health['error']))
                    elif time.monotonic() >= deadline:
                        raise Refused('encoder startup timeout')
                elif phase == 'probe' and future.done():
                    if args.ceiling:
                        record['ceiling']=future.result();stop_requested=True;continue
                    record['images'] = future.result()
                    record['speech']['post_load_health'] = check_health(args.speech_health)
                    future = pool.submit(asr)
                    asr_started = time.monotonic()
                    load_future = pool.submit(probe, url, key, args.images)
                    phase = 'post-asr'
                elif phase == 'post-asr':
                    if not future.done() and time.monotonic() - asr_started > args.asr_max_seconds:
                        raise SpeechLatency('speech-asr-latency-limit-while-pending')
                    if future.done():
                        speech_result = future.result()  # fail immediately, even while encoder load is pending
                    else:
                        speech_result = None
                    if load_future is not None and load_future.done():
                        record['speech_overlap_images'] = load_future.result()
                    if speech_result is None or (load_future is not None and not load_future.done()):
                        save()
                        time.sleep(.25)
                        continue
                    history=record['speech'].setdefault('asr_history',[])
                    if len(history)>=96:history.pop(1)
                    history.append(dict(speech_result,phase='post-load' if not history else 'service'))
                    record['speech']['post_load_asr'] = history[0]
                    record['speech']['latest_asr'] = speech_result
                    next_asr = time.monotonic() + 900
                    phase = 'service'
                    if args.service:record.update(verdict='RUNNING',reason='guarded-service; manual owner stop required',service_guard_ready=True)
                    if not args.service:
                        break
                elif phase == 'service' and time.monotonic() >= next_asr:
                    future = pool.submit(asr); asr_started = time.monotonic()
                    load_future = None; phase = 'post-asr'
                save()
                time.sleep(.25)
            if args.ceiling:
                record['ceiling']=ceiling.snapshot()
                if not record['ceiling'].get('complete'):raise Refused('ceiling-skipped-or-incomplete')
            if (not args.ceiling and (not record['images'] or not record['speech'].get('post_load_asr'))) or record['physical_peak_mib'] <= 0:
                raise Refused('incomplete real encoder/speech/memory evidence')
            record.update(verdict='PASS', reason='encoder-ceiling-serial-real-images-and-speech-guard' if args.ceiling else 'encoder-only-real-images-and-speech-guard')
    except Exception as exc:
        retry_after_stop = not args.ceiling and isinstance(exc, SpeechLatency)
        if args.ceiling:record['ceiling']=ceiling.snapshot()
        record.update(verdict='FAIL' if isinstance(exc, SpeechMismatch) else 'HOLD', reason=error_info(exc))
        print('PACKET_CLIENT_ERROR ' + json.dumps(record['reason']), file=sys.stderr, flush=True)
    finally:
        if args.ceiling:ceiling.cancel()
        cleanup_errors = []
        try:
            # Also check during early startup: an unobserved PID is not permission to leave the child running.
            found = command(docker + ['inspect', '--format', '{{.State.Pid}} {{index .Config.Labels "flashnext.vision.run"}}', name])
            value, label = found.split()
            if label != args.run_id:
                raise Refused('cleanup owner label differs')
            owned_pid = pid or int(value) or None
            command(docker + ['stop', '--time', '5', name], timeout=20)
            stopped = command(docker + ['inspect', '--format', '{{.State.Running}}', name]) == 'false'
            remaining = physical(clear['gpu_uuid'], owned_pid)['owned_mib'] if owned_pid else 0
            record['owned_gpu_released'] = stopped and remaining == 0
            if record['owned_gpu_released']:
                record['handoff'] = dict(state='OFF', container=name, gpu_pid=owned_pid,
                    gpu_uuid=clear['gpu_uuid'], run_id=args.run_id, sampled_owned_mib=remaining)
                lock.close()  # no GPU reservation while a CPU-only ASR retry finishes
                save()
                print('VISION_ENCODER_GPU_OFF ' + json.dumps(record['handoff']), flush=True)
            if proc:
                proc.wait(timeout=20)
        except Exception as exc:
            cleanup_errors.append(error_info(exc))
        try:
            record['speech']['after_stop_health'] = check_health(args.speech_health)
            if args.ceiling and record['owned_gpu_released']:
                record['speech']['after_stop_asr']=ceiling.post_stop(asr,record['speech']['pre_asr'])
        except Exception as exc:
            cleanup_errors.append(error_info(exc))
        record['cleanup_errors'] = cleanup_errors
        if not record['owned_gpu_released']:
            record.update(verdict='HOLD', reason='owned-gpu-release-not-proven')
        if cleanup_errors:
            record.update(verdict='HOLD', reason='cleanup-or-post-speech-check')
        if retry_after_stop and record['owned_gpu_released']:
            try:
                retry_delay(args.speech_health)
                if future is not None and future.done():
                    try:
                        future.result()
                    except SpeechLatency:
                        pass
                elif future is not None:
                    raise Refused('first-ASR-call-still-pending; no overlapping retry')
                record['speech']['after_stop_asr_retry'] = asr()
                record.update(verdict='HOLD', reason='speech-recovered-after-encoder-stop; mixed-load-ASR-unproven')
            except (SpeechMismatch, SpeechLatency) as exc:
                record.update(verdict='FAIL', reason=error_info(exc))
                print('PACKET_CLIENT_ERROR ' + json.dumps(record['reason']), file=sys.stderr, flush=True)
            except Exception as exc:
                record.update(verdict='HOLD', reason=error_info(exc))
                print('PACKET_CLIENT_ERROR ' + json.dumps(record['reason']), file=sys.stderr, flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        save()
        lock.close()
    print('VISION_ENCODER_JOB ' + record['verdict'] + ' ' + str(record['reason']))
    return {'PASS': 0, 'HOLD': 2, 'FAIL': 1}[record['verdict']]


if __name__ == '__main__':
    try:
        code = main()
    except SystemExit as exc:
        if exc.code:
            print('PACKET_CLIENT_ERROR ' + json.dumps(error_info(exc)), file=sys.stderr)
        raise
    except Exception as exc:
        print('PACKET_CLIENT_ERROR ' + json.dumps(error_info(exc)), file=sys.stderr)
        print('VISION_ENCODER_JOB HOLD ' + type(exc).__name__)
        raise SystemExit(2)
    raise SystemExit(code)
