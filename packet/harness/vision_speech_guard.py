"""VM-only speech and memory safety checks. Does not repair or stop speech."""
import hashlib
import json
import re
from pathlib import Path
import time
import urllib.request

HEALTH = 'http://<encoder-host>:9030/health'
ASR = 'http://<encoder-host>:9030/asr?language=en&output_format=json'
FIXTURE = '<hub-dir>/scratch/infra/asr-fixture/asr10.wav'
FIXTURE_SHA_PREFIX = 'e5aca12363f30fb7'
GPU_LIMIT_MIB = 3072
ALLOCATOR_LIMIT_MIB = 2048
ASR_EXPECTED = ('This is a short test of the transcription service. The quick brown fox jumps over '
    'the lazy 1 2 dog. 3 4 5 6 7 8 9 10 This is only a test.')


class Refused(RuntimeError):
    pass


class SpeechMismatch(Refused):
    pass


class SpeechLatency(Refused):
    pass


def check_health(url=HEALTH):
    start = time.monotonic()
    with urllib.request.urlopen(url, timeout=2) as reply:
        if reply.status != 200:
            raise Refused('speech-health-not-200')
        reply.read(4096)
    return dict(status=200, elapsed_s=time.monotonic() - start)


def check_asr(url, fixture, expected, max_seconds, text_field='text'):
    if expected.strip() != ASR_EXPECTED or max_seconds != 10 or text_field != 'text':
        raise Refused('ASR contract differs from infra reference')
    path = Path(fixture)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if (not digest.startswith(FIXTURE_SHA_PREFIX) or path.stat().st_mode & 0o777 != 0o644
            or not data or len(data) > 8 * 2**20):
        raise Refused('speech-fixture-identity-or-mode')
    boundary = 'flashnext-vision-asr-check'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="audio_file"; '
            'filename="asr10.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    body += data + f'\r\n--{boundary}--\r\n'.encode()
    request = urllib.request.Request(url, data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=60) as reply:
            if reply.status != 200:
                raise Refused('speech-asr-not-200')
            result = json.loads(reply.read(1024 * 1024))
    except TimeoutError as exc:
        raise SpeechLatency('speech-asr-60s-HTTP-timeout') from exc
    elapsed = time.monotonic() - start
    actual = result.get(text_field)
    if not isinstance(actual, str) or not expected.strip():
        raise Refused('speech-asr-schema-or-expectation')
    correct = actual.strip() == expected.strip()
    receipt = dict(status=200, correct=correct, elapsed_s=elapsed,
                   latency_limit_s=max_seconds, fixture_sha256=digest)
    if not correct:
        raise SpeechMismatch('speech-asr-text-mismatch ' + json.dumps(receipt, sort_keys=True))
    if elapsed > max_seconds:
        raise SpeechLatency('speech-asr-latency ' + json.dumps(receipt, sort_keys=True))
    return receipt


def validate_clearance(path, run_id):
    value = json.loads(Path(path).read_text())
    if (value.get('schema') != 'vision-infra-clearance/1' or value.get('cleared') is not True
            or value.get('run_id') != run_id or value.get('speech_must_remain_up') is not True
            or value.get('gpu_limit_mib') != GPU_LIMIT_MIB
            or not isinstance(value.get('gpu_uuid'), str)
            or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value['gpu_uuid'])):
        raise Refused('infra-clearance-missing-or-not-bound')
    return value


def check_memory(owned_mib, free_mib, *, started=True):
    if (owned_mib is None or free_mib is None or owned_mib < 0 or free_mib < 0
            or owned_mib > GPU_LIMIT_MIB or free_mib < (768 if started else 4096)):
        raise Refused('encoder-physical-memory-or-headroom')
    return dict(owned_mib=owned_mib, free_mib=free_mib, limit_mib=GPU_LIMIT_MIB)
