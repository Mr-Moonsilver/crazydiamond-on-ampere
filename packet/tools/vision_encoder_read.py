#!/usr/bin/env python3
"""Encoder-alone result; never qualifies box integration or image answers."""
import argparse
import json
from pathlib import Path
import sys
import traceback


def read(value):
    reasons = []
    if value.get('verdict') != 'PASS':
        reasons.append('job-not-PASS')
    peak = value.get('physical_peak_mib')
    if (not value.get('owned_gpu_released') or value.get('cleanup_errors')
            or type(peak) not in (int, float) or not 0 < peak <= 3072
            or value.get('memory_samples', 0) < 2):
        reasons.append('physical-peak-or-owned-release-unproven')
    weights = value.get('loaded') or {}
    if weights.get('tensors') != 333 or weights.get('bytes') != 897862112:
        reasons.append('vision-weights-unproven')
    speech = value.get('speech') or {}
    for name in ('pre_health', 'post_load_health', 'after_stop_health'):
        if (speech.get(name) or {}).get('status') != 200:
            reasons.append('speech-' + name)
    for name in ('pre_asr', 'post_load_asr'):
        item = speech.get(name) or {}
        if (item.get('correct') is not True or item.get('status') != 200
                or item.get('elapsed_s') is None or item.get('latency_limit_s') is None
                or item['elapsed_s'] > item['latency_limit_s']):
            reasons.append('speech-' + name)
    if len(value.get('images') or []) < 3 or not value.get('speech_overlap_images'):
        reasons.append('real-image-or-overlap-evidence-missing')
    for image in (value.get('images') or []) + (value.get('speech_overlap_images') or []):
        metrics = image.get('metrics') or {}
        if (not 0 < metrics.get('visual_tokens', 0) <= 1024
                or not 0 < metrics.get('peak_allocated_bytes', 0) <= 2048 * 2**20):
            reasons.append('allocator-or-image-bound')
    return dict(verdict='HOLD' if reasons else 'PASS', reasons=reasons,
        scope='encoder-only; box/native-MM integration and visual answers unqualified',
        measured_physical_peak_mib=value.get('physical_peak_mib'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record', type=Path, required=True)
    args = p.parse_args()
    result = read(json.loads(args.record.read_text()))
    print(json.dumps(result, sort_keys=True))
    print('VISION_ENCODER_READ ' + result['verdict'] + ' ' + (','.join(result['reasons']) or 'encoder-only-evidence-complete'))
    return 0 if result['verdict'] == 'PASS' else 2


if __name__ == '__main__':
    try:
        code = main()
    except SystemExit as exc:
        if exc.code:
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}', file=sys.stderr)
        raise
    except Exception as exc:
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}', file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(code)
