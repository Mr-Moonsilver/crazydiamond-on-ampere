#!/usr/bin/env python3
"""Owner handback: verify the guard PID, request stop, prove only its encoder is OFF."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def handoff(run_dir, timeout=30):
    owner = json.loads((run_dir / 'owner.json').read_text())
    if owner.get('run_dir') != str(run_dir.resolve()) or owner.get('source') != 'vision_encoder_job.py':
        raise ValueError('owner record differs from requested run')
    pid = owner['pid']
    if type(pid) is not int or pid <= 1:
        raise ValueError('invalid owner PID')
    state_path = run_dir / 'result.json'
    value = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if not value.get('owned_gpu_released'):
        args = Path('/proc/%d/cmdline' % pid).read_bytes().decode().split('\0')
        if ('vision_encoder_job.py' not in [Path(v).name for v in args if v]
                or not any(a == '--run-id' and b == owner['run_id'] for a, b in zip(args, args[1:]))
                or not any(a == '--run-dir' and b == str(run_dir) for a, b in zip(args, args[1:]))):
            raise ValueError('PID is not this owned encoder guard')
        os.kill(pid, signal.SIGTERM)
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = json.loads(state_path.read_text()) if state_path.is_file() else {}
        receipt = value.get('handoff') or {}
        if value.get('owned_gpu_released') and receipt.get('state') == 'OFF' and receipt.get('run_id') == owner['run_id']:
            name = 'flashnext-vision-' + owner['run_id']
            if receipt.get('container') != name or receipt.get('sampled_owned_mib') != 0:
                raise ValueError('handoff receipt does not prove this encoder is free')
            actual = subprocess.run(['sudo', '-n', 'docker', 'inspect', '--format',
                '{{.State.Running}} {{index .Config.Labels "flashnext.vision.run"}}', name],
                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
            if actual != 'false ' + owner['run_id']:
                raise ValueError('encoder container is not stopped with the matching owner')
            return receipt
        time.sleep(.2)
    raise TimeoutError('owned GPU OFF proof did not arrive; stand-in must not start')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    receipt = handoff(args.run_dir)
    print('VISION_ENCODER_OFF ' + json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except SystemExit as exc:
        if exc.code:
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}', file=sys.stderr)
        raise
    except Exception as exc:
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}', file=sys.stderr)
        raise SystemExit(2)
