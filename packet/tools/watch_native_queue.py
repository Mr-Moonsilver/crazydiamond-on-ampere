#!/usr/bin/env python3
"""Read-only boxro watch. Local immutable logs; no engine or box mutation.

Start only for the owner-armed named queue. Production restoration is owner-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(rb'(?i)(?:VLLM_API_KEY|api_key|authorization|x-api-key)'
                    rb'[^\n]{0,16}[\s\[\x27\x22:=]([0-9a-f]{64})')


def freeze(path, payload):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(payload)


def read(item):
    name, argv = item
    try:
        proc = subprocess.run([str(ROOT / 'tools/boxro.sh'), *argv], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=25)
        # Docker emits application stderr through the client stderr channel.
        data = proc.stdout + (proc.stderr if name == 'container-log' and not proc.returncode else b'')
        return name, proc.returncode, data
    except subprocess.TimeoutExpired:
        return name, 124, b''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--relay-owner', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'q18a-native[0-9]+', args.name)
    out = args.out.absolute()
    assert not out.exists() and not out.is_symlink()
    out.mkdir(mode=0o700)
    prefix = '/home/<service-user>/s0/' + args.name
    commands = {
        'owner': ('tail', '-n', '50', prefix + '-run.log'),
        'queue': ('tail', '-n', '500', prefix + '.log'),
        'unit': ('systemctl', 'show', '--property=ActiveState,SubState,MainPID,ControlPID,NRestarts', 'vllm-flash'),
        'gpu': ('nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'),
        'containers': ('docker', 'ps'),
    }
    relayed, captured = set(), set()
    last_full = None
    row1_passed = False
    deadline = time.monotonic() + 5 * 3600
    with (out / 'observations.jsonl').open('x') as journal:
        while time.monotonic() < deadline:
            started = time.monotonic()
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            with ThreadPoolExecutor(max_workers=5) as pool:
                replies = list(pool.map(read, commands.items()))
            safe = [(name, rc, data) for name, rc, data in replies if not SECRET.search(data)]
            rows = {name: data.decode(errors='replace').splitlines() for name, rc, data in safe if rc == 0}
            raw = [line for name in ('owner', 'queue') for line in rows.get(name, [])
                   if line.startswith(('STEP ', 'DECLARE rc=', 'QUEUE rc=', 'OWNER_CHAIN DONE',
                                       'Q18A-OVERLAP DONE', 'PRE_STAGE PASS'))]
            fresh = list(dict.fromkeys(line for line in raw if line not in relayed))
            step_edges = [line for line in fresh if line.startswith('STEP ')]
            containers = [line for line in rows.get('containers', []) if 'qwen-flash-prod' in line]
            row = dict(observed_utc=stamp, raw_lines=fresh, unit=rows.get('unit', []),
                       gpu=rows.get('gpu', []), containers=containers,
                       read_errors={name: rc for name, rc, _ in replies if rc},
                       withheld_outputs=[name for name, _, data in replies if SECRET.search(data)],
                       box_mutation=False)
            # Full startup polling preserves the original --rm container; every
            # newly observed phase edge always requests a full log as well.
            full = bool(step_edges) or not row1_passed
            if containers:
                argv = ('docker', 'logs', 'qwen-flash-prod')
                if not full:
                    argv += ('--tail', '1000')
                _, rc, payload = read(('container-log', argv))
                row['container_log_rc'] = rc
                if not rc and not SECRET.search(payload):
                    filename = 'container-' + stamp + '.log'
                    freeze(out / filename, payload)
                    snapshot = dict(file=filename, sha256=hashlib.sha256(payload).hexdigest(),
                                    bytes=len(payload), lines=len(payload.splitlines()),
                                    observed_utc=stamp, full_log=full, containers=containers)
                    row['snapshot'] = snapshot
                    if full:
                        last_full = snapshot
                elif not rc:
                    row['withheld_container_log_sha256'] = hashlib.sha256(payload).hexdigest()
            for line in step_edges:
                phase = line.split()[1]
                label = 'first-health' if phase == 'restart' and ' PASS' in line else phase
                if label in captured or last_full is None:
                    continue
                # Receipt names the actual observation time/container listing.
                # If the original disappeared, the last snapshot is descriptive
                # and is never labelled a fresh or complete failure capture.
                payload = (out / last_full['file']).read_bytes()
                filename = 'edge-' + label + '-' + stamp + '.log'
                freeze(out / filename, payload)
                receipt = dict(last_full, file=filename, raw_step=line,
                               edge_observed_utc=stamp,
                               source_observed_this_poll=last_full['observed_utc'] == stamp,
                               box_mutation=False)
                freeze(out / (filename + '.receipt.json'),
                       (json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode())
                row.setdefault('edge_captures', []).append(receipt)
                captured.add(label)
            if fresh:
                message = args.name + ' boxro raw lines ' + stamp + ':\n' + '\n'.join(fresh)
                if row.get('edge_captures'):
                    message += '\nImmutable local captures: ' + str(out) + '\n' + json.dumps(row['edge_captures'], sort_keys=True)
                if args.relay_owner:
                    sent = subprocess.run(['herdr', 'agent', 'prompt', 'flash-next', message],
                                          capture_output=True, timeout=30)
                    row['relay_rc'] = sent.returncode
                    if sent.returncode == 0:
                        relayed.update(fresh)
                else:
                    relayed.update(fresh)
            journal.write(json.dumps(row, sort_keys=True) + '\n')
            journal.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
            if any(line.startswith('OWNER_CHAIN DONE') for line in raw):
                break
            row1_passed |= any(line.startswith('STEP row1 PASS') for line in raw)
            time.sleep(max(0, (20 if row1_passed else 3) - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
