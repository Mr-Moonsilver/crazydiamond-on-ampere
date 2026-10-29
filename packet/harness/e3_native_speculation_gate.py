"""Require the actual engine startup log to declare speculative_config=None.

This observes configuration; it never edits settings or overrides MTP. Missing
or ambiguous startup evidence is INCONCLUSIVE, an enabled configuration FAIL.
No full log line or configuration value is included in the small receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def grade(text):
    clean = re.sub(r'\x1b\[[0-9;]*m', '', text)
    starts = [line for line in clean.splitlines()
              if re.search(r'\[core\.py:\d+\] Initializing a V1 LLM engine .*with config:', line)]
    if not starts:
        return dict(verdict='INCONCLUSIVE', rc=2, reason='engine-startup-config-missing', observations=0)
    fields = [re.search(r'\bspeculative_config\s*=\s*([^,]+)', line) for line in starts]
    if any(field is not None and field.group(1).strip() != 'None' for field in fields):
        return dict(verdict='FAIL', rc=1, reason='engine-speculative-config-enabled', observations=len(starts))
    if any(field is None for field in fields):
        return dict(verdict='INCONCLUSIVE', rc=2, reason='engine-speculative-config-unobserved', observations=len(starts))
    return dict(verdict='PASS', rc=0, reason='engine-speculative-config-none', observations=len(starts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    row = grade(raw.decode('utf-8', errors='replace'))
    row.update(schema='native-engine-speculation-gate/1', log_sha256=hashlib.sha256(raw).hexdigest(),
               settings_modified=False)
    with args.out.open('x') as stream:
        json.dump(row, stream, indent=2)
        stream.write('\n')
    print(json.dumps(row), flush=True)
    return row['rc']


if __name__ == '__main__':
    raise SystemExit(main())
