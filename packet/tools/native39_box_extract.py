#!/usr/bin/env python3
"""Native39: a small extract of retained box artifacts for the trunk queue tests.

The trunk checks read three box formats this tree has no copy of: the host
observer's followed container log (docker --timestamps lines) and host-disk
rows, the identity capture, the native35 performance grade, and the plain W1
client's output.  This writes harness/fixtures/native39_box_extract.json once
from the retained fn-main records (each source file's sha256 recorded):
  * q18a-native35-p-c run 1 (31 Oct 2026, overlay without the bind line):
    first lines and the 'Starting vLLM server' line of engine-stream.log, the
    count of bind lines (0), host-disk container ids, the captured command's
    --host values and defaults EXTRA_ARGS, the whole performance grade, the
    startup engine_epoch;
  * q18b-w1ab2 run 1 arm B0 (31 Oct 2026, EXTRA_ARGS ... --host 127.0.0.1 on
    the held-a overlay): the 'Found duplicate keys --host' warning and the
    'Starting vLLM server' line of its followed log, the first and last rows of
    its health.tsv, and its w1_trace output without the raw events.
No secret is read: the identity's env block is not copied, and no auth file is
opened.  Refuses to overwrite.
"""
import hashlib
import json
import lzma
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
import e3_native39_trunk_checks as checks  # noqa: E402

FN = Path('<hub-dir>/agents/flash-next/fn-main/results/box')
PC = FN / 'q18a-native35-p-c-run-1/box/n35pc/q18a-native35-p-c'
B0 = FN / 'q18b-w1ab2-run-1/box/q18b-w1ab2/B0'
OUT = ROOT / 'harness/fixtures/native39_box_extract.json'
SOURCES = dict(p_c_engine_stream=PC / 'engine-stream.log.xz', p_c_engine_stream_err=PC / 'engine-stream.err',
               p_c_host_disk=PC / 'host-disk.jsonl', p_c_identity_start=PC / 'engine.identity-start.json.xz',
               p_c_grade=PC / 'native35-performance-grade.json', p_c_startup_grade=PC / 'startup-grade.json',
               w1ab2_b0_follow=B0 / 'engine.follow.1.log', w1ab2_b0_health=B0 / 'health.tsv',
               w1ab2_b0_w1=B0 / 'w1.json')


def text(path):
    data = path.read_bytes()
    return (lzma.decompress(data) if path.suffix == '.xz' else data).decode(errors='replace')


def lines_matching(body, pattern):
    return [line for line in body.splitlines() if re.search(pattern, line)]


def main():
    assert not OUT.exists(), 'extract exists'
    sources = {name: dict(path=str(path.relative_to(FN.parents[1])), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
               for name, path in SOURCES.items()}
    log, err = text(SOURCES['p_c_engine_stream']), text(SOURCES['p_c_engine_stream_err'])
    identity = json.loads(text(SOURCES['p_c_identity_start']))['identity']
    args = identity['command']['args']
    rows = [json.loads(line) for line in text(SOURCES['p_c_host_disk']).splitlines()]
    follow = text(SOURCES['w1ab2_b0_follow'])
    health = text(SOURCES['w1ab2_b0_health']).splitlines()
    w1 = json.loads(text(SOURCES['w1ab2_b0_w1']))
    extract = dict(
        schema='native39-box-extract/1', sources=sources,
        p_c=dict(engine_stream=dict(lines=len(log.splitlines()), head=log.splitlines()[:3],
                                    served=lines_matching(log, checks.SERVED_RE.pattern),
                                    bind_lines=len(lines_matching(log + err, checks.BIND_RE.pattern))),
                 host_disk=dict(rows=len(rows), container_ids=sorted({r['container_id'] for r in rows})),
                 identity=dict(host_args=[args[j + 1] for j, x in enumerate(args[:-1]) if x == '--host'],
                               defaults_extra_args=identity['defaults'].get('EXTRA_ARGS')),
                 grade=json.loads(text(SOURCES['p_c_grade'])),
                 startup_engine_epoch=json.loads(text(SOURCES['p_c_startup_grade'])).get('engine_epoch')),
        w1ab2_b0=dict(follow=dict(duplicate_host_warning=lines_matching(follow, r'Found duplicate keys --host'),
                                  served=lines_matching(follow, checks.SERVED_RE.pattern),
                                  bind_lines=len(lines_matching(follow, checks.BIND_RE.pattern))),
                      health=dict(rows=len(health), first=health[:3], last=health[-2:]),
                      w1={k: v for k, v in w1.items() if k != 'raw'}))
    with OUT.open('x') as f:
        json.dump(extract, f, sort_keys=True, indent=1)
        f.write('\n')
    OUT.chmod(0o664)
    print('native39 box extract written', OUT.stat().st_size, hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
