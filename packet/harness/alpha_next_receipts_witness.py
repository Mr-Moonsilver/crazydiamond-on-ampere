#!/usr/bin/env python3
"""CPU witness for the alpha-next receipt writer (PLAN section 2): old log path tears, new path does not.

Setup: WRITERS writer processes and one noise process share one os.pipe as their fd 1; the parent reads the pipe.

- old path: each writer emits records through `logging` -> a StreamHandler on the default buffered `sys.stdout`
  (`open(1, 'w')`, 8 KiB buffer on a pipe) decorated by `vllm_add_prefix` (fixture leaf, the pinned vLLM
  `system_utils._add_prefix`), with the old 12000-byte part budget. The records carry the same pid/seq/sha identity
  as the new path (the real `ReceiptWriter` formats them; only its final write goes to the logger), so one grader
  reads both and the transport is the only difference.
- new path: each writer installs the real `native_measurement.ReceiptWriter` on fd 1 and emits through the real
  `emit_receipt_lines` / `emit_receipt_kv`, with the new 3968-byte budget.
- noise: one process writes 12 KB vLLM-style lines through the old path the whole time (vLLM's own output).

Records are a mix of multi-part progress parts (real `ProgressTrace.lines`), anatomy parts (real `AnatomyTrace`),
`native-alloc` and `memory` k=v lines. The parent grades the stream with `harness/receipt_lines.check`.

usage: alpha_next_receipts_witness.py [--records N] [--writers 4] [--out DIR]   (CPU only; no box, no GPU)
"""
import io
import json
import logging
import multiprocessing
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'patches' / 's5_engine'))

import receipt_lines  # noqa: E402

OLD_BUDGET = 12000
NOISE_BYTES = 12 * 1024
NOISE_PAUSE_S = 0.0005
INNER_HEADER = re.compile(r'\((?:Worker_\w+|EngineCore\w*) pid=\d+\) ')


# --- fixture leaf alpha-next-receipts-vllm-add-prefix --------------------------------------------------------------
# Replaces vLLM `vllm/utils/system_utils._add_prefix` (called by `decorate_logs`, `multiproc_executor.py:1067, 1095`).
# The pinned source is not under reference/vllm-src. Copied from a hub tree whose 491 overlapping files are
# byte-identical to reference/vllm-src (system_utils.py sha256 837e6ff0ab524c06c048d6b42d4f35b77baba2e9d9e57837fe333ca0aaee1cfb,
# lines 201-234); declared equivalent, not extracted from the image. The only change: the colour branch is fixed to
# the non-tty prefix (`envs` is not importable here), which is the box's case (its lines carry no colour codes).
def vllm_add_prefix(file, worker_name, pid):
    """Add colored prefix to file output for log decoration."""
    prefix = f"({worker_name} pid={pid}) "
    # Use the original write to avoid nesting prefixes on repeated calls.
    file_write = getattr(file, "_original_write", file.write)

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:  # type: ignore[attr-defined]
            file_write(prefix)
        idx = 0
        while (next_idx := s.find("\n", idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx])
            if next_idx == len(s):
                file.start_new_line = True  # type: ignore[attr-defined]
                return
            file_write(prefix)
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # type: ignore[attr-defined]

    file.start_new_line = True  # type: ignore[attr-defined]
    file._original_write = file_write  # type: ignore[attr-defined]
    file.write = write_with_prefix  # type: ignore[method-assign]


def old_path_logger(name, stream=None):
    """vLLM's worker log path: decorated default sys.stdout on fd 1, a StreamHandler, INFO.

    ``stream`` (tests): a stream to decorate instead of a new default ``open(1, 'w')``.
    """
    if stream is None:
        # The default object: TextIOWrapper over a BufferedWriter whose size is the pipe's st_blksize.
        stream = open(1, 'w', closefd=False)
        sys.stdout = stream
    vllm_add_prefix(stream, name, os.getpid())
    logger = logging.getLogger('witness.%s.%d' % (name, id(stream)))
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter('%(levelname)s %(asctime)s [executor_hooks.py:1303] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def records_for(nm, rng, rank, kind):
    """One record's texts: ('lines', [parts]) or ('kv', text)."""
    if kind == 'progress':
        trace = nm.ProgressTrace('worker', rank)
        for i in range(rng.randrange(1, 160)):
            t = rng.randrange(1 << 40)
            trace.record('r', i, 'execute_model', t, t + rng.randrange(10 ** 6), t + rng.randrange(10 ** 7), False, True)
        return 'lines', trace.lines(txn=rng.randrange(1000), chunk_start=0, chunk_end=1024)
    if kind == 'anatomy':
        trace = nm.AnatomyTrace('worker', rank)
        for i in range(rng.randrange(1, 120)):
            t = rng.randrange(1 << 40)
            trace.record('x', i, 'qsa-attended', rng.randrange(48), rng.randrange(3), t, t + 5, t + 9000)
        return 'lines', trace.lines(txn=rng.randrange(1000), stage=rank // 2)
    if kind == 'alloc':
        return 'kv', ('flash-s5 native-alloc rank=%d txn_seq=%d chunk=0-1024 union_retire=1 union_sets=%d '
                      'union_retired=0 union_single=1 union_live_max=1 indexer_layers=1 indexer_rows=1 indexer_width=1 '
                      'indexer_chunks=1 indexer_peak=1 indexer_total=1 indexer_errors=0' % (rank, rng.randrange(999), rng.randrange(9)))
    return 'kv', ('flash-s5 memory point=open rank=%d txn_seq=%d chunk=0-1024 allocated=%d reserved=2 inactive_split=3 '
                  'alloc_retries=0 ooms=0 free=1 total=2 sync_all_streams=0 device_allocs=1 device_frees=1'
                  % (rank, rng.randrange(999), rng.randrange(1 << 30)))


KINDS = ('progress', 'anatomy', 'alloc', 'memory')


def writer_main(mode, fd, rank, target, seed, done):
    """Child: fd becomes fd 1; emit records until `target` records have been written."""
    os.dup2(fd, 1)
    import native_measurement as nm
    rng = random.Random(seed)
    logger = old_path_logger('Worker_PP%d_TP%d_EP%d' % (rank // 2, rank % 2, rank))
    if mode == 'old':
        nm.PROGRESS_LINE_BYTES = OLD_BUDGET
        writer = nm.ReceiptWriter(1, write=lambda _fd, data: (logger.info(data.decode().rstrip('\n')), len(data))[1])
        writer.limit = OLD_BUDGET + 256  # old path: the 12000-byte budget, through the logger
        nm.install_receipt_writer(writer)
    else:
        writer = nm.install_receipt_writer(nm.ReceiptWriter())
    while writer.seq < target:
        form, texts = records_for(nm, rng, rank, rng.choice(KINDS))
        if form == 'lines':
            nm.emit_receipt_lines(texts, logger)
        else:
            nm.emit_receipt_kv(texts, logger)
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    done.put(writer.stats())


def noise_main(fd, stop, seed):
    os.dup2(fd, 1)
    rng = random.Random(seed)
    logger = old_path_logger('Worker_PP9_TP9_EP9')
    lines = 0
    while not stop.is_set():
        size = rng.randrange(NOISE_BYTES // 2, NOISE_BYTES)
        logger.info('noise ' + ''.join(rng.choice('abcdefghij') for _ in range(64)) * (size // 64))
        lines += 1
        time.sleep(NOISE_PAUSE_S)
    sys.stdout.flush()


def run(mode, *, writers=4, records=100_000, seed=1, noise=True, keep=None):
    """One witness run; returns the receipt_lines report plus counts. `keep`: path to write the raw stream to."""
    ctx = multiprocessing.get_context('fork')
    r, w = os.pipe()
    done = ctx.Queue()
    stop = ctx.Event()
    per = -(-records // writers)
    procs = [ctx.Process(target=writer_main, args=(mode, w, rank, per, seed * 100 + rank, done)) for rank in range(writers)]
    noise_proc = ctx.Process(target=noise_main, args=(w, stop, seed)) if noise else None
    started = time.monotonic()
    for p in procs + ([noise_proc] if noise_proc else []):
        p.start()
    os.close(w)
    stats = []

    def finisher():
        for _ in procs:
            stats.append(done.get())
        for p in procs:
            p.join()
        stop.set()
    t = threading.Thread(target=finisher, daemon=True)
    t.start()
    counts = dict(lines=0, bytes=0, spliced=0)
    sink = open(keep, 'wb') if keep else None

    def stream():
        tail = b''
        while True:
            data = os.read(r, 1 << 20)
            if sink:
                sink.write(data)
            counts['bytes'] += len(data)
            if not data:
                break
            pieces = (tail + data).split(b'\n')
            tail = pieces.pop()
            for piece in pieces:
                line = piece.decode(errors='replace') + '\n'
                counts['lines'] += 1
                found = receipt_lines._record_of(line)
                if found and (sum(line.count(p) for p in receipt_lines.PREFIXES) > 1 or INNER_HEADER.search(found[1])):
                    counts['spliced'] += 1
                yield line
        if tail:
            counts['lines'] += 1
            yield tail.decode(errors='replace')
    report = receipt_lines.check(stream())
    os.close(r)
    if sink:
        sink.close()
    t.join()
    if noise_proc:
        noise_proc.join()
    bad = len(report['missing_seq']) + len(report['bad_sha']) + len(report['incomplete'])
    return dict(mode=mode, writers=writers, records_target=per * writers, records_read=report['records'],
                writer_stats=stats, physical_lines=counts['lines'], bytes=counts['bytes'],
                lines_with_spliced_writer=counts['spliced'], torn_or_lost=bad,
                missing_seq=len(report['missing_seq']), bad_sha=len(report['bad_sha']),
                incomplete=len(report['incomplete']), unsafe_path=report['unsafe_path'],
                seconds=round(time.monotonic() - started, 3),
                seq_contiguous=not report['missing_seq'],
                per_process={k: dict(first_seq=v['first_seq'], last_seq=v['last_seq'], records=v['records'], kinds=v['kinds'])
                             for k, v in report['per_process'].items()})


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--records', type=int, default=100_000)
    parser.add_argument('--writers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', default=None)
    args = parser.parse_args(argv)
    result = {}
    for mode in ('old', 'new'):
        result[mode] = run(mode, writers=args.writers, records=args.records, seed=args.seed)
        print(json.dumps({k: v for k, v in result[mode].items() if k not in ('per_process', 'writer_stats')}, sort_keys=True),
              flush=True)
    result['pass'] = (result['old']['torn_or_lost'] > 0 and result['new']['torn_or_lost'] == 0
                      and not result['new']['unsafe_path'] and result['new']['records_read'] >= args.records)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print('witness: ' + ('PASS' if result['pass'] else 'FAIL'))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
