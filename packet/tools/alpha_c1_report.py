#!/usr/bin/env python3
"""C1 post-run report over the box record: C0's report (tools/alpha_c0_report.py), regrouped by chunk for two chunks in flight.

On an overlap arm one engine progress line is flushed per commit and may hold the successor's early rows; rows carry no
transaction. The `flash-s5 stage-overlap txn=T seqs=RANGES successor=S` line names each chunk's command sequences, so rows are
regrouped by transaction before C0's per-chunk figures are taken. A row no overlap line names stays with its flush (serial and
local arms log no such line, so their chunks are C0's, unchanged).

Added per chunk, beside C0's figures:
  slow_rank       per stage, per own TP rank, merged time from send to that rank's reply over the stage's commands; the slower
                  rank and its seconds (the lead's addition 2). Median per arm and per history band next to the pooled medians.
  commit_interval_s   apply of this chunk's last command after the previous chunk's (the throughput interval when chunks overlap;
                  C0's wall_s then spans into the neighbours).
Reads the record, changes nothing. No GPU, no box action.
"""
import alpha_c0_report as c0
# The grade's rule (harness/e3_alpha_c1_box.py), not a second one; alpha_c0_report put the harness on the path.
from e3_alpha_c1_box import OVERLAP_LINE, TARGETS, overlap_owners, regroup, seqs_of, slow_rank, slow_summary  # noqa: E402,F401


def chunk_figures(chunks, workers, by_rank):
    """C0's chunk_report per regrouped chunk, plus slow_rank and the commit interval."""
    rows, previous = [], None
    for chunk in chunks:
        report = c0.chunk_report(chunk, workers, by_rank)
        report['txn'] = chunk['identity'].get('txn')
        report['slow_rank'] = slow_rank(chunk)
        applied = [r['t_apply'] for r in chunk['rows'].get('n', []) if type(r.get('t_apply')) is int]
        last = max(applied) if applied else None
        report['commit_interval_s'] = (last - previous) / 1e9 if last is not None and previous is not None else None
        previous = last if last is not None else previous
        rows.append(report)
    return rows

