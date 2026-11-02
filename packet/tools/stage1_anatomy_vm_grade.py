#!/usr/bin/env python3
"""Grade a stage1-anatomy VM record: mechanics only (one rank, no PP/TP; no stage-1 number comes from it).

    python3 tools/stage1_anatomy_vm_grade.py RECORD OUT

PASS needs the wrapper's PASS and every check below; a check that cannot be read is INC; a check that reads wrong is
FAIL. The archived lines are also read back through tools/stage1_anatomy_report.py (unstamped, so every row is
unassigned: there is no engine flush on the VM).
"""
import json
import lzma
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_anatomy_report as report  # noqa: E402

REQUIRED_INSTALLED = ('runner.execute_model', 'runner.sample_tokens')
FORWARD_TARGETS = ('runner._model_forward', 'cgm.run_fullgraph', 'cgm.run_pw_graph', 'split_cgm.run_fullgraph')
ZERO_COUNTERS = ('observer_errors', 'log_failures', 'rows_dropped', 'events_pool_empty')


def read_json(record, name):
    path = os.path.join(record, name)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def grade(record):
    checks, notes = {}, {}
    result, probe = read_json(record, 'result.json'), read_json(record, 'probe.json')
    checks['wrapper_pass'] = None if result is None else result.get('verdict') == 'PASS'
    s = (probe or {}).get('stage1_anatomy')
    if s is None:
        checks['summary_present'] = None if probe is None else False
        return checks, notes, None
    installed = s.get('installed') or []
    counters = s.get('counters') or {}
    checks.update(
        recorder_installed=s.get('recorder_installed') is True,
        lines_written=s.get('handler_lines', 0) > 0 and s.get('flushes', 0) > 0,
        lines_parse=s.get('parse_errors') == [] and s.get('incomplete_flushes') == [],
        lines_bounded=0 < s.get('max_line_bytes', 0) < s.get('line_limit_bytes', 0),
        progress_lines=s.get('progress_lines', 0) > 0,
        stock_rows=(s.get('kinds') or {}).get('e', 0) > 0 and (s.get('kinds') or {}).get('s', 0) > 0,
        wrapped_names=all(n in installed for n in REQUIRED_INSTALLED) and any(n in installed for n in FORWARD_TARGETS),
        forward_seam='forward' in (s.get('seam_rows') or {}),
        events_resolve=s.get('rows_with_device_forward', 0) > 0,
        calibration=bool((s.get('calibration') or {})),
        counters_zero=all(counters.get(k, 0) == 0 for k in ZERO_COUNTERS),
        replies_seen=(s.get('output_rows_async_thread', 0) > 0 if (s.get('identity') or {}).get('async_scheduling')
                      else s.get('output_rows_main_thread', 0) > 0),
        rpcs_match=s.get('rpcs', 0) >= s.get('main_rows', 0) > 0)
    notes.update(identity=s.get('identity'), installed=installed, missing=s.get('missing'), counters=counters,
                 calibration=s.get('calibration'), kinds=s.get('kinds'), seam_rows=s.get('seam_rows'),
                 wall_ns_p50=s.get('wall_ns_p50'), observer_ns_p50=s.get('observer_ns_p50'),
                 rows_with_device_forward=s.get('rows_with_device_forward'), execute_rows=s.get('execute_rows'),
                 output_rows_async_thread=s.get('output_rows_async_thread'),
                 output_rows_main_thread=s.get('output_rows_main_thread'), rpcs=s.get('rpcs'),
                 releases=s.get('releases'), handler_lines=s.get('handler_lines'), max_line_bytes=s.get('max_line_bytes'))
    archive = os.path.join(record, 'stage1-anatomy-lines.jsonl.xz')
    readback = None
    if os.path.isfile(archive):
        with lzma.open(archive, 'rt') as handle:
            lines = [json.loads(line) for line in handle]
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, 'engine-current.log'), 'w') as handle:
                handle.write('\n'.join('INFO ' + t for t in lines) + '\n')
            got = report.analyse(root, native=False)
        readback = dict(handler_lines=got['handler_lines'], torn=got['torn_handler_lines'],
                        incomplete=len(got['incomplete_flushes']), gaps=len(got['line_gaps']), rows=got['rows'],
                        per_rank_kind=(got['groups'].get('unassigned') or {}).get('per_rank_kind'))
        checks['analyser_reads'] = (got['handler_lines'] == s.get('handler_lines') and got['torn_handler_lines'] == 0
                                    and not got['incomplete_flushes'] and got['rows']['main'] > 0)
    else:
        checks['analyser_reads'] = None
    return checks, notes, readback


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    record, out = argv
    if os.path.exists(out):
        raise SystemExit('output exists: ' + out)
    checks, notes, readback = grade(record)
    verdict = ('FAIL' if any(v is False for v in checks.values()) else
               'INC' if any(v is None for v in checks.values()) else 'PASS')
    doc = dict(schema='stage1-anatomy-vm-grade/1', record=os.path.abspath(record), verdict=verdict,
               rc={'PASS': 0, 'FAIL': 1, 'INC': 2}[verdict], checks=checks, notes=notes, analyser_readback=readback,
               scope='mechanics only on the real vLLM worker and model runner (one rank, no PP/TP); '
                     'no stage-specific number comes from this record')
    with open(out, 'x') as handle:
        json.dump(doc, handle, indent=1, sort_keys=True, default=str)
    print(json.dumps(dict(verdict=verdict, rc=doc['rc'], checks=checks), sort_keys=True))
    return doc['rc']


if __name__ == '__main__':
    raise SystemExit(main())
