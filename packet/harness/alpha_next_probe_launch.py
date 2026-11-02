#!/usr/bin/env python3
"""alpha-next engine self-run: run ONE frozen probe with its receipt lines read from the PLAN 2 writer's own records (declared).

The frozen probes (e3_native41_progress_probe.py and the held-g, Step A/B and C0 probes that reuse its Capture) collect
receipt lines with a logging.Handler on the emit sites' loggers. PLAN 2 sends every receipt through native_measurement's
process ReceiptWriter (one os.write per record on a private dup of fd 1) and no longer through the logger, so the probes saw no
line (g4 engine self-run: 39 cases "progress-lines-bounded {longest: 0}", HARNESS; the records were on the case's stdout).

What this launcher does, before the probe runs:
  * installs a REAL native_measurement.ReceiptWriter as the process writer (the engine's own install_receipt_writer() calls are
    idempotent and keep it). Only its ``write`` seam is replaced (fixture leaf probe-launch-receipt-sink, declared in
    alpha_next_probe_fixture_leaves.json): the bytes it would os.write to fd 1 are taken in memory (and copied to fd 2 so the
    case log still shows them). Numbering, identity, splitting and the record limit are the writer's own code.
  * reads every record back with receipt_lines.parse_line (the box grader's parser, the same one box.receipt_read uses over the
    engine-stream log) and hands the bare text (identity stripped, as the grader sees it) to the probe's capture handlers on the
    emit-site loggers, once per handler, as the logger route did.
  * at the end, runs receipt_lines.check over the captured records: any missing seq, bad sha or incomplete record makes the
    launcher exit 3 when the probe itself passed (a writer framing defect is never hidden). unsafe_path is reported only: the
    probe's stdout is a file in this container (unsafe:not-fifo, by construction); the box FIFO proof is owed to the run.
The logger fallback (no writer installed) stays covered by test_alpha_next_receipts. Nothing in the engine changes.

usage: python3 -B alpha_next_probe_launch.py PROBE.py [probe args...]
Over an overlay without the PLAN 2 writer (the negative's Step B overlay) it installs nothing and reports writer=absent.
"""
import importlib.abc
import importlib.util
import logging
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 'native_measurement'
EMIT_LOGGERS = ('engine_driver', 'executor_hooks', 'stage_progress', 'worker_bridge', 'native_measurement')
STATE = dict(writer='absent', modules=[], records=0, delivered=0)
LINES = []
_SEEN = set()
_WRITER = []


def _sink(fd, data):
    """The writer's os.write leaf: one call per record, exactly the bytes the box pipe would carry."""
    os.write(2, data)
    text = data.decode('utf-8', 'replace')
    for line in text.splitlines(keepends=True):
        LINES.append(line)
        STATE['records'] += 1
        record = receipt_lines.parse_line(line)
        if record and record.get('text'):
            _deliver(record['text'])
    return len(data)


def _deliver(text):
    handlers = []
    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(lg, logging.Logger) and name.rpartition('.')[2] in EMIT_LOGGERS:
            handlers += [h for h in lg.handlers if h not in handlers]
    record = logging.LogRecord('alpha-next-probe-launch', logging.INFO, __file__, 0, text, None, None)
    for handler in handlers:
        handler.handle(record)
        STATE['delivered'] += 1


def _install(module, name):
    STATE['modules'].append(name)
    if not hasattr(module, 'ReceiptWriter') or not hasattr(module, 'install_receipt_writer'):
        return
    if _WRITER:
        module.install_receipt_writer(_WRITER[0])     # a second copy of the module (plain vs package name): the same writer
        return
    writer = module.ReceiptWriter(fd=1, write=_sink)
    module.install_receipt_writer(writer)
    _WRITER.append(writer)
    STATE['writer'] = 'real-ReceiptWriter-with-sink'


class _Hook(importlib.abc.MetaPathFinder):
    """After any module named ``native_measurement`` (plain or package-qualified) executes, install the writer above."""

    def find_spec(self, name, path, target=None):
        if name.rpartition('.')[2] != TARGET or name in _SEEN:
            return None
        _SEEN.add(name)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            return spec
        original = spec.loader.exec_module

        def exec_module(module, _original=original, _name=name):
            _original(module)
            _install(module, _name)
        spec.loader.exec_module = exec_module
        return spec


def _integrity():
    if not LINES:
        return {}, 0
    report = receipt_lines.check(LINES, require_start=False)
    counts = {k: len(report[k]) for k in ('missing_seq', 'bad_sha', 'incomplete', 'unsafe_path')}
    return counts, counts['missing_seq'] + counts['bad_sha'] + counts['incomplete']


def main(argv):
    assert len(argv) >= 2 and argv[1].endswith('_probe.py') and os.path.basename(argv[1]) == argv[1], argv
    assert not any(n.rpartition('.')[2] == TARGET for n in sys.modules), 'native_measurement imported before the hook'
    sys.meta_path.insert(0, _Hook())
    probe = os.path.join(HERE, argv[1])
    sys.argv = [probe] + argv[2:]
    code = 0
    try:
        runpy.run_path(probe, run_name='__main__')
    except SystemExit as exc:
        code = exc.code
    counts, defects = _integrity()
    print('alpha-next probe launcher (declared, PLAN 2): writer=%s modules=%s records=%d delivered=%d integrity=%s'
          % (STATE['writer'], ','.join(STATE['modules']) or 'none', STATE['records'], STATE['delivered'],
             counts or 'no-records'), file=sys.stderr)
    if defects and code in (0, None):
        print('alpha-next probe launcher: receipt framing defects, exit 3', file=sys.stderr)
        code = 3
    sys.exit(code)


sys.path.insert(0, HERE)
import receipt_lines  # noqa: E402

if __name__ == '__main__':
    main(sys.argv)
