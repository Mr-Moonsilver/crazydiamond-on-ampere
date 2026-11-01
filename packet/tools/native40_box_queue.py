#!/usr/bin/env python3
"""Derive native40's box scripts from committed native39 held-b texts.

Only names, the 40-minute outer budget/window, and one pre-row1 unarmed
plain-client check change. No settings, model, grading or restore-body change.
"""
import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OLD = 'q18a_native39_trunk'
NEW = 'q18a_native40_unarmed_hold'
ADD = '''# Native40: no native admission has been armed in this queue. A plain
# stock prefill must remain held on split until the automatic typed FULL return.
timeout --signal=TERM --kill-after=15s 180s python3 "$H/e3_native40_unarmed_box.py" \\
  --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" \\
  --timeout-s 100 --control-timeout-s 30 --out "$Q/unarmed-stock-prefill.json" \\
  > "$Q/unarmed-stock-prefill.log" 2>&1
rc=$?; record unarmed-stock-prefill "$rc" 'plain prefill: unarmed split hold -> automatic FULL -> HTTP completion, all-rank empty FULL'
[[ $rc == 0 ]] || { DONE=1; exit "$rc"; }

'''


def derive():
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = (ROOT / 'harness/runs' / (OLD + suffix)).read_text()
        text = text.replace(OLD, NEW).replace('q18a-native39-trunk', 'q18a-native40-unarmed-hold')
        if suffix == '.draft.sh':
            old = 'exec timeout --signal=TERM --kill-after=900s 16200s'
            assert text.count(old) == 1
            text = text.replace(old, old.replace('16200s', '2400s'))
            text = text.replace('from native39_trunk_gate import trunk_window_ok as window_ok',
                                'from native40_box_gate import window_ok')
            mark = 'python3 "$H/e3_native_request_plans.py"'
            assert text.count(mark) == 1
            text = text.replace(mark, ADD + mark)
            interface = '  e3_native39_w1.py e3_native39_round.py; do'
            assert text.count(interface) == 1
            text = text.replace(interface, interface.replace('; do', ' e3_native40_unarmed_box.py native40_box_gate.py; do'))
            text = text.replace('its 16200 s timeout', 'its 2400 s timeout')
        out['harness/runs/' + NEW + suffix] = text
    return out


def derive_tools():
    text = (ROOT / 'tools/native39_trunk_cpu_probe.py').read_text()
    line = "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native39_trunk_queue.py'))"
    assert text.count(line) == 1
    text = text.replace(line, line + "\nsuite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native40_box_queue.py'))")
    text = re.sub(r'^EXPECTED_TESTS=(\d+).*$', lambda m: 'EXPECTED_TESTS=' + str(int(m.group(1)) + 6)
                  + '  # inherited native39 cases plus 6 native40 queue CLI/window cases', text, flags=re.M)
    text = text.replace("owner.QNAME=='q18a-native39-trunk'", "owner.QNAME=='q18a-native40-unarmed-hold'")
    text = text.replace('q18a_native39_trunk_settings.sh', 'q18a_native40_unarmed_hold_settings.sh')
    text = text.replace('active_and_cleanup=native39_trunk_gate.TRUNK_ACTIVE_AND_CLEANUP',
                        'import native40_box_gate\nactive_and_cleanup=native40_box_gate.ACTIVE_S+native40_box_gate.CLEANUP_S')
    text = text.replace('slot_receipts=slot_receipts+trunk_slots',
                        "slot_receipts=slot_receipts+trunk_slots+sys.modules['test_native40_box_queue'].SLOT_RECEIPTS")
    wrapper = (ROOT / 'tools/native39_trunk_bundle_selfrun.py').read_text().replace(
        'native39_trunk_cpu_probe.py', 'native40_box_cpu_probe.py').replace(
        "schema='native39-trunk-bundle-selfrun/1'", "schema='native40-box-bundle-selfrun/1'")
    return {'tools/native40_box_cpu_probe.py': text, 'tools/native40_box_selfrun.py': wrapper}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rewrite', action='store_true')
    a = p.parse_args()
    for name, text in (derive() | derive_tools()).items():
        path = ROOT / name
        if a.rewrite:
            path.write_text(text)
        else:
            assert path.read_text() == text, name
    print('native40 queue derivation PASS')


if __name__ == '__main__':
    main()
