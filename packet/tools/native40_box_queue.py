#!/usr/bin/env python3
"""Derive native40's box scripts from committed native39 held-b texts.

Only names, the 40-minute outer budget/window, and one pre-row1 unarmed
plain-client check change. No settings, model, grading or restore-body change.
"""
import argparse
from pathlib import Path

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rewrite', action='store_true')
    a = p.parse_args()
    for name, text in derive().items():
        path = ROOT / name
        if a.rewrite:
            path.write_text(text)
        else:
            assert path.read_text() == text, name
    print('native40 queue derivation PASS')


if __name__ == '__main__':
    main()
