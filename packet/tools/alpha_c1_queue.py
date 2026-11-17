#!/usr/bin/env python3
"""Derive the q18a-alpha-c1 box scripts from the committed q18a-alpha-c0 texts (C0 l). Nothing a past queue ran is edited.

Kept from C0, byte for byte: everything alpha_c0_queue kept from alpha-perf (pins, preflight, startup, bind, host headroom,
the unarmed check, the end identity, both band manifests and their /tokenize guards, the le-fix death scan and watch, the M3
allocator receipts, the owned cleanup/restore tail), C0's switch FLASH_S5_STAGE_LOCAL=1 and its asserts, and C0's
no-window note. Declared deltas, each applied exactly once:
  names    q18a_alpha_c0 -> q18a_alpha_c1, q18a-alpha-c0 -> q18a-alpha-c1 (only these two forms; C0's own module, gate and
           assert texts stay C0's where they are kept);
  budget   the draft reads alpha_c1_gate (9,000 s box budget less cleanup; LAST_ARM_ENTRY_S the rule; no absolute window);
  switch   FLASH_S5_STAGE_OVERLAP=1 in EXTRA_ENV beside C0's FLASH_S5_STAGE_LOCAL=1, asserted in declaration and preflight;
  files    e3_alpha_c1_box.py and alpha_c1_gate.py added to the preflight interface list and the declaration pins;
  body     sixteen arms: band 1 mirrored over serial, local and overlap, solo and live (rounds 0 and 1), then band 2 (rounds 2
           and 3) on alpha-perf's 131,094-token manifest, each behind a typed FULL; no optional arm; the grade is e3_alpha_c1_box.
Labels carry the round, so every cache salt stays unique and the band follows from the index as in alpha-perf.
No model setting, gate threshold, launcher or restore change. --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import alpha_c0_queue as c0  # noqa: E402

OLD, NEW = 'q18a_alpha_c0', 'q18a_alpha_c1'
OLD_Q, NEW_Q = 'q18a-alpha-c0', 'q18a-alpha-c1'
SWITCH = 'FLASH_S5_STAGE_OVERLAP=1'
ARMS = ('serial-solo-0', 'local-solo-0', 'overlap-solo-0', 'overlap-live-0', 'local-live-0', 'serial-live-0',
        'serial-live-1', 'local-live-1', 'overlap-live-1', 'overlap-solo-1', 'local-solo-1', 'serial-solo-1',
        'serial-solo-2', 'overlap-solo-2', 'overlap-live-3', 'serial-live-3')
HEADER_NOTE = ('# C1: this queue compares two chunks in flight (stage overlap, FLASH_S5_STAGE_OVERLAP) with C0\'s stage-local path and\n'
               '# the frozen serial path on one engine, matched per arm by the typed native_stage_local control: band 1 mirrored\n'
               '# rounds at 65,445 tokens, band 2 at 131,094 tokens. Mechanism and numerics gate (crash-free, engagement with an early\n'
               '# open, one command in flight per stage, same generated tokens, OOM); speed and memory are reported; release\n'
               '# readiness is not assessed. C0\'s text follows.\n#\n')
LOOP_OLD = ('  for _label in serial-solo-0 local-solo-0 local-live-0 serial-live-0 serial-live-1 local-live-1 local-solo-1 serial-solo-1; do\n'
            '    IFS=- read -r _mode _cohort _pair <<< "$_label"\n'
            '    _workload=$H/workloads/q18a-native41-progress.json\n')
LOOP_NEW = ('  for _label in %s; do\n'
            '    IFS=- read -r _mode _cohort _pair <<< "$_label"\n'
            '    if [[ $_pair == [01] ]]; then _workload=$H/workloads/q18a-native41-progress.json\n'
            '    else _workload=$H/workloads/q18a-alpha-perf-band2.json; fi\n' % ' '.join(ARMS))
BODY_SWAPS = (
    ('# C0: band 1 ABBA (serial-solo-0 .. serial-solo-1). Each arm selects\n'
     '# burst and its serial or local path by typed control while paused idle FULL. Speed and memory are descriptive. Every\n'
     '# timeout below is one alpha_c0_gate cap (instruments, arm_entries, alpha_c0_arms, ingress, end_full_idle, grade); each arm\n',
     '# C1: band 1 mirrored (serial-solo-0 .. serial-solo-1), then band 2 (serial-solo-2 .. serial-live-3). Each arm selects\n'
     '# burst and its serial, local or overlap path by typed control while paused idle FULL. Speed and memory are descriptive. Every\n'
     '# timeout below is one alpha_c1_gate cap (instruments, arm_entries, alpha_c1_arms, ingress, end_full_idle, grade); each arm\n'),
    (LOOP_OLD, LOOP_NEW),
    ('1500s python3 "$H/e3_alpha_c0_box.py" arm', '1500s python3 "$H/e3_alpha_c1_box.py" arm'),
    ("'auto arm: 8 standing decoders, burst plus the serial or local selection, fresh, append; live keeps them, solo cancels'",
     "'auto arm: 8 standing decoders, burst plus the serial, local or overlap selection, fresh, append; live keeps them, solo cancels'"),
    ('timeout 600s python3 "$H/e3_alpha_c0_box.py" grade', 'timeout 600s python3 "$H/e3_alpha_c1_box.py" grade'),
    ('--report "$Q/alpha-c0-report.md" --out "$Q/alpha-c0-grade.json" > "$Q/alpha-c0-grade.log" 2>&1\n'
     "record alpha-c0 \"$?\" 'per-arm mechanism (crash-free, TP commitment, engagement), complete progress lines; speed and memory descriptive'\n",
     '--report "$Q/alpha-c1-report.md" --out "$Q/alpha-c1-grade.json" > "$Q/alpha-c1-grade.log" 2>&1\n'
     "record alpha-c1 \"$?\" 'per-arm mechanism (crash-free, TP commitment, engagement, early open, one command in flight, OOM); speed and memory descriptive'\n"),
)
BUDGET_OLD = c0.BUDGET_NEW
BUDGET_NEW = 'from alpha_c1_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES\n'
DRAFT_SWITCH = "assert env.get('FLASH_S5_STAGE_OVERLAP') == '1', 'alpha-c1 stage-overlap switch'\n"
DECLARE_SWITCH = "assert env.get('FLASH_S5_STAGE_OVERLAP')=='1', 'alpha-c1 runs with the stage-overlap switch installed'\n"
SETTINGS_TOKEN = 'FLASH_S5_NATIVE_QSA_POLICY=dense FLASH_S5_STAGE_LOCAL=1 '
SETTINGS_ANCHOR = '# C0: FLASH_S5_STAGE_LOCAL=1 below,'
SETTINGS_NOTE = ('# C1: FLASH_S5_STAGE_OVERLAP=1 below, the only setting added to C0; each arm selects serial, local or overlap by '
                 'typed control.\n')
INTERFACE_OLD = '  e3_alpha_c0_box.py alpha_c0_gate.py; do'
INTERFACE_NEW = '  e3_alpha_c0_box.py alpha_c0_gate.py \\\n  e3_alpha_c1_box.py alpha_c1_gate.py; do'
PINS_OLD = 'e3_alpha_c0_box.py alpha_c0_gate.py '
PINS_NEW = PINS_OLD + 'e3_alpha_c1_box.py alpha_c1_gate.py '
DRAFT_SWAPS = (
    ("perf=p.parent/'alpha-c0-grade.json'", "perf=p.parent/'alpha-c1-grade.json'"),
    ("'missing':['alpha-c0-grade-absent']", "'missing':['alpha-c1-grade-absent']"),
    ('# arms are replaced by the held-f arms. Every cap is alpha_c0_gate.PHASES.',
     '# arms are replaced by the held-f arms. Every cap is alpha_c1_gate.PHASES.'),
    ('# number. Speed is descriptive. Alpha-perf text above; every cap here is alpha_c0_gate.\n',
     '# number. Speed is descriptive. Alpha-perf text above; every cap here is alpha_c1_gate.\n'),
)


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:120], text.count(old))
    return text.replace(old, new)


def rename(text):
    return text.replace(OLD, NEW).replace(OLD_Q, NEW_Q)


def body():
    text = c0.BODY
    for old, new in BODY_SWAPS:
        text = replace_once(text, old, new)
    return text


BODY = body()


def derive():
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = (ROOT / 'harness/runs' / (OLD + suffix)).read_text()
        if suffix == '.draft.sh':
            text = replace_once(text, c0.BODY, BODY)
            text = replace_once(text, BUDGET_OLD, BUDGET_NEW)
            text = replace_once(text, INTERFACE_OLD, INTERFACE_NEW)
            text = replace_once(text, c0.DRAFT_SWITCH, c0.DRAFT_SWITCH + DRAFT_SWITCH)
            for a, b in DRAFT_SWAPS:
                text = replace_once(text, a, b)
            first = text.index('# C0: this queue compares')
            text = text[:first] + HEADER_NOTE + text[first:]
        elif suffix == '_settings.sh':
            text = replace_once(text, SETTINGS_TOKEN, SETTINGS_TOKEN + SWITCH + ' ')
            text = replace_once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
        else:
            text = replace_once(text, c0.DECLARE_SWITCH, c0.DECLARE_SWITCH + DECLARE_SWITCH)
            text = replace_once(text, PINS_OLD, PINS_NEW)
        out['harness/runs/' + NEW + suffix] = rename(text)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    for name, text in derive().items():
        path = ROOT / name
        if a.check:
            assert path.read_text() == text, name
        else:
            path.write_text(text)
            path.chmod(0o664)
    print('alpha-c1 queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
