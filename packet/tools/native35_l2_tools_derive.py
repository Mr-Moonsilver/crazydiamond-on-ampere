#!/usr/bin/env python3
"""Derive the native35 L2 self-run tools from the frozen native35 ones.

native35_l2_cpu_probe.py = native35_cpu_probe.py with: arm L2 only, L's lifetime
window, EXPECTED_TESTS 125 -> 136 (the 11 test_native35_l2 cases, which the
probe's existing test_native35_*.py discovery already collects), the L2 slot
receipt appended to the 16 R/O/L slot receipts, and the L2 CLI/timing receipts
asserted and recorded (native35_l2='PASS'). Everything else is byte-identical.
native35_l2_selfrun.py = native35_selfrun.py with the L2 probe path, the L2
schema name and arm choice L2; image, mounts and PROBE_ENV are unchanged.
Every replacement must match exactly once.
"""
from pathlib import Path
TOOLS=Path(__file__).resolve().parent

PROBE_SWAPS=(
    ('"""Native35 CPU-only plugin discovery/source, wrapper, grade and builder-view checks; never used by box workers.\n\n'
     'FLASH_NATIVE35_ARM (R, O or L) selects the frozen owner runner, its window\n'
     'boundaries and the provenance binding; the native34 diagnostic is installed in\n'
     'its arm-O/readback configuration for every arm so the same plugin checks run.\n"""',
     '"""Native35 L2 CPU-only plugin discovery/source, wrapper, grade, builder-view and row4 window-order checks; never used by box workers.\n\n'
     "The native35 probe with FLASH_NATIVE35_ARM=L2 (the frozen L2 owner runner, L's\n"
     'lifetime window and the provenance binding) and the L2 receipts added; the\n'
     'native34 diagnostic is installed in its arm-O/readback configuration exactly as\n'
     'for R/O/L so the same plugin checks run.\n"""'),
    ('EXPECTED_TESTS=125  # 98 native30-34 cases plus 27 native35 cases; pinned into the frozen runner gate',
     'EXPECTED_TESTS=136  # 125 native30-35 R/O/L cases plus 11 native35 L2 cases; pinned into the frozen runner gate'),
    ("role=os.environ['FLASH_NATIVE35_ARM'];assert role in ('R','O','L') and os.environ['FLASH_NATIVE34_ARM']=='O'",
     "role=os.environ['FLASH_NATIVE35_ARM'];assert role=='L2' and os.environ['FLASH_NATIVE34_ARM']=='O'"),
    ("active_and_cleanup=native35_gate.LIFETIME_ACTIVE_AND_CLEANUP if role=='L' else weekend_gate.ACTIVE_AND_CLEANUP",
     "active_and_cleanup=native35_gate.LIFETIME_ACTIVE_AND_CLEANUP  # L2 keeps L's lifetime window"),
    ("assert {r['draft'] for r in slot_receipts}>={'q18a_native35_r.draft.sh','q18a_native35_o.draft.sh','q18a_native35_l.draft.sh'}\n",
     "assert {r['draft'] for r in slot_receipts}>={'q18a_native35_r.draft.sh','q18a_native35_o.draft.sh','q18a_native35_l.draft.sh'}\n"
     "l2_slots=sys.modules['test_native35_l2'].SLOT_RECEIPTS\n"
     "assert len(l2_slots)==1 and l2_slots[0]['draft']=='q18a_native35_l2.draft.sh' and l2_slots[0]['real_wall_clock'] and l2_slots[0]['active_s']==7200\n"
     "slot_receipts=slot_receipts+l2_slots\n"),
    ('record=dict(native35_arm=role,',
     '# Native35 L2: exact row4 argv through the L2 CLI, and the real wait functions on the box ordering.\n'
     "l2_cli=sys.modules['test_native35_l2'].CLI_RECEIPTS\n"
     "assert len(l2_cli)==1 and l2_cli[0]['runner']=='ResumeWindowPlacementRunner' and l2_cli[0]['first_network_call']=='prepare-full' and l2_cli[0]['substitution_restored'] is True\n"
     "l2_timing=sys.modules['test_native35_l2'].TIMING_RECEIPTS\n"
     "assert sorted(r['case'] for r in l2_timing)==['box-shaped','early-finish','fast-prefill','prefill-never-opens']\n"
     "assert [r for r in l2_timing if r['case']=='box-shaped'][0]['parent']=='placement-window-incomplete'\n"
     "Path('/out/native35-l2-check.json').write_text(json.dumps(dict(cli=l2_cli,timing=l2_timing),sort_keys=True,indent=2)+'\\n')\n"
     "record=dict(native35_l2='PASS',native35_l2_cli_receipts=l2_cli,native35_l2_timing_receipts=l2_timing,native35_arm=role,"),
)
SELFRUN_SWAPS=(
    ('"""Run exact frozen native35 bytes in the pinned image, CPU only."""',
     '"""Run exact frozen native35 L2 bytes in the pinned image, CPU only (native35_selfrun with the L2 probe)."""'),
    ("'/source/tools/native35_cpu_probe.py'", "'/source/tools/native35_l2_cpu_probe.py'"),
    ("schema='native35-selfrun/1'", "schema='native35-l2-selfrun/1'"),
    ("p.add_argument('--arm',choices=('R','O','L'),required=True)", "p.add_argument('--arm',choices=('L2',),required=True)"),
)


def apply(text,swaps):
    for old,new in swaps:
        assert text.count(old)==1,(old[:80],text.count(old));text=text.replace(old,new)
    return text


def generate(tools=TOOLS):
    return {tools/'native35_l2_cpu_probe.py':apply((tools/'native35_cpu_probe.py').read_text(),PROBE_SWAPS),
            tools/'native35_l2_selfrun.py':apply((tools/'native35_selfrun.py').read_text(),SELFRUN_SWAPS)}


if __name__=='__main__':
    import sys
    check='--check' in sys.argv
    for path,text in generate().items():
        if check:assert path.read_text()==text,path.name
        else:
            assert not path.exists(),path.name+' exists; regenerate only under --check'
            path.write_text(text)
    print('native35 l2 tools '+('verified' if check else 'written'))
