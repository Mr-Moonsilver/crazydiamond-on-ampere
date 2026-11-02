#!/usr/bin/env python3
"""alpha-next-c1 one-card VM input: tools/alpha_c1_vm_input.py, unchanged in every step, over the alpha-next-c1 bundle's overlay.

Plain run: both C1 switches off (FLASH_S5_STAGE_LOCAL=0, FLASH_S5_STAGE_OVERLAP=0) and every alpha-next switch at its default (off);
stage run (--stage): both on, so the overlay's own selection functions report overlap selected and OverlapSegmentDriver importable
(C1's selection record, alpha_c1_selection in the probe's GPU section). The B5 scenario, the wrappers, the GPU flock (flock -w 5400 on
VM <vm-id>), the staging and the fetch are C1's. One declared change: build() admits run names alpha-next-c1-* (C1's text admits only
alpha-c1-*), by one exact-once swap of its source, exec'd into C1's module so its main() calls it. Not run by this pack: the VM step
is the lead's, under the flock, after the hub chain stops ('stopped-before-delivery: VM run pending').

    python3 tools/alpha_next_c1_vm_input.py --bundle BUNDLE --name alpha-next-c1-TAG-plain --work DIR --cpu-selfrun --vm-run --record DIR
    python3 tools/alpha_next_c1_vm_input.py --bundle BUNDLE --name alpha-next-c1-TAG-stage --work DIR --stage --cpu-selfrun --vm-run --record DIR
"""
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alpha_c1_vm_input as base  # noqa: E402

PREFIX = 'alpha-next-c1-'
SWAPS = (("    if not candidate.startswith('alpha-c1-') or candidate == out.name or out.exists():\n"
          "        raise ValueError('need a fresh --work/<alpha-c1-*>-input')\n",
          "    if not candidate.startswith('alpha-next-c1-') or candidate == out.name or out.exists():\n"
          "        raise ValueError('need a fresh --work/<alpha-next-c1-*>-input')\n"),)


def derive_build(module=base):
    text = inspect.getsource(module.build)
    for old, new in SWAPS:
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new)
    return text


BUILD_TEXT = derive_build()
exec(compile(BUILD_TEXT, 'alpha_next_c1_vm_input:build', 'exec'), base.__dict__)
# The switches the probe text installs (C1's STAGE_CALL, patched into the probe): plain both off, stage both on.
SWITCHES_TEXT = ("ALPHA_C1_SWITCHES = {False: {'FLASH_S5_STAGE_LOCAL': '0', 'FLASH_S5_STAGE_OVERLAP': '0'},\n"
                 "                     True: {'FLASH_S5_STAGE_LOCAL': '1', 'FLASH_S5_STAGE_OVERLAP': '1'}}\n")
assert base.STAGE_CALL.count(SWITCHES_TEXT) == 1, 'C1 probe switches changed'

if __name__ == '__main__':
    raise SystemExit(base.main())
