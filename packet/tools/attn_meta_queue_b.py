#!/usr/bin/env python3
"""Derive the q18a-attn-meta box scripts on the c1b base: tools/attn_meta_queue.py's derivation (chain a ran it; never edited; superseded
by name) with the declared changes below. QNAME stays q18a-attn-meta (no attn-meta draft has run on the box), so the runs/ names stay
and their bytes move; build-a keeps the old bytes and git the history. Pattern: c1b-h's tools/alpha_next_c1b_queue_h.py.

  box    every call of e3_attn_meta_box.py -> e3_attn_meta_box_b.py (the c1b open-rows rule and ring read); the interface and pin
         lists add it and the modules it imports (e3_alpha_next_c1b_box.py, alpha_next_c1b_arms.py, e3_c1_open_rows.py);
  death  the lead's health-scoped rule (3 Nov 16:49; returning-append 14904f18, byte for byte): every name of native43_engine_death.py
         and alpha_next_stop_race.py in the draft -> their _h names (host_crash_detected's scan, engine_watched's watch, the stop-race
         call, its comment, the interface list); the declaration pins both _h files too (the old files stay pinned: old tests in the
         bundle use them);
  settings unchanged. No model setting, gate threshold, launcher or restore change. --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attn_meta_queue as base  # noqa: E402

ROOT = base.ROOT
DRAFT = 'harness/runs/' + base.NEW + '.draft.sh'
DECLARE = 'harness/runs/' + base.NEW + '_declare.sh'
SETTINGS = 'harness/runs/' + base.NEW + '_settings.sh'
DRAFT_SWAPS = (('native43_engine_death.py', 'native43_engine_death_h.py', 3),
               ('alpha_next_stop_race.py', 'alpha_next_stop_race_h.py', 3),
               ('"$H/e3_attn_meta_box.py"', '"$H/e3_attn_meta_box_b.py"', 4))
H_FILES = 'native43_engine_death_h.py alpha_next_stop_race_h.py'
BOX_FILES = 'e3_attn_meta_box_b.py e3_alpha_next_c1b_box.py alpha_next_c1b_arms.py e3_c1_open_rows.py'
INTERFACE_OLD = base.FILES + '; do'
INTERFACE_NEW = base.FILES + ' \\\n  ' + BOX_FILES + '; do'
PINS_OLD = base.PINS_NEW
PINS_NEW = PINS_OLD + BOX_FILES + ' ' + H_FILES + ' '


def __getattr__(name):                      # everything else is the base tool's
    return getattr(base, name)


def body():
    """The base body with the box call swapped (the death names are outside the body)."""
    return base.body().replace(DRAFT_SWAPS[2][0], DRAFT_SWAPS[2][1])


def derive():
    out = dict(base.derive())
    draft = out[DRAFT]
    for old, new, count in DRAFT_SWAPS:
        assert draft.count(old) == count, (old, draft.count(old))
        draft = draft.replace(old, new)
    assert draft.count(INTERFACE_OLD) == 1, 'the interface list moved'
    out[DRAFT] = draft.replace(INTERFACE_OLD, INTERFACE_NEW)
    assert out[DECLARE].count(PINS_OLD) == 1, 'the declaration pin list moved'
    out[DECLARE] = out[DECLARE].replace(PINS_OLD, PINS_NEW)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true')
    p.add_argument('--table', action='store_true')
    p.add_argument('--replace-derived', action='store_true',
                   help='write a derived text that differs from the one on disk (only for a text no queue has run on the box)')
    a = p.parse_args(argv)
    if a.table:
        print(base.table_text())
        return
    for name, text in derive().items():
        path = ROOT / name
        if a.check:
            assert path.read_text() == text, name
        else:
            assert a.replace_derived or not path.exists() or path.read_text() == text, 'refusing to change a derived text: ' + name
            path.write_text(text)
            path.chmod(0o664)
    print('attn-meta queue (b) ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
