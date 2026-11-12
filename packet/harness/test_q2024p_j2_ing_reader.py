#!/usr/bin/env python3
"""Ingress packets: the exploratory ing reader (tools/q2024p_j2_ing_reader.py). CPU only.

Synthetic records in fresh mkdtemp directories (left in place, never removed), built by the yr/yrs reader fixtures; the
status rows carry native_admission.ingress_prefix (engine) and native_admission.ingress_api (API process) as the engine
and router write them (ingress_incremental.PrefixCounter.snapshot / Ingress.snapshot). No engine, torch or box access.
"""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent / 'tools', HERE.parent / 'patches' / 's5_engine'):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import q2024p_j2_ing_reader as reader  # noqa: E402
import test_q2024p_j2_yr_reader as yrt  # noqa: E402

base = yrt.base
T0 = base.T0


def prefix(compared, through=0):
    return dict(schema='s5-ingress-prefix/1', unit=176, entries=8,
                counts=dict(first=8, compared=compared, through_reply=through, through_reply_but_last=through),
                delta_vs_prev_prompt_end={'-7..-1': compared - through, 'whole_reply': through},
                sums=dict(lcp=100000 * compared, prev_prompt=100004 * compared, prev_output=600 * compared,
                          new_prompt=103000 * compared))


def api(spliced, full, selected=True):
    return dict(schema='s5-ingress-incremental/1', process='api', disabled=None, entries=8, bytes=10 ** 7,
                counts=dict(spliced=spliced, full=full, verified=spliced // 16),
                fallbacks={'no-entry': full}, sums=dict(splice_us=9000 * spliced, full_us=200000 * full,
                                                         chars_total=400000 * spliced, chars_retokenised=12000 * spliced),
                splice_ms={'<=10': spliced}, full_ms={'<=500': full},
                binding=dict(selected=selected, flips={'on:release': 1}))


def build(on=True, sp_splices=0, ing='1', lines=(), turns=()):
    rec = base.real_record()
    arms = {label: base.arm(label, rec, rec) for label in reader.ARMS}
    for label, arm in arms.items():
        n3 = label.startswith('n3')
        for row, (c, s, f) in zip(arm['statuses'], ((0, 0, 0), (10, 12, 2))):
            adm = dict(queue_policy={})
            if n3:
                adm['ingress_prefix'] = prefix(c, through=0 if c == 0 else 1)
            if on:
                adm['ingress_api'] = api(s if n3 else (sp_splices if c else 0), f, selected=n3)
            row['native_admission'] = adm
    root = base.record(arms, list(lines))
    for phase in ('start', 'end'):
        path = root / ('engine.identity-%s.json' % phase)
        doc = json.loads(path.read_text())
        doc['identity']['env'].update({reader.yr.Y1_ENV: '0', reader.yr.Y2_ENV: '0', reader.ING_ENV: ing})
        path.write_text(json.dumps(doc))
    yrt.turns_file(root, 'n3.g2-live-0', list(turns))
    return root


class Blocks(unittest.TestCase):
    def test_on_record_reads_both_blocks(self):
        result = reader.collect(build(), 'on')
        arm = result['arms']['n3.g2-live-0']
        p, a = arm['ingress_prefix'], arm['ingress_api']
        self.assertEqual(p['counts'], dict(compared=10, through_reply=1, through_reply_but_last=1))
        self.assertEqual(p['delta_vs_prev_prompt_end'], {'-7..-1': 9, 'whole_reply': 1})
        self.assertEqual(p['means'], dict(lcp=100000.0, prev_prompt=100004.0, prev_output=600.0, new_prompt=103000.0))
        self.assertEqual(a['counts'], dict(spliced=12, full=2))
        self.assertEqual((a['splice_ms_mean'], a['full_ms_mean'], a['retokenised_share']), (9.0, 200.0, 0.03))
        self.assertEqual(a['fallbacks'], {'no-entry': 2})
        self.assertFalse([n for n in result['notes'] if 'ingress' in n or 'sp-arm' in n or 'INGRESS' in n], result['notes'])
        text = reader.render([result])
        self.assertIn('prefix: counts=', text)
        self.assertIn('api: counts=', text)

    def test_notes(self):
        self.assertIn('on-record-lacks-ingress_api:n3.g2-live-0', reader.collect(build(on=False), 'on')['notes'])
        self.assertIn('off-record-carries-ingress_api:n3.g2-live-0', reader.collect(build(ing='0'), 'off')['notes'])
        notes = reader.collect(build(sp_splices=3), 'on')['notes']
        self.assertIn('sp-arm-spliced:sp.g2-live-0:3', notes)
        self.assertTrue(any(n.startswith('identity-FLASH_S5_INGRESS_INCREMENTAL-on') for n in
                            reader.collect(build(ing='0'), 'on')['notes']))


class SentAdmit(unittest.TestCase):
    def test_per_kind_and_size(self):
        lines = [yrt.gap(2, 'request_admitted', request_key='k1', prompt_tokens=90000),
                 yrt.gap(10.25, 'request_admitted', request_key='k2', prompt_tokens=125000),
                 yrt.gap(20.5, 'request_admitted', request_key='k3', prompt_tokens=70000)]
        turns = [dict(t=2.7, kind='append', prompt_tokens=90000),       # sent T0+1.7, admitted T0+2: 0.3 s
                 dict(t=10.9, kind='append', prompt_tokens=125000),     # 0.35 s
                 dict(t=21.2, kind='fresh', prompt_tokens=70000),       # 0.3 s
                 dict(t=30.0, kind='append', prompt_tokens=1)]          # unplaced
        sa = reader.collect(build(lines=lines, turns=turns), 'on')['arms']['n3.g2-live-0']['sent_admit']
        self.assertEqual((sa['turns'], sa['unplaced']), (4, 1))
        self.assertEqual(sa['by_kind']['append']['n'], 2)
        self.assertAlmostEqual(sa['by_kind']['append']['mean'], 0.325, places=3)
        self.assertEqual(sorted(sa['by_size']), ['append 80-100k', 'append >=120k', 'fresh <80k'])

    def test_render_has_no_verdict_words(self):
        text = reader.render([reader.collect(build(), 'on')]).split('-- ingress', 1)[1]
        for word in ('PASS', 'FAIL', 'GO', 'verdict='):
            self.assertNotIn(word, text)


if __name__ == '__main__':
    unittest.main()
