#!/usr/bin/env python3
"""Night-pack family reader (tools/q2024p_j2_np_reader.py). CPU only.

Synthetic records in fresh mkdtemp directories (left in place, never removed), built by the ing/yr/yrs reader fixtures,
with the night-pack status-block fields as the engine writes them (scheduler_hooks yield_window not_served / next_left /
gap_by_path; ingress_incremental verify / timing / reply_aligned / divergence_windows) and the new log fields
(request_admitted arrival_s / admit_wall_s; the proof-miss append_dec / append_out / running; the Y2 append receipt's
path). No engine, torch or box access.
"""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent / 'tools', HERE.parent / 'patches' / 's5_engine'):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

import q2024p_j2_np_reader as reader  # noqa: E402
import test_q2024p_j2_ing_reader as ingt  # noqa: E402
import test_q2024p_j2_yr_reader as yrt  # noqa: E402

base = yrt.base
T0 = base.T0


def family_env(root, variant, **override):
    flags = dict(reader.FAMILY[variant])
    env = {e: flags[k] for k, e in reader.ENVS}
    if flags['ingress'] == '1':
        env.update(reader.VERIFY)
    env.update(override)
    for phase in ('start', 'end'):
        path = root / ('engine.identity-%s.json' % phase)
        doc = json.loads(path.read_text())
        doc['identity']['env'].update(env)
        path.write_text(json.dumps(doc))


def miss_line(s, dec, out, running):
    return ('%s flash-s5 yield-reentry proof-miss stage=barrier field=prefill-requests rank=0 got=1 want=0 retry=1 '
            'bound=4 ends=0 elapsed_ms=170.2 settle=0 append_dec=%s append_out=%s running=%s\n' % (base.at(s), dec, out, running))


def receipt(s, **row):
    return '%s %s%s\n' % (base.at(s), reader.yr.WINDOW_RECEIPT, json.dumps(row))


def build(variant='all', lines=(), turns=(), window=True, **env):
    root = ingt.build(lines=lines, turns=turns)
    family_env(root, variant, **env)
    for label in reader.ARMS:
        path = root / ('arm-' + label + '.json')
        arm = json.loads(path.read_text())
        for row, n in zip(arm['statuses'], (0, 1)):
            adm = row.setdefault('native_admission', {})
            if window and label.startswith('n3'):
                block = yrt.window_block(2 * n)
                block.update(not_served={'unsampled': 3 * n, 'abort': n}, next_left=n,
                             gap_by_path={'full': dict(n=2 * n, total=800.0 * n, max=500.0),
                                          'o3': dict(n=n, total=150.0 * n, max=150.0)})
                adm.setdefault('queue_policy', {})['yield_window'] = block
            if 'ingress_api' in adm:
                adm['ingress_api'].update(verify_every=1, verify_async=True,
                                          verify=dict(pending=0, verified=10 * n, errors=0, verify_us=2000000 * n,
                                                      lag_us=5000 * n, ms={'<=500': 10 * n}, mismatches=[]),
                                          timing=dict(async_n3_n=12 * n, async_n3_us=12 * 60000 * n,
                                                      run_n3_n=12 * n, run_n3_us=12 * 20000 * n))
            if 'ingress_prefix' in adm:
                adm['ingress_prefix'].update(reply_aligned={'whole': 8 * n, 'into': 2 * n}, reply_offset={'0': 10 * n},
                                             max_entries=32, divergence_windows=[dict(k=5, d=0, exact=True, new=[1, 2],
                                                                                      old=[1, 3], start=3)] if n else [])
                adm['ingress_prefix']['counts']['after_eviction'] = n
        path.write_text(json.dumps(arm))
    return root


class Family(unittest.TestCase):
    def test_table_is_the_brief(self):
        f = reader.FAMILY
        self.assertEqual(sorted(f), ['all', 'hgb', 'ing', 'off', 'set', 'y12'])
        self.assertEqual(set(f['off'].values()), {'0'})
        self.assertEqual(f['y12'], dict(f['off'], y1='1', y2='1'))
        self.assertEqual(f['ing'], dict(f['off'], ingress='1'))
        self.assertEqual(f['all'], dict(f['off'], y1='1', y2='1', ingress='1'))
        self.assertEqual(f['set'], dict(f['off'], settle='1'))
        self.assertEqual(f['hgb'], dict(f['off'], batch='1'))
        self.assertEqual((reader.variant_of('off-n2'), reader.variant_of('y12'), reader.variant_of('off-v1b3')),
                         ('off', 'y12', 'off'))
        with self.assertRaises(ValueError):
            reader.variant_of('y1')

    def test_identity_notes(self):
        self.assertEqual(reader.collect(build(), 'all')['family_notes'], [])
        notes = reader.collect(build(**{reader.VERIFY_EVERY_ENV: '16'}), 'all')['family_notes']
        self.assertEqual(notes, ['identity-all-FLASH_S5_INGRESS_VERIFY_EVERY-start-16',
                                 'identity-all-FLASH_S5_INGRESS_VERIFY_EVERY-end-16'])
        notes = reader.collect(build('off', window=False, **{reader.ing.ING_ENV: '0'}), 'off-n2')['family_notes']
        self.assertEqual(notes, [])


class Arms(unittest.TestCase):
    def test_agent_seconds_and_split(self):
        lines = [yrt.gap(2, 'request_admitted', request_key='k1', prompt_tokens=90000, arrival_s=T0 + 1.8,
                         admit_wall_s=T0 + 2.0),
                 yrt.gap(10.25, 'request_admitted', request_key='k2', prompt_tokens=125000)]
        turns = [dict(t=2.7, kind='append', prompt_tokens=90000, ttft=1.5, seconds=10.0, completion_tokens=400),
                 dict(t=10.9, kind='append', prompt_tokens=125000, ttft=2.5, seconds=12.0, completion_tokens=500),
                 dict(t=21.2, kind='fresh', prompt_tokens=70000)]
        arm = reader.collect(build(lines=lines, turns=turns), 'all')['arms']['n3.g2-live-0']
        self.assertEqual(arm['agent_seconds']['all'], dict(n=2, ttft_s=4.0, decode_s=18.0, output_tokens=900))
        self.assertEqual(arm['agent_seconds']['append']['n'], 2)
        split = arm['admit_split']
        self.assertEqual(split['stamped'], 1, 'the unstamped (V1-shaped) admission is joined but not split')
        self.assertAlmostEqual(split['by_kind']['append']['sent_to_arrival']['mean'], 0.1, places=3)
        self.assertAlmostEqual(split['by_kind']['append']['arrival_to_admit']['mean'], 0.2, places=3)

    def test_lever_extras(self):
        lines = [miss_line(5, 0, 1, 3), miss_line(6, 1, 1, 3), receipt(7, event='append', index=2, gap_ms=300.0, path='full'),
                 receipt(8, event='append', index=3, gap_ms=100.0, path='o3'), receipt(9, event='append', index=1, gap_ms=None)]
        arm = reader.collect(build(lines=lines), 'all')['arms']['n3.g2-live-0']
        self.assertEqual(arm['proof_miss_fields'], dict(n=2, append_dec={'0': 1, '1': 1}, append_out={'1': 2}, running={'3': 2}))
        we = arm['window_extra']
        self.assertEqual((we['not_served'], we['next_left']), ({'unsampled': 3, 'abort': 1}, 1))
        self.assertEqual(we['gap_by_path'], {'full': dict(n=2, mean=400.0, max_lifetime=500.0),
                                             'o3': dict(n=1, mean=150.0, max_lifetime=150.0)})
        self.assertEqual(sorted(we['receipts_by_path']), ['full', 'o3'])
        ix = arm['ingress_extra']
        self.assertEqual(ix['api']['verify']['verified'], 10)
        self.assertEqual((ix['api']['run_ms_mean'], ix['api']['queue_pickup_ms_mean']), ({'n3': 20.0}, {'n3': 40.0}))
        self.assertEqual(ix['prefix']['reply_aligned'], {'whole': 8, 'into': 2})
        self.assertEqual((ix['prefix']['after_eviction'], len(ix['prefix']['divergence_windows'])), (1, 1))

    def test_render_order_and_no_verdict_words(self):
        text = reader.render([reader.collect(build(), 'all')])
        first = next(line for line in text.splitlines() if line.startswith('  n3.g2-live-0'))
        self.assertLess(first.index('turns/h='), first.index('x_sp='))
        self.assertLess(first.index('x_sp='), first.index('native='))
        self.assertLess(first.index('native='), first.index('agent_s'))
        body = text.split('\n', 1)[1]
        for word in ('PASS', 'FAIL', ' GO', 'verdict='):
            self.assertNotIn(word, body)
        self.assertIn('window per_window=', text)
        self.assertIn('api counts=', text)
        self.assertIn('prefix counts=', text)


if __name__ == '__main__':
    unittest.main()
