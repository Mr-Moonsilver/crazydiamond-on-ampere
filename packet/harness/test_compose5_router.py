"""compose-5: one real-router smoke for every action the box client sends with this queue's settings, and each kind of arm end to end over it (a1-trunk's fixture, test_a1_trunk_router).

Only fastapi's request and response objects and the serving leaf are leaves. The replay's own HTTP is test_compose5_box.StreamTurn; here the replay's turns are scripted, because the fixture serves prompts, not chat. No native chunk runs here, by construction.
"""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import test_a1_trunk_router as a1r    # first: its imports put the engine sources on the path
import compose5_arms as T
import compose4_replay as replay
import e3_compose4_box as box4
import e3_compose5_box as box

iwr, held = a1r.iwr, a1r.held
HERE = Path(__file__).resolve().parent
SETTINGS = HERE / 'runs' / 'q18a_compose_throughput_b2_settings.sh'
RUN = 'compose-5-router'
QUICK = {'pair': a1r.QUICK['pair'], 'none': ()}
SHORT = dict(T.REPLAY, AGENTS=2, DURATION=0.4, WARMUP=0.0, START_MIN=60000, START_MAX=60000, TOOL_MIN=0.01, TOOL_MAX=0.02)


def scripted(_url, body, _key, _origin, **_):
    return dict(ttft=0.01, seconds=0.02, text='ok', finish='length', arrivals_ms=[0, 1], completion_tokens=body['max_tokens'], prompt_tokens=60000, cached_tokens=0)


class Base(a1r.Base):
    async def asyncSetUp(self):
        with mock.patch.object(a1r, 'SETTINGS', SETTINGS):
            await super().asyncSetUp()
        self.driver.controller.chunk_tokens = T.CHUNK_TOKENS    # the box's: the fixture's own is under the bound's 1024 rows, and the stage refuses that
        # coverage-throughput: the yield's launch bounds as the engine parses them from this queue's settings (a1's fixture pins its own 2048)
        self.driver.controller.append_yield = a1r.sh.AppendYieldConfig.from_env(self.env_box)
        assert self.driver.controller.append_yield.max_new == T.LAUNCH['max_new'] == 8192


class Exposure(Base):
    async def test_the_settings_install_every_launch_switch(self):
        for name, value in T.LAUNCH_ENV.items():
            self.assertEqual(self.env_box.get(name), str(value), name)
        for name, value in (('FLASH_S5_INWAIT', '1'), ('FLASH_S5_PLE_PRIVATE', '1'), ('FLASH_S5_PLE_WITNESS', '1')):    # connections: the fixture's env holds FLASH names only; test_compose5_queue reads EXTRA_ENV
            self.assertEqual(self.env_box.get(name), value, name)
        self.assertEqual(self.env_box.get('FLASH_S5_CHUNK_TOKENS'), str(T.CHUNK_TOKENS))

    async def test_every_box_action_gets_a_200(self):
        def smoke():
            client = self.client()
            client.prepare('full', 'full-prefill-solo')
            worker = iwr.am.Worker(client.control, a1r.RUN, client.epoch)
            out = {}
            for action, fields in box.ACTIONS:
                try:
                    if action == 'qualification_worker':
                        out[action] = worker.call(iwr.am.SELECT, {'on': True, 'slots_fast': True, 'prepare_slices': True, 'score_warm': True, 'cow_fast': True, 'reserved_cap': True, 'cow_host_filter': True, 'ple_private': True, 'ple_witness': False})
                    else:
                        out[action] = client.call(action, **{name: a1r.SAMPLE[name] for name in fields})
                except Exception as exc:  # noqa: BLE001 - the test reads it
                    out[action] = repr(exc)
            return out
        out = await self.off_loop(smoke)
        self.assertEqual({k: v for k, v in out.items() if type(v) is str}, {})
        self.assertEqual(set(out), {a for a, _f in box.ACTIONS})


class ArmEnd(a1r.ArmEnd):
    test_a_live_on_arm_runs_its_control_and_its_returners = None
    test_an_off_arm_sends_enabled_false_and_several_return_at_once = None
    test_a_solo_arm_has_no_returner_and_cancels_the_standing_streams = None

    async def asyncSetUp(self):
        with mock.patch.object(a1r, 'SETTINGS', SETTINGS):
            await super().asyncSetUp()
        self.driver.controller.chunk_tokens = T.CHUNK_TOKENS
        self.driver.controller.append_yield = a1r.sh.AppendYieldConfig.from_env(self.env_box)     # coverage-throughput: as Base

    async def arm(self, sel, cohort='live', index=0):
        real = replay.run

        def short(url, model, key, run_id, label, sink=None, cfg=None, stream=None):
            return real(url, model, key, run_id, label, cfg=dict(SHORT, SEED=cfg['SEED']), stream=stream.around(scripted), finish_bound_s=5.0, sink=sink)

        def run():
            client = self.client()
            client.prepare('full', 'full-prefill-solo')
            client.resume()
            name = T.label(sel, 2, cohort, index)
            out = Path(self.tmp.name) / ('arm-%s.json' % name)
            argv = ['arm', '--url', self.server.url, '--model', 'fixture-model', '--run-id', a1r.RUN, '--epoch', str(self.driver.engine_epoch), '--c1', sel, '--size', '2', '--cohort', cohort, '--index', str(index),
                    '--workload', self.workload, '--control-timeout-s', '30', '--out', str(out)]
            mark, served = len(self.server.session_actions), len(self.server.completions)
            with mock.patch.object(box, 'memory_read', lambda: dict(schema=box.MEMORY_SCHEMA, gpus=None, rss_kib=None, errors=['stub'])), \
                    mock.patch.object(box4.shared, '_package_gate', return_value={'ok': True}), mock.patch.object(box4.base, 'SETTLE_S', 0.2), mock.patch.object(box4.base, 'CLOSE_TIMEOUT_S', 1.0), \
                    mock.patch.object(box4.replay, 'run', short), mock.patch.object(box4, 'POLL_S', 0.1), mock.patch.object(T, 'ONE_STREAM_WINDOW_S', 0.3), mock.patch.dict(T.PLANS, QUICK), contextlib.redirect_stdout(io.StringIO()):
                rc_ = box.main(argv)
            return rc_, json.loads(out.read_text()), self.server.session_actions[mark:], self.server.completions[served:], out
        return await self.off_loop(run)

    def assert_control(self, actions):
        self.assertEqual(actions[:3], ['status', 'pause', 'native_append_yield'], actions[:12])
        self.assertLess(actions.index('native_append_yield'), actions.index('native_stage_local'))
        self.assertLess(actions.index('native_stage_local'), actions.index('release_auto_arm'))

    async def test_the_live_smoke_runs_its_control_and_names_what_a_fixture_cannot_engage(self):
        rc_, record, actions, served, _out = await self.arm('sm')
        self.assertNotIn('error', record, record.get('error'))
        self.assertEqual((rc_, record['reasons']), (0, []))
        self.assert_control(actions)
        self.assertIs(record['yield_control']['enabled'], True)
        self.assertEqual(len(served), T.by_label()['sm.g2-live-0']['generations'])
        self.assertIs((record['ple_private'] or {}).get('requested'), True)
        self.assertIs(record['inwait']['exact_acknowledged'], True)
        ok, reasons = box.smoke_check(record)
        self.assertFalse(ok)
        self.assertNotIn('smoke-reserved-cap-not-requested', reasons)
        self.assertNotIn('smoke-exact-successor-not-acknowledged', reasons)

    async def test_the_one_stream_arm_opens_one_and_reads_one_window(self):
        rc_, record, actions, served, _out = await self.arm('dec1')
        self.assertNotIn('error', record, record.get('error'))
        self.assertEqual((rc_, record['reasons'], record['kind']), (0, [], 'one-stream'))
        self.assert_control(actions)
        self.assertEqual(len(served), 1)
        self.assertEqual(len(record['decode_only']['stream_rates_tokens_per_s']), 1)

    async def test_the_replay_arm_keeps_its_turns_beside_the_record_with_its_seed(self):
        for sel, index, seed in (('ny', 0, 1), ('nt', 1, 2)):
            with self.subTest(sel):
                rc_, record, actions, _served, out = await self.arm(sel, index=index)
                name = T.label(sel, 2, 'live', index)
                self.assertNotIn('error', record, record.get('error'))
                self.assertEqual((rc_, record['reasons'], record['kind']), (0, [], 'replay'))
                self.assert_control(actions)
                self.assertIs(record['yield_control']['enabled'], T.YIELD[sel])     # coverage-throughput: on for nt
                self.assertIs(self.driver.controller.append_yield.enabled, T.YIELD[sel])
                self.assertIs((record['ple_private'] or {}).get('requested'), False)
                self.assertIs((record['reserved_cap'] or {}).get('requested'), False)
                # coverage-soak-fix build-2: the CPU CoW byte check stays off on both rows (the worker RSS owner)
                self.assertIs((record['cow_verify'] or {}).get('requested'), False)
                self.assertEqual(record['inwait']['bound_acknowledged'], T.by_label()[name]['chunk_bound'])
                self.assertEqual((record['inwait']['hold_requested'], record['inwait']['hold_acknowledged']), (T.NATIVE_HOLD[sel],) * 2)
                self.assertEqual(self.driver.controller.native_hold_s, T.NATIVE_HOLD[sel])
                self.assertEqual((record['inwait']['residue_requested'], record['inwait']['residue_acknowledged']), (T.NATIVE_RESIDUE[sel],) * 2)
                self.assertEqual(self.driver.controller.native_residue, T.NATIVE_RESIDUE[sel])
                # the real driver's status, through the real box: the decline record and its coverage counter reach the arm's record
                for status in [s for s in record['statuses'] if s['label'] in ('entry', 'after')]:
                    self.assertEqual(status['append_yield']['declines']['coverage']['schema'], 's5-prefill-coverage/1', status['label'])
                self.assertEqual(record['statuses'][-1]['append_yield']['declines']['residue']['on'], T.NATIVE_RESIDUE[sel])
                header, *turns = [json.loads(line) for line in Path(str(out) + '.turns.jsonl').read_text().splitlines()]
                self.assertEqual((header['header'], header['label'], header['replay_start_s']), (True, name, record['replay_start_s']))
                self.assertEqual((header['replay_config']['SEED'], record['replay_config']['SEED']), (seed, seed))
                self.assertEqual(record['replay_requests_sent'], len(turns))
                import compose5_rule as rule
                self.assertIsNone(rule.journal_problem(record['label'], record, *rule.load_turns(str(out) + '.turns.jsonl')), 'the arm\'s real record and journal agree under the reader')
                self.assertGreater(len(turns), 0)
                self.assertEqual((record['replay_still_open'], record['replay_open_requests']), ([], []))
                self.assertEqual((record['memory']['start']['schema'], record['memory']['end']['errors']), (box.MEMORY_SCHEMA, ['stub']))
                self.assertEqual(record['replay_config']['AGENTS'], 2)
                self.assertIsNotNone(record['polled'])
                self.assertTrue(set(record['control_actions']) <= box.ACTION_NAMES)


if __name__ == '__main__':
    unittest.main()
