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
SETTINGS = HERE / 'runs' / 'q18a_parity_o3_settings.sh'
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

    async def test_the_o3_pair_arm_sends_its_natives_and_selects_o3(self):
        for sel in T.O3_PAIR:
            with self.subTest(sel):
                rc_, record, actions, served, _out = await self.arm(sel)
                self.assertNotIn('error', record, record.get('error'))
                self.assertEqual(record['kind'], 'o3-pair')
                # this fixture's completion leaf streams no logprobs: the arm names each native's evidence invalid, never a pass
                self.assertEqual((rc_, record['reasons']), (2, ['o3-pair-native-%d-incomplete' % i for i in range(T.O3_PAIR_NATIVES)]))
                self.assertEqual({n['malformed'] for n in record['natives']}, {'teacher_logprob_evidence_invalid'})
                natives = [c for c in served if c.get('max_tokens') == T.O3_PAIR_TOKENS]    # the fixture keeps max_tokens and the salt of each body
                self.assertEqual(len(natives), T.O3_PAIR_NATIVES)
                self.assertEqual(len({c['cache_salt'] for c in natives}), T.O3_PAIR_NATIVES)
                self.assert_control(actions)
                self.assertEqual(len(served), T.by_label()['%s.g2-live-0' % sel]['generations'], 'two standing decoders and three natives, as the ingress check counts')
                self.assertEqual((record['inwait']['o3_requested'], record['inwait']['o3_acknowledged']), (T.O3_SPLIT[sel],) * 2)
                self.assertIs(self.driver.controller.o3_selected, T.O3_SPLIT[sel])
                self.assertEqual(len(record['natives']), T.O3_PAIR_NATIVES)
                self.assertEqual(len({n['salt_sha256'] for n in record['natives']}), T.O3_PAIR_NATIVES, 'one salt per native: no cache hit')
                self.assertEqual({n['prompt_sha256'] for n in record['natives']}, {record['prompt_sha256']}, 'one prompt: the numerics compare like with like')
                self.assertEqual({(n['request']['temperature'], n['request']['logprobs'], n['request']['max_tokens']) for n in record['natives']}, {(0.0, T.O3_LOGPROBS, T.O3_PAIR_TOKENS)})
                self.assertTrue(box.o3_arm_read(record)['read'])

    async def test_the_one_stream_arm_opens_one_and_reads_one_window(self):
        rc_, record, actions, served, _out = await self.arm('dec1')
        self.assertNotIn('error', record, record.get('error'))
        self.assertEqual((rc_, record['reasons'], record['kind']), (0, [], 'one-stream'))
        self.assert_control(actions)
        self.assertEqual(len(served), 1)
        self.assertEqual(len(record['decode_only']['stream_rates_tokens_per_s']), 1)

    async def test_the_replay_arm_keeps_its_turns_beside_the_record_with_its_seed(self):
        for sel, index, seed in (('sp', 0, 1), ('nt', 1, 2), ('no', 1, 2)):
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
                # parity-2 A: the terminal successor through the real router, session and control to the stage driver
                self.assertEqual((record['inwait']['terminal_requested'], record['inwait']['terminal_acknowledged']), (T.TERMINAL_SUCCESSOR[sel],) * 2)
                self.assertIs(self.driver.terminal_successor, T.TERMINAL_SUCCESSOR[sel])
                # lane-o3: o3_split through the real router, session and control to the stage driver and the controller the policy reads
                self.assertEqual((record['inwait']['o3_requested'], record['inwait']['o3_acknowledged']), (T.O3_SPLIT[sel],) * 2)
                self.assertIs(self.driver.controller.o3_selected, T.O3_SPLIT[sel])
                o3 = box.o3_arm_read(record)
                self.assertTrue(o3['read'], 'the policy\'s O3 record reaches the arm\'s before and after status rows')
                self.assertEqual(o3['kept'], 0)
                self.assertEqual(type(o3['route_moves']), int)
                self.assertEqual({k: type(record['stage_delta'].get('step_iw_' + k)) for k in ('terminal_admits', 'terminal_mismatch')}, dict(terminal_admits=int, terminal_mismatch=int), 'the counters reach the arm record')
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

    async def test_sp_keeps_the_lease_for_its_arm_and_the_next_arm_takes_the_lane_back(self):
        """parity-base: sp sends ny's controls (the release too), then pause and resume, and reads the lease held before any turn, at every
        poll and at the end, with attn-meta, the lazy receive and the decode merge off; nt run next on the same engine enters as after any arm
        and its release hands the policy back, with every decode feature on again (the real driver, AutoArmPolicy and workers)."""
        policy = self.driver.auto_arm
        self.assertIsNotNone(policy)
        polls = mock.patch.object(box4.Poller.__init__, '__defaults__', (0.05,))     # Poller's period is bound at def time; POLL_S's patch misses it
        with polls:
            rc_, record, actions, _served, _out = await self.arm('sp', index=0)
        self.assertNotIn('error', record, record.get('error'))
        self.assertEqual((rc_, record['reasons'], record['kind']), (0, [], 'replay'))
        self.assert_control(actions)
        at = actions.index('release_auto_arm')
        self.assertEqual(actions[at:at + 5], ['release_auto_arm', 'resume', 'pause', 'resume', 'status'], actions)
        self.assertEqual(actions.count('release_auto_arm'), 1)
        lease = record['native_off']
        self.assertEqual({k: lease[k] for k in ('requested', 'paused', 'resumed', 'manual_owned', 'active', 'route', 'scheduler_requests')},
                         dict(requested=True, paused=True, resumed=True, manual_owned=True, active=False, route='full', scheduler_requests=0))
        status = {s['label']: s for s in record['statuses']}
        self.assertEqual([status[k]['auto_arm']['manual_owned'] for k in ('entry', 'lease', 'before', 'after')], [True] * 4)
        self.assertGreater(len(record['polled']['rows']), 0)
        self.assertEqual({r['manual_owned'] for r in record['polled']['rows']}, {True})
        self.assertEqual({str(r): v['after']['selected'] for r, v in record['attn_meta']['select'].items()}, {str(r): False for r in range(4)})
        self.assertEqual((record['step']['acknowledged_lazy'], record['step']['acknowledged_merge']), (False, 'off'))
        self.assertEqual((record['yield_control']['enabled'], record['inwait']['hold_acknowledged'], record['inwait']['residue_acknowledged']), (False, 0, 0))
        self.assertIs(policy.manual_owned, True)
        self.assertIs(self.driver.lazy_recv, False)
        # the next arm, as the table runs it after sp.g2-live-0
        with polls:
            rc_, record, actions, _served, _out = await self.arm('nt', index=0)
        self.assertNotIn('error', record, record.get('error'))
        self.assertEqual((rc_, record['reasons']), (0, []))
        self.assertEqual(record['native_off'], dict(requested=False))
        at = actions.index('release_auto_arm')
        self.assertEqual(actions[at:at + 3], ['release_auto_arm', 'resume', 'status'], actions)     # no lease kept: the end read's pause comes after the record
        status = {s['label']: s for s in record['statuses']}
        self.assertEqual([status[k]['auto_arm']['manual_owned'] for k in ('entry', 'before', 'after')], [True, False, False])
        self.assertEqual({r['manual_owned'] for r in record['polled']['rows']}, {False})
        self.assertEqual({str(r): v['after']['selected'] for r, v in record['attn_meta']['select'].items()}, {str(r): True for r in range(4)})
        self.assertEqual((record['step']['acknowledged_lazy'], record['step']['acknowledged_merge']), (True, 'chunk'))
        self.assertEqual((record['yield_control']['enabled'], record['inwait']['hold_acknowledged'], record['inwait']['residue_acknowledged']), (True, T.HOLD_S, 1))
        self.assertIs(self.driver.lazy_recv, True)


if __name__ == '__main__':
    unittest.main()
