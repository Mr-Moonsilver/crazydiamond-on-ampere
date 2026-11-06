"""compose-5: every place a turn goes stock instead of native, counted per arm. CPU only.

c4b's FAIL rested on 4 fresh turns nobody could explain: nothing counted a decline. These run the real AutoArmPolicy, over a
stand-in scheduler for each site and over the connected yield rig for the status path, and the real reader over its records.
"""
import enum
import json
import re
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'patches/s5_engine')]

import compose4_rule as rule  # noqa: E402
import native_auto_arm as auto  # noqa: E402


class Pause(enum.Enum):
    UNPAUSED = 0
    PAUSED_ALL = 1


class Ref(NS):
    """The policy holds its head by weakref, as it does a real Request; a bare SimpleNamespace can't be referenced."""


class Queue(list):
    def peek_request(self):
        return self[0]


def req(uid='r1', tokens=70000, **fields):
    base = dict(request_id=uid, status=NS(name='WAITING'), num_computed_tokens=0, num_output_tokens=0, num_preemptions=0, resumable=False, lora_request=None,
                pooling_params=None, structured_output_request=None, ec_transfer_params=None, sampling_params=NS(prompt_logprobs=None, logprobs=None), num_tokens=tokens,
                num_prompt_tokens=tokens, is_prefill_chunk=False)      # build-2 (b): the pinned Request fields K2's tail reads
    base.update(fields)
    return Ref(**base)


def decoder(uid='d1', **fields):
    base = dict(request_id=uid, is_prefill_chunk=False, num_computed_tokens=10, num_prompt_tokens=10, lora_request=None, resumable=False)
    base.update(fields)
    return NS(**base)


class Stage:
    """The policy's real code over the fields it reads; nothing here decides anything."""

    def __init__(self, *heads):
        self.queue = Queue(heads)
        self.shared, self.start, self.free = 0, 0, 10 ** 7
        manager = NS(get_computed_blocks=lambda r: (NS(blocks=()), self.start, self.shared), watermark_blocks=0,
                     coordinator=NS(get_num_blocks_to_allocate=lambda **kw: kw['num_tokens'] - kw['total_computed_tokens']),
                     block_pool=NS(get_num_free_blocks=lambda: self.free))
        self.scheduler = NS(running=[decoder()], max_num_running_reqs=64, requests={r.request_id: r for r in heads}, kv_cache_manager=manager, max_num_scheduled_tokens=1024,
                            policy=NS(name='FCFS'), vllm_config=NS(parallel_config=NS(data_parallel_size=1), speculative_config=None), connector=None,
                            pause_state=Pause.UNPAUSED, set_pause_state=lambda state: None)
        self.refuses = False
        self.controller = NS(_queue=lambda s: self.queue, chunk_tokens=1024, minimum_stage_rows=1, _request_refuses=lambda s, r: self.refuses, _aligned_split=lambda s, r, limit, **kw: limit)
        self.driver = NS(controller=self.controller, _split_ready=False, active=False, admission_intent=lambda: None, _control_route=None, _bootstrapped=True, error=None, _engine_paused=lambda: False,
                         clock=lambda: 0.0, _control_timeout_s=1.0, _record=lambda *a, **k: None, engine_epoch=1, submit_control=lambda command: NS(done=lambda: False), control_tick=lambda: None)
        self.engine = NS(scheduler=self.scheduler, batch_queue=[], is_running=lambda: True)
        self.policy = auto.AutoArmPolicy(self.engine, self.driver)


def stock_takes(stage, request):
    request.status = NS(name='RUNNING')
    if request in stage.queue:
        stage.queue.remove(request)
    return stage.policy.tick()


class EverySite(unittest.TestCase):
    """Each site, by the state that reaches it: the tick count moves, stock's admission gives the turn its one count."""

    def cases(self):
        def cooldown(s): s.policy.cooldown = True
        def attr(owner, name, value):
            def go(s): setattr(s.__dict__[owner] if owner in s.__dict__ else owner, name, value)
            return go
        yield 'cooldown', cooldown
        yield 'gate-manual-owned', lambda s: setattr(s.policy, 'manual_owned', True)
        yield 'gate-route-or-work', lambda s: setattr(s.driver, '_split_ready', True)
        yield 'gate-route-or-work', lambda s: setattr(s.driver, 'active', True)
        yield 'gate-route-or-work', lambda s: setattr(s.driver, 'admission_intent', lambda: {})
        yield 'gate-route-or-work', lambda s: setattr(s.driver, '_control_route', 'x')
        yield 'gate-not-bootstrapped', lambda s: setattr(s.driver, '_bootstrapped', False)
        yield 'gate-driver-stopped', lambda s: setattr(s.driver, 'error', 'x')
        yield 'gate-engine-paused', lambda s: setattr(s.driver, '_engine_paused', lambda: True)
        yield 'gate-policy', lambda s: setattr(s.scheduler, 'policy', NS(name='PRIORITY'))
        yield 'gate-no-batch-queue', lambda s: setattr(s.engine, 'batch_queue', None)
        yield 'gate-config', lambda s: setattr(s.scheduler.vllm_config.parallel_config, 'data_parallel_size', 2)
        yield 'gate-config', lambda s: setattr(s.scheduler.vllm_config, 'speculative_config', object())
        yield 'gate-config', lambda s: setattr(s.scheduler, 'connector', object())
        yield 'gate-not-running', lambda s: setattr(s.engine, 'is_running', lambda: False)
        yield 'cohort-empty', lambda s: setattr(s.scheduler, 'running', [])
        yield 'cohort-over', lambda s: setattr(s.scheduler, 'max_num_running_reqs', 1)
        yield 'cohort-prefill', lambda s: setattr(s.scheduler, 'running', [decoder(is_prefill_chunk=True)])
        yield 'cohort-prefill', lambda s: setattr(s.scheduler, 'running', [decoder(num_computed_tokens=9)])
        yield 'cohort-other', lambda s: setattr(s.scheduler, 'running', [decoder(lora_request=object())])
        yield 'cohort-other', lambda s: setattr(s.scheduler, 'running', [decoder(resumable=True)])
        yield 'cohort-uid', lambda s: setattr(s.scheduler, 'running', [decoder(uid=7)])
        yield 'rejected', lambda s: s.policy.__dict__['rejected'].update(r1=lambda: s.queue[0])
        yield 'plan-controller', lambda s: setattr(s, 'refuses', True)
        yield 'plan-transfer', lambda s: setattr(s.queue[0], 'ec_transfer_params', {'a': 1})
        yield 'plan-prompt-logprobs', lambda s: setattr(s.queue[0].sampling_params, 'prompt_logprobs', 1)
        yield 'plan-shared', lambda s: setattr(s, 'shared', 1)
        yield 'plan-below-floor', lambda s: setattr(s.queue[0], 'num_tokens', auto.MIN_NATIVE_ROWS)
        yield 'plan-kv-blocks', lambda s: setattr(s, 'free', 0)
        for field, value in (('num_computed_tokens', 1), ('num_output_tokens', 1), ('num_preemptions', 1), ('resumable', True), ('lora_request', object()), ('pooling_params', object()),
                             ('structured_output_request', object()), ('status', NS(name='RUNNING'))):
            yield 'plan-state', (lambda f, v: lambda s: setattr(s.queue[0], f, v))(field, value)

    def test_each_state_lands_on_its_site_and_nothing_else(self):
        seen = set()
        for site, arrange in self.cases():
            with self.subTest(site=site):
                stage = Stage(req())
                arrange(stage)
                self.assertIs(stage.policy.tick(), False)
                self.assertEqual(stage.policy.phase, 'idle')
                record = stage.policy.decline_record()
                self.assertEqual({k: v for k, v in record['ticks'].items() if v}, {site: 1})
                self.assertEqual({k: v for k, v in record['turns'].items() if v}, {})
                head = stage.queue[0]
                stock_takes(stage, head)
                record = stage.policy.decline_record()
                self.assertEqual({k: v for k, v in record['turns'].items() if v}, {site: 1})
                self.assertEqual((record['pending'], record['read_errors']), (0, 0))
                seen.add(site)
        self.assertLessEqual(seen, set(auto.DECLINE_SITES))
        self.assertEqual(set(auto.DECLINE_SITES) - seen, {'tail', 'plan-generation-logprobs', 'plan-uid', 'plan-empty'}, 'the three below, and tail')

    def test_generation_logprobs_count_only_outside_inwait(self):
        with unittest.mock.patch.dict('os.environ', {'FLASH_S5_INWAIT': '0'}):
            stage = Stage(req(sampling_params=NS(prompt_logprobs=None, logprobs=5)))
            stage.policy.tick()
            self.assertEqual(stage.policy.decline_ticks['plan-generation-logprobs'], 1)
        with unittest.mock.patch.dict('os.environ', {'FLASH_S5_INWAIT': '1'}):
            stage = Stage(req(sampling_params=NS(prompt_logprobs=None, logprobs=5)))
            stage.policy.tick()
            self.assertEqual(stage.policy.phase, 'barrier')

    def test_a_uid_the_contract_refuses_on_the_head(self):
        stage = Stage(req(uid=7))
        stage.scheduler.requests = {7: stage.queue[0]}
        stage.policy.tick()
        self.assertEqual(stage.policy.decline_ticks['plan-uid'], 1)

    def test_an_empty_plan(self):
        stage = Stage(req())
        with unittest.mock.patch.object(auto, 'staged_chunk_plan', lambda *a: ()):
            stage.policy.tick()
        self.assertEqual(stage.policy.decline_ticks['plan-empty'], 1)

    def test_tail_lets_stock_take_the_head_and_counts_it(self):
        bulk, head = req('bulk', status=NS(name='RUNNING'), num_output_tokens=0), req('r2')
        stage = Stage(head)
        stage.scheduler.requests['bulk'] = bulk
        stage.policy.phase = 'tail'
        stage.policy.request_ref = lambda: bulk
        self.assertIs(stage.policy.tick(), False)
        self.assertEqual(stage.policy.phase, 'tail')
        self.assertEqual(stage.policy.decline_ticks['tail'], 1)
        stock_takes(stage, head)
        self.assertEqual(stage.policy.decline_turns['tail'], 1)


class Selection(unittest.TestCase):
    def test_a_clean_head_is_selected_with_no_count(self):
        stage = Stage(req())
        self.assertIs(stage.policy.tick(), True)
        self.assertEqual(stage.policy.phase, 'barrier')
        record = stage.policy.decline_record()
        self.assertEqual((sum(record['ticks'].values()), sum(record['turns'].values()), record['pending']), (0, 0, 0))

    def test_no_waiting_head_no_count(self):
        for arrange in (lambda s: setattr(s.policy, 'cooldown', True), lambda s: setattr(s.driver, 'active', True), lambda s: setattr(s.scheduler, 'running', [])):
            stage = Stage()
            arrange(stage)
            self.assertIs(stage.policy.tick(), False)
            self.assertEqual(sum(stage.policy.decline_ticks.values()), 0)

    def test_a_decline_then_native_counts_no_stock_turn(self):
        stage = Stage(req())
        stage.scheduler.running = [decoder(is_prefill_chunk=True)]
        stage.policy.tick()
        stage.scheduler.running = [decoder()]
        self.assertIs(stage.policy.tick(), True)
        record = stage.policy.decline_record()
        self.assertEqual(record['ticks']['cohort-prefill'], 1)
        self.assertEqual(sum(record['turns'].values()), 0)
        self.assertEqual(record['selected_after_decline'], 1)

    def test_waiting_ticks_count_each_but_the_turn_once_at_its_last_site(self):
        stage = Stage(req())
        stage.scheduler.running = [decoder(is_prefill_chunk=True)]
        for _ in range(3):
            stage.policy.tick()
        stage.scheduler.running = []
        stage.policy.tick()
        stock_takes(stage, stage.queue[0])
        record = stage.policy.decline_record()
        self.assertEqual((record['ticks']['cohort-prefill'], record['ticks']['cohort-empty']), (3, 1))
        self.assertEqual({k: v for k, v in record['turns'].items() if v}, {'cohort-empty': 1})

    def test_an_abort_while_waiting_is_no_stock_turn(self):
        head = req()
        stage = Stage(head)
        stage.scheduler.running = []
        stage.policy.tick()
        stage.queue.clear()
        stage.scheduler.requests.pop('r1')
        stage.policy.tick()
        record = stage.policy.decline_record()
        self.assertEqual((sum(record['turns'].values()), record['left_while_waiting'], record['pending']), (0, 1, 0))

    def test_restore_and_rollback_reasons_are_counted(self):
        stage = Stage(req())
        stage.driver.append_yield_end = lambda reason: None
        stage.policy.old_pause = Pause.UNPAUSED
        stage.policy.phase = 'drain'
        stage.policy._rollback('worker-barrier-cohort-refused')
        record = stage.policy.decline_record()
        self.assertEqual(record['rollbacks'], {'worker-barrier-cohort-refused': 1})
        self.assertEqual(record['restores'], {'worker-barrier-cohort-refused': 1})


class NeverStopsTheEngine(unittest.TestCase):
    def test_a_bad_queue_read_is_counted_not_raised(self):
        stage = Stage(req())
        stage.policy.cooldown = True
        stage.controller._queue = lambda s: 1 / 0
        self.assertIs(stage.policy.tick(), False)
        self.assertEqual(stage.policy.decline_record()['read_errors'], 1)

    def test_a_bad_status_read_is_counted_and_the_pending_cleared(self):
        class Bad:
            request_id = 'r1'
            @property
            def status(self):
                raise RuntimeError('x')
        stage = Stage(req())
        bad = Bad()
        stage.scheduler.requests['r1'] = bad
        stage.policy._decline_pending['r1'] = (lambda: bad, 'cooldown')
        stage.queue.clear()
        self.assertIs(stage.policy.tick(), False)
        record = stage.policy.decline_record()
        self.assertEqual((record['read_errors'], record['pending']), (1, 0))

    def test_the_pending_map_is_bounded(self):
        stage = Stage()
        heads = [req('r%d' % i) for i in range(auto.MAX_PENDING_DECLINES + 5)]
        stage.scheduler.running = []
        for head in heads:
            stage.queue[:] = [head]
            stage.scheduler.requests[head.request_id] = head
            stage.policy.tick()
        record = stage.policy.decline_record()
        self.assertEqual((record['pending'], record['dropped']), (auto.MAX_PENDING_DECLINES, 5))


class NumbersOnly(unittest.TestCase):
    def test_the_record_holds_no_request_uid(self):
        uid = 'chatcmpl-secret-looking-uid'
        stage = Stage(req(uid))
        stage.scheduler.running = []
        stage.policy.tick()
        self.assertNotIn(uid, json.dumps(stage.policy.decline_record()))

    def test_the_log_line_holds_site_and_length_only(self):
        uid = 'chatcmpl-secret-looking-uid'
        stage = Stage(req(uid, tokens=59318))
        stage.scheduler.running = []
        lines = []
        stage.policy._decline_log = lambda site, request: lines.append('flash-s5 decline site=%s num_tokens=%d' % (site, request.num_tokens))
        stage.policy.tick()
        stock_takes(stage, stage.queue[0])
        self.assertEqual(lines, ['flash-s5 decline site=cohort-empty num_tokens=59318'])
        self.assertTrue(rule.DECLINE_LINE.search(lines[0]))

    def test_no_site_name_trips_a_box_watcher(self):
        # tools/box-watch-haiku-c.sh and native43_engine_death read every log line
        words = re.compile(r'error|fail|fatal|oom|stall|hang|crash|traceback|dead|died|watchdog|shutdown|memory', re.I)
        self.assertEqual([s for s in auto.DECLINE_SITES if words.search(s)], [])


class RealLogger(unittest.TestCase):
    def test_the_line_goes_through_the_engines_progress_logger(self):
        import engine_driver
        lines = []
        with unittest.mock.patch.object(engine_driver, '_progress_logger', lambda: NS(info=lambda form, *args: lines.append(form % args))):
            stage = Stage(req(tokens=59318))
            stage.scheduler.running = []
            stage.policy.tick()
            stock_takes(stage, stage.queue[0])
        self.assertEqual(lines, ['flash-s5 decline site=cohort-empty num_tokens=59318'])
        self.assertEqual(stage.policy.decline_record()['read_errors'], 0)


class StatusPath(unittest.TestCase):
    """The connected rig: the record a box arm carries."""

    def test_the_real_driver_record_carries_the_counters(self):
        import test_append_yield as ay
        rig = ay.Rig(driver='staged')
        record = rig.d.append_yield_record()
        self.assertEqual(record['declines']['schema'], 's5-auto-declines/1')
        self.assertEqual(set(record['declines']['ticks']), set(auto.DECLINE_SITES))
        json.dumps(record)

    def test_a_stand_in_policy_reads_none(self):
        import test_append_yield as ay
        rig = ay.Rig(driver='staged')
        rig.d.auto_arm = NS(phase='native', manual_owned=False)
        self.assertIsNone(rig.d.append_yield_record()['declines'])


class Reader(unittest.TestCase):
    def status(self, label, **fields):
        declines = dict(schema='s5-auto-declines/1', ticks={}, turns={}, restores={}, rollbacks={}, pending=0, armed=0, selected_after_decline=0, left_while_waiting=0, dropped=0, read_errors=0)
        declines.update(fields)
        return dict(label=label, append_yield=dict(declines=declines))

    def test_an_arm_reads_its_delta_nonzero_only(self):
        record = dict(statuses=[self.status('entry', turns={'tail': 2, 'plan-shared': 1}, armed=3), self.status('before', turns={'tail': 2}),
                                self.status('after', turns={'tail': 5, 'plan-shared': 1}, ticks={'tail': 9}, restores={'full-tail-completed': 4}, armed=5, pending=1)])
        read = rule.decline_read(record)
        self.assertEqual(read['turns'], {'tail': 3})
        self.assertEqual(read['ticks'], {'tail': 9})
        self.assertEqual(read['restores'], {'full-tail-completed': 4})
        self.assertEqual((read['armed'], read['pending']), (2, 1))

    def test_a_record_from_before_the_counters_reads_none(self):
        self.assertIsNone(rule.decline_read(dict(statuses=[dict(label='entry', append_yield={}), dict(label='after', append_yield={})])))
        self.assertIsNone(rule.decline_read(None))

    def log(self, text):
        handle = tempfile.NamedTemporaryFile('w', suffix='.log', delete=False)
        handle.write(text)
        handle.close()
        return handle.name

    def test_a_flagged_turn_gets_its_site_from_the_log(self):
        path = self.log('2026-11-05T05:40:13.500000000Z (EngineCore pid=86) INFO [native_auto_arm.py:1] flash-s5 decline site=plan-kv-blocks num_tokens=59318\nnoise\n')
        declines = rule.read_declines(path)
        send = rule.epoch('2026-11-05T05:40:13.000000000Z')
        self.assertEqual(rule.site_of(declines, 59318, send), 'plan-kv-blocks')
        self.assertIsNone(rule.site_of(declines, 59319, send), 'another length')
        self.assertIsNone(rule.site_of(declines, 59318, send + 10), 'before its send')
        self.assertIsNone(rule.site_of(declines, 59318, send - 500), 'too late to be this turn')

    def test_the_real_c4b_shape_with_no_lines_reads_empty(self):
        self.assertEqual(rule.read_declines(self.log('2026-11-05T05:40:13Z flash-s5 memory point=open rank=1\n')), [])
        self.assertEqual(rule.read_declines('/nonexistent/x'), [])

    def test_stock_turns_by_kind_and_site(self):
        path = self.log('2026-11-05T05:40:13.500000000Z flash-s5 decline site=cohort-prefill num_tokens=60000\n2026-11-05T05:40:20.000000000Z flash-s5 decline site=plan-below-floor num_tokens=90000\n')
        origin = rule.epoch('2026-11-05T05:40:00.000000000Z')
        turns = rule.stamp_turns([dict(kind='fresh', t=13.0, prompt_tokens=60000), dict(kind='append', t=19.0, prompt_tokens=90000), dict(kind='fresh', t=30.0, prompt_tokens=70000), dict(kind='append', t=1.0, prompt_tokens=90000)], origin)
        self.assertEqual(rule.stock_turn_sites(turns, rule.read_declines(path), origin + 5), {'fresh': {'cohort-prefill': 1}, 'append': {'plan-below-floor': 1}})


if __name__ == '__main__':
    unittest.main()
