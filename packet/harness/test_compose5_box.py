"""compose-5's table, box binding and rule. CPU only.

compose-4's rule tests run again over this table, through the wrapper, with ny as the control: what held for ctl has to hold for ny. On top, the frozen pair's price and the r1 read over known numbers.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import compose4_arms as T4
import compose4_rule
import compose5_arms as T
import compose5_rule as rule
import e3_compose4_box as box4
import e3_compose5_box as box
import test_compose4_box as tb

tam, tap, ta1 = tb.tam, tb.tap, tb.ta1
SM, DEC1, CMP, NY = rule.SM, 'dec1.g2-live-0', rule.ON, rule.OFF      # the rule's labels: compose-5's, for its own suites; dnr's table has no cmp
W0, R0, W1, R1 = 'ny.g2-live-0', 'nt.g2-live-0', 'ny.g2-live-1', 'nt.g2-live-1'     # this table: ny as dnr ran it, nr the hold with the residue
# build-2b24: the table is one pair (nt.0 ny.0); W1/R1 are the second pair of TWO_PAIRS, under which the grade's own tests run
# (the grade reads T.arms() at call time and is shape-generic: windows, the last arm to 'after', nr-first-to-last).
TWO_PAIRS = (T.SMOKE_ROW, ('core', 'dec1', 'live', 0)) + tuple(('core', sel, 'live', i) for i in range(2) for sel in (('nt', 'ny') if i % 2 == 0 else ('ny', 'nt')))


class TwoPairs(unittest.TestCase):
    """build-2b24: the grade's tests over build-2b's two-pair shape (ABBA), patched in for the test only."""
    def setUp(self):
        super().setUp()
        for name, value in (('ARM_TABLE', TWO_PAIRS), ('REPLAY_SEEDS', {0: 1, 1: 2}), ('BAND1_INDEXES', (0, 1)), ('INDEXES', (0, 1))):
            patcher = mock.patch.object(T, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

PAIRS = rule.PRICE_PAIRS


def frozen(*, bulk=20.0, decode=56.0, returners=((500, 1.5), (1500, 2.5)), selected=2):
    return dict(client_basis=dict(bulk_complete_s=bulk, prefill_tokens_per_s=65445 / bulk, decode_window=dict(aggregate_tokens_per_s=8 * decode, stream_rates_tokens_per_s=[decode] * 8, gap_ms_p50=20.0, gap_ms_p99=70.0, gap_ms_max=500.0)),
                statuses=[dict(label='after', append_yield=dict(counters=dict(selected=selected)))], returners=[dict(suffix_tokens=s, ttft_s=t) for s, t in returners])


class Fixture5(tb.Fixture):
    """compose-4's record dir with the frozen arms this table adds, whole and quiet."""
    def __init__(self, tmp):
        super().__init__(tmp)
        for on, off in PAIRS:
            self.write_record(on, wall_start_s=tb.ORIGIN - 200, wall_end_s=tb.ORIGIN - 150, **frozen())
            self.write_record(off, wall_start_s=tb.ORIGIN - 140, wall_end_s=tb.ORIGIN - 90, **frozen(selected=0))


class UnderThisTable(unittest.TestCase):
    def setUp(self):
        for name, value in (('CTL', NY), ('rule', rule), ('Fixture', Fixture5)):
            patcher = mock.patch.object(tb, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        super().setUp()


class Rule(UnderThisTable, tb.Rule):
    """Every rule test of compose-4's, here over the wrapper, whose module hands them compose-4's helpers."""


class StoppedArm(UnderThisTable, tb.StoppedArm):
    pass


class JournalAgainstRecord(UnderThisTable, tb.JournalAgainstRecord):
    def test_a_missing_journal_is_inc_on_either_arm(self):   # compose-4's names the old control's file
        healthy_turns = self.fx.turns[CMP]
        self.assertTrue(healthy_turns)
        self.fx.done()
        path = self.fx.root / ('arm-%s.json.turns.jsonl' % NY)
        path.rename(path.with_suffix('.moved'))
        word, _lines, facts = rule.report(str(self.fx.root))
        self.assertEqual(word, 'INC')
        self.assertEqual(facts['arms'][NY]['journal_problem'], 'journal-missing-or-no-header')
        self.assertIn('journal: ' + NY, ' '.join(facts['reasons']))


class Table(unittest.TestCase):
    def test_the_order(self):
        self.assertEqual(T.labels(), [SM, DEC1, R0, W0], 'build-2b24: one pair, nt then ny')

    def test_each_pair_differs_in_the_plan_alone_and_shares_a_seed(self):
        rows = T.by_label()
        for on, off, seed in tuple(('nt.g2-live-%d' % i, 'ny.g2-live-%d' % i, i + 1) for i in range(T.PAIRS)):
            self.assertEqual(sorted(k for k in rows[on] if rows[on][k] != rows[off][k]), ['c1', 'chunk_bound', 'label', 'native_hold', 'native_residue', 'selection', 'yield_on'], (on, off))
            self.assertEqual((rows[on]['yield_on'], rows[off]['yield_on']), (True, False), 'coverage-throughput: the yield on nt')
            self.assertEqual((rows[on]['chunk_bound'], rows[off]['chunk_bound']), ([97152, 1024], [52624, 1024]), 'build-2b24: b24 on nt, b13 on ny')
            self.assertEqual((rows[on]['native_hold'], rows[off]['native_hold']), (T.HOLD_S, 0))
            self.assertEqual((rows[on]['native_residue'], rows[off]['native_residue']), (1, 0))
            self.assertEqual((T.replay_config(rows[on])['SEED'], T.replay_config(rows[off])['SEED']), (seed, seed))

    def test_a_replays_config_is_compose_4s_but_for_the_seed(self):
        self.assertEqual(T.PAIRS, 1, 'build-2b24: one pair')
        row = T.by_label()[W0]
        self.assertEqual({k: v for k, v in T.replay_config(row).items() if T4.REPLAY[k] != v}, dict(DURATION=480.0, WARMUP=0.0), "soak: dnr's turn mix and seed 1, 480 s arms, every turn graded")
        self.assertEqual(T4.REPLAY['SEED'], 1, "the table's own stays as it is")

    def test_the_cap_and_the_warm_block_are_off_on_every_arm(self):
        self.assertEqual({(r['reserved_cap'], r['score_warm']) for r in T.arms()}, {(False, False)})

    def test_the_private_ple_buffer_only_off_on_the_replays_and_the_witness_on_none(self):
        for row in T.arms():
            self.assertEqual((row['ple_private'], row['ple_witness']), (row['kind'] != 'replay', False), row['label'])

    def test_compose_4s_ctl_is_the_base(self):
        c4 = T4.by_label()['ctl.g2-live-0']
        row = T.by_label()[W0]
        # coverage-soak-fix build-2: cow_verify off (the worker RSS owner, test_cow_verify_rss)
        self.assertEqual(sorted(k for k in row if k not in c4 or row[k] != c4[k]), ['c1', 'chunk_bound', 'cow_verify', 'label', 'native_hold', 'native_residue', 'reserved_cap', 'score_warm', 'selection'])
        self.assertEqual((row['chunk_bound'], row['reserved_cap'], row['yield_on'], row['ple_private'], row['cow_verify'], c4['cow_verify']), ([52624, 1024], False, False, False, False, True))

    def test_the_replay_and_the_ceiling_are_compose_4s(self):
        self.assertEqual(set(T.REPLAY), set(T4.REPLAY))
        self.assertEqual({k for k in T.REPLAY if T.REPLAY[k] != T4.REPLAY[k]}, {'DURATION', 'WARMUP'}, "soak: dnr's sizes, its own length")
        self.assertEqual(T4.REPLAY['DURATION'], 1080.0, "compose-4's own stays as it is")

    def test_the_hold_and_the_residue_are_on_nr_alone(self):
        self.assertEqual({r['label']: r['native_hold'] for r in T.arms() if r['native_hold']}, {R0: T.HOLD_S})
        self.assertEqual({r['label'] for r in T.arms() if r['native_residue']}, {R0})
        self.assertEqual(T.MAX_PROMPT_TOKENS, 135000)
        self.assertEqual((T.RULE_PREFILL_TOK_S, T.RULE_DECODE_VS_FIXED, T.RULE_MIN_NATIVE_BULKS), (T4.RULE_PREFILL_TOK_S, T4.RULE_DECODE_VS_FIXED, T4.RULE_MIN_NATIVE_BULKS))

    def test_a_replays_request_count_is_in_its_record(self):
        for label in (W0, R0):
            self.assertIsNone(T.by_label()[label]['generations'])

    def test_a_matched_off_only_for_the_on_arms(self):
        rows = T.by_label()
        self.assertEqual(T.matched_off(rows[R0]), W0)
        self.assertIsNone(T.matched_off(rows[W0]))
        self.assertIsNone(T.matched_off(rows[SM]))


class CapReadBack(unittest.TestCase):
    def test_the_real_records(self):
        def rec(state):
            return dict(reserved_cap=dict(select={str(k): dict(after=dict(reserved_cap=state), before={}, order={}, counters={}) for k in range(4)}))
        self.assertIs(box.cap_selected(rec(dict(cap_bytes=0, flushes=0, fraction=0.0, selected=False))), False, "dn's")
        self.assertIs(box.cap_selected(rec(dict(cap_bytes=23369547776, flushes=1, fraction=0.92, overdraft_refused=0, overdrafts=0, selected=True))), True, "c4b's")
        self.assertIsNone(box.cap_selected(dict()))


class ReplaySeed(unittest.TestCase):
    """The seed reaches the run and the journal header, by the real replay_arm of compose-4's box up to the run call."""

    def go(self, label):
        row = T.by_label()[label]
        seen = {}

        class Stop(Exception):
            pass

        def fake(a, row_, client, poll_client, *, run=None):
            seen['header'] = box4.T.REPLAY
            try:
                run('u', 'm', 'k', 'r', row_['label'], sink=None)
            except Stop:
                pass
            return dict(label=row_['label'])

        def real_run(*args, cfg=None, **kwargs):
            seen['cfg'] = cfg
            raise Stop()
        with mock.patch.object(box, '_replay_arm', fake), mock.patch.object(box._box.replay, 'run', real_run), mock.patch.object(box, 'memory_read', dict):
            box.replay_arm(None, row, None, None)
        return seen

    def test_each_index_gets_its_seed_and_the_table_is_put_back(self):
        saved = T.REPLAY
        for label, seed in ((W0, 1), (R0, 1)):
            seen = self.go(label)
            self.assertEqual((seen['cfg']['SEED'], seen['header']['SEED']), (seed, seed), label)
        self.assertIs(T.REPLAY, saved)
        self.assertIs(box4.T, T4)


WORKER_BASE = (25732, 25230, 6646, 2795)      # MiB, the soak's four Worker_PP at nr.0's start (build-2 record)


def worker(pid, mib, anon=None, maps=900):
    return dict(pid=pid, comm='VLLM::Worker_PP%d' % (pid - 4210), rss_kib=mib << 10, vmrss_kib=mib << 10, rssanon_kib=(anon or mib // 2) << 10,
                rssfile_kib=1 << 20, rssshmem_kib=(mib - (anon or mib // 2)) << 10, vmlck_kib=0, vmpin_kib=0, vmdata_kib=mib << 10, maps=maps, errors=[])


def memory(free=(1464, 1574, 1352, 1462), rss=40 << 20, grow=(0, 0, 0, 0), workers=True):
    """coverage-soak-fix: four Worker_PP rows (pid 4210..4213), each grown by ``grow`` MiB; workers=False is a read without them."""
    rows = [worker(4210 + k, base + g) for k, (base, g) in enumerate(zip(WORKER_BASE, grow))] if workers else None
    return dict(schema=box.MEMORY_SCHEMA, wall_s=1.0, gpus={str(k): dict(used_mib=24576 - f, total_mib=24576, free_mib=f) for k, f in enumerate(free)},
                rss_kib=rss, processes=7, top=[['python3', rss]], workers=rows, errors=[])


class Soak(TwoPairs):
    """coverage-soak: a hang named, memory read at the traffic's start and end, growth flagged on nr."""

    def test_engine_memory_reads_nvidia_smi_and_docker_top(self):
        def run(cmd, **kw):
            if cmd[0] == 'nvidia-smi':
                return mock.Mock(returncode=0, stdout='0, 23112, 24576\n1, 23002, 24576\n2, 23224, 24576\n3, 23114, 24576\n')
            self.assertEqual(cmd, ['docker', 'top', 'qwen-flash-prod', '-eo', 'pid,rss,comm'])
            return mock.Mock(returncode=0, stdout='PID RSS COMMAND\n4101 812345 python3\n4210 9123456 VLLM::Worker_PP0_TP0\n4211 9000000 VLLM::Worker_PP0_TP1\n')
        out = box.engine_memory(run=run, clock=lambda: 7.0)
        self.assertEqual(out['gpus']['0'], dict(used_mib=23112, total_mib=24576, free_mib=1464))
        self.assertEqual((out['rss_kib'], out['processes'], out['top'][0], out['errors']), (812345 + 9123456 + 9000000, 3, ['VLLM::Worker_PP0_TP0', 9123456], []))

    def test_engine_memory_never_raises(self):
        def broken(cmd, **kw):
            raise OSError('no docker')
        out = box.engine_memory(run=broken)
        self.assertEqual((out['gpus'], out['rss_kib'], out['errors']), (None, None, ['nvidia-smi:OSError', 'docker-top:OSError']))
        out = box.engine_memory(run=lambda cmd, **kw: mock.Mock(returncode=1, stdout=''))
        self.assertEqual(out['errors'], ['nvidia-smi-rc-1', 'docker-top-rc-1'])

    def test_tracking_names_what_is_open_and_forgets_what_ended(self):
        import threading
        inflight, lock, gate = {}, threading.Lock(), threading.Event()

        def stream(url, body, key, origin, **kw):
            if body.get('block'):
                gate.wait(5)
            if body.get('boom'):
                raise TimeoutError('read')
            return 'ok'
        wrapped = box.tracking(stream, inflight, lock, clock=lambda: 130.0)
        body = lambda session, n, **kw: dict(messages=[{'role': 'system', 'content': 'You are coding agent 3, session %d.' % session}] + [{}] * (n - 1), **kw)
        self.assertEqual(wrapped('u', body(1, 2), None, 10.0), 'ok')
        with self.assertRaises(TimeoutError):
            wrapped('u', body(1, 4, boom=True), None, 10.0)
        self.assertEqual(inflight, {})
        th = threading.Thread(target=wrapped, args=('u', body(2, 6, block=True), None, 10.0), name='compose4-agent-3')
        th.start()
        for _ in range(100):
            if inflight:
                break
            threading.Event().wait(0.01)
        self.assertEqual(inflight, {'compose4-agent-3': dict(agent=3, session=2, kind='append', turn_messages=6, sent_s=120.0)})
        gate.set()
        th.join(5)
        self.assertEqual(inflight, {})

    def test_the_arm_record_carries_open_requests_and_memory(self):
        import threading
        row, gate = T.by_label()[R0], threading.Event()
        reads = iter((memory(), memory(free=(1400, 1574, 1352, 1462)), memory(free=(1460, 1574, 1352, 1462))))

        def fake(a, row_, client, poll_client, *, run=None):
            run('u', 'm', 'k', 'r', row_['label'], sink=None)
            return dict(label=row_['label'], complete=False, reasons=['replay-turns-still-open:1'])

        def real_run(*args, cfg=None, stream=None, **kwargs):
            body = dict(messages=[{'role': 'system', 'content': 'You are coding agent 5, session 1.'}, {}])
            threading.Thread(target=stream, args=('u', body, None, 0.0), name='compose4-agent-5', daemon=True).start()
            for _ in range(100):
                threading.Event().wait(0.01)
                if gate.is_set():
                    break
            return cfg, {}, [], ['compose4-agent-5']

        def slow(*args, **kw):
            gate.set()
            threading.Event().wait(2)
        with mock.patch.object(box, '_replay_arm', fake), mock.patch.object(box._box.replay, 'run', real_run), \
                mock.patch.object(box._box.replay, 'stream_turn', slow), mock.patch.object(box, 'memory_read', lambda: next(reads)), mock.patch.object(box.time, 'sleep'):
            rec = box.replay_arm(None, row, None, None)
        self.assertEqual([(r['agent'], r['session'], r['kind']) for r in rec['replay_open_requests']], [(5, 1, 'fresh')])
        self.assertEqual((rec['memory']['end']['gpus']['0']['free_mib'], rec['memory']['end']['settle_s'], rec['memory']['end']['first_gpus']['0']['free_mib']), (1460, 2.0, 1400))

    def test_the_end_read_waits_for_the_release_at_most_ten_seconds(self):
        slept = []
        held = iter([memory(free=(400, 400, 400, 400))] * 6)
        end = box.settled_end(memory(), read=lambda: next(held), sleep=slept.append)
        self.assertEqual((end['settle_s'], len(slept), end['gpus']['0']['free_mib']), (10.0, 5, 400))
        end = box.settled_end(memory(), read=lambda: memory(free=(1460, 1574, 1352, 1462)), sleep=slept.append)
        self.assertEqual((end['settle_s'], 'first_gpus' in end), (0.0, False))
        end = box.settled_end(dict(gpus=None), read=lambda: memory(), sleep=slept.append)
        self.assertEqual(end['settle_s'], 0.0, 'no start read: nothing to wait for')

    def arms(self, mem=None, open_=None):
        arms = Grade().records()
        for rec in arms:
            if rec.get('label') in (R0, R1, W0, W1):
                rec['memory'] = dict(start=memory(), end=memory())
                rec.update((mem or {}).get(rec['label'], {}))
                if rec['label'] in (open_ or ()):
                    rec['replay_open_requests'] = [dict(agent=2, session=1, kind='append', turn_messages=8, sent_s=471.2)]
                    rec['replay_still_open'] = ['compose4-agent-2']
        return arms

    def test_a_request_open_after_the_drain_fails_nr_by_name_and_is_read_on_ny(self):
        result = Grade().grade_of(self.arms(open_=(R1, W0)))
        self.assertIn(R1 + ':request-open-after-the-drain', result['failures'])
        self.assertNotIn(W0 + ':request-open-after-the-drain', result['failures'])
        self.assertEqual(result['open_requests'][R1]['requests'][0]['sent_s'], 471.2)
        self.assertEqual(result['open_requests'][W0]['agents'], ['compose4-agent-2'])

    def test_memory_flat_passes_and_growth_on_nr_is_flagged_not_failed(self):
        result = Grade().grade_of(self.arms())
        self.assertEqual((result['verdict'], result['memory_flags']), ('PASS', {}))
        self.assertEqual(result['memory'][R0]['ranks']['0'], dict(free_start_mib=1464, free_end_mib=1464, free_drop_mib=0))
        grown = {R1: dict(memory=dict(start=memory(), end=memory(free=(1300, 1574, 1352, 1462), rss=(40 << 20) + (300 << 10))))}
        result = Grade().grade_of(self.arms(mem=grown))
        self.assertEqual(result['verdict'], 'PASS', 'a flag, not a failure')
        self.assertEqual(result['memory_flags'][R1], ['vram-free-down-rank0-164MiB', 'engine-rss-up-300MiB'])
        self.assertNotIn('nr-first-start-to-last-start', result['memory_flags'], "nr.1's start is as nr.0's")
        drift = {R1: dict(memory=dict(start=memory(free=(1364, 1574, 1352, 1462), rss=(40 << 20) + (300 << 10)), end=memory()))}
        result = Grade().grade_of(self.arms(mem=drift))
        self.assertEqual(result['memory_flags']['nr-first-start-to-last-start'], ['vram-free-down-rank0-100MiB', 'engine-rss-up-300MiB'])
        self.assertEqual(result['verdict'], 'PASS')

    def test_memory_on_ny_is_read_not_flagged_and_an_unread_nr_is_flagged(self):
        result = Grade().grade_of(self.arms(mem={W0: dict(memory=dict(start=memory(), end=memory(free=(100, 100, 100, 100)))), R0: dict(memory=dict(start=None, end=memory()))}))
        self.assertEqual(result['memory'][W0]['ranks']['1']['free_drop_mib'], 1474)
        self.assertNotIn(W0, result['memory_flags'])
        self.assertEqual(result['memory_flags'][R0], ['memory-not-read'])
        # coverage-soak-fix: an nr arm with no worker read at its start fails; growth there must not pass unread
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['failures'], [R0 + ':worker-rss-not-read'])


class WorkerRss(TwoPairs):
    """coverage-soak-fix (lead, 6 Nov 06:25): per Worker_PP RSS fails an nr arm at 256 MiB, or 32 B per native-append history token."""

    def grade(self, label=R1, grow=(0, 0, 0, 0), history=None, start=None):
        arms = Soak().arms(mem={label: dict(memory=dict(start=start or memory(), end=memory(grow=grow)))})
        if history is not None:
            for rec in arms:
                for status in rec.get('statuses') or ():
                    cov = ((status.get('append_yield') or {}).get('declines') or {}).get('coverage')
                    if cov is not None and rec.get('label') == label and status.get('label') == 'after':
                        cov['append_history_tokens'] = history
        return Grade().grade_of(arms)

    def test_flat_workers_pass(self):
        result = self.grade()
        self.assertEqual(result['verdict'], 'PASS', result['failures'])
        self.assertEqual(result['worker_rss'][R1]['fails'], [])

    def test_the_soaks_growth_fails_by_size_and_by_token(self):
        # nr.1 of build-2: 2046, 2054, 1765, 2023 MiB over 7.624M append-history tokens (about 280 B per token)
        result = self.grade(grow=(2046, 2054, 1765, 2023), history=7624000)
        fails = [f for f in result['failures'] if f.startswith(R1)]
        self.assertIn(R1 + ':worker-rss-up-pid4210-2046MiB', fails)
        self.assertIn(R1 + ':worker-rss-per-append-token-pid4212-243B', fails)
        self.assertEqual(len(fails), 8)

    def test_under_the_size_a_per_token_rate_still_fails(self):
        result = self.grade(grow=(100, 0, 0, 0), history=1000000)          # 105 B per token
        self.assertEqual([f for f in result['failures'] if f.startswith(R1)], [R1 + ':worker-rss-per-append-token-pid4210-105B'])
        result = self.grade(grow=(100, 0, 0, 0), history=10000000)         # 10 B per token, under 256 MiB
        self.assertEqual(result['verdict'], 'PASS', result['failures'])

    def test_with_no_append_history_only_the_size_counts(self):
        self.assertEqual(self.grade(grow=(255, 0, 0, 0)).get('verdict'), 'PASS')
        self.assertIn(R1 + ':worker-rss-up-pid4210-256MiB', self.grade(grow=(256, 0, 0, 0))['failures'])

    def test_a_ny_arm_is_read_not_failed(self):
        result = self.grade(label=W1, grow=(3000, 3000, 3000, 3000), history=1000)
        self.assertEqual(result['verdict'], 'PASS', result['failures'])
        self.assertNotIn(W1, result['worker_rss'])
        self.assertEqual(result['memory'][W1]['workers']['rows']['4210']['rss_kib']['growth'], 3000 << 10)

    def test_a_changed_pid_or_a_missing_worker_read_fails(self):
        start = memory()
        start['workers'][0] = dict(start['workers'][0], pid=4300)
        self.assertIn(R1 + ':worker-pid-changed-4300', self.grade(start=start)['failures'])
        self.assertIn(R1 + ':worker-rss-not-read', self.grade(start=memory(workers=False))['failures'])

    def test_the_delta_carries_every_status_field_and_the_maps(self):
        result = self.grade(grow=(64, 0, 0, 0))
        row = result['memory'][R1]['workers']['rows']['4210']
        self.assertEqual(sorted(row), sorted(['comm', 'rss_kib', 'maps'] + [f.lower() + '_kib' for f in box.STATUS_FIELDS]))
        self.assertEqual((row['rssanon_kib']['growth'], row['maps']['growth']), (32 << 10, 0))

    def test_the_report_has_a_worker_line_per_nr_arm(self):
        result = self.grade(grow=(300, 0, 0, 0))
        with box._bound():
            text = box.report(result)
        self.assertIn('- %s worker RSS: worker-rss-up-pid4210-300MiB' % R1, text)
        self.assertIn('- %s worker RSS: no failure' % R0, text)

    def test_engine_memory_reads_each_workers_proc(self):
        with tempfile.TemporaryDirectory() as proc:
            for pid in (4210, 4211):
                Path(proc, str(pid)).mkdir()
                Path(proc, str(pid), 'status').write_text('Name:\tVLLM::Worker\nVmRSS:\t  9123456 kB\nRssAnon:\t 5000000 kB\n'
                                                          'RssFile:\t 23456 kB\nRssShmem:\t 4100000 kB\nVmLck:\t0 kB\nVmPin:\t 12 kB\nVmData:\t 7000000 kB\n')
                Path(proc, str(pid), 'maps').write_text('a\n' * (100 + pid - 4210))

            def run(cmd, **kw):
                if cmd[0] == 'nvidia-smi':
                    return mock.Mock(returncode=0, stdout='0, 23112, 24576\n')
                return mock.Mock(returncode=0, stdout='PID RSS COMMAND\n4101 812345 python3\n4210 9123456 VLLM::Worker_PP0_TP0\n4211 9000000 VLLM::Worker_PP0_TP1\n4299 5 VLLM::Worker_PP1\n')
            out = box.engine_memory(run=run, status=lambda pid: box.worker_status(pid, proc=proc))
        rows = {w['pid']: w for w in out['workers']}
        self.assertEqual(sorted(rows), [4210, 4211, 4299])
        self.assertEqual((rows[4210]['rssanon_kib'], rows[4210]['rssshmem_kib'], rows[4210]['vmpin_kib'], rows[4210]['maps']), (5000000, 4100000, 12, 100))
        self.assertEqual((rows[4211]['maps'], rows[4211]['rss_kib'], rows[4211]['comm']), (101, 9000000, 'VLLM::Worker_PP0_TP1'))
        self.assertEqual(rows[4299]['errors'], ['status:FileNotFoundError', 'maps:FileNotFoundError'])
        self.assertEqual(out['errors'], [])


class BulkHoldGrade(TwoPairs):
    """coverage-throughput build-2b (lead, 6 Nov): a yielded bulk held at the positive frontier FAILs nt loudly, with its own line."""
    cov = property(lambda self: Coverage())

    def arms(self, label, **held):
        arms = self.cov.arms_with(label)
        rec = next(a for a in arms if a.get('label') == label)
        later = [rec['statuses'][-1]] + [s for a in arms if a.get('label') in T.labels()[T.labels().index(label) + 1:] for s in a['statuses']]
        for status in later:
            declines = (status.get('append_yield') or {}).get('declines')
            if declines is not None:
                declines['yield_bulk_hold'] = dict(dict(bulks=0, deferrals=0, expired=0, errors=0), **held)
        return arms

    def test_a_held_bulk_fails_nt_with_its_own_line(self):
        result = self.cov.grade.grade_of(self.arms(R1, bulks=1, deferrals=40))
        self.assertIn(R1 + ':yield-bulk-held', result['failures'])
        self.assertEqual(result['coverage'][R1]['yield_bulk_hold'], dict(bulks=1, deferrals=40, expired=0, errors=0))
        with box._bound():
            self.assertIn('- %s YIELD BULK HELD: bulks 1, deferrals 40, expired 0, errors 0' % R1, box.report(result))

    def test_none_held_passes_and_prints_nothing(self):
        result = self.cov.grade.grade_of(self.arms(R1))
        self.assertEqual([f for f in result['failures'] if f.startswith(R1)], [])
        with box._bound():
            self.assertNotIn('YIELD BULK HELD', box.report(result))

    def test_on_ny_it_is_read_not_failed(self):
        result = self.cov.grade.grade_of(self.arms(W1, bulks=1))
        self.assertEqual([f for f in result['failures'] if f.startswith(W1)], [])


class GeometryGrade(TwoPairs):
    """coverage-soak-fix: the geometry-tail class is native-served on nr, on its own line; without its native turns it fails."""
    cov = property(lambda self: Coverage())      # defined below this class

    def arms(self, rows, turns, max_rows=None):
        arms = self.cov.arms_with(R1, stock={'handback': 10, 'geometry-tail': rows}, native_turns=10)
        rec = next(a for a in arms if a.get('label') == R1)
        for status in rec['statuses'][1:]:
            c = ((status.get('append_yield') or {}).get('declines') or {}).get('coverage')
            if c is not None:
                c['geometry_tail'] = dict(turns=turns, max_r=48, max_rows=max_rows or 48)
        return arms

    def test_geometry_tail_rows_after_their_turns_pass_and_count_native(self):
        result = self.cov.grade.grade_of(self.arms(107, 4))
        self.assertEqual([f for f in result['failures'] if f.startswith(R1)], [])
        cov = result['coverage'][R1]
        self.assertEqual(cov['geometry_tail'], dict(turns=4, rows=107, live_rows=107, requests=107, max_r=48, max_rows=48))
        self.assertEqual(cov['live_native_share'], 1.0)
        with box._bound():
            self.assertIn('- %s geometry-tail: turns 4, rows 107 (107 live), max r 48, max rows 48 (engine life)' % R1,
                          box.report(result))
            # build-2 (c): the tail's rows and share on the line after, next to native share (served and strict)
            text = box.report(result)
            total = sum(sum(cov['rows'][s][p].values()) for s in ('live', 'idle') for p in ('native', 'stock'))
            line = '- %s native share %s (strict %s) | geometry-tail rows 107 of %d prefill rows, share %s' % (
                R1, cov['all_native_share'], cov['all_native_share_strict'], total, round(107 / total, 6))
            self.assertIn(line, text)
            self.assertLess(cov['all_native_share_strict'], cov['all_native_share'])
            lines = text.splitlines()
            self.assertEqual(lines.index(line), lines.index('- %s geometry-tail: turns 4, rows 107 (107 live), max r 48, max rows 48 (engine life)' % R1) + 1)

    def test_rows_with_no_turn_or_too_many_fail(self):
        self.assertIn(R1 + ':geometry-tail-not-of-a-native-turn', self.cov.grade.grade_of(self.arms(20, 0))['failures'])
        self.assertIn(R1 + ':geometry-tail-not-of-a-native-turn', self.cov.grade.grade_of(self.arms(300, 4))['failures'])
        self.assertIn(R1 + ':geometry-tail-not-of-a-native-turn', self.cov.grade.grade_of(self.arms(48, 1, max_rows=65))['failures'])

    def test_on_a_ny_arm_it_is_read_not_failed(self):
        arms = self.cov.arms_with(W1, stock={'geometry-tail': 20})
        result = self.cov.grade.grade_of(arms)
        self.assertEqual([f for f in result['failures'] if f.startswith(W1)], [])


class Smoke(unittest.TestCase):
    """What this table adds to compose-4's smoke; b13h's numbers in the record are dp8d's (q18a-dp8-c32, arm-b13h.g2-live-0.json)."""

    def full(self):
        row = T.by_label()[SM]
        rec = ta1.record_of(row)
        rec.update(ple_private=dict(counters=dict(scopes={'0': 9, '1': 9})), score_warm=dict(counters=dict(allocs={str(k): 1 for k in range(4)})), reserved_cap=dict(requested=False, select={str(k): dict(after=dict(reserved_cap=dict(cap_bytes=0, flushes=0, fraction=0.0, selected=False))) for k in range(4)}, counters=dict(flushes={str(k): 0 for k in range(4)})),
                   cow_host_filter=dict(requested=True, counters=dict(kept={str(k): 0 for k in range(4)}, skipped={str(k): 37 for k in range(4)}, unfiltered={str(k): 0 for k in range(4)})))
        rec.setdefault('inwait', {}).update(exact_acknowledged=True, bound_requested=[52624, 1024], bound_acknowledged=[52624, 1024], hold_requested=0, hold_acknowledged=0, residue_requested=0, residue_acknowledged=0)
        rec.setdefault('progress_delta', {})['chunks'] = 40
        rec.setdefault('fresh', {})['prompt_tokens_observed'] = 65445
        return rec

    def assert_stops(self, change, reason):
        rec = self.full()
        change(rec)
        ok, reasons = box.smoke_check(rec)
        self.assertFalse(ok)
        self.assertIn(reason, reasons)

    def test_dp8ds_numbers_pass(self):
        self.assertEqual(box.smoke_check(self.full()), (True, []))

    def test_the_cap_selected_stops_the_table(self):
        self.assert_stops(lambda r: r['reserved_cap'].update(requested=True), 'smoke-reserved-cap-not-off')
        self.assert_stops(lambda r: r.pop('reserved_cap'), 'smoke-reserved-cap-not-off')
        # The numbers are c4b's sm on r1 and dn's sm, the real shape of the rank's answer.
        self.assert_stops(lambda r: r['reserved_cap']['select'].update({'1': dict(after=dict(reserved_cap=dict(cap_bytes=23369547776, flushes=1, fraction=0.9238, selected=True)))}), 'smoke-reserved-cap-selected')
        self.assert_stops(lambda r: r['reserved_cap']['select'].pop('3'), 'smoke-reserved-cap-selected')
        self.assert_stops(lambda r: r['reserved_cap'].update(select=None), 'smoke-reserved-cap-selected')

    def test_the_bound_not_acknowledged_stops_the_table(self):
        self.assert_stops(lambda r: r['inwait'].update(bound_acknowledged=None), 'smoke-chunk-bound-not-acknowledged')

    def test_the_hold_not_acknowledged_stops_the_table(self):
        self.assert_stops(lambda r: r['inwait'].update(hold_acknowledged=None), 'smoke-native-hold-not-acknowledged')
        self.assert_stops(lambda r: r['inwait'].update(hold_acknowledged=600), 'smoke-native-hold-not-acknowledged')
        self.assert_stops(lambda r: r['inwait'].pop('hold_acknowledged'), 'smoke-native-hold-not-acknowledged')

    def test_the_residue_not_acknowledged_stops_the_table(self):
        self.assert_stops(lambda r: r['inwait'].update(residue_acknowledged=None), 'smoke-native-residue-not-acknowledged')
        self.assert_stops(lambda r: r['inwait'].update(residue_acknowledged=1), 'smoke-native-residue-not-acknowledged')
        self.assert_stops(lambda r: r['inwait'].pop('residue_acknowledged'), 'smoke-native-residue-not-acknowledged')

    def test_a_launch_that_fell_back_to_1024_stops_the_table(self):
        # compose-4's smoke at 1024: 71 chunks, its 66 and the 5 of its returners and append. That's the control: under b13, with the same 5, 45.
        self.assert_stops(lambda r: r['progress_delta'].update(chunks=71), 'smoke-chunk-rows-not-requested')
        self.assert_stops(lambda r: r['progress_delta'].update(chunks=66), 'smoke-chunk-rows-not-requested')
        rec = self.full()
        rec['progress_delta']['chunks'] = 45
        self.assertEqual(box.smoke_check(rec), (True, []))

    def test_no_count_to_read_stops_the_table(self):
        self.assert_stops(lambda r: r['progress_delta'].pop('chunks'), 'smoke-chunk-rows-not-requested')
        self.assert_stops(lambda r: r['fresh'].pop('prompt_tokens_observed'), 'smoke-chunk-rows-not-requested')

    def test_the_host_filter_not_engaged_on_a_rank_stops_the_table(self):
        self.assert_stops(lambda r: r['cow_host_filter']['counters']['skipped'].update({'2': 0}), 'smoke-host-filter-not-engaged')
        self.assert_stops(lambda r: r['cow_host_filter'].update(counters={}), 'smoke-host-filter-not-engaged')
        self.assert_stops(lambda r: r['cow_host_filter']['counters']['unfiltered'].update({'3': 1}), 'smoke-host-filter-unfiltered')


REAL_TORCH = (" (Worker_PP0_TP0_EP0 pid=96) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 194.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 164.94 MiB is free. Including non-PyTorch memory\n",
              " (Worker_PP0_TP1_EP1 pid=102) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 194.00 MiB. GPU 1 has a total capacity of 23.56 GiB of which 60.94 MiB is free. Including non-PyTorch memory\n")     # dp8d's cf, band 2


class CrashRead(UnderThisTable, unittest.TestCase):
    """A read on an engine death, never the verdict: compose-4's clause already says FAIL."""

    def fixture(self, tmp, *, death_at=None, turns=True):
        fx = Fixture5(tmp)
        tb.healthy(fx)
        tb.healthy(fx, NY)
        if not turns:
            fx.turns[CMP] = []
        if death_at is not None:
            fx.lines.append(tb.memory_line(death_at - 1, 'open', 1, 900, 97152, 99200, free=60))
            fx.lines.append(tb.memory_line(death_at - 1, 'open', 0, 900, 97152, 99200, free=164))
            (fx.root / 'engine-stream.err').write_text(''.join(tb.stamp(death_at) + line for line in REAL_TORCH))
        return fx

    def test_a_clean_run_prints_nothing_of_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            _word, lines, facts = self.fixture(tmp).done()
        self.assertFalse([x for x in lines if x.startswith('CRASH')])
        self.assertIsNone(facts['crash'])

    def test_a_death_names_the_arm_each_ranks_chunk_and_torchs_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            word, lines, facts = self.fixture(tmp, death_at=tb.ORIGIN + 100).done()
        self.assertEqual(word, 'FAIL', 'the verdict is the old clause')
        crash = facts['crash']
        self.assertEqual(crash['arm'], CMP)
        self.assertEqual({k: crash['ranks'][k]['chunk'] for k in ('0', '1')}, {'0': '97152-99200', '1': '97152-99200'})
        self.assertEqual(crash['tried']['0']['tried_mib'], 194.0)
        self.assertEqual(crash['tried']['1']['free_mib'], 60.94)
        self.assertEqual(crash['tried']['0']['position_if_score_at_2048'], 99328, "dp8d's chunk ended at 99200: the width rounds up to 64")
        self.assertEqual(crash['tried']['0']['position_if_score_at_1024'], 198656)
        self.assertGreater(crash['turns']['finished'], 0)
        self.assertTrue([x for x in lines if x.startswith('CRASH GPU 1')])

    def test_the_verdict_is_the_same_with_the_read_or_without(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(rule, 'crash_read', return_value=None):
            without = self.fixture(tmp, death_at=tb.ORIGIN + 100).done()[0]
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.fixture(tmp, death_at=tb.ORIGIN + 100).done()[0], without)

    def test_a_death_between_arms_says_after_which(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self.fixture(tmp)
            record = json.loads((fx.root / ('arm-%s.json' % PAIRS[0][0])).read_text())
            self.assertEqual(rule.arm_at(str(fx.root), (PAIRS[0][0],), record['wall_end_s'] + 5), ('after ' + PAIRS[0][0], None))

    def test_a_line_without_a_size_or_a_stamp_is_kept_going(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = self.fixture(tmp)
            (fx.root / 'engine-stream.err').write_text('garbage OutOfMemoryError\n' + tb.stamp(tb.ORIGIN) + ' FLASH_S5_WORKER_FAILURE rank=1 cause=OutOfMemoryError\n')
            crash = rule.crash_read(str(fx.root))
        self.assertEqual(crash['tried'], {})
        self.assertIsNotNone(crash['first_death_s'])


class JsonCarriesTheReasons(UnderThisTable, unittest.TestCase):
    """Codex, REVIEW-compose5 item 1: a word this rule changes has its reason in the JSON too, not only in the text."""

    def over_the_bound(self, tmp):
        fx = Fixture5(tmp)
        tb.healthy(fx)
        tb.healthy(fx, NY)
        fx.lines.append(tb.memory_line(tb.ORIGIN + 50, 'open', 0, 9000, 52624, 54672))
        return fx.done()

    def test_a_chunk_past_the_bound_is_inc_with_its_reason_in_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            word, lines, facts = self.over_the_bound(tmp)
        self.assertEqual((word, facts['rule']), ('INC', 'INC'))
        self.assertEqual(facts['reasons'], ['chunk-rows: 1 chunks past the bound over 1024 rows'])
        self.assertIn('REASON ' + facts['reasons'][0], lines)

    def test_a_stock_rescue_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture5(tmp)
            tb.healthy(fx)
            tb.healthy(fx, NY)
            with mock.patch.object(rule, 'stock_rescues', return_value=2):
                word, lines, facts = fx.done()
        self.assertEqual(word, 'INC')
        self.assertEqual(facts['reasons'], ['stock-rescue: cmp 2'])
        self.assertIn('REASON stock-rescue: cmp 2', lines)

    def test_the_base_reasons_stay_and_a_clean_run_has_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture5(tmp)
            tb.healthy(fx)
            tb.healthy(fx, NY)
            word, _lines, facts = fx.done()
        self.assertEqual((word, facts['reasons']), ('PASS', []))
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture5(tmp)
            tb.healthy(fx)
            tb.healthy(fx, NY)
            fx.lines.append(tb.memory_line(tb.ORIGIN + 50, 'open', 0, 9000, 52624, 54672))
            fx.lines.append(tb.stamp(tb.ORIGIN) + ' ERROR torch.OutOfMemoryError: CUDA out of memory\n')
            word, _lines, facts = fx.done()
        self.assertEqual(word, 'FAIL', 'INC never lowers a FAIL')
        self.assertTrue(any('OutOfMemoryError' in x for x in facts['reasons']))
        self.assertIn('chunk-rows: 1 chunks past the bound over 1024 rows', facts['reasons'])


class ChunkRowsOfADeadArm(UnderThisTable, unittest.TestCase):
    """Codex, REVIEW-compose5 item 2: a crashed cmp has no final record; its count is partial or unavailable, never zero."""

    def crashed(self, tmp, *, death=True):
        fx = Fixture5(tmp)
        tb.healthy(fx, starts=(320, 500, 700))
        fx.lines.append(tb.memory_line(tb.ORIGIN + 900, 'open', 0, 9000, 97152, 98176))
        fx.lines.append(tb.memory_line(tb.ORIGIN + 1000, 'open', 0, 9001, 97152, 98176))      # past the death line: not this arm's
        (fx.root / 'engine-stream.log').write_text(''.join(fx.lines))
        fx.write_turns(CMP)
        if death:
            (fx.root / 'engine-stream.err').write_text(tb.stamp(tb.ORIGIN + 950) + REAL_TORCH[0])
        return rule.report(str(fx.root))

    def test_the_stopped_arm_counts_its_real_rows_marked_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            word, lines, facts = self.crashed(tmp)
        self.assertEqual(word, 'FAIL')
        got = facts['chunk_rows'][CMP]
        self.assertIs(got['partial'], True)
        self.assertGreater(got['below_bound'], 0)
        self.assertEqual(got['below_max'], 1024)
        self.assertTrue(any(line.startswith('CHUNK ROWS %s' % CMP) and '"partial": true' in line for line in lines))

    def test_the_stopped_arm_ends_at_the_death_line_not_past_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            with_death = self.crashed(tmp)[2]['chunk_rows'][CMP]
        with tempfile.TemporaryDirectory() as tmp:
            without = self.crashed(tmp, death=False)[2]['chunk_rows'][CMP]
        self.assertEqual(without['past_bound'] - with_death['past_bound'], 1, 'the row after the death line is counted only without a death line')

    def test_an_arm_that_never_ran_is_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            _word, lines, facts = self.crashed(tmp)
        got = facts['chunk_rows'][NY]
        self.assertEqual(got, dict(unavailable='no final record and no journal header'))
        self.assertNotIn('below_bound', got)
        self.assertTrue(any(line.startswith('CHUNK ROWS %s' % NY) and 'unavailable' in line for line in lines))

    def test_a_whole_arm_is_not_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture5(tmp)
            tb.healthy(fx)
            tb.healthy(fx, NY)
            facts = fx.done()[2]
        self.assertNotIn('partial', facts['chunk_rows'][CMP])
        self.assertGreater(facts['chunk_rows'][CMP]['below_bound'], 0)

    def test_a_partial_span_past_the_bound_over_1024_still_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture5(tmp)
            tb.healthy(fx, starts=(320, 500, 700))
            fx.lines.append(tb.memory_line(tb.ORIGIN + 900, 'open', 0, 9000, 97152, 99200))
            (fx.root / 'engine-stream.log').write_text(''.join(fx.lines))
            fx.write_turns(CMP)
            facts = rule.report(str(fx.root))[2]
        self.assertEqual(facts['chunk_rows'][CMP]['past_over_1024'], 1)
        self.assertIn('chunk-rows: 1 chunks past the bound over 1024 rows', facts['reasons'])


class ChunkRows(unittest.TestCase):
    def test_the_read_sees_each_side_of_the_bound_on_rank_0_and_inside_the_span(self):
        rows = [dict(point='open', rank=0, t=5, a=0, b=2048), dict(point='open', rank=0, t=5, a=2048, b=4048), dict(point='open', rank=0, t=6, a=52624, b=53648), dict(point='open', rank=0, t=6, a=53648, b=55696),
                dict(point='released', rank=0, t=6, a=53648, b=55696), dict(point='open', rank=1, t=6, a=53648, b=55696), dict(point='open', rank=0, t=50, a=53648, b=55696)]
        self.assertEqual(rule.chunk_rows(rows, 1, 10), dict(below_bound=2, below_over_1024=2, below_max=2048, past_bound=2, past_over_1024=1, past_max=2048))


class BoxBinding(unittest.TestCase):
    def test_the_table_is_bound_for_the_call_and_put_back(self):
        seen = []
        with mock.patch.object(box4, 'replay_arm', side_effect=lambda *a, **k: seen.append(box4.T) or {}):
            with box._bound():
                self.assertIs(box4.T, T)
        self.assertIs(box4.T, T4)
        self.assertEqual(box4.ARM_SCHEMA, 'compose4-arm/1')

    def test_a_raise_puts_it_back_too(self):
        with self.assertRaises(ValueError):
            with box._bound():
                raise ValueError('x')
        self.assertIs(box4.T, T4)

    def test_the_argparse_takes_this_tables_selections_and_refuses_the_old_control(self):
        with mock.patch.object(box4, 'run_arm', return_value=0) as arm, tempfile.TemporaryDirectory() as tmp:
            args = ['arm', '--url', 'u', '--model', 'm', '--run-id', 'r', '--epoch', '1', '--c1', 'ny', '--size', '2', '--cohort', 'live', '--index', '0', '--workload', 'w', '--out', tmp + '/x.json']
            self.assertEqual(box.main(args), 0)
            self.assertEqual((arm.call_args[0][0].c1, arm.call_args[0][0].index), ('ny', 0))
            with self.assertRaises(SystemExit):
                box.main(args[:15] + ['1'] + args[16:])        # build-2b24: no second pair
            with self.assertRaises(SystemExit):
                box.main(args[:9] + ['nw'] + args[10:])

    def test_the_smoke_goes_the_live_path_and_a_replay_the_replay_path(self):
        import types

        def go(c1, index):
            a = types.SimpleNamespace(c1=c1, size=2, cohort='live', index=index, url='u', run_id='r', epoch=1, control_timeout_s=1.0, out=str(Path(tmp) / ('%s-%d.json' % (c1, index))))
            with mock.patch.object(box4.refusal, 'RefusalKeepingClient'), mock.patch.object(box4.a1, 'load_config', return_value=({}, 1)), \
                    mock.patch.object(box4.a1, 'arm', return_value=dict(label='live', complete=True)) as live, mock.patch.object(box4, 'replay_arm', return_value=dict(label='replay', complete=True)) as play:
                self.assertEqual(box.run_arm(a), 0)
            return (live.call_args or play.call_args)[0][1], bool(live.called), bool(play.called)
        with tempfile.TemporaryDirectory() as tmp:
            row, live, play = go('sm', 0)
            self.assertEqual((row['label'], row['yield_on'], row['index'], live, play), (SM, True, 0, True, False))
            row, live, play = go('ny', 0)
            self.assertEqual((row['label'], row['yield_on'], row['ple_private'], live, play), (W0, False, False, False, True))

    def test_the_grade_names_this_run_and_reads_every_arm(self):
        grade = Grade()
        result = grade.grade_of(grade.records())
        self.assertEqual((result['verdict'], result['failures'], result['missing'], result['schema']), ('PASS', [], [], 'compose5-grade/1'))
        self.assertEqual(sorted(result['arms']), sorted(T.labels()))
        with box._bound():
            text = box.report(result)
        self.assertTrue(text.startswith('# q18a-compose-throughput-b24'))
        self.assertIn('compose5_rule.py', text)


def coverage(native=0, stock=None, idle_stock=None, errors=0, steps=0):
    """The engine's counter as decline_record carries it (scheduler_hooks.PrefillCoverage.record), cumulative."""
    rows = dict(live=dict(native=dict(plan=native) if native else {}, stock=dict(stock or {})), idle=dict(native={}, stock=dict(idle_stock or {})))
    return dict(schema=box.COVERAGE_SCHEMA, rows=rows, requests=rows, steps=steps, errors=errors, open_native=0)


def with_coverage(rec, start, end):
    rec['statuses'] = [dict(label='entry', append_yield=dict(declines=dict(coverage=start)))] + list(rec.get('statuses') or ()) + [
        dict(label='after', append_yield=dict(counters={}, declines=dict(coverage=end)))]
    return rec


class Grade(TwoPairs):
    def records(self):
        out = []
        for k, row in enumerate(T.arms()):
            rec = dict(ta1.record_of(dict(row)), schema=box.ARM_SCHEMA, kind=row['kind'], label=row['label'])
            rec['inwait'] = dict(rec.get('inwait') or {}, bound_requested=row['chunk_bound'], bound_acknowledged=row['chunk_bound'], hold_requested=row['native_hold'], hold_acknowledged=row['native_hold'],
                                 residue_requested=row['native_residue'], residue_acknowledged=row['native_residue'])
            with_coverage(rec, coverage(1000 * k, steps=10 * k), coverage(1000 * k + 900, steps=10 * k + 9))
            if row['kind'] == 'replay':
                rec['memory'] = dict(start=memory(), end=memory())     # coverage-soak-fix: the worker reads, flat
            out.append(rec)
        return ta1.tbox.decode_records() + out

    def grade_of(self, arms, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'engine.log'
            log.write_text(''.join('INFO x %s\n' % line for line in tam.STARTUP))
            known = [a for a in arms if a.get('label')]
            return box.grade(arms, {a['label']: tap.PASS for a in known}, tap.CHECKS, [str(log)], {a['label']: tap.ALIVE for a in known}, **kw)

    def test_a_bound_the_engine_did_not_acknowledge_fails_on_every_arm(self):
        for row in T.arms():
            for reply in (None, [0, 1024], [52624, 2048]):
                arms = self.records()
                next(a for a in arms if a.get('label') == row['label'])['inwait']['bound_acknowledged'] = reply
                result = self.grade_of(arms)
                self.assertEqual(result['verdict'], 'FAIL', (row['label'], reply))
                self.assertIn(row['label'] + ':chunk-bound-not-acknowledged', result['failures'])

    def test_a_hold_the_engine_did_not_acknowledge_fails_on_every_arm(self):
        for row in T.arms():
            for reply in (None, 0 if row['native_hold'] else 600, 1):
                arms = self.records()
                next(a for a in arms if a.get('label') == row['label'])['inwait']['hold_acknowledged'] = reply
                result = self.grade_of(arms)
                self.assertEqual(result['verdict'], 'FAIL', (row['label'], reply))
                self.assertIn(row['label'] + ':native-hold-not-acknowledged', result['failures'])

    def test_a_cap_a_rank_says_is_selected_fails_that_arm(self):
        arms = self.records()
        next(a for a in arms if a.get('label') == R0)['reserved_cap'] = dict(requested=False, select={str(k): dict(after=dict(reserved_cap=dict(cap_bytes=1, selected=True))) for k in range(4)})
        self.assertIn(R0 + ':reserved-cap-selected', self.grade_of(arms)['failures'])
        self.assertNotIn(W0 + ':reserved-cap-selected', self.grade_of(arms)['failures'])

    def test_a_missing_arm_is_inc(self):
        arms = [a for a in self.records() if a['label'] != R1]
        result = self.grade_of(arms)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn(R1 + ':no-arm-record', result['missing'])

    def test_either_off_arm_that_yielded_fails(self):
        for label in (W0, W1):          # coverage-throughput: nt has the yield on (lever 1); ny's two arms keep it off
            with self.subTest(label):
                arms = self.records()
                off = next(a for a in arms if a['label'] == label)
                off['statuses'][-1]['append_yield']['counters'] = dict(selected=9)
                off['statuses'][-1]['label'] = 'after'
                self.assertIn(label + ':yield-off-selected-a-yield', self.grade_of(arms)['failures'])


class Coverage(TwoPairs):
    """The operator's rule on the residue arms, from the engine's counter in the status rows (scheduler_hooks.PrefillCoverage)."""
    grade = Grade()

    def arms_with(self, label, *, stock=None, errors=0, drop=False, native_turns=None, max_rows=None, idle_stock=None):
        """Counts added inside this arm's window: its 'after' and every later status (the counter is cumulative)."""
        arms = self.grade.records()
        rec = next(a for a in arms if a.get('label') == label)
        later = [rec['statuses'][-1]] + [s for a in arms if a.get('label') in T.labels()[T.labels().index(label) + 1:] for s in a['statuses']]
        for status in later:
            cov = ((status.get('append_yield') or {}).get('declines') or {}).get('coverage')
            if cov is not None:
                cov['rows']['live']['stock'] = dict(stock or {})
                if idle_stock is not None:
                    cov['rows']['idle']['stock'] = dict(idle_stock)
                cov['errors'] += errors
                if native_turns is not None:
                    cov['native_turns'] = native_turns
                if max_rows is not None:
                    cov['handback_max_rows'] = max_rows
        if drop:
            rec['statuses'][0]['append_yield'].pop('declines')
        return arms

    def test_whole_native_passes_and_the_share_is_one(self):
        result = self.grade.grade_of(self.grade.records())
        self.assertEqual(result['verdict'], 'PASS', result['failures'])
        self.assertEqual(result['coverage'][R0]['live_native_share'], 1.0)
        self.assertEqual(result['coverage'][R0]['to'], 'next-entry')
        self.assertEqual(result['coverage'][R1]['to'], 'after')
        self.assertEqual(result['coverage'][R0]['rows']['live']['native'], {'plan': 1000})

    def test_a_stock_row_with_a_decoder_live_fails_a_residue_arm_by_site(self):
        result = self.grade.grade_of(self.arms_with(R0, stock={'no-seam': 3}))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn(R0 + ':stock-prefill-with-decoder-live', result['failures'])
        self.assertEqual(result['coverage'][R0]['rows']['live']['stock'], {'no-seam': 3})

    def test_the_one_row_handback_of_a_native_turn_is_native_served(self):
        result = self.grade.grade_of(self.arms_with(R1, stock={'handback': 12}, native_turns=12))
        self.assertEqual([f for f in result['failures'] if f.startswith(R1)], [])
        cov = result['coverage'][R1]
        self.assertEqual(cov['handback'], dict(turns=12, rows=12, excess_rows=0, requests=12, native_turns=12, max_rows_per_turn=1,
                                               every_row_after_a_native_turn=True))
        self.assertEqual((cov['live_native_share'], cov['all_native_share']), (1.0, 1.0))
        self.assertEqual(cov['live_native_share_strict'], round(900 / 912, 6))     # the last arm: entry to its after

    def test_a_handback_row_with_no_native_turn_before_it_fails(self):
        result = self.grade.grade_of(self.arms_with(R1, stock={'handback': 12}, native_turns=11))
        self.assertEqual([f for f in result['failures'] if f.startswith(R1)], [R1 + ':handback-not-one-row-of-a-native-turn'])

    def test_an_orphan_handback_fails_its_own_window_not_the_later_ones(self):
        result = self.grade.grade_of(self.arms_with(R0, stock={'handback': 2}, native_turns=1))
        self.assertIn(R0 + ':handback-not-one-row-of-a-native-turn', result['failures'])
        self.assertEqual([f for f in result['failures'] if f.startswith(R1)], [])

    def test_a_longer_handback_is_a_stock_row_and_fails(self):
        result = self.grade.grade_of(self.arms_with(R1, stock={'handback': 3, 'handback-excess': 2}, native_turns=3, max_rows=3))
        self.assertEqual([f for f in result['failures'] if f.startswith(R1)],
                         [R1 + ':handback-not-one-row-of-a-native-turn', R1 + ':stock-prefill-with-decoder-live'])
        self.assertEqual((result['coverage'][R1]['handback']['max_rows_per_turn'], result['coverage'][R1]['handback']['rows']), (3, 5))
        self.assertEqual(result['coverage'][R1]['live_native_share'], round(903 / 905, 6))

    def test_the_hold_valve_expired_fails_a_residue_arm(self):
        result = self.grade.grade_of(self.arms_with(R0, stock={'expired': 1}))
        self.assertEqual([f for f in result['failures'] if f.startswith(R0)], [R0 + ':stock-prefill-with-decoder-live'])

    def test_idle_stock_rows_fail_a_residue_arm_too_and_both_shares_are_given(self):
        result = self.grade.grade_of(self.arms_with(R0, idle_stock={'no-seam': 250, 'handback': 2}, native_turns=2))
        self.assertEqual([f for f in result['failures'] if f.startswith(R0)], [R0 + ':stock-prefill-idle'])
        self.assertEqual((result['coverage'][R0]['live_native_share'], result['coverage'][R0]['all_native_share']), (1.0, round(1002 / 1252, 6)))
        self.assertEqual(result['coverage'][R0]['all_native_share_strict'], round(1000 / 1252, 6))

    def test_the_hold_arm_is_read_not_failed(self):
        result = self.grade.grade_of(self.arms_with(W0, stock={'plan-shared': 500}))
        self.assertEqual(result['verdict'], 'PASS', result['failures'])
        self.assertEqual(result['coverage'][W0]['live_native_share'], round(1000 / 1500, 6))

    def test_counter_errors_or_no_counter_fail_any_replay_arm(self):
        self.assertIn(W1 + ':coverage-errors', self.grade.grade_of(self.arms_with(W1, errors=1))['failures'])
        self.assertIn(R0 + ':coverage-not-read', self.grade.grade_of(self.arms_with(R0, drop=True))['failures'])

    def test_a_residue_the_engine_did_not_acknowledge_fails(self):
        for row in T.arms():
            arms = self.grade.records()
            next(a for a in arms if a.get('label') == row['label'])['inwait']['residue_acknowledged'] = None
            self.assertIn(row['label'] + ':native-residue-not-acknowledged', self.grade.grade_of(arms)['failures'])

    def test_a_count_that_went_down_is_not_read(self):
        self.assertIsNone(box.coverage_delta(dict(append_yield=dict(declines=dict(coverage=coverage(10)))),
                                             dict(append_yield=dict(declines=dict(coverage=coverage(5))))))

    def test_the_status_row_keeps_the_decline_record(self):
        reply = dict(append_yield=dict(schema='s5-append-yield/1', declines=dict(coverage=coverage(7))), route='full')
        with box._bound():
            row = box4.status_row('entry', reply)
        self.assertEqual(row['append_yield']['declines']['coverage']['rows']['live']['native'], {'plan': 7})
        self.assertNotIn('declines', box4.status_row('entry', reply)['append_yield'], "compose-4's own stays as it is")


class Reads(UnderThisTable):
    def fixture(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Fixture5(tmp.name)

    def healthy(self, fx=None):
        fx = fx or self.fixture()
        tb.healthy(fx)
        tb.healthy(fx, NY)
        return fx

    def test_the_price_over_known_numbers(self):
        fx = self.healthy()
        fx.write_record(PAIRS[0][0], **frozen(bulk=23.0, decode=57.12))
        fx.write_record(PAIRS[0][1], **frozen(bulk=21.0, decode=56.0, selected=0))
        fx.write_record(PAIRS[1][0], **frozen(bulk=23.5, decode=56.0))
        fx.write_record(PAIRS[1][1], **frozen(bulk=21.5, decode=56.0, selected=0))
        _word, lines, facts = fx.done()
        price = facts['price']
        self.assertEqual([p['bulk_extra_s'] for p in price['pairs']], [2.0, 2.0])
        self.assertEqual([p['bulk_on_over_off'] for p in price['pairs']], [round(23 / 21, 4), round(23.5 / 21.5, 4)])
        self.assertEqual(price['pairs'][0]['decode_per_stream_on_over_off'], 1.02)
        self.assertEqual((price['on_on_bulk_spread_s'], price['off_off_bulk_spread_s']), (0.5, 0.5))
        self.assertEqual(price['bulk_ratio_spread'], round(abs(round(23 / 21, 4) - round(23.5 / 21.5, 4)), 4))
        self.assertEqual(sum(1 for line in lines if line.startswith('PRICE fon')), 2)
        self.assertIn('PRICE NOISE', '\n'.join(lines))

    def test_the_price_is_never_a_word(self):
        calm = self.healthy()
        word, _l, _f = calm.done()
        dear = self.healthy()
        dear.write_record(PAIRS[0][0], **frozen(bulk=90.0))
        dear.write_record(PAIRS[1][0], **frozen(bulk=90.0))
        self.assertEqual(dear.done()[0], word)

    def test_a_frozen_arm_with_no_record_reads_as_none_and_the_rule_says_inc_or_worse(self):
        fx = self.healthy()
        path = fx.root / ('arm-%s.json' % PAIRS[1][1])
        path.rename(path.with_suffix('.gone'))
        (fx.root / ('arm-%s.log' % PAIRS[1][1])).rename(fx.root / 'gone.log')
        word, lines, facts = fx.done()
        self.assertEqual(word, 'INC', lines[-3:])
        self.assertIsNone(facts['price']['pairs'][1]['off_read'])
        self.assertIn('not-run: ' + PAIRS[1][1], facts['reasons'])

    def test_r1_is_read_on_the_smoke_and_on_the_pair_and_is_not_in_the_verdict(self):
        fx = self.healthy()
        fx.write_record(SM, wall_start_s=tb.ORIGIN - 400, wall_end_s=tb.ORIGIN - 340, returners=[dict(suffix_tokens=500, ttft_s=1.5)], score_warm=dict(counters=dict(rewarms={'1': 7}, allocs={'1': 8})), reserved_cap=dict(counters=dict(overdrafts={'1': 3}, overdraft_refused={'1': 0})))
        word, lines, facts = fx.done()
        self.assertEqual((facts['r1'][SM]['rewarms'], facts['r1'][SM]['allocs'], facts['r1'][SM]['overdrafts']), (7, 8, 3))
        self.assertIn('SMOKE R1', '\n'.join(lines))
        clean = self.healthy().done()[0]
        self.assertEqual(word, clean, 'a read, not a verdict')

    def test_a_stock_rescue_on_cmp_is_inc_and_on_ny_is_nothing(self):
        fx = self.healthy()
        clean = fx.done()[0]
        self.assertEqual(clean, 'PASS')
        rescued = dict(stock_rescue=dict(counters=dict(attempts={'1': 2}, refused={'1': 0})))
        self.assertEqual(rule.stock_rescues(dict(record=rescued)), 2)
        self.assertEqual(rule.stock_rescues(dict(record={})), 0)
        path = fx.root / ('arm-%s.json' % CMP)
        record = json.loads(path.read_text())
        record.update(rescued)
        path.write_text(json.dumps(record))
        word, lines, facts = rule.report(str(fx.root))
        self.assertEqual((word, facts['stock_rescues'], lines[-1]), ('INC', 2, 'RULE INC'))
        self.assertIn('REASON stock-rescue: cmp 2', lines)

    def test_no_line_names_the_old_control_and_compose_4s_rule_is_put_back(self):
        _word, lines, _facts = self.healthy().done()
        self.assertFalse([line for line in lines if ' ctl' in line or 'ctl ' in line], lines)
        self.assertEqual((compose4_rule.OFF, compose4_rule.ARMS[-1]), ('ctl.g2-live-0', 'ctl.g2-live-0'))

    def test_it_runs_as_the_lead_runs_it_with_the_standard_library_alone(self):
        fx = self.healthy()
        fx.done()
        proc = subprocess.run([sys.executable, '-I', rule.__file__, str(fx.root)], capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:] + proc.stdout[-800:])
        self.assertIn('RULE PASS', proc.stdout)
        source = Path(rule.__file__).read_text()
        self.assertNotRegex(source, r'(?m)^(import|from) (compose5_arms|e3_|torch|numpy)')


if __name__ == '__main__':
    unittest.main()
