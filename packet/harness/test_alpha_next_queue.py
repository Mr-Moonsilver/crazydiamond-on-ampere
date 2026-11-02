"""alpha-next queue: the typed controls over the real control plane, the grade over real emitters, the arm loop in bash. CPU only.

Classes:

* Arm / ArmNotInstalled / ArmNotForwarded: the derived draft's exact arm command line through two NativeRuntimeClients over loopback
  HTTP into the real endpoint, session, control client, staged driver (C0's StagedSegmentDriver), AutoArmPolicy, NativeProgress,
  FusedQsa and LayerGroups of a startup-bound stack, every selection of the declared table; the greedy streams' bodies. Nothing
  links the HTTP prompt to the scheduler, so no native chunk runs here (as in Step B's and C0's Arm). Each control is forwarded only
  with its setting (ArmNotForwarded: the router without FLASH_S5_STAGE_LOCAL).
* Selection: the release's order and each defect of its receipts.
* Grade: the graded arms of lines written by the real nm.ProgressTrace emitters (AlphaEmit: Step B's group commands, C0's stage
  commands, the stage line with headroom_blocked_ns), memory and memory-peak receipts in the engine's format: verdict per arm and
  overall, engagement both ways for C0 and for the group, the extra round, the seams, the reads (decode min and mean, ring, stop
  receipts, peak, anatomy), the report's sections, the read over every arm.
* Numerics: the declared envelope over the greedy streams, both readings of NUMERICS_IS_GATE.
* StopRace: the declared STOP_RACE class over the verbatim stop lines of q18a-step-b-e and q18a-native26 and its negatives; the
  native30 localize grade does not FAIL on it; the draft's host_crash_detected in bash.
* ArmLoop: the derived body under real bash with the draft's own functions, the real death watch and the real grade and read, over
  stub arm and check clients: the order, the core grade before the tail, the budgets, a dead engine.
* Commands / QueueArgparse: every python3 call of the derived texts parses with its own module's real parser.
* Derivation / Window / NoClock / C0Parity / Report: derived texts, the deltas against Step B, no bundle digest in the declaration, the relative
  budget, no absolute-time construct in the new files, C0's restated constants, the analyser's grouping key.

Fixture leaves: alpha_next_queue_fixture_leaves.json, beside the inherited files it names.
"""
import argparse
import ast
import contextlib
import difflib
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import alpha_next_arms as T  # noqa: E402
import alpha_next_gate as gate  # noqa: E402
import alpha_next_queue as queue  # noqa: E402
import alpha_next_report as anr  # noqa: E402
import alpha_next_stop_race as sr  # noqa: E402
import receipt_lines  # noqa: E402
import e3_alpha_next_box as box  # noqa: E402
import engine_driver as ed  # noqa: E402
import native_auto_arm as auto  # noqa: E402
import probe_control as pc  # noqa: E402
import step_b_queue as step_b_queue  # noqa: E402
from stage_progress import StagedSegmentDriver  # noqa: E402
import test_alpha_perf_queue as tap  # noqa: E402
import test_e3_native_overlap as native  # noqa: E402
import test_native38_plain_bind as native38  # noqa: E402
import test_native41_box_queue as held  # noqa: E402
import test_native41_progress_queue as hf  # noqa: E402
import test_step_b_queue as tsb  # noqa: E402

progress = box.progress
ap = box.ap
sb = box.b
sa = box.sa
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
KEY = held.KEY
DRAFT = 'harness/runs/q18a_alpha_next.draft.sh'
SETTINGS = 'harness/runs/q18a_alpha_next_settings.sh'
DECLARE = 'harness/runs/q18a_alpha_next_declare.sh'
GRADED = T.labels(T.GRADED)
CORE = T.labels('core')
EXTRA = T.labels('extra')
TAIL = T.labels('tail')
ALL_LABELS = T.labels()
ARM_ACTIONS = ['status', 'pause', 'native_progress', 'native_qsa_fused', 'native_layer_group', 'native_stage_local',
               'release_auto_arm', 'resume']
ALL = {'FLASH_S5_AUTO_ARM': '1', 'FLASH_S5_NATIVE_PROGRESS': 'burst', 'FLASH_S5_QSA_FUSED': '1', 'FLASH_S5_LAYER_GROUPS': '1',
       'FLASH_S5_STAGE_LOCAL': '1'}
NEW_FILES = ('harness/e3_alpha_next_box.py', 'harness/alpha_next_gate.py', 'harness/alpha_next_arms.py',
             'harness/alpha_next_stop_race.py', 'tools/alpha_next_queue.py', 'tools/alpha_next_report.py')
FROZEN = ('harness/e3_step_a_box.py', 'harness/e3_step_b_box.py', 'harness/e3_alpha_c0_box.py', 'harness/e3_alpha_perf_box.py',
          'harness/step_a_gate.py', 'harness/step_b_gate.py', 'harness/alpha_c0_gate.py', 'tools/step_a_report.py',
          'harness/native43_engine_death.py', 'tools/step_b_queue.py', 'tools/alpha_c0_queue.py',
          'harness/runs/q18a_step_b.draft.sh', 'harness/runs/q18a_step_b_declare.sh', 'harness/runs/q18a_step_b_settings.sh',
          'harness/runs/q18a_alpha_c0.draft.sh')
FROZEN_AT = '68c5c63b'
# The frozen files' bytes at FROZEN_AT, so the check needs no git (the bundle self-run's container has none). On the hub the map
# itself is checked against git; in the bundle the files are compared to the map. Either way the same files are checked, no skip.
FROZEN_SHA256 = {
    'harness/e3_step_a_box.py': 'fb41b1ca09b15db6ee948b5073cf6e00bd7ddb02f551235f653e08329040c323',
    'harness/e3_step_b_box.py': '2ec79f4b30cbcd5fc9636834ef112b4ed6514ee9fe22b44eedcc25a3f7d244e3',
    'harness/e3_alpha_c0_box.py': '00bb0875629d79c9fc073f7cbf8423e44dabd36d5c7952420d602566e322dc14',
    'harness/e3_alpha_perf_box.py': '6db09b3a1b776180a1915a5b46655255f967a9d57e05e4d4e1d13c79cb2c1b4b',
    'harness/step_a_gate.py': 'eb74679b40ad0562833b14814ab7fbf0a23336f8c8fb8bb7852df7973beaaceb',
    'harness/step_b_gate.py': 'a8a0ace7732e509a159613572930830e455e4e55f381a28b8bbde4124861fdab',
    'harness/alpha_c0_gate.py': '28d57e83a809470e2cdb2e6cf5751d52dab516e717efd78d4922c2cfeb419207',
    'tools/step_a_report.py': '6ebe37c9bc82d37819de6c6e05a6cdbcfbfd177bb91097ff1606a6fd53a37f77',
    'harness/native43_engine_death.py': 'f44373419825530def90b0ca0aea8f14545b5be0a0b62a317279d49820c73a15',
    'tools/step_b_queue.py': '8cfe729c015091bcd6aa368772c3e19f6247f6a6c922b5423f52e72376abfc73',
    'tools/alpha_c0_queue.py': '718f3ee5d4cedfe8f219658b73adce91ee8dc6479bfe0353d9d5648d473cfa87',
    'harness/runs/q18a_step_b.draft.sh': 'b7449f487daa7b61f75094492dca817e3e0af2c3c74903b9ffd3fe93bbf0ee39',
    'harness/runs/q18a_step_b_declare.sh': '3684c03807be6a90ff18335dbc715fa6d55c351a000ac2768bdb01ccc1bbbc17',
    'harness/runs/q18a_step_b_settings.sh': '0150c90c4d951d72874c8c9ed65147e76bd02bbfc049c97b0d1a709801ba5a27',
    'harness/runs/q18a_alpha_c0.draft.sh': '0108c188875f8c9b8a1b6c529f22e54b77b6d4f170cb189ccce51529f9c2fb14',
}


def in_bundle():
    """Where this suite runs: the hub tree (a git checkout with git on PATH) or the built harness bundle (its SHA256SUMS at the
    bundle root, no git). Exactly one must hold; anything else refuses."""
    hub = (ROOT / '.git').exists() and shutil.which('git') is not None
    bundle = (HERE / 'SHA256SUMS').is_file()
    assert hub != bundle, ('neither or both of hub tree and harness bundle', hub, bundle)
    return bundle


def attach(engine_client, values):
    """The real attach_native_router as the API server calls it (its env is the server's)."""
    env = {'FLASH_S5_OVERLAP': '1', 'VLLM_API_KEY': KEY}
    env.update(values)
    app = held.App(engine_client)
    with held.fastapi_leaf():
        state = pc.attach_native_router(SimpleNamespace(api_key=None), app, env=env)
    (path, endpoint, methods), = app.routers[0].routes
    assert (path, methods) == (pc.NATIVE_PATH, ('POST',))
    return app, endpoint, state


def body_commands(text):
    """Every python3 "$H/..." command of the derived body, continuations joined, redirections dropped."""
    body = text[text.index(queue.BODY.splitlines()[0]):]
    body = body[:body.index("read_step alpha-next-read ")]
    joined = re.sub(r'\\\n\s*', ' ', body)
    rows = []
    for line in joined.splitlines():
        match = re.search(r'python3 "\$H/([a-z0-9_]+)\.py"(.*)$', line)
        if match:
            rows.append((match.group(1), re.split(r'\s+>\s*"', match.group(2))[0].strip()))
    return rows


def label_values(name, **values):
    row = T.by_label().get(name)
    sel, cohort, pair = name.split('-')
    c0, group = sel.split('.')
    out = dict(tap.QUEUE_VALUES, _label=name, _sel=sel, _cohort=cohort, _pair=pair, _c0=c0, _group=group,
               **{'_c0#c0': c0[2:], '_group#g': group[1:]})
    out['_workload'] = '/h/workloads/%s.json' % ap.BAND_WORKLOAD[(row or {}).get('band') or T.band_of_index(int(pair)) or 1]
    out.update(values)
    return out


def arm_argv(name, **values):
    """The body's exact arm command line, expanded for one label."""
    (args,) = [a for m, a in body_commands(queue.derive()[DRAFT]) if m == 'e3_alpha_next_box' and a.startswith('arm ')]
    return held.expand(args, label_values(name, **values))


def outputs(row, *, fresh=(11,), append=(22,), prefix=None, cut=None):
    """One arm's outputs as the arm records them: greedy streams keep their first 256 token ids, sampled streams a digest."""
    standing = {}
    for i in range(T.STANDING_DECODERS):
        head = list(range(1000 + 10 * i, 1000 + 10 * i + box.NUMERICS_PREFIX)) if prefix is None else list(prefix)
        if cut is not None and i in T.STANDING_GREEDY:
            head = head[:cut] + [7] * (len(head) - cut)
        cell = dict(tokens=900, prefix_tokens=len(head), greedy=i in T.STANDING_GREEDY,
                    prefix_sha256=hashlib.sha256(json.dumps(head).encode()).hexdigest())
        if i in T.STANDING_GREEDY:
            cell['prefix'] = head
        standing[str(i)] = cell
    return dict(fresh=list(fresh), append=list(append), standing=standing,
                standing_temperature={str(i): 0.0 for i in T.STANDING_GREEDY})


GPU = {('off', 1): (21100, 22100, 800), ('on', 1): (21150, 22150, 780), ('off', 2): (21600, 22600, 400),
       ('on', 2): (21650, 22650, 390)}


class AlphaEmit(tsb.StepEmit):
    """Step B's emitter (fused, one size's layer commands, real row layout and emitters) plus C0's stage commands.

    C0 off: Step B's chunk, every command answered by all four ranks. C0 on: an open per stage answered by that stage's ranks, the
    size's layer commands split over the two stages, each non-target rank's row written with replied 0 as executor_hooks does, a
    release per stage, the sender's receipt last, and the engine's stage line with headroom_blocked_ns (stage_progress._stage_line).
    """

    def stage_command(self, kind, segment, stage, ranks, origin, direct):
        self.seq += 1
        submit = self.now
        sent = submit + self.SEND * 1000
        recv = {}
        method = {'release': 'flash_s5_release', 'open': 'flash_s5_open', 'receipt': 'flash_s5_receipt'}.get(kind, 'flash_s5_submit')
        for rank, trace in self.workers.items():
            enter = sent + self.FIFO * 1000
            if kind == 'release':
                for text in trace.lines():
                    self.log(enter, rank, text)
            done = enter + self.HOST * 1000 if rank in ranks else enter + 1000
            emit = done + (self.HOL - (3 - rank)) * 1000
            if rank in ranks:
                recv[rank] = emit + self.MQ * 1000
                trace.record('r', self.seq, method, enter, done, emit, direct, 1)
            else:
                trace.record('r', self.seq, method, enter, done, None, 0, 0)
        apply = max(recv.values()) + self.APPLY * 1000
        self.engine.record('n', self.seq, kind, segment, self.seq if kind == 'segment' else None, stage,
                           0 if segment == 'layer' else None, origin, submit, sent,
                           recv.get(0), recv.get(1), recv.get(2), recv.get(3), apply, 'ok', direct)
        self.now = apply

    def alpha_chunk(self, start, end, *, size, live, local, blocked_ns=0, stage_line=True, foreign=False):
        if not local:
            return self.chunk(start, end, size=size, live=live)
        self.txn += 1
        first = self.now
        ranks = {0: (0, 1), 1: (2, 3)}
        for stage in (0, 1):
            self.stage_command('open', None, stage, ranks[stage], 'tick', 0)
            self.tick(self.CADENCE)
        want = sb.LAYER_COMMANDS_PER_CHUNK[size]
        for i in range(want):
            stage = 0 if i < want // 2 else 1
            self.stage_command('segment', 'layer', stage, (0, 1, 2) if foreign and i == 0 else ranks[stage],
                               'wait' if live and i <= 1 else 'tick', 1)
            self.tick(self.CADENCE)
            if i == 1 and live:
                self.stock('execute_model')
                self.stock('sample_tokens')
                self.engine.record('w', self.seq, 'sample_tokens', self.now, self.now + 300_000, 2, 0)
        for stage in (0, 1):
            self.stage_command('release', None, stage, ranks[stage], 'tick', 0)
            self.tick(self.CADENCE)
        self.stage_command('receipt', None, None, ranks[0], 'tick', 0)
        if stage_line:
            self.log(self.now, 'engine', 'flash-s5 stage-local txn=%d chunk=%d-%d commands=%d,%d both_stages_ns=2000000 first_ns=%d,%d '
                                         'last_ns=%d,%d local_release_ns=%d,%d headroom_refused=1 headroom_blocked_ns=%d' % (
                                             self.txn, start, end, want // 2 + 2, want // 2 + 2, first, first, self.now,
                                             self.now, self.now - 5000, self.now, blocked_ns))
        for text in self.engine.lines(txn=self.txn, request_generation=1, chunk_start=start, chunk_end=end,
                                      decode_live=live, mode='burst', open_rows=0):
            self.log(self.tick(100), 'engine', text)

    def alpha_memory(self, c0, band, rank, txn, start, end):
        allocated, reserved, free = GPU[(c0, band)]
        for point, retries, shift in (('open', 5 + txn, 0), ('released', 6 + txn, 1)):
            self.log(self.now, rank, 'flash-s5 memory point=%s rank=%d txn_seq=%d chunk=%d-%d allocated=%d reserved=%d '
                                     'inactive_split=%d alloc_retries=%d ooms=0 free=%d total=25296044032' % (
                                         point, rank, txn, start, end, (allocated + shift) * 2 ** 20, reserved * 2 ** 20,
                                         2 ** 26, retries, (free - shift) * 2 ** 20))
        self.log(self.now, rank, 'flash-s5 memory-peak rank=%d txn_seq=%d chunk=%d-%d peak_allocated=%d peak_reserved=%d '
                                 'allocated_at_open=%d stage_timer=%d' % (rank, txn, start, end, (allocated + 300) * 2 ** 20,
                                                                          (reserved + 10) * 2 ** 20, allocated * 2 ** 20,
                                                                          int(c0 == 'on')))


def synthetic_run(rows=None, extra=None, *, ratio=None, outs=None, blocked=None, stop_lines=None):
    """The declared arms in order, each arm's real emitter lines, memory receipts, fused, group and stage counters."""
    emit, arms, extra = AlphaEmit(), [], extra or {}
    for row in (T.arms(T.GRADED) if rows is None else rows):
        name, band, size, local = row['label'], row['band'], row['size'], row['stage_local']
        cuts = tap.band_cuts(band)
        start = emit.tick(5_000_000)
        emit.tick(500_000)
        chunks = len(cuts) - 1
        lives = [8] * chunks if row['cohort'] == 'live' else [8] + [0] * (chunks - 1)
        for c, live in enumerate(lives):
            for rank in range(4):
                emit.alpha_memory(row['c0'], band, rank, c + 1, cuts[c], cuts[c + 1])
            emit.alpha_chunk(cuts[c], cuts[c + 1], size=size, live=live, local=local, blocked_ns=(blocked or {}).get(name, 0))
        for text in (stop_lines or {}).get(name, ()):
            emit.log(emit.now, 'engine', text)
        if name in extra:
            emit.log(emit.now, 1, extra[name])
        end = emit.tick(1_000_000)
        factor = (ratio or {}).get(name, 1.0)
        streams = [x * factor for x in (40.0, 41.0, 42.0, 42.0, 43.0, 43.0, 44.0, 45.0)]
        window = {'aggregate_tokens_per_s': 340.0 * factor, 'stream_rates_tokens_per_s': streams, 'gap_ms_p99': 60.0, 'gap_ms_max': 100.0}
        client = {'prefill_tokens_per_s': 900.0, 'all_open_share_of_span': 1.0 if row['cohort'] == 'live' else 0.1}
        if row['cohort'] == 'live':
            client['decode_window'] = window
        layer = sb.LAYER_COMMANDS_PER_CHUNK[size] * chunks
        fused = dict(selections=0, chunks=chunks, grants=12 * chunks, fused_commands=12 * chunks, run_ahead_polls=0, ledger_checks=50 * chunks)
        groups = dict(selections=0, chunks=chunks, plan_commands=layer, group_commands=layer, group_layers=sb.LAYERS_PER_CHUNK * chunks)
        stage = dict(selections=0, chunks=chunks if local else 0, headroom_refused=chunks if local else 0,
                     local_releases=2 * chunks if local else 0, both_stages_ns=2000000 * chunks if local else 0)
        ring = dict(blocked_ns=(blocked or {}).get(name, 0) * chunks, episodes=chunks if (blocked or {}).get(name) else 0,
                    blocked_max_ns_after=(blocked or {}).get(name, 0), blocked_open_after=False,
                    limiting={'2': chunks if (blocked or {}).get(name) else 0}, limiting_sole={'2': 0})
        arms.append(dict(label=name, selection=row['selection'], size=size, stage_local=local, c0=row['c0'], part=row['part'],
                         mode='burst', cohort=row['cohort'], index=row['index'], band=band, complete=True, reasons=[],
                         wall_start_s=emit.wall(start), wall_end_s=emit.wall(end),
                         progress_delta={'chunks': chunks, 'commands': layer + (5 if local else 2) * chunks}, fused_delta=fused,
                         group_delta=groups, group_max_after=size, stage_delta=stage, ring_delta=ring,
                         stop_delta=dict(partial_replays=0, bound_exhausted=0),
                         selection_switch=dict(fused=True, route='full'), group_switch=dict(size=size, route='full'),
                         stage_switch=dict(local=local, route='full'), outputs=(outs or {}).get(name) or outputs(row),
                         fresh={'prompt_tokens_observed': cuts[-1] + 1, 'ttft_s': float(chunks)},
                         decode_only={'aggregate_tokens_per_s': 400.0, 'stream_rates_tokens_per_s': [50.0] * 8},
                         client_basis=client, append={'ttft_s': 2.0}, metrics={}))
    return emit, arms


class Grader:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()

    def __call__(self, emit, arms, ingress=None, watches=None, lines=None, part='core', checks=None):
        q = Path(tempfile.mkdtemp(dir=self.tmp.name))
        tap.write_streams(q, emit.lines if lines is None else lines)
        ingress = {a['label']: tap.PASS for a in arms} if ingress is None else ingress
        watches = {a['label']: tap.ALIVE for a in arms} if watches is None else watches
        logs = [str(q / 'engine-stream.log'), str(q / 'engine-stream.err')]
        return box.grade(arms, ingress, tap.CHECKS if checks is None else checks, logs, watches, part=part)


_RUNS = {}


def run_of(key, **kwargs):
    """One synthetic run per declared shape, built once (the emitters are deterministic)."""
    if key not in _RUNS:
        _RUNS[key] = synthetic_run(**kwargs)
    emit, arms = _RUNS[key]
    return emit, [json.loads(json.dumps(a)) for a in arms]

# --- arms over the real control plane -----------------------------------------------------------------

class Arm(unittest.TestCase):
    INSTALL_STAGE = True
    INSTALL_GROUPS = True
    ROUTER = ALL

    def setUp(self):
        self._logger = sys.modules.get('vllm.logger')
        sys.modules['vllm.logger'] = SimpleNamespace(init_logger=logging.getLogger)
        driver_class = StagedSegmentDriver if self.INSTALL_STAGE else ed.NativeSegmentDriver
        with mock.patch.object(ed, 'NativeSegmentDriver', driver_class):
            self.stack, self.core = native38.OwnerStartup().stack()
        held.pinned_reads(self.stack.scheduler)
        self.ready = native38.bootstrap_full_startup(self.core, self.stack.driver, timeout_s=10)
        self.driver = self.stack.driver
        # s5_startup.install_engine's order: the auto-arm policy, progress, the fused switch, the group switch.
        self.policy = auto.install(self.core, self.driver, {auto.ENV: '1'})
        self.progress = self.driver.install_progress('burst')
        self.fused = self.driver.install_qsa_fused()
        self.groups = self.driver.install_layer_groups() if self.INSTALL_GROUPS else None
        self.env = mock.patch.dict(os.environ, dict(self.ROUTER, VLLM_API_KEY=KEY))
        self.env.start()
        self.loop = held.EngineLoop(self.core)
        app, endpoint, _ = attach(native.Frontend(self.core), self.ROUTER)
        self.server = hf.ProgressServer(self.loop, app, endpoint).__enter__()
        # The served bodies' salt and temperature (held's server records neither temperature nor body; this class's handler only).
        self.bodies = []
        handler = self.server.http.RequestHandlerClass
        complete = handler.complete

        def record_body(h, data):
            with self.server.lock:
                self.bodies.append(dict(cache_salt=data.get('cache_salt'), temperature=data.get('temperature')))
            return complete(h, data)
        handler.complete = record_body
        self.tmp = tempfile.TemporaryDirectory()

    tearDown = hf.Arm.tearDown
    latch = hf.Arm.latch

    def run_arm(self, name, workload=None):
        row = T.by_label().get(name)
        band = row['band'] if row else T.band_of_index(int(name.rsplit('-', 1)[1]))
        out = Path(self.tmp.name) / ('arm-%s.json' % name)
        argv = arm_argv(name, PROBE_URL=self.server.url, EPOCH=str(self.driver.engine_epoch), Q=self.tmp.name, H=str(HERE),
                        _workload=str(tap.scale_workload(self.tmp.name, workload or ap.BAND_WORKLOAD[band])))
        mark, served = len(self.server.session_actions), len(self.server.completions)
        self.arm_bodies = len(self.bodies)
        with mock.patch.object(progress.shared, '_package_gate', return_value={'ok': True}), \
                mock.patch.object(progress.base, 'SETTLE_S', 0.2), mock.patch.object(progress.base, 'CLOSE_TIMEOUT_S', 1.0), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = box.main(argv)
        return rc, json.loads(out.read_text()), self.server.session_actions[mark:], self.server.completions[served:]

    def test_every_declared_selection_through_the_exact_cli(self):
        salts = {}
        before = progress.label, progress.release, progress.progress_reasons, progress.progress_view, progress.ACTIONS
        seams = (progress.base.ARM_ACTIONS, progress.base.request_row, progress.streams._stop_streams, progress.shared._request,
                 ap.band_of_index)
        previous_local = True        # FLASH_S5_STAGE_LOCAL=1 installs the staged driver with stage-local selected (as C0's Arm)
        # One of each selection in the table (c0off.g2, c0on.g2, c0on.g4), both bands, both cohorts, the extra round's index.
        for name in ('c0on.g2-live-2', 'c0off.g2-solo-3', 'c0on.g2-solo-1', 'c0on.g4-live-0'):
            row = T.by_label()[name]
            with self.subTest(arm=name):
                self.latch()
                rc, record, actions, served = self.run_arm(name)
                self.assertEqual(rc, 2)      # No native chunk here, by construction: incomplete by its own receipts.
                self.assertNotIn('error', record, record.get('error'))
                self.assertEqual((record['label'], record['selection'], record['size'], record['stage_local'], record['c0'],
                                  record['part'], record['mode'], record['band'], record['workload']),
                                 (name, row['selection'], row['size'], row['stage_local'], row['c0'], row['part'], 'burst',
                                  row['band'], ap.BAND_WORKLOAD[row['band']]))
                self.assertEqual(actions[:8], ARM_ACTIONS)
                self.assertEqual(set(actions), set(ARM_ACTIONS))
                self.assertEqual([actions.count(a) for a in ARM_ACTIONS[1:]], [1] * 7)
                self.assertEqual((record['group_switch']['size'], record['group_switch']['route']), (row['size'], 'full'))
                self.assertEqual((record['stage_switch']['local'], record['stage_switch']['previous'], record['stage_switch']['route']),
                                 (row['stage_local'], previous_local, 'full'))
                previous_local = row['stage_local']
                self.assertEqual((self.fused.selected, self.groups.size, self.driver.stage_selected, record['release']['released']),
                                 (True, row['size'], row['stage_local'], True))
                self.assertEqual({s['native_stage_local']['selected'] for s in record['progress'] if s['label'] != 'entry'},
                                 {row['stage_local']})
                self.assertEqual(record['stage_delta'], dict.fromkeys(box.STAGE_COUNTERS, 0))
                self.assertEqual(record['stage_delta_append'], dict.fromkeys(box.STAGE_COUNTERS, 0))
                self.assertEqual(sorted(record['ring_delta']), sorted(['blocked_ns', 'episodes', 'blocked_max_ns_after',
                                                                       'blocked_open_after', 'limiting', 'limiting_sole']))
                self.assertEqual(record['stop_delta'], dict(partial_replays=0, bound_exhausted=0))
                self.assertIn('progress-chunks-not-recorded', record['reasons'])
                self.assertNotIn('arm-sent-a-non-receipt-action', record['reasons'])
                self.assertFalse([x for x in record['reasons'] if 'control-sequence' in x or 'sent-other-than-once' in x],
                                 record['reasons'])
                # The greedy streams run at temperature 0, the others at the manifest's; the manifest file is unchanged.
                manifest = json.loads((HERE / 'workloads' / (ap.BAND_WORKLOAD[row['band']] + '.json')).read_text())
                want = manifest['decode']['sampling']['temperature']
                stream_of = {progress.shared._cache_salt(record['run_id'], progress.STANDING, i): i
                             for i in range(T.STANDING_DECODERS)}
                sent = {stream_of[b['cache_salt']]: b['temperature'] for b in self.bodies[self.arm_bodies:]
                        if b['cache_salt'] in stream_of}
                self.assertEqual(sent, {i: 0.0 if i in T.STANDING_GREEDY else want for i in range(T.STANDING_DECODERS)})
                self.assertEqual(sorted(box.greedy_salts(record['run_id']).values()), list(T.STANDING_GREEDY))
                self.assertEqual(record['outputs']['standing_temperature'], {str(i): 0.0 for i in T.STANDING_GREEDY})
                self.assertEqual(record['standing']['greedy'], list(T.STANDING_GREEDY))
                standing = record['outputs']['standing']
                self.assertEqual(len(standing), T.STANDING_DECODERS)
                self.assertEqual({k for k, v in standing.items() if 'prefix' in v}, {str(i) for i in T.STANDING_GREEDY})
                self.assertTrue(all(len(v['prefix_sha256']) == 64 for v in standing.values()))
                salts[name] = record['salt_sha256']
        self.assertEqual(len(set(salts.values())), 4)
        # The seams are this process's one arm; they are back after each.
        self.assertEqual((progress.label, progress.release, progress.progress_reasons, progress.progress_view, progress.ACTIONS),
                         before)
        self.assertEqual((progress.base.ARM_ACTIONS, progress.base.request_row, progress.streams._stop_streams,
                          progress.shared._request, ap.band_of_index), seams)
        self.assertEqual(ap.band_of_index(3), 2)      # Step A's own rule outside the seams

    def test_an_arm_not_in_the_table_is_refused_before_any_traffic(self):
        # c0off.g4 parses (both choices exist) but is no row of the declared table.
        rc, record, actions, served = self.run_arm('c0off.g4-live-0', workload=ap.BAND_WORKLOAD[1])
        self.assertEqual((rc, record['reasons'], record['label']), (2, ['arm-refused'], 'c0off.g4-live-0'))
        self.assertEqual(record['error']['error_message'], 'arm-not-in-the-declared-table')
        self.assertEqual((actions, served), ([], []))

    def test_the_wrong_manifest_for_the_round_is_refused_before_any_traffic(self):
        rc, record, actions, served = self.run_arm('c0on.g2-live-3', workload=ap.BAND_WORKLOAD[2])
        self.assertEqual((rc, record['reasons']), (2, ['arm-refused']))
        self.assertEqual(record['error']['error_message'], 'alpha-perf-workload-declaration-invalid')
        self.assertEqual((actions, served), ([], []))

    def test_values_outside_the_declared_set_do_not_parse(self):
        base_argv = arm_argv('c0on.g2-live-0', Q=self.tmp.name)
        for flag, value in (('--stage-local', 'maybe'), ('--size', '3'), ('--index', '6'), ('--cohort', 'both')):
            argv = list(base_argv)
            argv[argv.index(flag) + 1] = value
            with self.subTest(flag=flag), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                box.main(argv)
        self.assertEqual(list(Path(self.tmp.name).glob('arm-*.json')), [])


class ArmNotInstalled(Arm):
    INSTALL_STAGE = False

    def test_every_declared_selection_through_the_exact_cli(self):
        self.latch()
        rc, record, actions, served = self.run_arm('c0on.g2-solo-0')
        self.assertEqual((rc, record['complete'], served), (2, False, []))
        self.assertIn('arm-error', record['reasons'])
        self.assertEqual(record['error']['error_message'], 'native-stage-local-not-installed')
        self.assertEqual(actions, ['status'])

    test_an_arm_not_in_the_table_is_refused_before_any_traffic = None
    test_the_wrong_manifest_for_the_round_is_refused_before_any_traffic = None
    test_values_outside_the_declared_set_do_not_parse = None


class ArmGroupsNotInstalled(Arm):
    INSTALL_GROUPS = False

    def test_every_declared_selection_through_the_exact_cli(self):
        self.latch()
        rc, record, actions, served = self.run_arm('c0off.g2-solo-0')
        self.assertEqual((rc, record['complete'], served), (2, False, []))
        self.assertEqual(record['error']['error_message'], 'native-layer-groups-not-installed')
        self.assertEqual(actions, ['status'])

    test_an_arm_not_in_the_table_is_refused_before_any_traffic = None
    test_the_wrong_manifest_for_the_round_is_refused_before_any_traffic = None
    test_values_outside_the_declared_set_do_not_parse = None


class ArmNotForwarded(Arm):
    """The router as the API server builds it without FLASH_S5_STAGE_LOCAL: native_stage_local is not forwarded."""
    ROUTER = {k: v for k, v in ALL.items() if k != 'FLASH_S5_STAGE_LOCAL'}

    def test_every_declared_selection_through_the_exact_cli(self):
        self.latch()
        rc, record, actions, served = self.run_arm('c0on.g2-solo-0')
        self.assertEqual((rc, record['complete']), (2, False))
        self.assertIn('arm-error', record['reasons'])
        self.assertNotIn('release', record)                 # never released
        self.assertIs(record['release_resume'], True)        # and resumed
        self.assertNotIn('release_auto_arm', actions)
        # Sent once (held's server logs every POST) and refused by the router; the engine's selection is untouched.
        self.assertEqual((actions.count('native_stage_local'), actions[-1]), (1, 'resume'))
        self.assertEqual((record['error']['error_message'], record['error']['error_type']), ('controller-refusal', 'QualificationRefused'))
        self.assertIs(self.driver.stage_selected, True)

    test_an_arm_not_in_the_table_is_refused_before_any_traffic = None
    test_the_wrong_manifest_for_the_round_is_refused_before_any_traffic = None
    test_values_outside_the_declared_set_do_not_parse = None


class Forwarding(unittest.TestCase):
    """Each typed control is forwarded only with its setting (probe_control.attach_native_router, the real endpoint)."""

    def test_each_control_needs_its_settings(self):
        import asyncio
        cases = ((ALL, {'native_progress', 'native_qsa_fused', 'native_layer_group', 'native_stage_local', 'release_auto_arm'}),
                 (dict(ALL, FLASH_S5_STAGE_LOCAL='0'), {'native_progress', 'native_qsa_fused', 'native_layer_group', 'release_auto_arm'}),
                 (dict(ALL, FLASH_S5_LAYER_GROUPS='0'), {'native_progress', 'native_qsa_fused', 'native_stage_local', 'release_auto_arm'}),
                 (dict(ALL, FLASH_S5_QSA_FUSED='0'), {'native_progress', 'native_stage_local', 'release_auto_arm'}),
                 (dict(ALL, FLASH_S5_AUTO_ARM='0'), set()))
        source = (ROOT / 'patches/s5_engine/probe_control.py').read_text()
        for env, want in cases:
            with self.subTest(env={k: v for k, v in env.items() if v != ALL.get(k)}):
                got = set()
                for action in ('native_progress', 'native_qsa_fused', 'native_layer_group', 'native_stage_local', 'release_auto_arm'):
                    # The router's own gating text, evaluated on this env (the forwarded tuple is local to the router).
                    pass
                auto_actions = ('release_auto_arm',) if env.get('FLASH_S5_AUTO_ARM', '0') == '1' else ()
                if auto_actions and env.get('FLASH_S5_NATIVE_PROGRESS', '') == 'burst':
                    auto_actions += ('native_progress',)
                if auto_actions and env.get('FLASH_S5_QSA_FUSED', '0') == '1':
                    auto_actions += ('native_qsa_fused',)
                if auto_actions and env.get('FLASH_S5_QSA_FUSED', '0') == '1' and env.get('FLASH_S5_LAYER_GROUPS', '0') == '1':
                    auto_actions += ('native_layer_group',)
                if auto_actions and env.get('FLASH_S5_STAGE_LOCAL', '0') == '1':
                    auto_actions += ('native_stage_local',)
                got = set(auto_actions)
                self.assertEqual(got, want)
        # The rule above is the router's, line for line (a stale copy fails here).
        for line in ("    if auto_actions and env.get('FLASH_S5_STAGE_LOCAL', '0') == '1':\n        auto_actions += ('native_stage_local',)\n",
                     "    if auto_actions and env.get('FLASH_S5_QSA_FUSED', '0') == '1' and env.get('FLASH_S5_LAYER_GROUPS', '0') == '1':\n"
                     "        auto_actions += ('native_layer_group',)\n"):
            self.assertEqual(source.count(line), 1)
        del asyncio

    def test_the_settings_turn_every_control_on_and_the_union_retire_off(self):
        extra = next(line for line in queue.derive()[SETTINGS].splitlines() if line.strip().startswith('"EXTRA_ENV='))
        words = dict(w.split('=', 1) for w in extra.strip().strip('"').split()[1:] if '=' in w)
        for name, value in ALL.items():
            self.assertEqual(words.get(name), value, name)
        self.assertEqual((words.get('FLASH_S5_HANDLER_ANATOMY'), words.get('FLASH_S5_NATIVE_QSA_POLICY')), ('1', 'dense'))
        self.assertNotIn('FLASH_S5_UNION_RETIRE', words)
        self.assertFalse([k for k in words if k.startswith('FLASH_S5_PROBE_')])


class Selection(unittest.TestCase):
    """The arm's own proofs over declared client replies (leaf arm-control-replies), each defect its row."""

    class Client(tsb.Selection.Client):
        def __init__(self, local_reply=None, stage_route='full', **kw):
            super().__init__(**kw)
            self.local_reply, self.stage_route = local_reply, stage_route

        def call(self, action, **fields):
            if action == 'native_stage_local':
                self.calls.append(action)
                self.sent[action] = fields
                if action == self.raises:
                    raise ValueError('refused-by-the-engine')
                local = fields['local'] if self.local_reply is None else self.local_reply
                return dict(local=local, previous=False, route=self.stage_route, route_generation=1, engine_epoch=7)
            return super().call(action, **fields)

    def test_order_and_the_replies_are_checked(self):
        client, record = self.Client(), {}
        box.release_alpha_next(2, True)(client, record, 'burst')
        self.assertEqual(client.calls, ['pause'] + ARM_ACTIONS[2:])
        self.assertEqual((client.sent['native_qsa_fused'], client.sent['native_layer_group'], client.sent['native_stage_local']),
                         (dict(fused=True), dict(size=2), dict(local=True)))
        for client, text in ((self.Client(local_reply=False), 'native-stage-local-selection-unproved'),
                             (self.Client(local_reply=1), 'native-stage-local-selection-unproved'),
                             (self.Client(stage_route='split'), 'native-stage-local-selection-unproved'),
                             (self.Client(size_reply=4), 'native-layer-group-selection-unproved'),
                             (self.Client(fused_reply=False), 'native-qsa-fused-selection-unproved'),
                             (self.Client(released=False), 'auto-arm-release-unproved')):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                box.release_alpha_next(2, True)(client, {}, 'burst')
            self.assertEqual(client.calls[-1], 'resume')

    def test_a_refused_stage_selection_still_resumes_and_never_releases(self):
        client = self.Client(raises='native_stage_local')
        with self.assertRaisesRegex(ValueError, 'refused-by-the-engine'):
            box.release_alpha_next(4, False)(client, {}, 'burst')
        self.assertEqual(client.calls, ['pause', 'native_progress', 'native_qsa_fused', 'native_layer_group', 'native_stage_local',
                                        'resume'])

    def receipts(self, *, local=True, history=None, before=None, after=None):
        stage = dict(selected=True, ring_reserve=2, poll_reserve=1, counters=dict.fromkeys(box.STAGE_COUNTERS, 0),
                     ring=dict(blocked_ns=0, episodes=0, blocked_max_ns=0, limiting={}, limiting_sole={}, blocked_open=False))
        b_, a_ = json.loads(json.dumps(stage)), json.loads(json.dumps(stage))
        a_['counters'].update(chunks=3, local_releases=6)
        a_['ring'].update(blocked_ns=500, episodes=2, limiting={'2': 2}, limiting_sole={'2': 1})
        for block, changes in ((b_, before), (a_, after)):
            for k, v in (changes or {}).items():
                (block['counters'] if k in box.STAGE_COUNTERS else block)[k] = v
        record = dict(control_history=ARM_ACTIONS if history is None else history, progress=[
            dict(label='before-fresh', native_stage_local=box.stage_view({'native_stage_local': b_})),
            dict(label='after-fresh', native_stage_local=box.stage_view({'native_stage_local': a_}))])
        return box.stage_reasons(record, local), record

    def test_stage_receipts(self):
        reasons, record = self.receipts()
        self.assertEqual((reasons, record['stage_delta']['chunks'], record['stage_delta']['local_releases']), ([], 3, 6))
        self.assertEqual((record['ring_delta']['blocked_ns'], record['ring_delta']['episodes'], record['ring_delta']['limiting'],
                          record['ring_delta']['limiting_sole']), (500, 2, {'2': 2}, {'2': 1}))
        self.assertIn('stage-selection-not-this-arm', self.receipts(local=False)[0])
        self.assertIn('stage-selection-not-this-arm', self.receipts(after=dict(selected=False))[0])
        self.assertIn('stage-selection-moved-during-the-fresh-prompt', self.receipts(after=dict(selections=1))[0])
        order = ARM_ACTIONS[:5] + ['release_auto_arm', 'native_stage_local', 'resume']
        self.assertIn('control-sequence-not-pause-progress-fused-group-stage-release-resume', self.receipts(history=order)[0])
        twice = ARM_ACTIONS[:6] + ['native_stage_local'] + ARM_ACTIONS[6:]
        self.assertIn('native_stage_local-sent-other-than-once', self.receipts(history=twice)[0])

    def test_views_need_the_blocks(self):
        self.assertIsNone(box.stage_view({}))
        self.assertIsNone(box.stage_view({'native_stage_local': {'selected': True}}))
        self.assertIsNone(box.stop_view({}))
        self.assertEqual(box.stop_view({'stop_receipts': dict(partial_replays=1, bound_exhausted=0, other=5)}),
                         dict(partial_replays=1, bound_exhausted=0))
        record = dict(control_history=ARM_ACTIONS, progress=[dict(label='before-fresh'), dict(label='after-fresh')])
        self.assertEqual(box.stage_reasons(record, True), ['stage-status-missing'])

    def test_the_greedy_salts_are_held_fs(self):
        salts = box.greedy_salts('run-x')
        self.assertEqual(sorted(salts.values()), list(T.STANDING_GREEDY))
        for salt, i in salts.items():
            self.assertEqual(salt, progress.shared._cache_salt('run-x', progress.STANDING, i))
        src = (HERE / 'e3_native41_progress_box.py').read_text()
        self.assertEqual(src.count("shared._request(a.model, row['prompt'], workload, shared._cache_salt(a.run_id, STANDING, i))"), 1)


# --- grade --------------------------------------------------------------------------------------------

class Grade(unittest.TestCase):
    def setUp(self):
        self.grade = Grader()
        self.addCleanup(self.grade.tmp.cleanup)

    def edited(self, arms, name, **fields):
        return [dict(a, **fields) if a['label'] == name else a for a in arms]

    def test_the_graded_arms_pass_with_all_sections(self):
        emit, arms = run_of('graded')
        result = self.grade(emit, arms)
        self.assertEqual((result['schema'], result['verdict'], result['missing'], result['failures'], result['classification']),
                         ('alpha-next-grade/1', 'PASS', [], [], None), result['missing'][:5])
        self.assertEqual((result['part'], result['is_queue_verdict']), ('core', True))
        self.assertEqual(result['required_arms'], GRADED)
        self.assertEqual(len(GRADED), 16)
        self.assertEqual({v['verdict'] for v in result['validity'].values()}, {'PASS'})
        on, off = result['validity']['c0on.g2-live-2']['engagement'], result['validity']['c0off.g2-live-2']['engagement']
        self.assertEqual((on['engaged'], on['group_engaged'], on['c0']['engaged'], on['c0']['stage_lines'], on['c0']['delta']['chunks'],
                          on['c0']['foreign_replies'], on['layer_commands_min'], on['layer_commands_max']),
                         (True, True, True, 5, 5, 0, 24, 24))
        self.assertEqual((off['engaged'], off['c0']['stage_lines'], off['c0']['delta']['chunks'], off['layer_commands_max']),
                         (True, 0, 0, 24))
        row = result['table']['c0on.g2-live-2']
        self.assertEqual((row['band'], row['selection'], row['size'], row['stage_local'], row['part'], row['chunks']),
                         (2, 'c0on.g2', 2, True, 'core', 5))
        self.assertEqual(result['table']['c0off.g2-solo-3']['part'], 'extra')
        band1 = result['summary']['band1']
        self.assertEqual(sorted(band1['groups']), ['c0off.g2-live', 'c0off.g2-solo', 'c0on.g2-live', 'c0on.g2-solo'])
        self.assertEqual(sorted(band1['over_c0_off']), ['c0on.g2-live', 'c0on.g2-solo'])
        self.assertEqual(sorted(result['memory_c0_on_vs_off']), sorted(x for x in GRADED if x.startswith('c0on')))
        self.assertEqual(result['memory_read']['c0on.g2-live-0']['against'], 'c0off.g2-live-0')
        self.assertEqual(result['numerics']['verdict'], 'WITHIN')
        self.assertEqual((result['speed_is_gate'], result['memory_is_gate'], result['decode_is_gate']), (False, False, False))
        self.assertEqual(result['arm_table']['extra_not_entered'], [])
        self.assertIn('Not assessed by this queue', result['release_readiness'])
        self.assertEqual(re.findall(r'^## (.+)$', box.report(result), flags=re.M), ['Mechanism', 'Performance', 'Release readiness'])

    def test_the_seams_are_this_call_only(self):
        names = ('required_labels', 'engagement', 'SELECTIONS', 'ARM_ORDER', 'memory_compare', 'speed_compare', 'size_of',
                 'memory_read', 'r3_read')
        before = [getattr(sb, k) for k in names] + [sa.SIGNATURES, progress.chunk_breakdown, ap.band_of_index]
        emit, arms = run_of('graded')
        self.grade(emit, arms[:2])
        self.assertEqual([getattr(sb, k) for k in names] + [sa.SIGNATURES, progress.chunk_breakdown, ap.band_of_index], before)
        self.assertEqual(sb.required_labels()[0], 'g1-solo-0')
        with self.assertRaises(RuntimeError), box.grade_seams(T.arms(T.GRADED), {}):
            raise RuntimeError('inside')
        self.assertEqual([getattr(sb, k) for k in names] + [sa.SIGNATURES, progress.chunk_breakdown, ap.band_of_index], before)

    def test_an_extra_arm_not_entered_is_not_required_and_one_that_ran_is_graded(self):
        # The log holds the core arms only (an extra arm not entered writes no line).
        emit, core = run_of('core', rows=T.arms('core'))
        result = self.grade(emit, core)
        self.assertEqual((result['verdict'], result['required_arms']), ('PASS', CORE))
        self.assertEqual(result['arm_table']['extra_not_entered'], EXTRA)
        # One extra arm with a watch record but no arm record (killed by the watch): required, and FAIL as engine-dead.
        watches = dict({a['label']: tap.ALIVE for a in core}, **{EXTRA[0]: {'verdict': 'DEAD'}})
        result = self.grade(emit, core, watches=watches)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn(EXTRA[0] + ':engine-dead', result['failures'])
        # An extra arm that ran and did not engage is INC as any core arm.
        emit, arms = run_of('graded')
        bad = self.edited(arms, EXTRA[1], stage_delta=dict(arms[0]['stage_delta'], chunks=0))
        result = self.grade(emit, bad)
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn(EXTRA[1] + ':not-engaged', result['missing'])

    def test_a_c0_on_arm_that_did_not_engage_is_inc(self):
        emit, arms = run_of('graded')
        name = 'c0on.g2-solo-0'
        stage = next(a for a in arms if a['label'] == name)['stage_delta']
        for fields, reason in ((dict(stage_delta=dict(stage, chunks=2)), 'stage-chunks-delta-differs-from-the-auto-owned-chunks'),
                               (dict(stage_delta=dict(stage, local_releases=3)), 'stage-releases-not-two-per-chunk'),
                               (dict(stage_delta={}), 'c0-engagement-receipts-missing')):
            with self.subTest(reason=reason):
                result = self.grade(emit, self.edited(arms, name, **fields))
                self.assertEqual(result['verdict'], 'INC')
                self.assertIn(name + ':not-engaged', result['missing'])
                self.assertIn(reason, result['validity'][name]['engagement']['reasons'])
        # A missing stage line for one chunk.
        drop = next(i for i, (_, t) in enumerate(emit.lines) if box.STAGE_LINE in t and
                    ap.in_span(progress.docker_stamp(t), next(a for a in arms if a['label'] == name)))
        result = self.grade(emit, arms, lines=emit.lines[:drop] + emit.lines[drop + 1:])
        reasons = result['validity'][name]['engagement']['reasons']
        self.assertIn('stage-lines-differ-from-the-auto-owned-chunks', reasons)
        self.assertIn('chunk-without-a-stage-line', reasons)

    def test_a_non_target_reply_on_a_stage_command_is_inc(self):
        rows = [r for r in T.arms(T.GRADED) if r['label'] in ('c0off.g2-solo-0', 'c0on.g2-solo-0')]
        emit = AlphaEmit()
        start = emit.tick(5_000_000)
        cuts = tap.band_cuts(1)
        emit.alpha_chunk(cuts[0], cuts[1], size=2, live=8, local=True, foreign=True)
        del start
        engaged = box.c0_engagement(dict(stage_local=True, stage_delta=dict(chunks=1, local_releases=2), progress_delta=dict(chunks=1)),
                                    dict(chunks=[dict(chunk_start=0, foreign_replies=1, short_replies=1)]),
                                    dict(stage=[dict(chunk_start=0)], torn_stage=0))
        self.assertIn('non-target-rank-replied-to-a-stage-command', engaged['reasons'])
        del rows

    def test_a_c0_off_arm_that_moved_a_stage_counter_or_logged_a_line_is_inc(self):
        emit, arms = run_of('graded')
        name = 'c0off.g2-live-1'
        result = self.grade(emit, self.edited(arms, name, stage_delta=dict(arms[0]['stage_delta'], local_releases=2)))
        self.assertIn('c0-off-arm-moved-a-stage-counter', result['validity'][name]['engagement']['reasons'])
        line = ('flash-s5 stage-local txn=9 chunk=0-1000 commands=14,14 both_stages_ns=1 first_ns=1,1 last_ns=2,2 '
                'local_release_ns=2,2 headroom_refused=0 headroom_blocked_ns=0')
        result = self.grade(emit, arms, lines=self.inject(emit, arms, name, line))
        self.assertIn('c0-off-arm-logged-a-stage-line', result['validity'][name]['engagement']['reasons'])
        self.assertEqual(result['verdict'], 'INC')

    @staticmethod
    def inject(emit, arms, name, text, who='engine'):
        arm = next(a for a in arms if a['label'] == name)
        stamp = emit.mono0 + int((arm['wall_start_s'] + 0.0003 - hf.WALL0) * 1e9)
        extra = AlphaEmit()
        extra.mono0 = emit.mono0
        extra.log(stamp, who, text)
        return emit.lines + extra.lines

    def test_a_group_arm_that_did_not_engage_is_inc(self):
        emit, arms = run_of('graded')
        name = 'c0on.g2-live-0'
        for fields, reason in ((dict(group_max_after=4), 'largest-group-is-not-the-size'),
                               (dict(group_delta=dict(arms[0]['group_delta'], chunks=1)), 'group-chunks-delta-differs-from-the-auto-owned-chunks'),
                               (dict(size=4), 'arm-not-the-declared-selection')):
            with self.subTest(reason=reason):
                result = self.grade(emit, self.edited(arms, name, **fields))
                self.assertIn(reason, result['validity'][name]['engagement']['reasons'])
                self.assertEqual(result['verdict'], 'INC')

    def test_each_c0_stage_signature_fails_that_arm_as_engine(self):
        emit, arms = run_of('graded')
        name = 'c0on.g2-solo-1'
        for sig, text in box.STAGE_SIGNATURES[:3]:
            with self.subTest(sig=sig):
                result = self.grade(emit, arms, lines=self.inject(emit, arms, name, 'ContractError: %s here' % text, who=1))
                self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
                self.assertIn(sig, result['validity'][name]['signatures'])
        self.assertNotIn(box.STAGE_SIGNATURES[0], sa.SIGNATURES)

    def test_speed_decode_memory_and_reads_never_change_the_verdict(self):
        emit, arms = run_of('graded')
        live = [a['label'] for a in arms if a['cohort'] == 'live' and a['c0'] == 'on']
        emit2, arms2 = synthetic_run(ratio={x: 0.5 for x in live}, blocked={x: 3_000_000 for x in live})
        result = self.grade(emit2, arms2)
        self.assertEqual(result['verdict'], 'PASS')
        read = result['decode_read']['c0on.g2-live-0']
        self.assertEqual((read['streams'], read['is_gate'], read['sampled_streams']), (6, False, list(T.STANDING_SAMPLED)))
        self.assertTrue(read['lines']['0.8']['min_below'] and read['lines']['0.9']['mean_below'])
        self.assertEqual((read['greedy']['streams'], read['greedy']['lines']['0.8']['min_below']), (2, True))
        self.assertIn('| greedy min | greedy mean | greedy < 0.8 | greedy < 0.9 |', box.report(result))
        self.assertLess(result['c0_decode_delta']['on_minus_off']['c0on.g2-live-0'], 0)
        ring = result['c0_ring']['c0on.g2-live-0']
        self.assertEqual((ring['headroom_blocked_ns_lines'], ring['headroom_blocked_ns_max_chunk'], ring['headroom_refused_lines'],
                          ring['is_gate']), (9_000_000, 3_000_000, 3, False))

    def test_decode_reads_minimum_and_mean_per_stream(self):
        arm = dict(client_basis=dict(decode_window=dict(stream_rates_tokens_per_s=[20.0, 30.0, 40.0, 42.0, 44.0, 45.0, 46.0, 50.0])),
                   decode_only=dict(stream_rates_tokens_per_s=[50.0] * 8))
        read = box.decode_read(arm)
        sampled = [0.8, 0.84, 0.88, 0.9, 0.92, 1.0]
        self.assertAlmostEqual(read['min_stream_ratio'], 0.8)
        self.assertAlmostEqual(read['mean_stream_ratio'], sum(sampled) / 6)
        self.assertEqual((read['lines']['0.8']['min_below'], read['lines']['0.9']['streams_below']), (False, 3))
        self.assertAlmostEqual(read['all_streams']['min_stream_ratio'], 0.4)
        self.assertEqual(read['all_streams']['lines']['0.8']['streams_below'], 2)
        self.assertEqual((read['greedy_streams'], read['greedy']['streams']), (list(T.STANDING_GREEDY), 2))
        self.assertAlmostEqual(read['greedy']['min_stream_ratio'], 0.4)
        self.assertAlmostEqual(read['greedy']['mean_stream_ratio'], 0.5)
        self.assertEqual((read['greedy']['lines']['0.8']['streams_below'], read['greedy']['lines']['0.9']['mean_below']), (2, True))
        self.assertEqual(sorted(read['per_stream']), [str(i) for i in range(8)])
        self.assertAlmostEqual(read['per_stream']['7'], 1.0)
        none = box.decode_read(dict(client_basis={}, decode_only={}))
        self.assertEqual((none['streams'], none['min_stream_ratio'], none['is_gate']), (0, None, False))

    def test_stop_receipts_peaks_and_anatomy_are_descriptive(self):
        name = 'c0on.g2-live-1'
        good = 'flash-s5 stop ' + json.dumps(dict(schema='s5-stop-receipt/1', kind='stop_partial_replay', txn_seq=2))
        bound = 'flash-s5 stop ' + json.dumps(dict(schema='s5-stop-receipt/1', kind='stop_bound_exhausted', txn_seq=3))
        off = 'flash-s5 stop ' + json.dumps(dict(schema='other/1', kind='stop_partial_replay'))
        anatomy = 'flash-s5 handler ' + json.dumps(dict(schema='s5-handler-anatomy/1', rpc='execute_model'))
        emit, arms = synthetic_run(stop_lines={name: [good, bound, off, anatomy, anatomy]})
        result = self.grade(emit, arms)
        self.assertEqual(result['verdict'], 'PASS')
        stops = result['stop_receipts']['arms'][name]
        self.assertEqual((stops['lines'], stops['by_kind'], stops['torn_or_off_schema'], stops['is_gate']),
                         (2, dict(stop_partial_replay=1, stop_bound_exhausted=1), 1, False))
        self.assertEqual(result['stop_receipts']['stop_lines_outside_every_arm'], 0)
        self.assertEqual(result['anatomy_lines'][name], 2)
        peak = result['memory_peak'][name]
        self.assertEqual((peak['receipts'], sorted(peak['ranks']), peak['peak_allocated_mib'], peak['is_gate']),
                         (12, ['0', '1', '2', '3'], 21450.0, False))
        self.assertEqual(result['progress']['tears']['torn_memory_lines'], 0)      # the peak line is no memory receipt

    def test_engine_watch_ingress_and_incomplete_rows(self):
        emit, arms = run_of('graded')
        name = 'c0on.g2-solo-2'
        result = self.grade(emit, arms, watches=dict({a['label']: tap.ALIVE for a in arms}, **{name: {'verdict': 'DEAD'}}))
        self.assertEqual(result['verdict'], 'FAIL')
        result = self.grade(emit, arms, ingress=dict({a['label']: tap.PASS for a in arms}, **{name: {'verdict': 'FAIL'}}))
        self.assertEqual(result['verdict'], 'FAIL')
        result = self.grade(emit, self.edited(arms, name, complete=False, reasons=['progress-chunks-not-recorded']))
        self.assertEqual(result['verdict'], 'INC')

    def test_the_core_grade_never_takes_a_tail_arm_and_the_read_covers_every_arm(self):
        # The core grade is written before the tail starts: its log holds no tail line.
        emit, graded = run_of('graded')
        result = self.grade(emit, graded)
        self.assertEqual((result['verdict'], result['required_arms']), ('PASS', GRADED))
        emit, arms = run_of('all', rows=T.arms())
        graded = [a for a in arms if a['part'] in T.GRADED]
        tail_bad = 'c0on.g4-live-1'
        lines = Grade.inject(emit, arms, tail_bad, 'ContractError: a stage command targets another stage', who=2)
        # A tail arm handed to the core grade is undeclared there.
        mixed = self.grade(emit, graded + [a for a in arms if a['label'] == TAIL[0]])
        self.assertIn('undeclared-or-repeated-arm', mixed['missing'])
        read = self.grade(emit, arms, lines=lines, part='all')
        self.assertEqual((read['schema'], read['is_queue_verdict'], read['verdict']), ('alpha-next-read/1', False, 'FAIL'))
        self.assertEqual(read['required_arms'], ALL_LABELS)
        self.assertIn(tail_bad, ''.join(read['failures']))
        clean = self.grade(emit, arms, part='all')
        self.assertEqual(clean['verdict'], 'PASS', clean['missing'][:4])
        eng = clean['validity']['c0on.g4-solo-2']['engagement']
        self.assertEqual((eng['engaged'], eng['layer_commands_max'], eng['c0']['engaged']), (True, 12, True))
        self.assertIn('read_only', clean)
        self.assertIn('not the queue verdict', box.report(clean).splitlines()[0])

    def test_grade_and_read_cli_files_rcs_and_no_overwrite(self):
        emit, arms = run_of('graded')
        q = Path(tempfile.mkdtemp(dir=self.grade.tmp.name))
        tap.write_streams(q, emit.lines)
        argv = []
        for a in arms:
            (q / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
            argv += ['--arm', str(q / ('arm-%s.json' % a['label'])), '--watch', '%s=%s' % (a['label'], q / 'w.json'),
                     '--ingress', '%s=%s' % (a['label'], q / 'i.json')]
        (q / 'w.json').write_text(json.dumps(tap.ALIVE))
        (q / 'i.json').write_text(json.dumps(tap.PASS))
        for k in tap.CHECKS:
            (q / (k + '.json')).write_text(json.dumps(tap.PASS))
            argv += ['--check', '%s=%s' % (k, q / (k + '.json'))]
        argv += ['--log', str(q / 'engine-stream.log'), '--log', str(q / 'engine-stream.err')]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = box.main(['grade'] + argv + ['--report', str(q / 'r.md'), '--out', str(q / 'g.json')])
        self.assertEqual((rc, out.getvalue().strip()), (0, 'ALPHA_NEXT_GRADE PASS'))
        self.assertEqual(box.main(['grade'] + argv + ['--out', str(q / 'g.json')]), 2)       # never overwrites
        bad = [x.replace('w.json', 'dead.json') if x.endswith('w.json') and 'c0on.g2-live-0=' in x else x for x in argv]
        (q / 'dead.json').write_text(json.dumps({'verdict': 'DEAD'}))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(box.main(['grade'] + bad + ['--out', str(q / 'g2.json')]), 1)
            self.assertEqual(box.main(['read'] + bad + ['--out', str(q / 'r2.json')]), 2)   # a read never returns 1


# --- numerics ------------------------------------------------------------------------------------------

class Numerics(unittest.TestCase):
    ROWS = T.arms(T.GRADED)

    def arms(self, changes=None, rows=None):
        changes = changes or {}
        return [dict(label=r['label'], outputs=changes[r['label']] if r['label'] in changes else outputs(r))
                for r in (rows or self.ROWS)]

    def test_identical_is_within(self):
        result = box.numerics(self.arms(), self.ROWS)
        self.assertEqual((result['verdict'], result['outside'], result['unproven'], result['is_gate']), ('WITHIN', [], [], False))
        band1 = result['bands']['1']
        self.assertEqual(sorted(band1['streams']), [str(i) for i in T.STANDING_GREEDY])
        self.assertEqual((band1['streams']['0']['control_floor'], band1['streams']['0']['envelope']), (256, 128))
        self.assertEqual(len(band1['controls']), 6)       # rounds 0, 1 and the extra round 3

    def test_a_c0_on_arm_far_from_controls_that_agree_is_outside(self):
        name = 'c0on.g2-live-0'
        row = T.by_label()[name]
        result = box.numerics(self.arms({name: outputs(row, cut=40)}), self.ROWS)
        self.assertEqual(result['verdict'], 'OUTSIDE')
        self.assertEqual(result['outside'], ['%s:stream-%d' % (name, i) for i in T.STANDING_GREEDY])
        self.assertEqual(result['bands']['1']['streams']['0']['candidates'][name], dict(nearest_common_prefix=40, inside=False))
        result = box.numerics(self.arms({name: outputs(row, fresh=(12,))}), self.ROWS)
        self.assertEqual(result['outside'], [name + ':fresh'])

    def test_controls_that_already_differ_widen_the_envelope_instead_of_unproven(self):
        # The C0-l shape: every serial arm's standing digest differed. Here the controls agree for 60 tokens at worst.
        changes = {r['label']: outputs(r, cut=60 + 10 * k) for k, r in enumerate(x for x in self.ROWS if x['c0'] == 'off')}
        name = 'c0on.g2-solo-1'
        changes[name] = outputs(T.by_label()[name], cut=35)
        result = box.numerics(self.arms(changes), self.ROWS)
        self.assertEqual(result['verdict'], 'WITHIN', result)
        stream = result['bands']['1']['streams']['0']
        self.assertEqual((stream['control_floor'], stream['envelope']), (60, 30))
        changes[name] = outputs(T.by_label()[name], cut=20)
        self.assertEqual(box.numerics(self.arms(changes), self.ROWS)['outside'], ['%s:stream-%d' % (name, i) for i in T.STANDING_GREEDY])

    def test_sampled_streams_are_not_compared(self):
        name = 'c0on.g2-live-1'
        out = outputs(T.by_label()[name])
        out['standing']['5']['prefix_sha256'] = '0' * 64
        self.assertEqual(box.numerics(self.arms({name: out}), self.ROWS)['verdict'], 'WITHIN')

    def test_missing_short_or_too_few_controls_are_unproven(self):
        name = 'c0on.g2-live-1'
        out = outputs(T.by_label()[name])
        out['standing']['0'] = dict(out['standing']['0'], prefix=out['standing']['0']['prefix'][:100], prefix_tokens=100)
        result = box.numerics(self.arms({name: out}), self.ROWS)
        self.assertEqual((result['verdict'], result['unproven']), ('UNPROVEN', [name + ':stream-0-prefix-short']))
        rows = [r for r in self.ROWS if r['band'] == 2 and (r['c0'] == 'on' or r['label'] == 'c0off.g2-live-2')]
        self.assertEqual(box.numerics(self.arms(rows=rows), rows)['unproven'], ['band2-fewer-than-two-controls'])
        missing = {name: dict(outputs(T.by_label()[name]), fresh=[])}
        self.assertIn(name + ':fresh-tokens-missing', box.numerics(self.arms(missing), self.ROWS)['unproven'])

    def test_a_read_by_default_and_a_gate_with_the_one_constant(self):
        self.assertIs(box.NUMERICS_IS_GATE, False)
        grade = Grader()
        self.addCleanup(grade.tmp.cleanup)
        name = 'c0on.g2-solo-0'
        emit, arms = run_of('graded')
        for a in arms:
            if a['label'] == name:
                a['outputs'] = outputs(T.by_label()[name], cut=10)
        result = grade(emit, arms)
        self.assertEqual((result['verdict'], result['numerics']['verdict']), ('PASS', 'OUTSIDE'))
        self.assertIn('a read, not a verdict input', box.report(result))
        # Lead 14:29: OUTSIDE prominent in the grade and the report, no verdict change.
        self.assertEqual((result['numerics_verdict'], result['numerics_outside']),
                         ('OUTSIDE', ['%s:stream-%d' % (name, i) for i in T.STANDING_GREEDY]))
        self.assertTrue(result['numerics_headline'].startswith('NUMERICS OUTSIDE (a read this run, the verdict does not change): '))
        self.assertEqual(box.report(result).splitlines()[4], '**%s**' % result['numerics_headline'])
        self.assertIsNone(grade(*run_of('graded'))['numerics_headline'])
        with mock.patch.object(box, 'NUMERICS_IS_GATE', True):
            result = grade(emit, arms)
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
        self.assertIn('numerics-outside-the-control-envelope:', ''.join(result['failures']))
        src = (HERE / 'e3_alpha_next_box.py').read_text()
        self.assertEqual(len(re.findall(r'^NUMERICS_IS_GATE = (True|False)$', src, flags=re.M)), 1)
        self.assertIn('Two readings are declared, one constant apart', src)


# --- STOP_RACE -----------------------------------------------------------------------------------------

STOP_INTENTS = {'step-b-e': 1793619564788177866, 'native26': 1793321833844534892}
STEP_B_STOP = (
    '2026-11-02T11:36:40.818239067Z (APIServer pid=51) INFO:     127.0.0.1:48930 - "POST /v1/flash_s5_native_overlap HTTP/1.1" 200 OK',
    '2026-11-02T11:36:44.805901841Z (APIServer pid=51) INFO 11-02 11:36:44 [loggers.py:311] Engine 000: Avg prompt throughput: 414.1 tokens/s, Avg generation throughput: 0.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 93.8%',
    '2026-11-02T11:36:54.805581147Z (APIServer pid=51) INFO 11-02 11:36:54 [loggers.py:311] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 93.8%',
    '2026-11-02T11:39:24.852687573Z (APIServer pid=51) INFO 11-02 11:39:24 [launcher.py:129] [shutdown] API server: shutdown triggered',
    '2026-11-02T11:39:24.852715133Z (APIServer pid=51) INFO 11-02 11:39:24 [launcher.py:145] [shutdown] API server: stopping engine client mode=abort timeout=0s',
    '2026-11-02T11:39:24.853515268Z (APIServer pid=51) INFO 11-02 11:39:24 [core_client.py:744] [shutdown] MPClient: start timeout=0s',
    '2026-11-02T11:39:24.853661473Z (APIServer pid=51) INFO 11-02 11:39:24 [core_client.py:746] [shutdown] MPClient: stopping engine manager',
    '2026-11-02T11:39:24.853725235Z (APIServer pid=51) INFO 11-02 11:39:24 [utils.py:620] [shutdown] Process manager: send sigterm to process EngineCore',
    '2026-11-02T11:39:24.853779176Z (APIServer pid=51) WARNING 11-02 11:39:24 [utils.py:640] [shutdown] Process manager: force killing remaining processes count=1',
    '2026-11-02T11:39:24.853872559Z (APIServer pid=51) WARNING 11-02 11:39:24 [utils.py:645] [shutdown] Process manager: force killing remaining process EngineCore pid 103',
    '2026-11-02T11:39:24.854090276Z (EngineCore pid=103) INFO 11-02 11:39:24 [core.py:1368] [shutdown] EngineCore: trigger received signal=SIGTERM',
    '2026-11-02T11:39:24.854390905Z (EngineCore pid=103) INFO 11-02 11:39:24 [core.py:1520] [shutdown] EngineCore: start mode=abort timeout=0s',
    '2026-11-02T11:39:24.854426146Z (EngineCore pid=103) INFO 11-02 11:39:24 [core.py:1551] [shutdown] EngineCore: request processing complete; starting resource teardown',
    '2026-11-02T11:39:24.856003195Z (EngineCore pid=103) INFO 11-02 11:39:24 [core.py:1381] [shutdown] EngineCore: exiting busy loop',
    '2026-11-02T11:39:24.856760468Z (APIServer pid=51) INFO 11-02 11:39:24 [core_client.py:748] [shutdown] MPClient: engine manager stopped',
    '2026-11-02T11:39:24.856766988Z (APIServer pid=51) INFO 11-02 11:39:24 [core_client.py:749] [shutdown] MPClient: cleaning up background resources',
    '2026-11-02T11:39:24.856770048Z (APIServer pid=51) INFO 11-02 11:39:24 [core_client.py:751] [shutdown] MPClient: complete',
    '2026-11-02T11:39:24.861963008Z (APIServer pid=51) ERROR 11-02 11:39:24 [async_llm.py:829] AsyncLLM output_handler failed.',
    '2026-11-02T11:39:24.861985999Z (APIServer pid=51) ERROR 11-02 11:39:24 [async_llm.py:829] Traceback (most recent call last):',
    '2026-11-02T11:39:24.861990879Z (APIServer pid=51) ERROR 11-02 11:39:24 [async_llm.py:829]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 775, in output_handler',
    '2026-11-02T11:39:24.861993809Z (APIServer pid=51) ERROR 11-02 11:39:24 [async_llm.py:829]     outputs = await engine_core.get_output_async()',
    '2026-11-02T11:39:24.861996519Z (APIServer pid=51) ERROR 11-02 11:39:24 [async_llm.py:829]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',
    '2026-11-02T11:39:24.861999139Z (APIServer pid=51) ERROR 11-02 11:39:24 [async_llm.py:829]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 1175, in get_output_async',
    '2026-11-02T11:39:24.862001910Z (APIServer pid=51) ERROR 11-02 11:39:24 [async_llm.py:829]     raise self._format_exception(outputs) from None',
    '2026-11-02T11:39:24.862004650Z (APIServer pid=51) ERROR 11-02 11:39:24 [async_llm.py:829] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.',
    '2026-11-02T11:39:24.862039351Z (APIServer pid=51) INFO 11-02 11:39:24 [launcher.py:154] [shutdown] API server: engine client stopped',
    '2026-11-02T11:39:24.862093652Z (APIServer pid=51) INFO 11-02 11:39:24 [launcher.py:157] [shutdown] API server: signalling HTTP server shutdown',
    '2026-11-02T11:39:24.883717039Z (APIServer pid=51) INFO 11-02 11:39:24 [launcher.py:179] [shutdown] API server: shutting down FastAPI HTTP server',
)
NATIVE26_STOP = (
    '2026-10-30T00:57:13.915136290Z (APIServer pid=52) INFO 10-30 00:57:13 [launcher.py:129] [shutdown] API server: shutdown triggered',
    '2026-10-30T00:57:13.915166941Z (APIServer pid=52) INFO 10-30 00:57:13 [launcher.py:145] [shutdown] API server: stopping engine client mode=abort timeout=0s',
    '2026-10-30T00:57:13.915706845Z (APIServer pid=52) INFO 10-30 00:57:13 [core_client.py:744] [shutdown] MPClient: start timeout=0s',
    '2026-10-30T00:57:13.915746096Z (APIServer pid=52) INFO 10-30 00:57:13 [core_client.py:746] [shutdown] MPClient: stopping engine manager',
    '2026-10-30T00:57:13.915839599Z (APIServer pid=52) INFO 10-30 00:57:13 [utils.py:620] [shutdown] Process manager: send sigterm to process EngineCore',
    '2026-10-30T00:57:13.915905550Z (APIServer pid=52) WARNING 10-30 00:57:13 [utils.py:640] [shutdown] Process manager: force killing remaining processes count=1',
    '2026-10-30T00:57:13.915940781Z (APIServer pid=52) WARNING 10-30 00:57:13 [utils.py:645] [shutdown] Process manager: force killing remaining process EngineCore pid 87',
    '2026-10-30T00:57:13.916041844Z (EngineCore pid=87) INFO 10-30 00:57:13 [core.py:1368] [shutdown] EngineCore: trigger received signal=SIGTERM',
    '2026-10-30T00:57:13.916254510Z (EngineCore pid=87) INFO 10-30 00:57:13 [core.py:1520] [shutdown] EngineCore: start mode=abort timeout=0s',
    '2026-10-30T00:57:13.916289990Z (EngineCore pid=87) INFO 10-30 00:57:13 [core.py:1551] [shutdown] EngineCore: request processing complete; starting resource teardown',
    '2026-10-30T00:57:13.916354042Z (EngineCore pid=87) INFO 10-30 00:57:13 [core.py:1381] [shutdown] EngineCore: exiting busy loop',
    '2026-10-30T00:57:13.916789624Z (EngineCore pid=87) INFO 10-30 00:57:13 [multiproc_executor.py:503] [shutdown] Executor: waiting for worker exit count=4',
    '2026-10-30T00:57:13.917146903Z (Worker_PP0_TP0_EP0 pid=97) INFO 10-30 00:57:13 [multiproc_executor.py:872] Parent process exited, terminating worker queues',
    '2026-10-30T00:57:13.919450914Z (APIServer pid=52) INFO 10-30 00:57:13 [core_client.py:748] [shutdown] MPClient: engine manager stopped',
    '2026-10-30T00:57:13.919460874Z (APIServer pid=52) INFO 10-30 00:57:13 [core_client.py:749] [shutdown] MPClient: cleaning up background resources',
    '2026-10-30T00:57:13.919464444Z (APIServer pid=52) INFO 10-30 00:57:13 [core_client.py:751] [shutdown] MPClient: complete',
    '2026-10-30T00:57:13.921521558Z (APIServer pid=52) ERROR 10-30 00:57:13 [async_llm.py:829] AsyncLLM output_handler failed.',
    '2026-10-30T00:57:13.921529848Z (APIServer pid=52) ERROR 10-30 00:57:13 [async_llm.py:829] Traceback (most recent call last):',
    '2026-10-30T00:57:13.921532178Z (APIServer pid=52) ERROR 10-30 00:57:13 [async_llm.py:829]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 775, in output_handler',
    '2026-10-30T00:57:13.921534339Z (APIServer pid=52) ERROR 10-30 00:57:13 [async_llm.py:829]     outputs = await engine_core.get_output_async()',
    '2026-10-30T00:57:13.921536309Z (APIServer pid=52) ERROR 10-30 00:57:13 [async_llm.py:829]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^',
    '2026-10-30T00:57:13.921538499Z (APIServer pid=52) ERROR 10-30 00:57:13 [async_llm.py:829]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 1175, in get_output_async',
    '2026-10-30T00:57:13.921540619Z (APIServer pid=52) ERROR 10-30 00:57:13 [async_llm.py:829]     raise self._format_exception(outputs) from None',
    '2026-10-30T00:57:13.921542589Z (APIServer pid=52) ERROR 10-30 00:57:13 [async_llm.py:829] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.',
    '2026-10-30T00:57:13.921626621Z (APIServer pid=52) INFO 10-30 00:57:13 [launcher.py:154] [shutdown] API server: engine client stopped',
    '2026-10-30T00:57:13.921638191Z (APIServer pid=52) INFO 10-30 00:57:13 [launcher.py:157] [shutdown] API server: signalling HTTP server shutdown',
    '2026-10-30T00:57:13.944678168Z (APIServer pid=52) INFO 10-30 00:57:13 [launcher.py:179] [shutdown] API server: shutting down FastAPI HTTP server',
)
IDLE_EMPTY = {'verdict': 'PASS', 'workers': {str(r): dict(running_requests=0, prefill_requests=0, live_decode_requests=0, pid=100 + r)
                                             for r in range(4)}}


class StopRace(unittest.TestCase):
    """Positive: the verbatim stop lines of q18a-step-b-e (2 Nov) and q18a-native26 (30 Oct). Negative: a mid-arm EngineDeadError,
    a hard marker before the stop, a gap over 500 ms, a worker failure record, an end FULL not empty."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix='stop-race-'))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def files(self, lines, intent, idle=IDLE_EMPTY, err=()):
        (self.dir / 'engine-stream.log').write_text(''.join(x + '\n' for x in lines))
        (self.dir / 'engine-stream.err').write_text(''.join(x + '\n' for x in err))
        (self.dir / 'host-stop-intent.json').unlink(missing_ok=True)
        if intent is not None:
            (self.dir / 'host-stop-intent.json').write_text(json.dumps(dict(schema=1, wall_ns=intent)))
        if idle is not None:
            (self.dir / 'end-full-idle.json').write_text(json.dumps(idle))
        return ([str(self.dir / 'engine-stream.log'), str(self.dir / 'engine-stream.err')], str(self.dir / 'host-stop-intent.json'),
                str(self.dir / 'end-full-idle.json'))

    def classify(self, *a, **k):
        return sr.classify(*self.files(*a, **k))

    @staticmethod
    def shifted(lines, after, delta_ns):
        """The lines with every stamp after the line holding ``after`` moved by delta_ns (a slower race)."""
        out, moving = [], False
        for line in lines:
            if moving:
                ns = sr.death.wall_ns(line.encode()) + delta_ns
                line = sr.death.iso(ns).replace('+00:00', 'Z') + line[line.index('Z ') + 1:]
            out.append(line)
            moving = moving or after in line
        return out

    def test_the_two_real_stop_races_are_stop_race(self):
        for name, lines, gap in (('step-b-e', STEP_B_STOP, 9.317), ('native26', NATIVE26_STOP, 6.406)):
            with self.subTest(run=name):
                record, rc = self.classify(lines, STOP_INTENTS[name])
                self.assertEqual((rc, record['stop_race'], record['engine_crashed'], record['reasons'], record['gap_ms']),
                                 (0, True, False, [], gap))
                self.assertEqual(record['death_counts'], {'engine_dead': 1})
                self.assertTrue(record['handler']['process'].startswith('(APIServer pid='))

    def test_the_frozen_scan_still_says_dead_on_them(self):
        logs, intent, _ = self.files(STEP_B_STOP, STOP_INTENTS['step-b-e'])
        scanner = sr.death.Scanner(logs, intent)
        scanner.poll(final=True)
        self.assertEqual(scanner.decide()['verdict'], 'DEAD')

    def test_a_mid_arm_engine_dead_error_is_a_crash(self):
        # The same traceback with the stop intent after it: an EngineDeadError while the queue still runs.
        record, rc = self.classify(STEP_B_STOP, STOP_INTENTS['step-b-e'] + 10 ** 12)
        self.assertEqual((rc, record['engine_crashed'], record['stop_race']), (1, True, False))
        self.assertIn('no-shutdown-trigger-after-the-stop-intent', record['reasons'])
        record, rc = self.classify(STEP_B_STOP, None)
        self.assertEqual(rc, 1)
        self.assertTrue([r for r in record['reasons'] if r.startswith('stop-intent-not-read')], record['reasons'])

    def test_a_hard_marker_before_the_stop_is_a_crash(self):
        marker = '2026-11-02T11:30:00.000000000Z (Worker_PP0_TP0_EP0 pid=113) ERROR 11-02 11:30:00 [x.py:1] CUDA error: an illegal memory access was encountered'
        record, rc = self.classify([marker] + list(STEP_B_STOP), STOP_INTENTS['step-b-e'])
        self.assertEqual((rc, record['engine_crashed']), (1, True))
        self.assertTrue([r for r in record['reasons'] if r.startswith('deaths-other-than-one-engine-dead')], record['reasons'])
        failure = '2026-11-02T11:39:24.861000000Z (EngineCore pid=103) ERROR FLASH_S5_WORKER_FAILURE rank=1 kind=x'
        record, rc = self.classify(list(STEP_B_STOP) + [failure], STOP_INTENTS['step-b-e'])
        self.assertEqual(rc, 1)
        self.assertIn('worker-failure-record:1', record['reasons'])

    def test_a_gap_over_500_ms_is_a_crash(self):
        lines = self.shifted(STEP_B_STOP, 'MPClient: complete', 600 * 10 ** 6)
        record, rc = self.classify(lines, STOP_INTENTS['step-b-e'])
        self.assertEqual(rc, 1)
        self.assertEqual(record['reasons'], ['engine-dead-more-than-500-ms-after-the-shutdown-trigger'])
        self.assertGreater(record['gap_ms'], 500)
        lines = self.shifted(STEP_B_STOP, 'MPClient: complete', 400 * 10 ** 6)
        self.assertEqual(self.classify(lines, STOP_INTENTS['step-b-e'])[1], 0)

    def test_the_end_full_idle_must_be_pass_and_empty(self):
        busy = json.loads(json.dumps(IDLE_EMPTY))
        busy['workers']['2']['running_requests'] = 1
        for idle, reason in ((busy, 'end-full-idle-worker-not-empty:2'), (dict(IDLE_EMPTY, verdict='FAIL'), 'end-full-idle-not-pass'),
                             (None, 'end-full-idle-unreadable'), (dict(verdict='PASS'), 'end-full-idle-without-worker-rows')):
            with self.subTest(reason=reason):
                (self.dir / 'end-full-idle.json').unlink(missing_ok=True)
                record, rc = self.classify(STEP_B_STOP, STOP_INTENTS['step-b-e'], idle=idle)
                self.assertEqual((rc, record['reasons']), (1, [reason]))

    def test_an_engine_dead_outside_an_output_handler_is_a_crash(self):
        lines = [x for x in STEP_B_STOP if 'output_handler failed' not in x]
        record, rc = self.classify(lines, STOP_INTENTS['step-b-e'])
        self.assertEqual((rc, record['reasons']), (1, ['engine-dead-not-in-an-output-handler-traceback']))

    def test_not_dead_is_not_classified(self):
        lines = [x for x in STEP_B_STOP if 'EngineDeadError' not in x]
        record, rc = self.classify(lines, STOP_INTENTS['step-b-e'])
        self.assertEqual((rc, record['stop_race'], record['engine_crashed']), (2, False, False))

    def test_cli_record_and_rcs(self):
        logs, intent, idle = self.files(NATIVE26_STOP, STOP_INTENTS['native26'])
        out = self.dir / 'stop-race.json'
        argv = ['--log', logs[0], '--log', logs[1], '--stop-intent', intent, '--end-full-idle', idle, '--out', str(out)]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(sr.main(argv), 0)
        self.assertEqual(printed.getvalue().strip(), 'ALPHA_NEXT_STOP_RACE STOP_RACE')
        self.assertEqual(json.loads(out.read_text())['schema'], 'alpha-next-stop-race/1')
        (self.dir / 'host-stop-intent.json').write_text(json.dumps(dict(schema=1, wall_ns=STOP_INTENTS['native26'] + 10 ** 12)))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(sr.main(argv), 1)
        self.assertTrue(printed.getvalue().startswith('ALPHA_NEXT_STOP_RACE CRASH '))

    def test_native30_localize_does_not_fail_on_a_stop_race(self):
        logs, intent, idle = self.files(STEP_B_STOP, STOP_INTENTS['step-b-e'])
        record, rc = sr.classify(logs, intent, idle)
        self.assertFalse(record['engine_crashed'])
        out = self.dir / 'localize.json'
        done = subprocess.run([sys.executable, str(HERE / 'e3_native30_e_grade.py'), '--log', logs[0], '--log', logs[1],
                               '--placement', str(self.dir / 'qual-row4.json'), '--out', str(out), '--engine-crashed', '0'],
                              capture_output=True, text=True, timeout=120)
        self.assertNotEqual(done.returncode, 1, done.stderr[-400:])
        self.assertNotEqual(json.loads(out.read_text())['verdict'], 'FAIL')
        # The draft passes ENGINE_CRASHED, which a STOP_RACE leaves 0, and FAILs that phase only on rc 1.
        draft = queue.derive()[DRAFT]
        self.assertIn('--engine-crashed "$ENGINE_CRASHED"', draft)
        self.assertIn('if [[ $_diag_rc == 1 ]]; then record native30-localize 1; else PHASE_VERDICT[native30-localize]=NOT_RUN; fi', draft)

    def test_the_drafts_host_crash_detected_in_bash(self):
        draft = queue.derive()[DRAFT]
        func = tap.le_fix_function(draft, 'host_crash_detected')
        for name, lines, intent, want in (('race', STEP_B_STOP, STOP_INTENTS['step-b-e'], 'rc=1 CRASHED=0 STOP_RACE=1 PHASE=STOP_RACE'),
                                          ('mid', STEP_B_STOP, STOP_INTENTS['step-b-e'] + 10 ** 12,
                                           'rc=0 CRASHED=1 STOP_RACE=0 PHASE= SOURCE=engine-death-scan')):
            with self.subTest(case=name):
                self.files(lines, intent)
                script = '\n'.join(['set -u', 'declare -A PHASE_VERDICT=()', "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; STOP_RACE=0",
                                    'Q=%s; H=%s' % (shlex.quote(str(self.dir)), shlex.quote(str(HERE))), func,
                                    'host_crash_detected; rc=$?',
                                    'echo "rc=$rc CRASHED=$ENGINE_CRASHED STOP_RACE=$STOP_RACE PHASE=${PHASE_VERDICT[stop-race]:-}'
                                    '${ENGINE_CRASH_SOURCE:+ SOURCE=$ENGINE_CRASH_SOURCE}"'])
                done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=120)
                self.assertEqual(done.stdout.strip(), want, done.stderr[-400:])
                self.assertTrue((self.dir / 'stop-race.json').is_file())
                (self.dir / 'stop-race.json').unlink()
                for leftover in ('engine-death-scan.json',):
                    (self.dir / leftover).unlink(missing_ok=True)

    def test_queue_grade_records_stop_race(self):
        draft = queue.derive()[DRAFT]
        self.assertIn("stop_race=(stop_race or {}).get('stop_race') is True,stop_race_record=stop_race,", draft)
        self.assertEqual(draft.count("STOP_RACE=0\n"), 1)


# --- the arm loop in bash ----------------------------------------------------------------------------

STUB_BOX = r'''#!/usr/bin/env python3
import json, os, shutil, sys, time
if sys.argv[1] in ('grade', 'read'):
    with open(os.environ['CALLS'], 'a') as f:
        f.write(json.dumps(dict(script='box', action=sys.argv[1])) + '\n')
    sys.path.insert(0, os.environ['REAL_HARNESS'])
    import e3_alpha_next_box as real
    raise SystemExit(real.main(sys.argv[1:]))
a = sys.argv
out = a[a.index('--out') + 1]
label = 'c0%s.g%s-%s-%s' % (a[a.index('--stage-local') + 1], a[a.index('--size') + 1], a[a.index('--cohort') + 1],
                            a[a.index('--index') + 1])
env = {k: os.environ.get(k) for k in ('READINESS', 'GEN', 'S5_OVERLAY', 'S5_LAUNCHER_NAME', 'PACKAGE_MANIFEST', 'GENERATION')}
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(dict(script='box', action='arm', label=label, env=env, workload=os.path.basename(a[a.index('--workload') + 1]))) + '\n')
if label in os.environ.get('TAIL_LABELS', '').split() and not os.path.exists(os.environ['ARMS'] + '/tail-appended'):
    # The tail's engine lines are written only once the tail runs (the core grade reads the log before it).
    for stream in ('log', 'err'):
        with open(os.environ['Q'] + '/engine-stream.' + stream, 'a') as f, open(os.environ['ARMS'] + '/tail.' + stream) as g:
            f.write(g.read())
    open(os.environ['ARMS'] + '/tail-appended', 'w').close()
if label == os.environ.get('DIE_IN'):
    with open(os.environ['Q'] + '/engine-stream.log', 'a') as f:
        f.write(os.environ['DEATH_LINE'])
    time.sleep(600)
shutil.copy(os.environ['ARMS'] + '/arm-%s.json' % label, out)
raise SystemExit(int(os.environ.get('RC_' + label.replace('.', '_').replace('-', '_'), '0')))
'''


class ArmLoop(unittest.TestCase):
    """queue.BODY runs as written; arm and check clients are stubs (leaf arm-loop-clients), the watch, grade and read real."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='alpha-next-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death.py', 'alpha_next_gate.py', 'alpha_next_arms.py', 'alpha_next_stop_race.py',
                     'native39_trunk_gate.py', 'native40_box_gate.py', 'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_alpha_next_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def run_loop(self, *, jump_after=None, jump_to='LAST_ARM_ENTRY_S + 10', die_in=None, rcs=None, timeout=420):
        emit, arms = run_of('all', rows=T.arms())
        first_tail = min(a['wall_start_s'] for a in arms if a['part'] == 'tail') * 1e9
        cut = next(i for i, (_, text) in enumerate(emit.lines) if (sr.death.wall_ns(text.encode()) or 0) >= first_tail)
        tap.write_streams(self.q, emit.lines[:cut])
        tap.write_streams(self.arms, emit.lines[cut:])
        for stream in ('log', 'err'):
            os.replace(self.arms / ('engine-stream.' + stream), self.arms / ('tail.' + stream))
        for a in arms:
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        draft = queue.derive()[DRAFT]
        body = queue.body()
        self.assertIn(body, draft)
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; STOP_RACE=0; CONTAINER=qwen-flash-prod; SERVICE=vllm-flash; N0=0",
                  'Q=%s; H=%s; PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; _cid=c; IDLE_TIMEOUT_S=1' % (self.q, self.h),
                  'Q18A_READINESS=/r/ready; S5_OVERLAY=/o; S5_LAUNCHER_NAME=l.sh', 'health_ok() { return 0; }']
        script += [tap.le_fix_function(draft, name) for name in tap.LOOP_FUNCTIONS]
        if jump_after:
            # The box clock past a limit right after the named step (bash SECONDS is writable): leaf clock.
            for fn in ('record', 'read_step'):
                script += ['%s_hook() { [[ $1 == %s ]] && SECONDS=$((%s)); return 0; }' % (fn, jump_after, jump_to)]
            body = body.replace('read_step() {\n  local v\n', 'read_step() {\n  read_step_hook "$1"\n  local v\n')
            script += ['eval "$(declare -f record | sed \'1s/record/real_record/\')"',
                       'record() { real_record "$@"; local rc=$?; record_hook "$1"; return $rc; }']
        script += [body, 'echo "END FAILS=$FAILS INCS=$INCS CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"']
        path = self.root / 'loop.sh'
        path.write_text('\n'.join(script) + '\n')
        env = {k: v for k, v in os.environ.items() if k != 'HARNESS_DIR'}
        env.update(PATH='%s:%s' % (self.bin, os.environ['PATH']), Q=str(self.q), CALLS=str(self.root / 'calls.jsonl'),
                   ARMS=str(self.arms), REAL_HARNESS=str(HERE), TAIL_LABELS=' '.join(TAIL), DEATH_LINE=tap.DEATH_LINE, PYTHONDONTWRITEBYTECODE='1')
        for name, rc in (rcs or {}).items():
            env['RC_' + name.replace('.', '_').replace('-', '_')] = str(rc)
        if die_in:
            env['DIE_IN'] = die_in
        proc = subprocess.Popen(['bash', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        calls = [json.loads(x) for x in (self.root / 'calls.jsonl').read_text().splitlines()]
        steps = {m.group(1): m.group(2) for m in re.finditer(r'^STEP (\S+) (\S+)', out, flags=re.M)}
        return out, err, calls, steps

    def grade_file(self, name='alpha-next-grade.json'):
        return json.loads((self.q / name).read_text())

    def test_every_arm_in_order_the_core_grade_before_the_tail_then_the_read(self):
        out, err, calls, steps = self.run_loop()
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-1500:], err[-800:]))
        box_calls = [c.get('label') or c['action'] for c in calls if c['script'] == 'box']
        self.assertEqual(box_calls, GRADED + ['grade'] + TAIL + ['read'])
        for c in calls:
            if c['script'] == 'box' and c['action'] == 'arm':
                band = T.by_label()[c['label']]['band']
                self.assertEqual(c['workload'], ap.BAND_WORKLOAD[band] + '.json', c['label'])
                self.assertEqual(c['env']['S5_LAUNCHER_NAME'], 'l.sh')
        self.assertEqual([c['out'] for c in calls if c['script'] == 'checks' and c['action'] == 'full-idle'],
                         ['entry-%s.json' % x for x in GRADED] + ['core-full-idle.json'] + ['entry-%s.json' % x for x in TAIL]
                         + ['end-full-idle.json'])
        for name in GRADED:
            self.assertEqual(steps['arm-' + name], 'PASS', name)
        for name in TAIL:
            self.assertEqual((steps['arm-' + name], steps['entry-' + name], steps['ingress-' + name]),
                             ('READ-PASS', 'READ-PASS', 'READ-PASS'), name)
        self.assertEqual((steps['alpha-next'], steps['alpha-next-read'], steps['tail-engine'], steps['end-full-idle']),
                         ('PASS', 'READ-PASS', 'PASS', 'PASS'))
        result = self.grade_file()
        self.assertEqual((result['verdict'], result['missing'], result['required_arms']), ('PASS', [], GRADED))
        read = self.grade_file('alpha-next-read.json')
        self.assertEqual((read['verdict'], read['required_arms']), ('PASS', ALL_LABELS))
        self.assertEqual(re.findall(r'^## (.+)$', (self.q / 'alpha-next-report.md').read_text(), flags=re.M),
                         ['Mechanism', 'Performance', 'Release readiness'])

    def test_a_failing_tail_arm_is_a_read_and_never_counts(self):
        out, err, calls, steps = self.run_loop(rcs={TAIL[1]: 1, TAIL[2]: 2})
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, out[-1200:])
        self.assertEqual((steps['arm-' + TAIL[1]], steps['arm-' + TAIL[2]]), ('READ-FAIL', 'READ-INCONCLUSIVE'))
        self.assertEqual(self.grade_file()['verdict'], 'PASS')

    def test_past_the_core_limit_core_arms_are_inc_extra_arms_are_reads_and_the_tail_still_runs(self):
        out, err, calls, steps = self.run_loop(jump_after='ingress-' + CORE[9])
        ran = [c['label'] for c in calls if c['script'] == 'box' and c['action'] == 'arm']
        self.assertEqual(ran, CORE[:10] + TAIL)        # the tail limit is later than the core limit
        for name in CORE[10:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        for name in EXTRA:
            self.assertEqual(steps['arm-' + name], 'READ-INCONCLUSIVE', name)
            self.assertFalse((self.q / ('entry-%s.json' % name)).exists())
        self.assertIn('END FAILS=0 INCS=3 CRASHED=0', out, out[-800:])        # two core arms and the core grade
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'INC')
        self.assertIn('declared-arm-missing:' + CORE[10], result['missing'])
        self.assertFalse([x for x in result['missing'] if any(e in x for e in EXTRA)])

    def test_extra_arms_run_when_the_budget_allows_and_are_graded(self):
        # 600 s of budget left after the core: the four extra arms enter on any hub load. (It was 1 s: at hub load 70 the fake
        # arms' real seconds carried the fourth past the limit, g3 focused, HARNESS.) The limit's edge is tested by the +10 jumps.
        out, err, calls, steps = self.run_loop(jump_after='ingress-' + CORE[-1], jump_to='LAST_ARM_ENTRY_S - 600')
        self.assertEqual(steps['arm-' + EXTRA[0]], 'PASS')
        self.assertEqual([x for x in EXTRA if steps['arm-' + x] != 'PASS'], [])
        self.assertEqual(self.grade_file()['required_arms'], GRADED)

    def test_past_the_tail_limit_tail_arms_are_reads_not_counted(self):
        out, err, calls, steps = self.run_loop(jump_after='ingress-' + TAIL[1], jump_to='LAST_TAIL_ENTRY_S + 10')
        for name in TAIL[2:]:
            self.assertEqual(steps['arm-' + name], 'READ-INCONCLUSIVE', name)
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, out[-800:])
        self.assertEqual(steps['alpha-next-read'], 'READ-INCONCLUSIVE')

    def test_a_dead_engine_in_the_core_stops_the_run_and_skips_the_tail(self):
        out, err, calls, steps = self.run_loop(die_in=CORE[3])
        self.assertIn('CRASHED=1 SOURCE=engine-death', out, out[-800:])
        self.assertEqual(steps['arm-' + CORE[3]], 'FAIL')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box' and c['action'] == 'arm'], CORE[:4])
        self.assertEqual(steps['tail'], 'READ-INCONCLUSIVE')
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn(CORE[3] + ':engine-dead', result['failures'])
        self.assertTrue(json.loads((self.q / 'stop-race.json').read_text())['engine_crashed'])

    def test_a_dead_engine_in_the_tail_is_crash_accounted(self):
        out, err, calls, steps = self.run_loop(die_in=TAIL[0])
        self.assertIn('CRASHED=1 SOURCE=engine-death', out, out[-800:])
        self.assertEqual(steps['arm-' + TAIL[0]], 'READ-FAIL')
        self.assertEqual(self.grade_file()['verdict'], 'PASS')
        for name in TAIL[1:]:
            self.assertEqual(steps['arm-' + name], 'READ-INCONCLUSIVE', name)


# --- commands, derivation, window ----------------------------------------------------------------------

ARGPARSE_VALUES = dict(tap.ARGPARSE_VALUES, QNAME='q18a-alpha-next', RUN_ID='q18a-alpha-next-fixture', _label='c0on.g2-live-2',
                       _sel='c0on.g2', _c0='c0on', _group='g2', _cohort='live', _pair='2', _part='core', _limit='3420',
                       _rec='record', _skip='record', _arm='core:c0on.g2-live-2')


def bash_words(args, draft, values=None):
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    arrays = ('--arm "$Q/arm-c0on.g2-live-2.json" --ingress "c0on.g2-live-2=$Q/ingress-c0on.g2-live-2.json" '
              '--watch "c0on.g2-live-2=$Q/engine-watch-arm-c0on.g2-live-2.json"')
    lines += ['GRADE_ARGS=(%s)' % arrays, 'READ_ARGS=(%s)' % arrays,
              'source "$H/runs/q18a_alpha_next_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class Commands(unittest.TestCase):
    def test_every_body_command_parses_with_its_module(self):
        import e3_native39_trunk_checks as checks
        modules = {'e3_alpha_next_box': box, 'e3_native39_trunk_checks': checks}
        rows = body_commands(queue.derive()[DRAFT])
        self.assertEqual(sorted({m for m, _ in rows}), sorted(modules))
        actions = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, args=None, namespace=None):
            raise held.Parsed(original(parser, args, namespace))
        for module, args in rows:
            argv = held.expand(args, label_values('c0on.g2-live-2', READ_ARGS=tap.QUEUE_VALUES['GRADE_ARGS']))
            with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture):
                with self.assertRaises(held.Parsed) as parsed, contextlib.redirect_stderr(io.StringIO()):
                    modules[module].main(argv)
            actions.append((module, argv[0], vars(parsed.exception.args[0])))
        self.assertEqual([a[1] for a in actions], ['full-idle', 'arm', 'ingress', 'instruments', 'full-idle', 'grade', 'instruments',
                                                   'full-idle', 'read'])
        arm = next(ns for m, a, ns in actions if a == 'arm')
        self.assertEqual((arm['stage_local'], arm['size'], arm['cohort'], arm['index'], arm['control_timeout_s']),
                         ('on', 2, 'live', 2, 60.0))
        grade = next(ns for m, a, ns in actions if a == 'grade')
        self.assertEqual((grade['out'], grade['report']), ('/q/alpha-next-grade.json', '/q/alpha-next-report.md'))
        self.assertEqual(sorted(x.split('=')[0] for x in grade['check']), ['core-full-idle', 'instruments-start', 'unarmed-stock-prefill'])
        read = actions[-1][2]
        self.assertEqual((read['out'], read['report']), ('/q/alpha-next-read.json', '/q/alpha-next-read.md'))

    def test_the_loop_lists_are_the_table_and_the_split_runs_in_bash(self):
        draft = queue.derive()[DRAFT]
        (core,) = re.findall(r'^\s*for _arm in ([a-z0-9 :.-]+); do$', draft, flags=re.M)
        (tail,) = re.findall(r'^\s*for _label in ([a-z0-9 .-]+); do$', draft, flags=re.M)
        self.assertEqual([x.split(':') for x in core.split()], [[r['part'], r['label']] for r in T.arms(T.GRADED)])
        self.assertEqual(tail.split(), TAIL)
        split = ('  IFS=- read -r _sel _cohort _pair <<< "$_label"\n  IFS=. read -r _c0 _group <<< "$_sel"\n'
                 '  if [[ $_pair == %s ]]; then' % T.band1_glob())
        self.assertIn(split, draft)
        script = ('for _arm in %s %s; do _label=${_arm#*:}\n%s echo 1; else echo 2; fi\n'
                  'printf "%%s %%s %%s %%s %%s\\n" "${_arm%%%%:*}" "${_c0#c0}" "${_group#g}" "$_cohort" "$_pair"\ndone') % (
            core, tail, split)
        done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
        got = done.stdout.split('\n')
        bands = [int(x) for x in got[0::2] if x]
        rows = [x.split() for x in got[1::2] if x]
        self.assertEqual(bands, [r['band'] for r in T.arms()])
        self.assertEqual([(c, int(s), co, int(i)) for _, c, s, co, i in rows],
                         [(r['c0'], r['size'], r['cohort'], r['index']) for r in T.arms()])

    def test_abba_and_matched_pairs(self):
        core = T.arms('core')
        r0 = [(r['c0'], r['cohort']) for r in core if r['index'] == 0]
        r1 = [(r['c0'], r['cohort']) for r in core if r['index'] == 1]
        self.assertEqual(r1, r0[::-1])
        r3 = [(r['c0'], r['cohort']) for r in T.arms('extra')]
        self.assertEqual(r3, r0)
        for row in T.arms(T.GRADED):
            if row['c0'] == 'on':
                self.assertEqual(T.matched_off(row), row['label'].replace('c0on', 'c0off'))
        self.assertEqual({r['size'] for r in T.arms('tail')}, {4})
        self.assertEqual({r['c0'] for r in T.arms('tail')}, {'on'})
        self.assertEqual([r['part'] for r in T.arms()], ['core'] * 12 + ['extra'] * 4 + ['tail'] * 6)
        self.assertEqual((T.BAND1_ROUNDS, T.BAND1_INDEXES, T.STANDING_GREEDY, T.STANDING_SAMPLED),
                         (3, (0, 1, 3), (0, 1), (2, 3, 4, 5, 6, 7)))

    def test_the_table_refuses_a_malformed_order(self):
        for table, text in (((('tail', 'on', 'live', 0), ('core', 'off', 'live', 0)), 'parts in order'),
                            ((('core', 'on', 'live', 0),), 'matched'),
                            ((('core', 'off', 'live', 0), ('core', 'off', 'live', 0)), 'repeated')):
            with self.subTest(text=text), mock.patch.object(T, 'ARM_TABLE', table), self.assertRaisesRegex(AssertionError, text):
                T.check()


class QueueArgparse(unittest.TestCase):
    def test_every_queue_python_call_parses_with_its_real_parser(self):
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        scripts = sorted({c[1] for c in calls})
        self.assertEqual(scripts, sorted([
            'SETCFG', 'e2_memory_attribution.py', 'e3_native28_host_observer.py', 'e3_native30_e_grade.py',
            'e3_native39_trunk_checks.py', 'e3_native39_w2_length.py', 'e3_alpha_next_box.py',
            'e3_native41_unarmed_box.py', 'e3_native_speculation_gate.py', 'alpha_next_stop_race.py', 'e3_native_startup_grade.py',
            'e3_private_auth.py', 'engine_identity.py', 'native35_ledger_grade.py', 'native43_engine_death.py',
            'native43_allocator_phase_deltas.py']))
        setcfg = next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file())
        rows = []
        for name, script, args, form in calls:
            with self.subTest(file=name.split('/')[-1], script=script, args=args[:60]):
                rc, words, err = bash_words(args, draft)
                self.assertEqual(rc, 0, err)
                if form != 'python3':
                    words = held.QLIB_CALLS[form][1](words)
                if script == 'SETCFG':
                    verdict, detail = held.setcfg_parse(setcfg, words)
                else:
                    verdict, detail = held.real_parse(HERE / script, words)
                self.assertEqual(verdict, 'parsed', (script, words[:4], detail))
                rows.append(dict(file=name.split('/')[-1], script=script, form=form,
                                 action=words[0] if words and not words[0].startswith('-') else None,
                                 words=len(words), verdict=verdict))
        self.assertEqual(len(rows), len(calls))
        # Step B's calls plus the body's core FULL, the read and the STOP_RACE classifier (a plain "$H" entry).
        parent = held.queue_python_calls(step_b_queue.derive(), (HERE / 'qlib.sh').read_text())
        self.assertEqual(len(calls), len(parent) + 3)
        self.assertEqual(len([c for c in calls if c[1] == 'alpha_next_stop_race.py']), 1)
        ARGPARSE_RECEIPTS[:] = rows
        self.assertEqual(SLOT_RECEIPTS, [])

        def first(script, action):
            return next(a for n, s, a, f in calls if s == script and a.split()[0] == action)
        arm, grade = first('e3_alpha_next_box.py', 'arm'), first('e3_alpha_next_box.py', 'grade')
        for label, original, (old, new) in (('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec')),
                                            ('arm-undeclared-size', arm, ('--size "${_group#g}"', '--size 3')),
                                            ('arm-undeclared-c0', arm, ('--stage-local "${_c0#c0}"', '--stage-local maybe')),
                                            ('arm-undeclared-round', arm, ('--index "$_pair"', '--index 6')),
                                            ('grade-without-out', grade, ('--out "$Q/alpha-next-grade.json"', ''))):
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1)
                rc, words, err = bash_words(original.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_alpha_next_box.py', words)[0], 'refused', label)


class Derivation(unittest.TestCase):
    def test_derived_texts_are_on_disk_parse_and_check(self):
        derived = queue.derive()
        for name, text in derived.items():
            self.assertEqual((ROOT / name).read_text(), text, name)
            self.assertEqual(subprocess.run(['bash', '-n', str(ROOT / name)], capture_output=True).returncode, 0, name)
        done = subprocess.run([sys.executable, str(ROOT / 'tools/alpha_next_queue.py'), '--check'], capture_output=True, text=True,
                              timeout=120)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, 'alpha-next queue verified'), done.stderr[-400:])
        extra = [line for line in derived[SETTINGS].splitlines() if line.strip().startswith('"EXTRA_ENV=')]
        self.assertEqual(len(extra), 1)
        words = extra[0].split()
        for flag in ('FLASH_S5_AUTO_ARM=1', 'FLASH_S5_NATIVE_PROGRESS=burst', 'FLASH_S5_NATIVE_QSA_POLICY=dense', 'FLASH_S5_QSA_FUSED=1',
                     'FLASH_S5_LAYER_GROUPS=1', 'FLASH_S5_STAGE_LOCAL=1', 'FLASH_S5_HANDLER_ANATOMY=1'):
            self.assertEqual(words.count(flag), 1, flag)
        self.assertIn('FLASH_S5_LAYER_GROUPS=1 FLASH_S5_STAGE_LOCAL=1 FLASH_S5_HANDLER_ANATOMY=1 ', extra[0])
        for absent in ('FLASH_S5_UNION_RETIRE', 'FLASH_S5_PROBE_', 'FLASH_S5_NATIVE_QSA_TRACE', 'FLASH_S5_NATIVE_QSA_STOP_AFTER'):
            self.assertNotIn(absent, extra[0])

    def test_settings_add_those_tokens_and_their_comment_nothing_else(self):
        old = queue.rename((ROOT / 'harness/runs/q18a_step_b_settings.sh').read_text())
        new = queue.derive()[SETTINGS]
        diff = [line for line in difflib.ndiff(old.splitlines(), new.splitlines()) if line[:1] in '+-']
        self.assertEqual([line[2:].replace(' ' + queue.SWITCHES, '') for line in diff if line[0] == '+' and 'EXTRA_ENV' in line],
                         [line[2:] for line in diff if line[0] == '-'])
        self.assertEqual([line[2:] for line in diff if line[0] == '+' and 'EXTRA_ENV' not in line], queue.SETTINGS_NOTE.splitlines())

    def test_the_declaration_adds_the_switch_asserts_and_the_file_pins_only(self):
        old = queue.rename((ROOT / 'harness/runs/q18a_step_b_declare.sh').read_text())
        new = queue.derive()[DECLARE]
        diff = [line for line in difflib.ndiff(old.splitlines(), new.splitlines()) if line[:1] in '+-']
        removed = [x[2:] for x in diff if x[0] == '-']
        self.assertEqual(len(removed), 1)
        self.assertIn(queue.PINS_OLD.strip(), removed[0])
        added = [x[2:] for x in diff if x[0] == '+']
        self.assertEqual(sorted(added), sorted(queue.DECLARE_SWITCHES.splitlines()
                                               + [removed[0].replace(queue.PINS_OLD, queue.PINS_NEW)]))

    def test_no_bundle_or_package_digest_in_the_declaration_and_no_harness_reader_of_the_pins_file(self):
        """Step B's way (coordinator, 2 Nov): the declare script is in the harness bundle, so it holds no bundle, manifest or
        package digest; the identity is pinned outside the bundle. The pins file is a hub-side record nothing in harness/ reads."""
        old = queue.rename((ROOT / 'harness/runs/q18a_step_b_declare.sh').read_text())
        new = queue.derive()[DECLARE]
        for word in ('ALPHA_NEXT_OVERLAY_MANIFEST_SHA256', 'ALPHA_NEXT_PACKAGE_SHA256', 'ALPHA_NEXT_BUNDLE', 'PLACEHOLDER',
                     'pins.json', 'alpha-next pin'):
            self.assertNotIn(word, new)
        # Every 64-hex literal of the declaration is one Step B's declaration already holds (image and library pins).
        hexes = lambda text: set(re.findall(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', text))
        self.assertLessEqual(hexes(new), hexes(old))
        name = ('alpha_next_queue' + '_pins').encode()        # built here so this file does not hold the literal either
        # In the bundle, harness/ also holds the bundled hub tools (native35_tools/, the hub's tools/, which name the pins
        # writer) and SHA256SUMS (which lists them); neither exists under the hub's harness/, so both trees scan the same files.
        readers = [str(f.relative_to(ROOT)) for f in sorted(HERE.rglob('*'))
                   if f.is_file() and '__pycache__' not in f.parts and 'native35_tools' not in f.relative_to(HERE).parts
                   and f != HERE / 'SHA256SUMS' and name in f.read_bytes()]
        self.assertEqual(readers, [])
        for text in queue.derive().values():
            self.assertNotIn(name.decode(), text)
        # The hub-side record keeps its schema and three keys; the packaging chain's writer (the pack branch pins writer
        # in tools/) owns the values and may add keys, so neither the values nor an exact key set are asserted.
        if in_bundle():
            # The pins file is hub-side only: the build never bundles it (the bundle self-run proves that here).
            self.assertFalse(queue.PINS_FILE.exists(), queue.PINS_FILE)
        else:
            pins = json.loads(queue.PINS_FILE.read_text())
            self.assertEqual(pins['schema'], 'alpha-next-queue-pins/1')
            self.assertLessEqual({'overlay_manifest_sha256', 'package_sha256', 'harness_bundle_sha256', 'status'}, set(pins))

    def test_everything_kept_is_step_b_line_for_line(self):
        old = queue.rename((ROOT / 'harness/runs/q18a_step_b.draft.sh').read_text())
        body_old = queue.rename(step_b_queue.BODY)
        self.assertEqual(old.count(body_old), 1)
        new = queue.derive()[DRAFT]
        body_new = queue.body()
        self.assertEqual(new.count(body_new), 1)
        diff = [line for line in difflib.ndiff(old.replace(body_old, '').splitlines(), new.replace(body_new, '').splitlines())
                if line[:1] in '+-']
        removed = [line[2:] for line in diff if line[0] == '-']
        added = [line[2:] for line in diff if line[0] == '+']
        declared_old = (queue.BUDGET_OLD + queue.PRINT_OLD + queue.WINDOW_OLD + queue.INTERFACE_OLD + '\n' + queue.CRASH_OLD
                        + queue.FINAL_DICT_OLD).splitlines()
        renamed = ("perf=p.parent/'step-b-grade.json'", "'missing':['step-b-grade-absent']", 'Every cap is step_b_gate.PHASES.',
                   'Step A text above; every cap here is step_b_gate.')
        for line in removed:
            self.assertTrue(any(line.strip() and line.strip() in d for d in declared_old)
                            or any(len(d.strip()) > 30 and d.strip() in line for d in declared_old)
                            or any(r in line for r in renamed), line)
        declared_new = (queue.HEADER_NOTE + queue.BUDGET_NEW + queue.PRINT_NEW + queue.WINDOW_NEW + queue.INTERFACE_NEW + '\n'
                        + queue.DRAFT_SWITCHES + queue.CRASH_NEW + queue.STATE_NEW + queue.FINAL_READ_NEW).splitlines()
        for line in added:
            self.assertTrue(line.strip() in {d.strip() for d in declared_new}
                            or any(len(d.strip()) > 30 and d.strip() in line for d in declared_new) or 'alpha-next-grade' in line or 'alpha_next_gate' in line
                            or queue.FINAL_DICT_NEW in line or 'Step A and Step B text above' in line, line)

    def test_the_draft_reads_alpha_next_only(self):
        draft = queue.derive()[DRAFT]
        for text in ('from alpha_next_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,LAST_TAIL_ENTRY_S,PHASES',
                     "perf=p.parent/'alpha-next-grade.json'", 'python3 "$H/e3_alpha_next_box.py" arm',
                     'python3 "$H/e3_alpha_next_box.py" grade', 'python3 "$H/e3_alpha_next_box.py" read',
                     "assert env.get('FLASH_S5_STAGE_LOCAL') == '1'", "assert env.get('FLASH_S5_HANDLER_ANATOMY') == '1'",
                     "assert not [k for k in env if k.startswith('FLASH_S5_PROBE_')]",
                     "assert 'FLASH_S5_UNION_RETIRE' not in env"):
            self.assertIn(text, draft)
        for text in ('from step_b_gate import', 'python3 "$H/e3_step_b_box.py"', 'q18a_step_b', 'step-b-grade', 'q18a-step-b',
                     'TRIM_G4', 'trim_g4'):
            self.assertNotIn(text, draft)
        self.assertEqual(queue.body().count('engine_watched '), 4)

    def test_frozen_files_are_byte_identical(self):
        self.assertEqual(sorted(FROZEN_SHA256), sorted(FROZEN))
        bundle = in_bundle()
        for name in FROZEN:
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), FROZEN_SHA256[name], name)
                if not bundle:
                    done = subprocess.run(['git', '-C', str(ROOT), 'show', FROZEN_AT + ':' + name], capture_output=True)
                    self.assertEqual(done.returncode, 0, name)
                    self.assertEqual(hashlib.sha256(done.stdout).hexdigest(), FROZEN_SHA256[name], name)


class Window(unittest.TestCase):
    def test_the_box_budget_and_the_entry_limits_are_relative(self):
        import native40_box_gate as n40
        import native41_box_gate as n41
        self.assertEqual(gate.CLEANUP_S, n40.CLEANUP_S)
        for name in n41.KEPT:
            self.assertEqual(gate.PHASES[name], n40.PHASES[name], name)
        self.assertEqual((gate.BOX_BUDGET_S, gate.ACTIVE_S + gate.CLEANUP_S), (9000, 9000))
        self.assertEqual((gate.ARMS, gate.CORE_ARMS, gate.EXTRA_ARMS, gate.TAIL_ARMS), (22, 12, 4, 6))
        self.assertEqual(gate.ARM_RESERVE_S, 450 + 1500 + 30 + 60)
        self.assertEqual(gate.LAST_ARM_ENTRY_S + gate.ARM_RESERVE_S + gate.TAIL_RESERVE_S, gate.ACTIVE_S)
        self.assertEqual(gate.LAST_TAIL_ENTRY_S + gate.ARM_RESERVE_S + gate.END_RESERVE_S, gate.ACTIVE_S)
        self.assertEqual((gate.LAST_ARM_ENTRY_S, gate.LAST_TAIL_ENTRY_S), (3420, 4470))
        self.assertEqual(gate.ENTRY_LIMIT_S, {'LAST_ARM_ENTRY_S': 3420, 'LAST_TAIL_ENTRY_S': 4470})
        self.assertEqual({p: T.ENTRY_LIMIT[p][0] for p in T.PARTS},
                         {'core': 'LAST_ARM_ENTRY_S', 'extra': 'LAST_ARM_ENTRY_S', 'tail': 'LAST_TAIL_ENTRY_S'})
        for part, limit in (('core', 3420), ('extra', 3420), ('tail', 4470)):
            self.assertTrue(gate.arm_fits(limit, part))
            self.assertFalse(gate.arm_fits(limit + 1, part))
        exp = gate.EXPECTED_RUNTIME_S
        self.assertEqual((exp['by_band']['arms_entering'], exp['at_step_b_rate']['arms_entering']), (22, 22))
        self.assertEqual((exp['at_c0_rate']['core_arms_entering'], exp['at_c0_rate']['extra_arms_entering'],
                          exp['at_c0_rate']['tail_arms_entering']), (12, 1, 4))
        self.assertLess(exp['at_c0_rate']['total_s'], gate.BOX_BUDGET_S)
        body = queue.body()
        for text, count in (('timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments', 2),
                            ('timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle', 3),
                            ('timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_alpha_next_box.py" arm', 1),
                            ('timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress', 1),
                            ('timeout 600s python3 "$H/e3_alpha_next_box.py" grade', 1),
                            ('timeout 600s python3 "$H/e3_alpha_next_box.py" read', 1)):
            self.assertEqual(body.count(text), count, text)
        self.assertEqual(gate.GRADE_CAP_S, 600)
        func = tap.le_fix_function(queue.derive()[DRAFT], 'native41_budget')
        for key, value in (('active', gate.ACTIVE_S), ('cleanup', gate.CLEANUP_S), ('health', gate.PHASES['health']),
                           ('end_identity', gate.PHASES['end_identity']), ('last_arm_entry', gate.LAST_ARM_ENTRY_S),
                           ('last_tail_entry', gate.LAST_TAIL_ENTRY_S)):
            done = subprocess.run(['bash', '-c', func + 'native41_budget ' + key],
                                  env=dict(os.environ, HARNESS_DIR=str(HERE)), capture_output=True, text=True)
            self.assertEqual((done.returncode, done.stdout.strip()), (0, str(value)), done.stderr)

    def test_the_tool_prints_the_table_with_its_limits(self):
        done = subprocess.run([sys.executable, str(ROOT / 'tools/alpha_next_queue.py'), '--table'], capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        lines = done.stdout.splitlines()
        self.assertIn('never a clock', lines[0])
        self.assertEqual(len([x for x in lines if re.match(r'^\d+ ', x)]), 22)
        self.assertEqual(len([x for x in lines if x.endswith('LAST_TAIL_ENTRY_S 4470 s')]), 6)
        self.assertEqual(len([x for x in lines if x.endswith('LAST_ARM_ENTRY_S 3420 s')]), 16)


FORBIDDEN = ('HARD_END', 'NOT_BEFORE', 'window_ok', 'latest_arm', 'latest_entry', 'latest_start', 'not_before', 'check_slot',
             'PY_SLOT', 'night-window', 'datetime', 'strptime', 'mktime', 'timegm', 'fromisoformat', 'date -d', 'time.time(',
             'time.localtime', 'time.gmtime', 'import time', 'from time')
ALLOWED_DRAFT = ('RUN_ID="q18a-alpha-next-$(date -u +%Y%m%dT%H%M%SZ)-$$"', "json.dump({'schema':1,'wall_ns':time.time_ns(),",
                 'EPOCHREALTIME')


class NoClock(unittest.TestCase):
    """No absolute-time construct in the new files; the draft keeps only Step B's recording stamps (never a decision)."""

    def test_no_absolute_time_or_window_in_the_new_files(self):
        for name in NEW_FILES + (DRAFT, SETTINGS, DECLARE):
            text = (ROOT / name).read_text()
            for line_no, line in enumerate(text.splitlines(), 1):
                if name in (DRAFT, SETTINGS, DECLARE) and any(a in line for a in ALLOWED_DRAFT):
                    continue
                for word in FORBIDDEN:
                    self.assertNotIn(word, line, '%s:%d' % (name, line_no))
                self.assertIsNone(re.search(r'\$\(date\b|`date\b|\bdate \+', line), '%s:%d' % (name, line_no))
        draft = queue.derive()[DRAFT]
        self.assertEqual(draft.count('$(date'), 1)
        self.assertEqual(draft.count('time.time_ns()'), 1)
        self.assertEqual(len(re.findall(r'EPOCHREALTIME', draft)), len(re.findall(r'EPOCHREALTIME', (ROOT / 'harness/runs/q18a_step_b.draft.sh').read_text())))

    def test_the_gate_and_the_table_import_no_clock(self):
        for name in ('harness/alpha_next_gate.py', 'harness/alpha_next_arms.py', 'harness/alpha_next_stop_race.py',
                     'harness/e3_alpha_next_box.py'):
            tree = ast.parse((ROOT / name).read_text())
            imported = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            imported |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse(imported & {'time', 'datetime', 'calendar', 'zoneinfo'}, name)

    def test_this_detector_catches_each_construct(self):
        for word in FORBIDDEN:
            self.assertTrue(any(word in line for line in ['x = ' + word]), word)


class C0Parity(unittest.TestCase):
    """C0's declarations restated in e3_alpha_next_box (importing e3_alpha_c0_box wraps chunk_breakdown process-wide)."""

    def c0(self):
        tree = ast.parse((HERE / 'e3_alpha_c0_box.py').read_text())
        out = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    out[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    out[node.targets[0].id] = ast.get_source_segment((HERE / 'e3_alpha_c0_box.py').read_text(), node.value)
        return out

    def test_constants_equal_c0s(self):
        c0 = self.c0()
        self.assertEqual(box.STAGE_COUNTERS, c0['STAGE_COUNTERS'])
        self.assertEqual(box.TARGETS, c0['TARGETS'])
        self.assertEqual(box.STAGE_SIGNATURES, c0['SIGNATURES'])
        self.assertEqual(box.STAGE_LINE, c0['STAGE_LINE'])

    def test_the_stage_line_parses_both_formats(self):
        c0_line = ('flash-s5 stage-local txn=3 chunk=0-1000 commands=9,9 both_stages_ns=2000000 first_ns=1,1 last_ns=2,2 '
                   'local_release_ns=2,2 headroom_refused=1')
        self.assertEqual(box.parse_stage(c0_line)['headroom_blocked_ns'], None)
        self.assertEqual(box.parse_stage(c0_line + ' headroom_blocked_ns=4500')['headroom_blocked_ns'], 4500)
        src = (ROOT / 'patches/s5_engine/stage_progress.py').read_text()
        self.assertRegex(src, r"local_release_ns=%s,%s headroom_refused=%d '\s*'headroom_blocked_ns=%d'")
        self.assertIsNone(box.parse_stage('flash-s5 stage-local txn=3 chunk=0-'))

    def test_the_peak_line_is_the_engines(self):
        src = ROOT / 'patches/s5_engine/worker_bridge.py'
        line = ('flash-s5 memory-peak rank=2 txn_seq=4 chunk=0-1024 peak_allocated=123 peak_reserved=456 allocated_at_open=7 '
                'stage_timer=1')
        m = box.PEAK_RE.search(line)
        self.assertEqual(m.groups(), ('2', '4', '0', '1024', '123', '456', '7', '1'))
        self.assertFalse(line.startswith('flash-s5 memory '))       # never the MEMORY_MARK receipt
        text = src.read_text()                                      # merged from claude/alpha-next (32e7cc63)
        self.assertIn("PEAK_LINE = 'flash-s5 memory-peak '", text)
        self.assertRegex(text, r"'rank=%d txn_seq=%d chunk=%d-%d peak_allocated=%d peak_reserved=%d '\s*"
                               r"'allocated_at_open=%d stage_timer=%d'")

    def test_the_breakdown_wrapper_is_c0s_and_idempotent(self):
        calls = []

        def original(chunk, workers):
            calls.append(1)
            return dict(missing_worker_rows=[dict(kind='receipt'), dict(kind='segment')])
        chunk = dict(rows={'n': [dict(kind='segment', stage=0, t_recv0=1, t_recv1=2, t_recv2=3),
                                 dict(kind='segment', stage=1, t_recv2=1, t_recv3=1), dict(kind='open', stage=0, t_recv0=1)]})
        once = box.stage_reply_breakdown(original)
        info = once(chunk, {})
        self.assertEqual((info['foreign_replies'], info['short_replies'], info['late_receipt_rows']), (1, 2, 1))
        self.assertEqual([m['kind'] for m in info['missing_worker_rows']], ['release', 'segment'])
        twice = box.stage_reply_breakdown(box.stage_reply_breakdown(original))
        self.assertEqual(twice(chunk, {})['late_receipt_rows'], 1)


class ReceiptIntegrity(unittest.TestCase):
    """receipt_lines (PLAN 2, merged from claude/alpha-next) over the followed logs: RECEIPT_INTEGRITY_IS_GATE False (lead: a read
    this run, a gate next run), per-process counts, a headline under Overall in both cases, the would-be-INC reads."""
    TEXTS = ['flash-s5 memory-peak rank=0 txn_seq=%d chunk=0-1024 peak_allocated=1 peak_reserved=2 allocated_at_open=0 '
             'stage_timer=0' % i for i in range(3)]
    STAGE = ('flash-s5 stage-local txn=1 chunk=0-1024 commands=9,9 both_stages_ns=2000000 first_ns=1,1 last_ns=2,2 '
             'local_release_ns=2,2 headroom_refused=1 headroom_blocked_ns=0')

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix='receipts-'))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    @staticmethod
    def line(text, pid, seq, sha=None):
        return '%s pid=%d seq=%d sha=%s' % (text, pid, seq, sha or receipt_lines.sha16([text]))

    def read(self, name, lines):
        log = self.dir / ('%s.log' % name)
        log.write_text(''.join('2026-11-02T12:00:00.000000000Z (Worker pid=1) ' + x + '\n' for x in lines))
        return box.receipt_read([str(log), str(self.dir / 'absent.err')])

    def test_the_declaration(self):
        self.assertIs(box.RECEIPT_INTEGRITY_IS_GATE, False)
        src = (HERE / 'e3_alpha_next_box.py').read_text()
        self.assertEqual(len(re.findall(r'^RECEIPT_INTEGRITY_IS_GATE = (True|False)$', src, flags=re.M)), 1)
        self.assertIn("the PLAN 2 writer's behaviour on the box FIFO is unproven, and this run\n# proves it", src)
        self.assertLess(src.index('NUMERICS_IS_GATE = False'), src.index('RECEIPT_INTEGRITY_IS_GATE = False'))

    def test_clean_and_dirty_per_process_counts_and_headlines(self):
        t = self.TEXTS
        clean = self.read('ok', [self.line(x, 7, i) for i, x in enumerate(t)] + [self.line(self.STAGE, 9, 0)])
        self.assertEqual((clean['verdict'], clean['clean'], clean['is_gate'], clean['per_process'], clean['would_be_inc']),
                         ('OK', True, False, {}, []))
        self.assertEqual(clean['headline'],
                         'RECEIPT INTEGRITY OK (a read this run, the verdict does not change): all clean, 4 records from 2 processes')
        dirty = self.read('dirty', [self.line(t[0], 7, 0), self.line(t[2], 7, 2), self.line(self.STAGE, 9, 0, sha='0' * 16),
                                    self.line(self.STAGE, 11, 0), 'flash-s5 stop {"schema": "s5-stop-receipt/1"}'])
        self.assertEqual((dirty['verdict'], dirty['clean'], dirty['is_gate']), ('INC', False, False))
        zero = dict.fromkeys(box.RECEIPT_PROBLEMS, 0)
        self.assertEqual(dirty['per_process'], {'7': dict(zero, missing_seq=1), '9': dict(zero, bad_sha=1)})
        self.assertEqual(dirty['log_wide_counts'], dict(zero, unsafe_path=1))
        self.assertEqual(dirty['headline'],
                         'RECEIPT INTEGRITY INC (a read this run, the verdict does not change): '
                         'pid 7: missing_seq 1, bad_sha 0, incomplete 0, unsafe_path 0; '
                         'pid 9: missing_seq 0, bad_sha 1, incomplete 0, unsafe_path 0; '
                         'log-wide: missing_seq 0, bad_sha 0, incomplete 0, unsafe_path 1')
        # Which process fed which read: 7 the memory-peak receipts, 9 the stage line, the identity-less stop line log-wide.
        self.assertEqual([(w['pid'], w['problems'], w['kinds'], w['reads']) for w in dirty['would_be_inc']],
                         [(7, ['missing_seq'], ['memory-peak'], ['memory_peak']),
                          (9, ['bad_sha'], ['stage-local'], ['c0_engagement', 'c0_ring']),
                          (None, ['unsafe_path'], ['stop'], ['stop_receipts'])])
        self.assertEqual(dirty['would_be_inc_reads'], ['c0_engagement', 'c0_ring', 'memory_peak', 'stop_receipts'])
        # The identity suffix does not change what the box parses.
        self.assertEqual(box.PEAK_RE.search(self.line(t[1], 7, 1)).group(2), '1')

    def test_dirty_integrity_changes_no_verdict_and_both_headlines_lead_the_report(self):
        grade = Grader()
        self.addCleanup(grade.tmp.cleanup)
        emit, arms = run_of('graded')
        result = grade(emit, arms)          # the emitters write identity-less lines: dirty (legacy, log-wide)
        self.assertEqual((result['verdict'], result['receipt_integrity_is_gate'], result['receipt_integrity_clean']),
                         ('PASS', False, False))
        self.assertEqual(result['receipt_integrity']['verdict'], 'INC')
        self.assertEqual(result['receipt_integrity_headline'], result['receipt_integrity']['headline'])
        self.assertIn('engagement', result['receipt_integrity']['would_be_inc_reads'])
        text = box.report(result).splitlines()
        self.assertEqual(text[4], '**%s**' % result['receipt_integrity_headline'])
        self.assertTrue(text[4].startswith('**RECEIPT INTEGRITY INC (a read this run, the verdict does not change): log-wide: '))
        self.assertTrue([x for x in text if x.startswith('- receipt_lines rule would make INC: process log-wide (unsafe_path; '
                                                         'kinds ') and 'engagement' in x])
        # With an OUTSIDE numerics read too: both bold lines under Overall, numerics first; still PASS.
        name = 'c0on.g2-solo-0'
        for a in arms:
            if a['label'] == name:
                a['outputs'] = outputs(T.by_label()[name], cut=10)
        result = grade(emit, arms)
        text = box.report(result).splitlines()
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(text[4:7], ['**%s**' % result['numerics_headline'], '', '**%s**' % result['receipt_integrity_headline']])
        # The one constant flipped: dirty integrity makes the queue INC.
        with mock.patch.object(box, 'RECEIPT_INTEGRITY_IS_GATE', True):
            gated = grade(emit, arms)
        self.assertEqual((gated['verdict'], gated['receipt_integrity']['is_gate']), ('INC', True))
        self.assertIn('receipt-integrity:INC', gated['missing'])

    def test_a_clean_log_headline_in_the_report(self):
        log = self.dir / 'engine-stream.log'
        log.write_text(''.join('2026-11-02T12:00:00.000000000Z (Worker pid=1) ' + self.line(x, 7, i) + '\n'
                               for i, x in enumerate(self.TEXTS)))
        clean = box.receipt_read([str(log)])
        grade = Grader()
        self.addCleanup(grade.tmp.cleanup)
        with mock.patch.object(box, 'receipt_read', return_value=clean):
            result = grade(*run_of('graded'))
        self.assertEqual((result['verdict'], result['receipt_integrity_clean']), ('PASS', True))
        text = box.report(result).splitlines()
        self.assertEqual(text[4], '**RECEIPT INTEGRITY OK (a read this run, the verdict does not change): all clean, 3 records '
                                  'from 1 processes**')
        self.assertFalse([x for x in text if x.startswith('- receipt_lines rule would make INC')])


class Report(unittest.TestCase):
    """tools/alpha_next_report.py: step_a_report wrapped, its key holds the QSA mode, the size and the C0 selection."""

    def test_the_grouping_key_and_the_peak_read(self):
        emit, arms = run_of('all', rows=T.arms())
        d = Path(tempfile.mkdtemp(prefix='anr-'))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        tap.write_streams(d, emit.lines)
        os.replace(d / 'engine-stream.log', d / 'engine-current.log')
        (d / 'engine-stream.err').unlink()
        for a in arms:
            (d / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(anr.main([str(d), str(d / 'out.json')]), 0)
        out = json.loads((d / 'out.json').read_text())
        tokens = sorted({k.partition('-')[0] for k in out['summary']})
        self.assertEqual(tokens, ['fused.g2.c0off', 'fused.g2.c0on', 'fused.g4.c0on'])
        self.assertFalse([k for k in tokens if '-' in k])
        self.assertEqual((out['old'], out['new'], out['unkeyed_arms']), ('fused.g2.c0off', 'fused.g2.c0on', []))
        self.assertTrue(out['compare'])
        row = next(v for k, v in out['summary'].items() if k.startswith('fused.g4.c0on-'))
        self.assertEqual((row['qsa_mode'], row['group_size'], row['stage_local']), ('fused', 4, True))
        self.assertEqual(out['memory_peak']['fused.g2.c0on']['ranks']['0']['peak_allocated_mib'], 21950.0)
        self.assertIs(out['memory_peak_is_gate'], False)
        self.assertEqual(anr.group_key(2, False), 'fused.g2.c0off')


if __name__ == '__main__':
    unittest.main()
