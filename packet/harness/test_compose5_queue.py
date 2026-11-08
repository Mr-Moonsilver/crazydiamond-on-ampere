#!/usr/bin/env python3
"""q18a-parity-o3f queue: the derivation from a1's committed texts, the name rules with the runner's real QNAME, the budget, the derived arm loop under real bash with the real grade, and every python call against its real parser. CPU only; no pass here is box evidence."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools')]
import test_inwait_queue as tiq  # noqa: E402  (first: paths)
import compose5_arms as T  # noqa: E402
import compose5_gate as gate  # noqa: E402
import compose5_queue as queue  # noqa: E402
import engine_identity  # noqa: E402
import inwait_queue_names as names  # noqa: E402
import test_compose5_box as tbox  # noqa: E402

tap, tq, tan = tiq.tap, tiq.tq, tiq.tan
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
RUNS = HERE / 'runs'
DRAFT = 'harness/runs/q18a_parity_n3s24q2024rc6144.draft.sh'
SETTINGS = 'harness/runs/q18a_parity_n3s24q2024rc6144_settings.sh'
DECLARE = 'harness/runs/q18a_parity_n3s24q2024rc6144_declare.sh'
CORE = T.labels('core')
DECODE = [d['label'] for d in T.DECODE_ARMS]
QNAME = 'q18a-parity-n3s24q2024rc6144'
STUB_BOX = tiq.STUB_BOX.replace('import e3_inwait_box as real', 'import e3_compose5_box as real')
assert STUB_BOX != tiq.STUB_BOX


class Derivation(unittest.TestCase):
    def test_the_files_are_the_derivation(self):
        derived = queue.derive()
        self.assertEqual(sorted(derived), sorted((DRAFT, SETTINGS, DECLARE)))
        for name, text in derived.items():
            self.assertEqual((ROOT / name).read_text(), text, name)
        self.assertIn(queue.final_body(), derived[DRAFT])

    def test_the_declared_deltas_are_there(self):
        derived = queue.derive()
        draft, settings, declare = derived[DRAFT], derived[SETTINGS], derived[DECLARE]
        self.assertIn('python3 "$H/e3_compose5_box.py" arm', draft)
        self.assertNotIn('e3_a1_trunk_box.py" arm', draft)
        self.assertIn('--generation "$(python3 "$H/compose5_arms.py" generations "$_label" "$Q/arm-$_label.json")"', draft)
        self.assertIn('--generation 8 ', draft, 'the decode arm keeps its 8')
        self.assertIn('for _arm in ' + queue.CORE + '; do', draft)
        self.assertNotRegex(draft, r'(?<![a-z])(on|off|mn|mo|ctl)\.g2-')
        env = re.findall(r'^\s*"EXTRA_ENV=([^"]*)"', settings, re.M)
        self.assertEqual(len(env), 1)
        pairs = dict(tok.split('=', 1) for tok in env[0].split() if '=' in tok)
        for name, value in T.LAUNCH_ENV.items():
            self.assertEqual(pairs[name], str(value), name)
        for name, value in (('CUDA_DEVICE_MAX_CONNECTIONS', '32'), ('FLASH_S5_PLE_PRIVATE', '1'), ('FLASH_S5_PLE_WITNESS', '1')):
            self.assertEqual(pairs[name], value, name)
            self.assertIn("assert env.get('%s') == " % name, draft)
            self.assertIn("assert env.get('%s')==" % name, declare)
        self.assertEqual((pairs['FLASH_S5_INWAIT'], pairs['FLASH_S5_AUTO_ARM']), ('1', '1'))
        self.assertEqual(pairs['FLASH_S5_CHUNK_TOKENS'], '2048', 'b13 is a plan under a 2048 launch')
        self.assertIn("assert env.get('FLASH_S5_CHUNK_TOKENS') == '2048'", draft)
        self.assertIn("assert env.get('FLASH_S5_CHUNK_TOKENS')=='2048'", declare)
        self.assertNotIn("'FLASH_S5_CHUNK_TOKENS' not in env", draft + declare)
        for module in ('e3_a1_trunk_box.py', 'a1_trunk_arms.py', 'compose4_replay.py', 'e3_compose4_box.py', 'compose4_arms.py', 'agent_replay.py', 'e3_inwait_box.py', 'inwait_arms.py', 'e3_native_refusal_client.py', 'inwait_gate.py'):
            self.assertIn(module, draft, 'the box imports it: the preflight lists it')
            self.assertIn(module, declare, 'the box imports it: the declaration lists it')
        self.assertNotRegex(draft, r'a1-trunk-(grade|read|report)', 'the records carry this queue\'s name')

    def test_the_default_chunk_is_1024_so_the_launch_must_say_2048(self):
        import async_handoff
        self.assertEqual(async_handoff.DEFAULT_CHUNK_TOKENS, 1024)

    def test_the_header_states_numerics_are_not_covered(self):
        self.assertIn('NUMERICS ARE NOT COVERED', queue.derive()[DRAFT])


def settings_rows():
    """The real settings text under real bash: what engine_identity declare --settings gets."""
    lines = ['set -u', 'H=%s' % shlex.quote(str(HERE))] + ['%s=%s' % (k, shlex.quote(v)) for k, v in ARGPARSE_VALUES.items()]
    lines += ['source "$H/runs/q18a_parity_n3s24q2024rc6144_settings.sh" || exit 97', "printf '%s\\0' \"${S5_SETTINGS[@]}\""]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    assert done.returncode == 0, done.stderr.decode()[-400:]
    return done.stdout.decode().split('\0')[:-1]


def declared_env():
    """EXTRA_ENV through engine_identity's own cleaning, the box's record's form (q18a-parity-na2 staging, 5 Nov 04:22: FLASH_S5_PLE_PRIVATE {'redacted_length': 1})."""
    (extra,) = [row.split('=', 1)[1] for row in settings_rows() if row.startswith('EXTRA_ENV=')]
    env = engine_identity.clean_value('EXTRA_ENV', extra)
    env['Q18A_TARGET_LABEL'] = ARGPARSE_VALUES.get('Q18A_TARGET_LABEL', 's5-overlap-80-80')
    # parity-n3s24q2024rc6144: the launcher's -e QSA_ROWCACHE_SLOTS from this settings row (setcfg over the box's 10240), and REQS from
    # MAX_SEQS (the box's 10), as the launcher passes them: they are not in EXTRA_ENV.
    (slots,) = [row.split('=', 1)[1] for row in settings_rows() if row.startswith('QSA_ROWCACHE_SLOTS=')]
    env.update(QSA_ROWCACHE_SLOTS=slots, QSA_ROWCACHE_REQS='10')
    return env


def real_env():
    # compose-4's staging on the box, and the one name this queue adds, as dp8d's identity declared it on the box (q18a-dp8-c32, engine.identity-start.json: "2048", plain).
    # coverage-throughput: and the yield's new-row bound this table raises (compose5_arms.YIELD_MAX_NEW), shown plain as a1's 2048 was.
    # build-2: and FLASH_S5_PREPARE_AHEAD, in the form engine_identity.clean_value gives it ('1', plain; checked below), not yet seen on the box.
    # build-2b24c64: and the logits cap, plain, as the box declared o4's 136 (q18a-o4-chunk2048, engine.identity-start.json: '136').
    # lane-o3: and FLASH_S5_O3, in the form engine_identity.clean_value gives it ('1', plain; checked below), not yet seen on the box.
    # lane-o3-na: and FLASH_S5_AHEAD_NOSYNC the same way ('1', plain; checked below), not yet seen on the box.
    # parity-n3s: and the decode SM count this table moves (compose5_arms.DECODE_SMS), plain as the box declared 44 here and 36 on dp4.
    return dict(json.loads((HERE / 'fixtures/compose4_declared_20261105_0422.json').read_text())['identity']['env'], FLASH_S5_CHUNK_TOKENS='2048',
                FLASH_S5_APPEND_YIELD_MAX_NEW=str(T.LAUNCH['max_new']), FLASH_S5_PREPARE_AHEAD='1', VLLM_SPARSE_INDEXER_MAX_LOGITS_MB='64', FLASH_S5_O3='1', FLASH_S5_AHEAD_NOSYNC='1',
                FLASH_S5_DECODE_SMS=str(T.DECODE_SMS),
                # parity-n3s24q2024rc6144: and the row cache's two launcher values, plain as n3s28q1352's request declared 10240 and 10
                # (q18a-parity-n3s28q1352-requests/startup.json identity.env), at this table's slots.
                QSA_ROWCACHE_SLOTS=str(T.ROWCACHE_SLOTS), QSA_ROWCACHE_REQS=str(T.ROWCACHE_REQS))


def declare_block(text):
    start = text.index("<<'PY_DECL'")
    start = text.index('\n', start) + 1
    return text[start:text.index('\nPY_DECL\n', start)] + '\n'


class DeclaredIdentity(unittest.TestCase):
    """Build-3's staging stopped at the declaration's FLASH_S5_PLE_PRIVATE line: the assert wanted the value, the record has its length. These run the preflight's own text over engine_identity's own output."""

    def run_block(self, block, env):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'startup.json'
            path.write_text(json.dumps(dict(identity=dict(env=env, command=dict(args=['--host', '0.0.0.0', '--host', '127.0.0.1'])))))
            return subprocess.run([sys.executable, '-I', '-c', block, str(path), env['Q18A_TARGET_LABEL']], capture_output=True, text=True, timeout=60)

    def test_the_real_record_of_the_refused_staging_now_passes_and_the_text_it_refused_on_still_fails(self):
        env = real_env()
        self.assertEqual(env['FLASH_S5_PLE_PRIVATE'], {'redacted_length': 1})
        block = declare_block(queue.derive()[DECLARE])
        done = self.run_block(block, env)
        self.assertEqual(done.returncode, 0, done.stderr[-600:])
        done = self.run_block(block.replace("assert env.get('FLASH_S5_PLE_PRIVATE')=={'redacted_length': 1}", "assert env.get('FLASH_S5_PLE_PRIVATE')=='1'"), env)
        self.assertEqual(done.returncode, 1)
        self.assertIn('AssertionError: installed for the whole run', done.stderr)

    def test_redacted_variants_of_the_real_record(self):
        """The redaction as it stands passes; the switch shown plain, another length, or a declared switch newly redacted stops the declaration."""
        block = declare_block(queue.derive()[DECLARE])
        for name, value, want in (('FLASH_S5_PLE_PRIVATE', engine_identity.clean_value('FLASH_S5_PLE_PRIVATE', '1'), 0),
                                  ('FLASH_S5_PLE_PRIVATE', '1', 1), ('FLASH_S5_PLE_PRIVATE', {'redacted_length': 2}, 1),
                                  ('FLASH_S5_PLE_WITNESS', {'redacted_length': 1}, 1), ('CUDA_DEVICE_MAX_CONNECTIONS', {'redacted_length': 2}, 1)):
            with self.subTest(name=name, value=value):
                self.assertEqual(self.run_block(block, dict(real_env(), **{name: value})).returncode, want)

    def test_prepare_ahead_is_declared_plain_and_both_preflights_refuse_it_missing_or_off(self):
        """build-2: the identity shows the switch as '1' (no redaction), and the declaration stops without it or at 0."""
        self.assertEqual(engine_identity.clean_value('FLASH_S5_PREPARE_AHEAD', '1'), '1')
        self.assertEqual(declared_env()['FLASH_S5_PREPARE_AHEAD'], '1')
        block = declare_block(queue.derive()[DECLARE])
        env = real_env()
        self.assertEqual(self.run_block(block, env).returncode, 0)
        for value in (None, '0'):
            bad = dict(env)
            if value is None:
                bad.pop('FLASH_S5_PREPARE_AHEAD')
            else:
                bad['FLASH_S5_PREPARE_AHEAD'] = value
            done = self.run_block(block, bad)
            self.assertEqual(done.returncode, 1, value)
            self.assertIn('AssertionError: build-2 prepare-ahead installed', done.stderr)
        self.assertIn("assert env.get('FLASH_S5_PREPARE_AHEAD') == '1', 'build-2 prepare-ahead installed'", queue.derive()[DRAFT])

    def test_the_logits_cap_is_declared_plain_and_both_preflights_refuse_it_missing_or_another_value(self):
        """build-2b24c64: the identity shows the cap as '64' (no redaction, as o4's '136'); the declaration stops without it, at stock or at o4's."""
        self.assertEqual(engine_identity.clean_value('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB', '64'), '64')
        self.assertEqual(declared_env()['VLLM_SPARSE_INDEXER_MAX_LOGITS_MB'], '64')
        block = declare_block(queue.derive()[DECLARE])
        self.assertEqual(self.run_block(block, real_env()).returncode, 0)
        for value in (None, '512', '136', 64):
            bad = dict(real_env())
            if value is None:
                bad.pop('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB')
            else:
                bad['VLLM_SPARSE_INDEXER_MAX_LOGITS_MB'] = value
            done = self.run_block(block, bad)
            self.assertEqual(done.returncode, 1, value)
            self.assertIn('AssertionError: build-2b24c64 indexer logits cap installed', done.stderr)
        draft = queue.derive()[DRAFT]
        self.assertIn("assert env.get('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB') == '64', 'build-2b24c64 indexer logits cap installed'", draft)
        self.assertEqual(draft.count('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB'), 2, 'the assert and the header: the cap is in EXTRA_ENV (settings), nowhere else')
        settings = queue.derive()[SETTINGS]
        self.assertEqual(settings.count(' VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 '), 1, 'once in EXTRA_ENV, for the whole run')

    def test_o3_is_declared_plain_and_both_preflights_refuse_it_missing_or_off(self):
        """lane-o3: FLASH_S5_O3=1 in EXTRA_ENV for the whole run; the identity shows it as '1'; the declaration stops without it or at '0'."""
        self.assertEqual(engine_identity.clean_value('FLASH_S5_O3', '1'), '1')
        self.assertEqual(declared_env()['FLASH_S5_O3'], '1')
        block = declare_block(queue.derive()[DECLARE])
        self.assertEqual(self.run_block(block, real_env()).returncode, 0)
        for value in (None, '0', 1):
            bad = dict(real_env())
            if value is None:
                bad.pop('FLASH_S5_O3')
            else:
                bad['FLASH_S5_O3'] = value
            done = self.run_block(block, bad)
            self.assertEqual(done.returncode, 1, value)
            self.assertIn('AssertionError: lane-o3 O3 installed', done.stderr)
        draft = queue.derive()[DRAFT]
        self.assertIn("assert env.get('FLASH_S5_O3') == '1', 'lane-o3 O3 installed'", draft)
        settings = queue.derive()[SETTINGS]
        self.assertEqual(settings.count(' FLASH_S5_O3=1 '), 1, 'once in EXTRA_ENV, for the whole run')

    def test_ahead_nosync_is_declared_plain_and_both_preflights_refuse_it_missing_or_off(self):
        """lane-o3-na: FLASH_S5_AHEAD_NOSYNC=1 in EXTRA_ENV for the whole run, after O3; the identity shows it as '1'; the declaration stops
        without it or at '0'."""
        self.assertEqual(engine_identity.clean_value('FLASH_S5_AHEAD_NOSYNC', '1'), '1')
        self.assertEqual(declared_env()['FLASH_S5_AHEAD_NOSYNC'], '1')
        block = declare_block(queue.derive()[DECLARE])
        self.assertEqual(self.run_block(block, real_env()).returncode, 0)
        for value in (None, '0', 1):
            bad = dict(real_env())
            if value is None:
                bad.pop('FLASH_S5_AHEAD_NOSYNC')
            else:
                bad['FLASH_S5_AHEAD_NOSYNC'] = value
            done = self.run_block(block, bad)
            self.assertEqual(done.returncode, 1, value)
            self.assertIn('AssertionError: parity-ahead ahead nosync installed', done.stderr)
        draft = queue.derive()[DRAFT]
        self.assertIn("assert env.get('FLASH_S5_AHEAD_NOSYNC') == '1', 'parity-ahead ahead nosync installed'", draft)
        settings = queue.derive()[SETTINGS]
        self.assertEqual(settings.count(' FLASH_S5_O3=1 FLASH_S5_AHEAD_NOSYNC=1 '), 1, 'once in EXTRA_ENV, for the whole run, after O3')
        self.assertEqual(settings.count('FLASH_S5_PREPARE_AHEAD=1'), 1, 'the nosync needs prepare-ahead, installed with it')

    def test_the_settings_here_declare_what_the_box_declared(self):
        """Every name this text puts in EXTRA_ENV reads on the box as it reads here; the two library hashes are this test's stand-ins."""
        real, here = real_env(), declared_env()
        self.assertEqual(set(here) - set(real), set())
        self.assertEqual(sorted(name for name in here if here[name] != real[name]), ['FLASH_S5_GREEN_LIB_SHA256', 'FLASH_S5_REPLAY_LIB_SHA256'])

    def test_a_declared_length_of_one_cannot_tell_0_from_1_so_the_settings_hold_the_value(self):
        """The preflight can't see the value: the settings text pins it here, and the smoke's scopes on ranks 0 and 1 show it engaged on the box."""
        self.assertEqual(engine_identity.clean_value('FLASH_S5_PLE_PRIVATE', '0'), engine_identity.clean_value('FLASH_S5_PLE_PRIVATE', '1'))
        (extra,) = [row.split('=', 1)[1] for row in settings_rows() if row.startswith('EXTRA_ENV=')]
        self.assertIn('FLASH_S5_PLE_PRIVATE=1', extra.split())

    def test_the_identity_redacts_the_private_switch_and_shows_the_others(self):
        env = declared_env()
        self.assertEqual(env['FLASH_S5_PLE_PRIVATE'], {'redacted_length': 1})
        self.assertEqual((env['FLASH_S5_PLE_WITNESS'], env['CUDA_DEVICE_MAX_CONNECTIONS']), ('1', '32'))
        self.assertEqual(dict(tok.split('=', 1) for tok in queue.SWITCHES.split())['FLASH_S5_PLE_PRIVATE'], '1', 'the value itself: the settings text')

    def test_rowcache_slots_are_a_launcher_setting_not_extra_env_and_both_preflights_refuse_stock_or_missing(self):
        """parity-n3s24q2024rc6144: QSA_ROWCACHE_SLOTS=6144 once, as its own S5_SETTINGS row (setcfg writes it over the box's 10240), never in
        EXTRA_ENV; the draft's and the declaration's asserts pass on the identity with it and refuse 10240, a missing key and a changed REQS."""
        rows = settings_rows()
        self.assertEqual([r for r in rows if r.startswith('QSA_ROWCACHE')], ['QSA_ROWCACHE_SLOTS=6144'])
        (extra,) = [row.split('=', 1)[1] for row in rows if row.startswith('EXTRA_ENV=')]
        self.assertNotIn('ROWCACHE', extra)
        self.assertEqual(queue.derive()[SETTINGS].count('QSA_ROWCACHE_SLOTS'), 1)
        for text, form in ((queue.derive()[DECLARE], '=='), (queue.derive()[DRAFT], ' == ')):
            self.assertEqual(text.count("assert env.get('QSA_ROWCACHE_SLOTS')%s'6144'" % form), 1)
            self.assertEqual(text.count("assert env.get('QSA_ROWCACHE_REQS')%s'10'" % form), 1)
        block = declare_block(queue.derive()[DECLARE])
        self.assertEqual(self.run_block(block, real_env()).returncode, 0)
        for bad in (dict(real_env(), QSA_ROWCACHE_SLOTS='10240'), {k: v for k, v in real_env().items() if k != 'QSA_ROWCACHE_SLOTS'},
                    dict(real_env(), QSA_ROWCACHE_REQS='8')):
            done = self.run_block(block, bad)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn('row cache', done.stderr)

    def test_the_declarations_preflight_passes_on_the_real_identity(self):
        done = self.run_block(declare_block(queue.derive()[DECLARE]), declared_env())
        self.assertEqual(done.returncode, 0, done.stderr[-600:])

    def test_the_text_build_3_ran_fails_as_it_did_on_the_box(self):
        block = declare_block(queue.derive()[DECLARE])
        old = block.replace("assert env.get('FLASH_S5_PLE_PRIVATE')=={'redacted_length': 1}", "assert env.get('FLASH_S5_PLE_PRIVATE')=='1'")
        self.assertNotEqual(old, block)
        done = self.run_block(old, declared_env())
        self.assertEqual(done.returncode, 1)
        self.assertIn('AssertionError: installed for the whole run', done.stderr)

    def test_each_switch_missing_fails_the_preflight(self):
        for name in ('CUDA_DEVICE_MAX_CONNECTIONS', 'FLASH_S5_PLE_PRIVATE', 'FLASH_S5_PLE_WITNESS', 'FLASH_S5_INWAIT', 'FLASH_S5_CHUNK_TOKENS'):
            with self.subTest(name):
                env = declared_env()
                del env[name]
                self.assertEqual(self.run_block(declare_block(queue.derive()[DECLARE]), env).returncode, 1)
        for value in ('1024', '4048', {'redacted_length': 4}):
            with self.subTest(value=value):
                env = dict(declared_env(), FLASH_S5_CHUNK_TOKENS=value)
                self.assertEqual(self.run_block(declare_block(queue.derive()[DECLARE]), env).returncode, 1)

    def test_the_identity_shows_the_chunk_plain_as_dp8ds_did(self):
        self.assertEqual(engine_identity.clean_value('FLASH_S5_CHUNK_TOKENS', '2048'), '2048')

    def test_the_drafts_preflight_env_lines_pass_on_the_real_identity(self):
        env = declared_env()
        lines = [line for line in queue.derive()[DRAFT].splitlines() if line.startswith(("assert env.get('", "assert '")) and ' in env' in line or re.match(r"assert env\.get\('\w+'\) == ('|\{'redacted_length': 1\})", line)]
        self.assertGreater(len(lines), 20)
        self.assertIn("assert env.get('FLASH_S5_PLE_PRIVATE') == {'redacted_length': 1}, 'installed for the whole run'", lines)
        done = subprocess.run([sys.executable, '-I', '-c', 'import json,sys\nenv=json.loads(sys.argv[1])\n' + '\n'.join(lines), json.dumps(env)], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr[-600:])

    def test_every_module_the_preflights_pin_is_a_file_here(self):
        texts = queue.derive()
        for text, names in ((texts[DECLARE], queue.DECLARE_FILES + queue.MODULES), (texts[DRAFT], queue.DRAFT_FILES.replace('; do', '') + queue.MODULES)):
            self.assertIn(names.split()[0], text)
            for name in names.split():
                self.assertTrue((HERE / name).is_file(), name)
        declared = re.search(r'for name in (.*?); do', texts[DECLARE], re.S).group(1)
        for name in declared.replace('\\\n', ' ').split():
            self.assertTrue((HERE / name).is_file() or name.startswith(('baselines/', 'canary_hub', 'flash_canary')), name)


class Names(unittest.TestCase):
    def test_the_real_qname_passes_every_name_rule(self):
        phases = ['q18a_parity_n3s24q2024rc6144.draft.sh', 'q18a_parity_n3s24q2024rc6144_declare.sh']
        self.assertEqual(names.check_names(QNAME, phases, RUNS, HERE / 'qlib.sh'), [])
        self.assertEqual(re.findall(r"^CANDIDATE = '([^']+)'$", (ROOT / 'tools/build_compose5.py').read_text(), re.M), [QNAME])
        for text in queue.derive().values():
            for name in re.findall(r'python3 "\$H/([a-z0-9_]+\.py)"', text) + re.findall(r'\$H/runs/([A-Za-z0-9_.]+\.sh)', text):
                self.assertTrue((HERE / name).is_file() or (RUNS / name).is_file(), name)


class Budget(unittest.TestCase):
    def test_relative_and_every_arm_enters_at_the_walls_and_at_one_and_a_half(self):
        source = (HERE / 'compose5_gate.py').read_text()
        self.assertNotRegex(source, r'import (time|datetime)')
        for key in ('nominal', 'slower'):
            est = gate.EXPECTED_RUNTIME_S[key]
            self.assertEqual((est['not_run'], est['arms_entering']), ([], len(gate.rows())), key)
            self.assertLessEqual(est['total_min'] * 60, gate.ACTIVE_S, key + ': the run ends before the TERM')
        self.assertEqual(gate.NOMINAL_S, 3990, 'parity-n3u: three 180 s pair arms and four 480 s replay arms with a 60 s tail, 66.5 min nominal (lane-o3-fix 5090)')
        self.assertEqual((gate.CAP, gate.BOX_BUDGET_S, gate.LAST_ARM_ENTRY_S), (1.4, 5586, 3838), 'parity-n3u: 1.4x so every arm enters at 1.5x, 93.1 min hard, the last entry at 64.0 min')
        self.assertGreater(gate.EXPECTED_RUNTIME_S['slower']['min_margin_s'], 0, 'every arm enters at 1.5x the walls')
        self.assertLessEqual(gate.BOX_BUDGET_S, gate.CAP * gate.NOMINAL_S, "the lead's hard cap, cleanup included")
        self.assertLessEqual(gate.BOX_BUDGET_S, 130 * 60, 'lane-o3-na: the hard cap, 128.9 min (the lead: up to ~130 min; box time free until Monday 08:00)')
        self.assertEqual(gate.ACTIVE_S + gate.CLEANUP_S, gate.BOX_BUDGET_S)
        self.assertIn('compose5_arms', gate.PHASES)

    def test_the_arm_timeout_holds_a_replay_and_its_finish_bound(self):
        import compose4_replay as replay
        self.assertLess(T.REPLAY['DURATION'] + replay.FINISH_BOUND_S + 60, 1500, 'the queue kills an arm at 1500 s')
        self.assertIn('--kill-after=30s 1500s python3 "$H/e3_compose5_box.py" arm', queue.derive()[DRAFT])


class ArmLoop(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='compose5-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death_h.py', 'compose5_gate.py', 'compose5_arms.py', 'compose4_arms.py', 'a1_trunk_arms.py', 'inwait_arms.py', 'attn_meta_arms.py', 'alpha_next_arms.py',
                     'alpha_next_stop_race_h.py', 'native39_trunk_gate.py', 'native40_box_gate.py', 'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_compose5_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def run_loop(self, timeout=420, smoke_rc=0):
        grade = tbox.Grade()
        for a in grade.records():
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        emit = tiq.tbox.run_of()[0]
        tap.write_streams(self.q, [(emit.lines[0][0], 'INFO x %s\n' % line) for line in tbox.tam.STARTUP])
        with open(self.q / 'engine-stream.log', 'a') as handle:
            handle.write(tbox.QUIET)       # build-2b24c64 scope: one native chunk's receipts, so the logits rule has read the lane
            handle.write(tbox.ahead_receipts(grade.records()))     # lane-o3-na: the ahead-split receipts in each nt row's window (na engaged)
        draft = queue.derive()[DRAFT]
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; STOP_RACE=0; CONTAINER=qwen-flash-prod; SERVICE=vllm-flash; N0=0",
                  'Q=%s; H=%s; PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; _cid=c; IDLE_TIMEOUT_S=1' % (self.q, self.h),
                  'Q18A_READINESS=/r/ready; S5_OVERLAY=/o; S5_LAUNCHER_NAME=l.sh', 'health_ok() { return 0; }']
        script += [tap.le_fix_function(draft, name) for name in tap.LOOP_FUNCTIONS]
        script += [queue.final_body(), 'echo "END FAILS=$FAILS INCS=$INCS CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"']
        path = self.root / 'loop.sh'
        path.write_text('\n'.join(script) + '\n')
        env = {k: v for k, v in os.environ.items() if k != 'HARNESS_DIR'}
        env.update(SMOKE_RC=str(smoke_rc), PATH='%s:%s' % (self.bin, os.environ['PATH']), Q=str(self.q), CALLS=str(self.root / 'calls.jsonl'), ARMS=str(self.arms),
                   REAL_HARNESS=str(HERE), DEATH_LINE=tap.DEATH_LINE, PYTHONDONTWRITEBYTECODE='1')
        proc = subprocess.Popen(['bash', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        calls = [json.loads(x) for x in (self.root / 'calls.jsonl').read_text().splitlines()]
        steps = {m.group(1): m.group(2) for m in re.finditer(r'^STEP (\S+) (\S+)', out, flags=re.M)}
        return out, err, calls, steps

    def test_every_arm_in_order_then_grade_and_read(self):
        out, err, calls, steps = self.run_loop()
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-2500:], err[-1500:]))
        box = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c.get('label') or c['action'] for c in box], CORE[:1] + ['smoke'] + DECODE + CORE[1:] + ['grade', 'read'])
        self.assertEqual(steps['smoke'], 'PASS')
        rows = T.by_label()
        for c in box:
            if c['action'] == 'arm':
                self.assertEqual(c['c1'], rows[c['label']]['c1'], c['label'])
        for name in DECODE + CORE:
            self.assertEqual(steps['arm-' + name], 'PASS', name)
        ingress = {c['out'][len('ingress-'):-len('.json')]: c for c in calls if c['script'] == 'checks' and c['action'] == 'ingress'}
        self.assertEqual(sorted(ingress), sorted(CORE + DECODE))
        self.assertEqual(steps['compose5'], 'PASS', out[-1500:])
        grade = json.loads((self.q / 'compose5-grade.json').read_text())
        self.assertEqual((grade['schema'], grade['verdict'], grade['missing']), ('compose5-grade/1', 'PASS', []))
        self.assertEqual(sorted(grade['arms']), sorted(CORE))
        self.assertTrue((self.q / 'compose5-report.md').read_text().startswith('# q18a-parity-n3s24q2024rc6144'))
        self.assertEqual(json.loads((self.q / 'compose5-read.json').read_text())['verdict'], 'PASS')

    def test_a_smoke_that_does_not_engage_stops_the_table(self):
        out, err, calls, steps = self.run_loop(smoke_rc=2)
        self.assertEqual(steps['smoke'], 'INCONCLUSIVE', out[-1500:])
        box = [c for c in calls if c['script'] == 'box']
        self.assertEqual([c.get('label') or c['action'] for c in box if c['action'] in ('arm', 'decode-arm')], CORE[:1], 'the smoke arm alone ran')
        for name in DECODE + CORE[1:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        self.assertIn('NOT_RUN: the exact-path smoke did not engage', out)


class Generations(unittest.TestCase):
    def call(self, label, record):
        return subprocess.run([sys.executable, '-B', str(HERE / 'compose5_arms.py'), 'generations', label, str(record)], capture_output=True, text=True, timeout=30)

    def test_the_table_tells_the_ingress_check_its_count_and_a_replay_its_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / 'arm.json'
            done = self.call(T.SMOKE_ARM, record)
            self.assertEqual((done.returncode, done.stdout.strip()), (0, str(T.by_label()[T.SMOKE_ARM]['generations'])))
            self.assertEqual(self.call('sp.g2-live-0', record).stdout.strip(), '0', 'no record: the queue runs no check over it, and 0 would not pass one')
            record.write_text(json.dumps(dict(replay_requests_sent=217)))
            self.assertEqual(self.call('sp.g2-live-0', record).stdout.strip(), '217')
            self.assertEqual(self.call('n3.g2-live-1', record).stdout.strip(), '217', 'lane-o3-fix: n3 is a replay arm too')
            self.assertEqual(self.call('po.g2-live-0', record).stdout.strip(), str(T.O3_PAIR_DECODERS + T.O3_PAIR_NATIVES), 'lane-o3-fix: the pair arms are back')
            self.assertNotEqual(self.call('nk.g2-live-0', record).returncode, 0, 'parity-n3u: nk is not an arm (sp is back)')


# parity-n3s: the runner hands the queue the table's decode SM count (compose5_spec's runner swap), not the parent fixture's 44.
ARGPARSE_VALUES = dict(tiq.ARGPARSE_VALUES, S5_REQUESTED_DECODE_SMS=str(T.DECODE_SMS), QNAME=QNAME, RUN_ID=QNAME + '-fixture', _label='n3.g2-live-0', _sel='n3.g2', _c1='n3', _group='g2', _arm='core:n3.g2-live-0', _pair='0', _cohort='live')


def bash_words(args, draft, values=None):
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    arrays = '--arm "$Q/arm-cmp.g2-live-0.json" --ingress "cmp.g2-live-0=$Q/ingress-cmp.g2-live-0.json" --watch "cmp.g2-live-0=$Q/engine-watch-arm-cmp.g2-live-0.json"'
    lines += ['GRADE_ARGS=(%s)' % arrays, 'READ_ARGS=(%s)' % arrays, 'source "$H/runs/q18a_parity_n3s24q2024rc6144_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class DecodeSplit(unittest.TestCase):
    """parity-n3s: the decode SM count is the table's (compose5_arms.DECODE_SMS) at every site compose-4 declared 44, legal by the
    decode-phase rule; the startup grade binds it exactly; the runner hands it to the queue."""

    def test_the_count_is_legal_and_not_n3us(self):
        self.assertEqual(T.legal_decode_sms(T.DECODE_SMS), str(T.DECODE_SMS))
        self.assertNotEqual(T.DECODE_SMS, T.N3U_DECODE_SMS)
        for bad in ('23', '27', '61', '028', '', 'x', '30.0', '-28'):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                T.legal_decode_sms(bad)
        self.assertEqual([T.legal_decode_sms(v) for v in (24, 28, 36, 44, 60)], ['24', '28', '36', '44', '60'])

    def test_every_site_carries_the_count_and_none_carries_44(self):
        d = queue.derive()
        draft, settings, declare = d[DRAFT], d[SETTINGS], d[DECLARE]
        n = T.DECODE_SMS
        self.assertEqual(settings.count('S5_REQUESTED_DECODE_SMS=${S5_REQUESTED_DECODE_SMS:-%d}' % n), 1)
        self.assertEqual(settings.count("[[ $S5_REQUESTED_DECODE_SMS == %d ]] || {\n  echo 'decode split remains declared %d' >&2; return 2; }" % (n, n)), 1)
        self.assertEqual(settings.count(' FLASH_S5_DECODE_SMS=%d ' % n), 1)
        self.assertEqual(draft.count('[[ $S5_REQUESTED_DECODE_SMS == %d ]] || fail_early decode-sms-must-be-%d' % (n, n)), 1)
        self.assertEqual(declare.count("assert env.get('FLASH_S5_DECODE_SMS')=='%d', 'decode split differs'" % n), 1)
        self.assertEqual(draft.count('python3 "$H/e3_split_startup_grade.py" --url'), 1)
        self.assertNotIn('python3 "$H/e3_native_startup_grade.py"', draft)
        self.assertIn('e3_native_startup_grade.py e3_split_startup_grade.py ', draft)
        self.assertIn('e3_native_startup_grade.py e3_split_startup_grade.py ', declare)
        for text in (draft, settings, declare):
            self.assertNotRegex(text, r'DECODE_SMS(=|==| == |\)==\'|:-)\'?44\b')

    def test_the_settings_refuse_n3us_count_and_take_the_tables(self):
        def source(value):
            values = dict(ARGPARSE_VALUES, S5_REQUESTED_DECODE_SMS=value)
            lines = ['set -u', 'H=%s' % shlex.quote(str(HERE))] + ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
            lines += ['source "$H/runs/%s" || exit 97' % SETTINGS.split('/')[-1], 'echo "$S5_REQUESTED_DECODE_SMS"']
            return subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, text=True, timeout=60)
        done = source(str(T.DECODE_SMS))
        self.assertEqual((done.returncode, done.stdout.strip()), (0, str(T.DECODE_SMS)), done.stderr[-300:])
        for other in ('44', '36' if T.DECODE_SMS != 36 else '28', '29'):
            with self.subTest(value=other):
                done = source(other)
                self.assertEqual(done.returncode, 97)
                self.assertIn('decode split remains declared %d' % T.DECODE_SMS, done.stderr)

    def test_the_startup_grade_binds_the_count_exactly(self):
        import e3_split_startup_grade as split
        self.assertEqual(split.REQUESTED, T.DECODE_SMS)
        native = (HERE / 'e3_native_startup_grade.py').read_text()
        for old, new in split.SWAPS:
            self.assertEqual(native.count(old), 1, old)
            self.assertIn('44', old)
            self.assertNotIn('44', new)
        ns = split.namespace()
        self.assertEqual(ns['REQUESTED'], T.DECODE_SMS)
        self.assertTrue(callable(ns['grade']) and callable(ns['main']))
        self.assertTrue(split.launch_matches({}))
        self.assertTrue(split.launch_matches({'S5_REQUESTED_DECODE_SMS': str(T.DECODE_SMS)}))
        self.assertFalse(split.launch_matches({'S5_REQUESTED_DECODE_SMS': '44'}))
        env = dict(os.environ, S5_REQUESTED_DECODE_SMS='44')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'startup-grade.json'
            done = subprocess.run([sys.executable, '-B', str(HERE / 'e3_split_startup_grade.py'), '--url', 'http://127.0.0.1:9', '--run-id', 'x',
                                   '--package', tmp, '--out', str(out)], capture_output=True, text=True, timeout=120, env=env, cwd=tmp)
            self.assertEqual(done.returncode, 2)
            self.assertIn('startup grade refused', done.stderr)
            self.assertFalse(out.exists(), 'refused before grading: no record')

    def test_the_runner_hands_the_queue_the_count(self):
        spec = json.loads((ROOT / 'tools/native43_spec.compose-5.json').read_text())
        swaps = [tuple(row) for row in spec['runner_swaps']]
        self.assertIn(("S5_REQUESTED_DECODE_SMS='44'", "S5_REQUESTED_DECODE_SMS='%d'" % T.DECODE_SMS), swaps)


class QueueArgparse(unittest.TestCase):
    def test_every_queue_python_call_parses_with_its_real_parser(self):
        held = tan.held
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        setcfg = next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file())
        rows = []
        for name, script, args, form in calls:
            if script == 'compose5_arms.py':
                continue        # no parser: argv read by hand, test Generations runs it
            with self.subTest(file=name.split('/')[-1], script=script, args=args[:60]):
                rc, words, err = bash_words(args, draft, dict(_label='decode-am1', _attn='on') if args.startswith('decode-arm') else None)
                self.assertEqual(rc, 0, err)
                if form != 'python3':
                    words = held.QLIB_CALLS[form][1](words)
                verdict, detail = held.setcfg_parse(setcfg, words) if script == 'SETCFG' else held.real_parse(HERE / script, words)
                self.assertEqual(verdict, 'parsed', (script, words[:4], detail))
                rows.append(dict(file=name.split('/')[-1], script=script, form=form, action=words[0] if words and not words[0].startswith('-') else None, words=len(words), verdict=verdict))
        self.assertTrue(rows)
        ARGPARSE_RECEIPTS[:] = rows
        self.assertEqual(SLOT_RECEIPTS, [])
        arm = next(a for _n, s, a, _f in calls if s == 'e3_compose5_box.py' and a.split()[0] == 'arm')
        for label, (old, new) in (('arm-unknown-option', ('--control-timeout-s', '--control-timeout-sec')), ('arm-undeclared-selection', ('--c1 "${_c1#c1}"', '--c1 ps1')), ('arm-undeclared-round', ('--index "$_pair"', '--index 16'))):
            with self.subTest(control=label):
                self.assertEqual(arm.count(old), 1, label)
                rc, words, err = bash_words(arm.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_compose5_box.py', words)[0], 'refused', label)


if __name__ == '__main__':
    unittest.main()
