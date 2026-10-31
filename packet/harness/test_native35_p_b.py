"""Native35 P-b: the P speed arm with the approved L2 row4 window fix, taken exactly.

Proves, on CPU: (1) the P-b queue texts are the P-a texts under fresh names with
only the declared swaps, and P-b's qualification phase is byte-identical to the
L2 queue's (P-a's is byte-identical to L's), so row4 alone runs the L2
resume-receipt collector; (2) the L2 runner file is the exact claude/native35-l2
29f8265 bytes; (3) every P-b script parses; (4) the exact P-b row4 argv reaches
its first network call through the real e3_native35_l2_placement CLI with
ResumeWindowPlacementRunner, restoring the parent runner afterwards. The
runner's own source-relation and box-timing proofs are test_native35_l2
(unchanged from claude/native35-l2); P's W1/W2/tail proofs execute the P-b
draft (test_native35_performance, test_native35_performance_queue).
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
from string import Template
import subprocess
import tempfile
import unittest
from unittest import mock

import e3_native_qualification_driver as driver_module
from e3_native_qualification_driver import ImmutableStore, NativePlacementRunner, canonical_sha
import e3_native35_l2_placement as l2
import test_e3_native_qualification_driver as base

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
ROW4_RECEIPTS = []
# harness/e3_native35_l2_placement.py at claude/native35-l2 29f8265 (opus-native35-l2).
L2_RUNNER_SHA256 = '9327e01cd9f923e31b101a53208743990e4f877c68d9f6de6225c26ee30fd406'
P_HEADER = ('# Native35-p: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> W1 -> two retained W2 rounds -> owned restore.\n')
PB_HEADER = ('# Native35-p-b: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 '
             '(L2 client window at the accepted resume receipt) -> W1 -> two retained W2 rounds -> owned restore.\n')
P_INTERFACE = '  e3_native35_performance.py native35_performance_grade.py native35p_gate.py; do\n'
PB_INTERFACE = '  e3_native35_performance.py native35_performance_grade.py native35p_gate.py \\\n  e3_native35_l2_placement.py; do\n'
P_PINS = '      e3_native35_performance.py native35_performance_grade.py native35p_gate.py \\\n'
PB_PINS = '      e3_native35_performance.py native35_performance_grade.py native35p_gate.py e3_native35_l2_placement.py \\\n'


def text(name):
    return (RUNS / name).read_text()


def phase(source):
    start = source.index('qualification_phase() {')
    return source[start:source.index('\n}\n', start) + 3]


def once(source, old, new):
    assert source.count(old) == 1, (old[:70], source.count(old))
    return source.replace(old, new)


def fresh_names(source):
    assert 'q18a-native35-p-b' not in source and 'q18a_native35_p_b' not in source
    return source.replace('q18a-native35-p', 'q18a-native35-p-b').replace('q18a_native35_p', 'q18a_native35_p_b')


class Derivation(unittest.TestCase):
    def test_pb_is_pa_under_fresh_names_with_the_exact_l2_row4_phase(self):
        pa, pb = text('q18a_native35_p.draft.sh'), text('q18a_native35_p_b.draft.sh')
        l, l2_queue = text('q18a_native35_l.draft.sh'), text('q18a_native35_l2.draft.sh')
        self.assertEqual(phase(pa), phase(l))
        self.assertEqual(phase(pb), phase(l2_queue))
        self.assertNotEqual(phase(pa), phase(pb))
        expected = fresh_names(once(once(once(pa, P_HEADER, PB_HEADER), phase(pa), phase(l2_queue)), P_INTERFACE, PB_INTERFACE))
        self.assertEqual(pb, expected)
        self.assertEqual(text('q18a_native35_p_b_settings.sh'), fresh_names(text('q18a_native35_p_settings.sh')))
        self.assertEqual(text('q18a_native35_p_b_declare.sh'), fresh_names(once(text('q18a_native35_p_declare.sh'), P_PINS, PB_PINS)))
        body = phase(pb)
        self.assertIn('local name=$1 rc cli=e3_q18a_qualification.py\n', body)
        self.assertIn('  if [[ $name == row4 ]]; then cli=e3_native35_l2_placement.py; fi\n', body)
        self.assertEqual(body.count('e3_native35_l2_placement.py'), 1)
        self.assertNotIn('e3_native34_causal.py', pb)
        settings = text('q18a_native35_p_b_settings.sh')
        self.assertIn(' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0 ', settings)
        self.assertNotIn('FLASH_NATIVE34_ARM', settings)
        self.assertIn('\nQ18A_FIRST_ADVANCE_READ_TIMEOUT_S=1200\n', settings)
        self.assertEqual(hashlib.sha256((HERE / 'e3_native35_l2_placement.py').read_bytes()).hexdigest(), L2_RUNNER_SHA256)

    def test_every_pb_script_parses_and_no_global_blocking(self):
        for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
            path = RUNS / ('q18a_native35_p_b' + suffix)
            self.assertEqual(subprocess.run(['bash', '-n', str(path)], capture_output=True).returncode, 0, path.name)
            self.assertNotIn('set -e\n', path.read_text())


def pb_row4_argv(directory):
    source = text('q18a_native35_p_b.draft.sh')
    invocation = phase(source).split('python3 "$H/$cli"', 1)[1].split('> "$Q/qual-$name.log"', 1)[0]
    values = dict(H=str(HERE), Q=str(directory), PROBE_URL='http://127.0.0.1:1', MODEL='qwen38-w8a8-tp4',
                  RUN_ID='q18a-native35-p-b-cpu', EPOCH='7', Q18A_STORE=str(directory / 'private/qualification-store'),
                  SNAPSHOT_MAX_BYTES=str(512 * 1024 ** 2), name='row4')
    return [Template(token).substitute(values) for token in shlex.split(invocation.replace('\\\n', ' '))], values


class Row4CLI(unittest.TestCase):
    def test_exact_pb_row4_argv_reaches_first_network_call_with_the_l2_runner(self):
        fixture_key = 'native35-p-b-cpu-transport-fixture'
        seen = []

        def refuse_request(connection, method, url, body=None, headers=None, **kwargs):
            command = json.loads(body)
            runners = [type(frame.frame.f_locals['self']) for frame in inspect.stack()
                       if frame.function == 'run' and isinstance(frame.frame.f_locals.get('self'), NativePlacementRunner)]
            seen.append(dict(host=connection.host, port=connection.port, method=method, url=url,
                             action=command.get('action'), target_route=command.get('target_route'),
                             epoch=command.get('engine_epoch'), runners=runners,
                             authorized=headers.get('Authorization') == 'Bearer ' + fixture_key))
            raise ConnectionRefusedError('declared CPU boundary')

        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            argv, values = pb_row4_argv(directory)
            self.assertEqual(argv[:2], ['--phase', 'row4'])
            # Declared leaves: on the box row1 and rule4_controls run first and leave the frozen
            # row1 plan (parsed, unused by row4) and the retained teacher artifact.
            (directory / 'private').mkdir()
            (directory / 'private/row1-plan.json').write_text(json.dumps({'schema': 'cpu-fixture-row1-plan'}))
            artifact = base.retained_teacher(ImmutableStore(values['Q18A_STORE']), {'schema': 'cpu-fixture', 'rows': []})
            env = dict(VLLM_API_KEY=fixture_key, Q18A_TEACHER_ARTIFACT_SHA=canonical_sha(artifact), Q18A_ROW1_SHA='e' * 64)
            with mock.patch.dict(os.environ, env), mock.patch('http.client.HTTPConnection.request', refuse_request):
                rc = l2.main(argv)
            self.assertIs(driver_module.NativePlacementRunner, NativePlacementRunner)
            record = json.loads((directory / 'qual-row4.json').read_text())
        self.assertEqual(rc, 2)
        self.assertEqual(record['verdict'], 'INCONCLUSIVE')
        self.assertEqual(len(seen), 1, seen)
        first = seen[0]
        self.assertEqual((first['host'], first['port'], first['method'], first['url']),
                         ('127.0.0.1', 1, 'POST', '/v1/flash_s5_native_overlap'))
        self.assertEqual((first['action'], first['target_route'], first['epoch']), ('prepare', 'full', 7))
        self.assertTrue(first['authorized'])
        self.assertEqual(first['runners'], [l2.ResumeWindowPlacementRunner])
        self.assertEqual(len(record['missing']), 1)
        self.assertTrue(record['missing'][0].startswith('collect:'), record['missing'])
        self.assertNotIn(fixture_key, json.dumps(record))
        paths = [HERE / 'workloads/q17b-correctness-precontrols.json', HERE / 'workloads/q18a-native-placement-prefill.json']
        ROW4_RECEIPTS.append(dict(case='exact-p-b-row4-argv', verdict='PASS', first_network_call='prepare-full',
                                  network_calls=len(seen), network_contact=False, runner='ResumeWindowPlacementRunner',
                                  runner_sha256=L2_RUNNER_SHA256, expected_refusal=record['missing'][0], substitution_restored=True,
                                  workload_and_plan_shas={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))


if __name__ == '__main__':
    unittest.main()
