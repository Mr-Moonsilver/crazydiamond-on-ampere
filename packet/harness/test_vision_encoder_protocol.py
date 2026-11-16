"""Whole guard protocol on real HTTP stubs; never invokes Docker, CUDA or the VM."""
import base64
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import queue
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
import vision_encoder_job as job
import vision_encoder_read as reader
import vision_encoder_service as service
import vision_speech_guard as guard
import vision_encoder_stop as stop
import build_vision_encoder_packet as packet


class FakeBackend:
    weights = dict(tensors=333, bytes=897862112)
    manifest = job.MODEL_MANIFEST_SHA
    backend = 'FLASH_ATTN_CPU_STUB'

    def __init__(self, *_):
        pass

    def encode(self, image):
        return dict(schema='flash-next-image-embeds/1', manifest_sha256=self.manifest,
            dtype='bfloat16', shape=[2, 2560], image_grid_thw=[[1, 2, 4]],
            embedding_sha256=hashlib.sha256(image).hexdigest(),
            content=dict(type='image_embeds', image_embeds=dict(image_embeds='CPU_STUB', image_grid_thw='CPU_STUB')),
            metrics=dict(visual_tokens=2, peak_allocated_bytes=1400 * 2**20,
                peak_reserved_bytes=1600 * 2**20, encode_event_ms=1, output_bytes=10240))


class Protocol(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='vision-protocol-'))
        self.state = dict(started=False, stopped=False, asr=0, memory_bad=False, health_bad=False,
                          post_asr_bad=False, label_bad=False, last_argv=None, stops=0)
        fixture = self.root / 'asr.wav'; fixture.write_bytes(b'RIFF_CPU_SYNTHETIC_ASR_FIXTURE'); fixture.chmod(0o644)
        self.key = self.root / 'key.env'; self.key.write_text('VISION_ENCODER_API_KEY=' + 'b' * 64); self.key.chmod(0o600)
        expected = self.root / 'expected.txt'; expected.write_text(guard.ASR_EXPECTED)
        clearance = self.root / 'clearance.json'
        clearance.write_text(json.dumps(dict(schema='vision-infra-clearance/1', cleared=True,
            run_id='cpu-test', speech_must_remain_up=True, gpu_limit_mib=3072,
            gpu_uuid='GPU-00000000-0000-0000-0000-000000000000')))
        self.images = []
        for name in ('red.png', 'table.png', 'diagram.png'):
            p = self.root / name; p.write_bytes(b'CPU_STUB_IMAGE_' + name.encode()); self.images.append(p)
        state = self.state

        class Speech(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                self.send_response(503 if state['health_bad'] and state['started'] else 200)
                self.end_headers(); self.wfile.write(b'{}')
            def do_POST(self):
                if self.path != '/asr?language=en&output_format=json':
                    raise AssertionError('infra ASR endpoint/query differs')
                body = self.rfile.read(int(self.headers['Content-Length']))
                if b'name="audio_file"' not in body or fixture.read_bytes() not in body:
                    raise AssertionError('real multipart ASR shape not exercised')
                state['asr'] += 1
                text = 'wrong' if state['post_asr_bad'] and state['asr'] >= 2 else guard.ASR_EXPECTED
                self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(dict(text=text)).encode())

        self.speech = ThreadingHTTPServer(('127.0.0.1', 0), Speech)
        self.encoder = service.Encoder(FakeBackend)
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), service.handler(self.encoder, 'b' * 64))
        for server in (self.speech, self.http):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        self.addCleanup(setattr, self.encoder, 'stopping', True)
        speech_url = 'http://127.0.0.1:' + str(self.speech.server_port)
        self.encoder_url = 'http://127.0.0.1:' + str(self.http.server_port)
        self.argv = ['--model', str(self.root), '--run-dir', str(self.root / 'run'), '--run-id', 'cpu-test',
            '--clearance', str(clearance), '--key-file', str(self.key), '--expected-asr', str(expected),
            '--asr-url', speech_url + '/asr?language=en&output_format=json', '--asr-max-seconds', '10', '--speech-health', speech_url + '/health',
            '--fixture', str(fixture), '--images', *map(str, self.images)]
        (self.root / 'actual-guard-argv.json').write_text(json.dumps(self.argv))
        self.fixture_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()

    def drive(self):
        state = self.state
        def command(argv, timeout=20):
            if 'image' in argv:
                return job.IMAGE
            if 'stop' in argv:
                state['stopped'] = True; state['stops'] += 1
                return 'owned encoder stopped'
            if '{{.State.Running}}' in argv:
                return 'false' if state['stopped'] else 'true'
            if 'inspect' in argv:
                return '345 ' + ('foreign-run' if state['label_bad'] else 'cpu-test')
            raise AssertionError('unexpected client command ' + repr(argv))
        class Child:
            def poll(self): return 0 if state['stopped'] else None
            def wait(self, timeout=20):
                if not state['stopped']:
                    raise AssertionError('child not stopped before wait')
                return 0
        def popen(argv, **kwargs):
            state['last_argv'] = argv; state['started'] = True
            return Child()
        def physical(gpu_uuid, pid=None):
            return dict(owned_mib=(4000 if state['memory_bad'] else 1800) if pid and not state['stopped'] else 0,
                        free_mib=8000)
        original_request = job.request
        original_open = open
        def local_request(url, path, key=None, body=None):
            return original_request(self.encoder_url, path, key, body)
        def local_open(path, *args, **kwargs):
            return original_open(self.root / 'job.lock' if str(path) == '/opt/flashnext/.job.lock' else path, *args, **kwargs)
        with patch.object(guard, 'FIXTURE_SHA_PREFIX', self.fixture_sha), \
                patch.object(job, 'verify_model'), \
                patch.object(job, 'command', side_effect=command), patch.object(job, 'physical', side_effect=physical), \
                patch.object(job.subprocess, 'Popen', side_effect=popen), patch.object(job, 'request', side_effect=local_request), \
                patch.object(job, 'open', side_effect=local_open, create=True), patch.object(job.signal, 'signal'), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = job.main(self.argv)
        value = json.loads((self.root / 'run/result.json').read_text())
        return code, value

    def test_whole_actual_argv_protocol_to_final_reader(self):
        code, value = self.drive()
        self.assertEqual(code, 0)
        self.assertTrue(value['owned_gpu_released'])
        self.assertEqual(self.state['stops'], 1)
        self.assertEqual(self.state['asr'], 2)  # no unregistered ASR loop
        self.assertEqual(reader.read(value)['verdict'], 'PASS')
        self.assertEqual(len(value['images']), 3)
        self.assertEqual(len(value['speech_overlap_images']), 3)
        argv = self.state['last_argv']
        self.assertIn(job.IMAGE, argv); self.assertNotIn('--rm', argv)
        self.assertEqual(argv.count('--tmpfs'), 1)
        env_items = [argv[i+1] for i, value in enumerate(argv[:-1]) if value == '-e']
        self.assertEqual(len(env_items), len({value.split('=', 1)[0] for value in env_items}))
        self.assertEqual(argv[-6:], ['--model', '/model', '--run-dir', '/artifacts', '--key-file', '/encoder.env'])
        self.assertNotIn('b' * 64, ' '.join(argv))
        result = subprocess.run([sys.executable, str(Path(reader.__file__)), '--record', str(self.root / 'run/result.json')],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('VISION_ENCODER_READ PASS', result.stdout)

    def test_memory_breach_stops_only_owned_child(self):
        self.state['memory_bad'] = True
        code, value = self.drive()
        self.assertEqual(code, 2); self.assertTrue(self.state['stopped'])
        self.assertTrue(value['owned_gpu_released']); self.assertEqual(self.state['stops'], 1)
        self.assertEqual(reader.read(value)['verdict'], 'HOLD')

    def test_speech_health_failure_stops_encoder(self):
        self.state['health_bad'] = True
        code, value = self.drive()
        self.assertEqual(code, 2); self.assertTrue(self.state['stopped'])
        self.assertTrue(value['owned_gpu_released'])
        self.assertEqual(value['images'], [])
        self.assertEqual(self.state['asr'], 1)

    def test_post_load_asr_failure_stops_encoder(self):
        self.state['post_asr_bad'] = True
        code, value = self.drive()
        self.assertEqual(code, 1); self.assertTrue(self.state['stopped'])
        self.assertEqual(self.state['asr'], 2)
        self.assertEqual(reader.read(value)['verdict'], 'HOLD')

    def test_missing_clearance_never_starts(self):
        (self.root / 'clearance.json').write_text('{}')
        with self.assertRaisesRegex(guard.Refused, 'clearance'):
            self.drive()
        self.assertFalse(self.state['started'])

    def test_uncleared_but_bound_run_never_starts(self):
        path = self.root / 'clearance.json'
        value = json.loads(path.read_text()); value['cleared'] = False
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(guard.Refused, 'clearance'):
            self.drive()
        self.assertFalse(self.state['started'])

    def test_foreign_container_is_never_stopped(self):
        self.state['label_bad'] = True
        code, value = self.drive()
        self.assertEqual(code, 2)
        self.assertEqual(self.state['stops'], 0)
        self.assertFalse(value['owned_gpu_released'])

    def test_unauthenticated_request_refuses(self):
        with self.assertRaisesRegex(guard.Refused, 'encoder-http-401'):
            job.request(self.encoder_url, '/v1/encode', body=dict(image_base64='eA=='))

    def test_reader_complete_record_actual_cli(self):
        _, value = self.drive()
        result = subprocess.run([sys.executable, str(Path(reader.__file__)), '--record', str(self.root / 'run/result.json')],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('VISION_ENCODER_READ PASS', result.stdout)
        value['physical_peak_mib'] = None
        self.assertEqual(reader.read(value)['verdict'], 'HOLD')
        self.assertEqual(reader.read(dict(value, physical_peak_mib=0))['verdict'], 'HOLD')

    def test_generated_service_argv_entire_http_protocol(self):
        self.drive()
        generated = json.loads((self.root / 'run/argv.json').read_text())
        start = generated.index('/packet/harness/vision_encoder_service.py') + 1
        argv = generated[start:]
        servers = queue.Queue()
        real_server = ThreadingHTTPServer
        real_path = Path
        failures = []
        def mapped_path(value):
            return real_path({'/model': str(self.root), '/artifacts': str(self.root / 'run'),
                              '/encoder.env': str(self.key)}.get(str(value), value))
        def mapped_server(address, callback):
            self.assertEqual(address, ('<encoder-host>', 9041))
            server = real_server(('127.0.0.1', 0), callback)
            servers.put(server)
            return server
        def run():
            try:
                service.main(argv)
            except BaseException as exc:
                failures.append(exc)
        with patch.object(service, 'VisionBackend', FakeBackend), patch.object(service, 'Path', side_effect=mapped_path), \
                patch.object(service, 'ThreadingHTTPServer', side_effect=mapped_server), patch.object(service.signal, 'signal'):
            thread = threading.Thread(target=run, daemon=True); thread.start()
            server = servers.get(timeout=5)
            url = 'http://127.0.0.1:' + str(server.server_port)
            try:
                health = job.request(url, '/health')
                self.assertTrue(health['ready'])
                result = job.probe(url, 'b' * 64, self.images)
                self.assertEqual(len(result), 3)
                self.assertEqual(job.request(url, '/health')['encodes'], 3)
            finally:
                server.shutdown(); thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])

    def test_generated_service_missing_model_mutant_refuses_before_backend(self):
        argv = ['--run-dir', str(self.root), '--key-file', str(self.key)]
        with patch.object(service, 'VisionBackend') as backend, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                service.main(argv)
            self.assertEqual(cm.exception.code, 2)
            backend.assert_not_called()

    def test_pre_latency_retries_once_before_gpu_start(self):
        calls = []
        def asr(*_):
            calls.append(self.state['started'])
            if len(calls) == 1:
                raise guard.SpeechLatency('first slow ASR')
            return dict(status=200, correct=True, elapsed_s=.1, latency_limit_s=10)
        with patch.object(job, 'check_asr', side_effect=asr), patch.object(job, 'retry_delay') as wait:
            code, value = self.drive()
            self.assertEqual(code, 0)
            self.assertEqual(calls, [False, False, True])
            wait.assert_called_once()
            self.assertEqual(value['speech']['pre_asr_attempts'], 2)

    def test_pre_two_latency_misses_fail_without_loading(self):
        with patch.object(job, 'check_asr', side_effect=guard.SpeechLatency('slow ASR')), patch.object(job, 'retry_delay'):
            code, value = self.drive()
        self.assertEqual((code, value['verdict']), (1, 'FAIL'))
        self.assertFalse(self.state['started'])

    def test_post_latency_stops_before_retry_recovery_is_hold(self):
        calls = []
        def asr(*_):
            calls.append((self.state['started'], self.state['stopped']))
            if len(calls) == 2:
                raise guard.SpeechLatency('loaded slow ASR')
            return dict(status=200, correct=True, elapsed_s=.1, latency_limit_s=10)
        with patch.object(job, 'check_asr', side_effect=asr), patch.object(job, 'retry_delay'):
            code, value = self.drive()
        self.assertEqual((code, value['verdict']), (2, 'HOLD'))
        self.assertEqual(calls, [(False, False), (True, False), (True, True)])
        self.assertTrue(value['owned_gpu_released'])

    def test_retry_wait_is_sixty_seconds_with_health(self):
        clock = [0.]
        def advance(seconds): clock[0] += seconds
        with patch.object(job.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(job.time, 'sleep', side_effect=advance), patch.object(job, 'check_health') as health:
            job.retry_delay('CPU_STUB')
        self.assertEqual(clock[0], 60.)
        self.assertEqual(health.call_count, 60)

    def test_reader_exception_keeps_text_and_raising_location(self):
        path = self.root / 'bad.json'; path.write_text('{')
        result = subprocess.run([sys.executable, str(Path(reader.__file__)), '--record', str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('JSONDecodeError', result.stderr)
        self.assertRegex(result.stderr, r'at .*:[0-9]+')

    def test_stop_rechecks_off_receipt_and_container_owner(self):
        _, value = self.drive()
        run = self.root / 'run'
        with patch.object(stop.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'false cpu-test')):
            self.assertEqual(stop.handoff(run)['state'], 'OFF')
        with patch.object(stop.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'true cpu-test')):
            with self.assertRaisesRegex(ValueError, 'not stopped'):
                stop.handoff(run)

    def test_stop_refuses_pid_reuse_before_signal(self):
        self.drive()
        run = self.root / 'run'
        state = json.loads((run / 'result.json').read_text())
        state['owned_gpu_released'] = False
        (run / 'result.json').write_text(json.dumps(state))
        with patch.object(stop.Path, 'read_bytes', return_value=b'python3\0some-other-job.py\0'), patch.object(stop.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'not this owned'):
                stop.handoff(run)
            kill.assert_not_called()

    def test_stop_actual_cli_to_off_verdict(self):
        self.drive()
        fakebin = self.root / 'stop-bin'; fakebin.mkdir()
        sudo = fakebin / 'sudo'
        sudo.write_text('#!/bin/sh\nprintf "false cpu-test\\n"\n')
        sudo.chmod(0o755)
        result = subprocess.run([sys.executable, str(Path(stop.__file__)), '--run-dir', str(self.root / 'run')],
            env=dict(os.environ, PATH=str(fakebin)+':'+os.environ['PATH']), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('VISION_ENCODER_OFF ', result.stdout)

    def test_physical_parser_counts_only_owned_gpu_pid(self):
        gpu = 'GPU-00000000-0000-0000-0000-000000000000'
        with patch.object(job, 'command', side_effect=['8000', '111, 7000, '+gpu+'\n345, 1800, '+gpu]):
            self.assertEqual(job.physical(gpu, 345), dict(owned_mib=1800, free_mib=8000))

    def test_frozen_start_wrapper_actual_argv_whole_protocol(self):
        frozen = self.root / 'packet'
        with contextlib.redirect_stdout(io.StringIO()):
            packet.build(frozen)
        fakebin = self.root / 'bin'; fakebin.mkdir()
        captured = self.root / 'wrapper-argv.json'
        python = fakebin / 'python3'
        python.write_text('#!' + sys.executable + '\nimport json,sys\nfrom pathlib import Path\n'
            + 'Path(' + repr(str(captured)) + ').write_text(json.dumps(sys.argv[1:]))\n')
        python.chmod(0o755)
        actual = subprocess.run(['bash', str(frozen / 'tools/vision_encoder_start.sh'), 'cpu-test',
            str(self.root / 'clearance.json')], env=dict(os.environ, PATH=str(fakebin)+':'+os.environ['PATH']),
            capture_output=True, text=True)
        self.assertEqual(actual.returncode, 0, actual.stderr)
        argv = json.loads(captured.read_text())
        self.assertEqual(argv[:2], ['-B', str(frozen / 'harness/vision_encoder_job.py')])
        previous = self.argv
        self.argv = argv[2:]
        for option in ('--model', '--run-dir', '--key-file', '--fixture', '--speech-health', '--asr-url'):
            self.argv[self.argv.index(option)+1] = previous[previous.index(option)+1]
        code, value = self.drive()
        self.assertEqual(code, 0)
        self.assertEqual(len(value['images']), 5)
        self.assertEqual(reader.read(value)['verdict'], 'PASS')
        with (frozen / 'harness/vision_encoder_service.py').open('a') as modified:
            modified.write('\n# altered packet\n')
        refused = subprocess.run(['bash', str(frozen / 'tools/vision_encoder_start.sh'), 'cpu-test',
            str(self.root / 'clearance.json')], env=dict(os.environ, PATH=str(fakebin)+':'+os.environ['PATH']),
            capture_output=True, text=True)
        self.assertNotEqual(refused.returncode, 0)


if __name__ == '__main__':
    unittest.main()
