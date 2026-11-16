#!/usr/bin/env python3
"""Bounded vision-only HTTP service; owner/infra guard must wrap every GPU launch."""
import argparse
import base64
from concurrent.futures import Future, TimeoutError
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import re
import signal
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_encoder_backend import VisionBackend

MAX_BODY = 12 * 2**20


def error_info(exc):
    frame = traceback.extract_tb(exc.__traceback__)[-1]
    return dict(type=type(exc).__name__, message=str(exc)[:256], file=Path(frame.filename).name, line=frame.lineno)


class Encoder:
    def __init__(self, factory):
        self.queue = queue.Queue(maxsize=8)
        self.loaded = False
        self.stopping = False
        self.error = None
        self.encodes = 0
        self.weights = None
        self.last_metrics = None
        self.thread = threading.Thread(target=self.work, args=(factory,), daemon=True)
        self.thread.start()

    def work(self, factory):
        try:
            backend = factory()
            self.weights = dict(backend.weights, manifest_sha256=backend.manifest, attention_backend=backend.backend)
            self.loaded = True
            while not self.stopping:
                try:
                    future, image, deadline = self.queue.get(timeout=.25)
                except queue.Empty:
                    continue
                if not future.set_running_or_notify_cancel():
                    continue
                if time.monotonic() >= deadline:
                    future.set_exception(ValueError('encoder queue deadline'))
                    continue
                try:
                    result = backend.encode(image)
                    self.encodes += 1
                    self.last_metrics = result['metrics']
                    future.set_result(result)
                except Exception as exc:
                    future.set_exception(exc)
                    if type(exc).__name__ not in ('ValueError', 'UnidentifiedImageError'):
                        raise
        except Exception as exc:
            self.error = error_info(exc)
            print('VISION_ENCODER_ERROR ' + json.dumps(self.error), file=sys.stderr, flush=True)
        finally:
            self.loaded = False
            self.stopping = True
            while not self.queue.empty():
                future, _, _ = self.queue.get_nowait()
                if not future.cancelled():
                    future.set_exception(RuntimeError('encoder stopping'))

    def status(self):
        return dict(schema='flash-next-vision-health/1', ready=self.loaded and not self.stopping,
            queued=self.queue.qsize(), encodes=self.encodes, weights=self.weights,
            last_metrics=self.last_metrics, error=self.error)


def handler(encoder, key):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # no request bodies, embeddings, bearer headers or image URLs in logs

        def reply(self, code, value):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, dict(error='unknown endpoint'))
            value = encoder.status()
            self.reply(200 if value['ready'] else 503, value)

        def do_POST(self):
            if self.path != '/v1/encode':
                return self.reply(404, dict(error='unknown endpoint'))
            supplied = self.headers.get('Authorization', '')
            if not supplied.isascii() or not hmac.compare_digest(supplied, 'Bearer ' + key):
                return self.reply(401, dict(error='Unauthorized'))
            if not encoder.loaded or encoder.stopping:
                return self.reply(503, dict(error='encoder unavailable'))
            try:
                count = int(self.headers.get('Content-Length', '-1'))
                if count <= 0 or count > MAX_BODY:
                    return self.reply(413, dict(error='body exceeds packet bound'))
                value = json.loads(self.rfile.read(count))
                if set(value) != {'image_base64'} or not isinstance(value['image_base64'], str):
                    raise ValueError('one image_base64 field required')
                image = base64.b64decode(value['image_base64'], validate=True)
                if not image or len(image) > 8 * 2**20:
                    raise ValueError('image byte bound')
                future = Future()
                encoder.queue.put_nowait((future, image, time.monotonic() + 30))
            except queue.Full:
                return self.reply(429, dict(error='encoder queue full'))
            except (ValueError, TypeError):
                return self.reply(400, dict(error='invalid bounded image request'))
            try:
                self.reply(200, future.result(timeout=30))
            except TimeoutError:
                future.cancel()
                self.reply(504, dict(error='encoder deadline'))
            except Exception as exc:
                code = 400 if type(exc).__name__ in ('ValueError', 'UnidentifiedImageError') else 500
                self.reply(code, dict(error=error_info(exc)))
    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--key-file', type=Path, required=True)
    args = p.parse_args(argv)
    if args.key_file.is_symlink() or args.key_file.stat().st_mode & 0o777 != 0o600:
        raise ValueError('encoder key file must be private regular 0600')
    fields = dict(line.split('=', 1) for line in args.key_file.read_text().splitlines() if '=' in line)
    key = fields.get('VISION_ENCODER_API_KEY', '')
    if not re.fullmatch('[0-9a-f]{64}', key):
        raise ValueError('encoder key identity missing')
    encoder = Encoder(lambda: VisionBackend(args.model, args.run_dir))
    server = ThreadingHTTPServer(('<encoder-host>', 9041), handler(encoder, key))
    server.daemon_threads = True
    def stop(*_):
        encoder.stopping = True
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        encoder.stopping = True
        server.server_close()


if __name__ == '__main__':
    try:
        main()
    except SystemExit as exc:
        if exc.code:
            print('PACKET_CLIENT_ERROR ' + json.dumps(error_info(exc)), file=sys.stderr)
        raise
    except Exception as exc:
        print('PACKET_CLIENT_ERROR ' + json.dumps(error_info(exc)), file=sys.stderr)
        raise SystemExit(2)
