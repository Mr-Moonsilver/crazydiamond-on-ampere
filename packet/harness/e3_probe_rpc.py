#!/usr/bin/env python3
"""One bounded authenticated loopback request; no retries, redirects or shell.

stdin/stdout JSON adapter for e3_w1_decode_driver.CommandController. The owner
pins this executable; PROBE_URL and the existing VLLM_API_KEY stay environment
only. --raw selects the fixed low-level diagnostic endpoint for owner checks.
"""
import argparse
import http.client
import ipaddress
import json
import math
import os
import sys
from urllib.parse import urlsplit

PATH = '/v1/flash_s5_decode_probe'
MAX_COMMAND = 16384
MAX_RESPONSE = 4 * 1024 * 1024


class TransportError(RuntimeError):
    pass


def target(url):
    try:
        u = urlsplit(url)
        if (u.scheme not in ('http', 'https') or u.username is not None
                or u.password is not None or u.path not in ('', '/')
                or u.query or u.fragment or not ipaddress.ip_address(u.hostname).is_loopback):
            raise ValueError()
        port = u.port or (443 if u.scheme == 'https' else 80)
        return u.scheme, u.hostname, port
    except (ValueError, TypeError):
        raise TransportError('invalid-loopback-base-url') from None


def exchange(command, *, url, key, timeout_s, raw=False):
    scheme, host, port = target(url)
    if (type(key) is not str or not key or '\n' in key or '\r' in key
            or type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise TransportError('invalid-auth-or-timeout')
    if type(command) is not dict:
        raise TransportError('command-not-object')
    data = json.dumps(command, allow_nan=False).encode()
    if len(data) > MAX_COMMAND:
        raise TransportError('command-too-large')
    connection = (http.client.HTTPSConnection if scheme == 'https' else http.client.HTTPConnection)(
        host, port, timeout=timeout_s)
    try:
        connection.request('POST', PATH if raw else PATH + '/session', body=data,
            headers={'Content-Type':'application/json', 'Authorization':'Bearer ' + key})
        response = connection.getresponse()
        if response.status != 200:
            # Never print response/error bodies or URLs; they can echo secrets.
            raise TransportError('http-status-' + str(response.status))
        body = response.read(MAX_RESPONSE + 1)
        if len(body) > MAX_RESPONSE:
            raise TransportError('response-too-large')
        result = json.loads(body)
        # Read-only status is deliberately sequence-independent (and remains
        # available after session quarantine), so it has no command_seq to
        # echo.  Its identity is the explicit read_only flag plus the
        # sequence_independent marker.
        read_only = (command.get('action') == 'status'
                     and command.get('read_only') is True)
        if read_only:
            identity_ok = (type(result) is dict
                           and type(result.get('schema')) is int
                           and result['schema'] == 1
                           and result.get('run_id') == command.get('run_id')
                           and result.get('action') == 'status'
                           and result.get('read_only') is True
                           and result.get('sequence_independent') is True)
        else:
            expected_seq = command.get('seq', 0) if raw else command.get('command_seq')
            identity_ok = (type(result) is dict
                           and type(result.get('schema')) is int
                           and result['schema'] == 1
                           and result.get('run_id') == command.get('run_id')
                           and result.get('action') == command.get('action')
                           and type(result.get('command_seq')) is int
                           and result['command_seq'] == expected_seq)
        if not identity_ok:
            raise TransportError('response-identity-differs')
        return result
    except TransportError:
        raise
    except Exception as exc:
        raise TransportError('transport-failed-' + type(exc).__name__) from None
    finally:
        connection.close()


def main(argv=None, env=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', action='store_true')
    args = parser.parse_args(argv)
    env = os.environ if env is None else env
    try:
        data = sys.stdin.buffer.read(MAX_COMMAND + 1)
        if len(data) > MAX_COMMAND:
            raise TransportError('command-too-large')
        command = json.loads(data)
        result = exchange(command, url=env.get('PROBE_URL'), key=env.get('VLLM_API_KEY'),
                          timeout_s=float(env.get('PROBE_TIMEOUT_S', '120')), raw=args.raw)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        message = str(exc) if type(exc) is TransportError else type(exc).__name__
        print(json.dumps({'error':message}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
