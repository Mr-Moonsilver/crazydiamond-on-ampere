#!/usr/bin/env python3
"""Narrow q17b private auth file, never a credential-bearing process argument.

The existing VLLM_API_KEY comes only from this process environment. prepare
creates a new 0600 file; check validates the retained file and compares it in
memory. No key, digest, subprocess, credential minting or cleanup is performed.
Docker consumes the file as root via --env-file; its path is public settings.
"""
import argparse
import hmac
import os
from pathlib import Path
import stat
import sys

KEY = 'VLLM_API_KEY'
MAX_BYTES = 8192


class PrivateAuthError(ValueError):
    pass


def _value(value):
    if (type(value) is not str or not 8 <= len(value.encode('utf-8')) <= 4096
            or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in value)
            or '#' in value):
        raise PrivateAuthError('private auth value unavailable or unsupported')
    return value


def read_private(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise PrivateAuthError('private auth path must be absolute without symlinks')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_uid != os.geteuid() or before.st_nlink != 1
                or not 1 <= before.st_size <= MAX_BYTES):
            raise PrivateAuthError('private auth metadata invalid')
        raw = os.read(fd, MAX_BYTES + 1)
        after = os.fstat(fd)
        stable = lambda st: (st.st_dev, st.st_ino, st.st_mode, st.st_uid, st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if stable(before) != stable(after):
            raise PrivateAuthError('private auth changed during read')
    finally:
        os.close(fd)
    prefix = (KEY + '=').encode()
    if not raw.startswith(prefix) or not raw.endswith(b'\n') or raw.count(b'\n') != 1:
        raise PrivateAuthError('private auth file shape invalid')
    return _value(raw[len(prefix):-1].decode('utf-8'))


def prepare(path, environ=None):
    environ = os.environ if environ is None else environ
    value = _value(environ.get(KEY))
    path = Path(path)
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise PrivateAuthError('private auth path must be absolute without symlinks')
    parent = path.parent.stat()
    if not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700 or parent.st_uid != os.geteuid():
        raise PrivateAuthError('private auth parent must be owned mode 0700')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        data = (KEY + '=' + value + '\n').encode()
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    check(path, environ)


def check(path, environ=None):
    environ = os.environ if environ is None else environ
    if not hmac.compare_digest(read_private(path), _value(environ.get(KEY))):
        raise PrivateAuthError('private auth differs from the existing client key')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check'))
    parser.add_argument('path', type=Path)
    args = parser.parse_args(argv)
    try:
        (prepare if args.action == 'prepare' else check)(args.path)
    except Exception:
        print('private auth INCONCLUSIVE: unavailable, changed or unsupported private input')
        return 2
    print('private auth PASS: ' + args.action + '; value not emitted')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
