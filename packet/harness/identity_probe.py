#!/usr/bin/env python3
"""Read-only mount hashing, sent to `docker exec -i CONTAINER python3 -c ...`.

Input/output are JSON over pipes. Never prints file contents or exception text.
No torch/CUDA imports. Files are read at container destinations, never host sources.
"""
import hashlib
import json
import os
import stat
import sys


def fingerprint(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def main():
    request = json.load(sys.stdin)
    budget = dict(bytes=0, files=0)

    def file_hash(path):
        # O_NOFOLLOW closes replacement-to-symlink races. Symlinks use a separate
        # explicit path below and their referent is hashed as well as their target.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as f:
            before = os.fstat(f.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('not regular')
            budget['files'] += 1
            budget['bytes'] += before.st_size
            if budget['files'] > request['max_files'] or budget['bytes'] > request['max_bytes']:
                raise ValueError('hash budget exceeded')
            h = hashlib.sha256()
            while True:
                block = f.read(4 * 1024 * 1024)
                if not block:
                    break
                h.update(block)
            if fingerprint(before) != fingerprint(os.fstat(f.fileno())):
                raise ValueError('file changed during read')
        if fingerprint(before) != fingerprint(os.stat(path, follow_symlinks=False)):
            raise ValueError('file replaced during read')
        return dict(kind='file', bytes=before.st_size, sha256=h.hexdigest())

    def leaf(path):
        s = os.lstat(path)
        if stat.S_ISREG(s.st_mode):
            return file_hash(path)
        if stat.S_ISLNK(s.st_mode):
            target = os.readlink(path)
            # Directory symlinks cannot silently escape the tree scope.
            real = os.path.realpath(path)
            referent = file_hash(real)
            if fingerprint(s) != fingerprint(os.lstat(path)) or target != os.readlink(path):
                raise ValueError('symlink changed')
            return dict(kind='symlink', target=target, referent=referent)
        raise ValueError('unsupported special file')

    def content(path, exclusions):
        if not os.path.isdir(path) or os.path.islink(path):
            if exclusions:
                raise ValueError('exclusions on a file')
            return leaf(path)
        paths = [e['path'] for e in exclusions]
        def excluded(rel):
            return any(rel == p or rel.startswith(p + '/') for p in paths)

        def inventory():
            out = []
            for root, dirs, files in os.walk(path, followlinks=False, onerror=lambda _: (_ for _ in ()).throw(OSError())):
                for name in sorted(dirs + files):
                    p = os.path.join(root, name)
                    rel = os.path.relpath(p, path)
                    if excluded(rel):
                        if name in dirs:
                            dirs.remove(name)
                        continue
                    s = os.lstat(p)
                    out.append((rel, fingerprint(s)))
                    if len(out) > request['max_files']:
                        raise ValueError('entry budget exceeded')
            return sorted(out)
        before = inventory()
        entries = []
        for rel, stamp in before:
            p = os.path.join(path, rel)
            item = dict(kind='directory') if stat.S_ISDIR(stamp[2]) else leaf(p)
            entries.append(dict(path=rel, **item))
        if before != inventory():
            raise ValueError('tree changed during read')
        return dict(kind='directory', entries=entries, exclusions=exclusions)

    result = []
    for mount in request['mounts']:
        try:
            c = content(mount['destination'], mount['exclusions'])
            # Watch absent runtime switches even when their surrounding mutable
            # directory is explicitly excluded. Existing paths get full hashes.
            watched = []
            for rel in mount['watch']:
                path = os.path.join(mount['destination'], rel)
                item = leaf(path) if os.path.lexists(path) else dict(kind='absent')
                watched.append(dict(path=rel, **item))
            result.append(dict(destination=mount['destination'], content=c, watch=watched))
        except Exception:
            result.append(dict(destination=mount['destination'], error='mount unreadable, unstable, unsupported or over budget'))
    json.dump(dict(mounts=result, bytes_read=budget['bytes'], files_read=budget['files']), sys.stdout)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('{"error":"invalid probe input or hashing failed"}')
        sys.exit(2)
