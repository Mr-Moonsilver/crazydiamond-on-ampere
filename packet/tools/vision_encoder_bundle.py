#!/usr/bin/env python3
"""CPU-only vision tensor extraction and SHA-verified staging. No GPU or downloads."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

CONFIG_SHA = '0fabfa21fab8bfe69f02234f7ae8df4ed91b785fca2e22011c1230f6a07e5329'
CONFIG_FILES = ('config.json', 'preprocessor_config.json', 'processor_config.json',
                'tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json',
                'vocab.json', 'merges.txt', 'chat_template.jinja')
REQUIRED = ('config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pack(source, stream):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save

    source = source.resolve(strict=True)
    config = (source / 'config.json').read_bytes()
    if sha(config) != CONFIG_SHA:
        raise ValueError('model config differs from released Flash-Next')
    index_bytes = (source / 'model.safetensors.index.json').read_bytes()
    mapping = json.loads(index_bytes)['weight_map']
    prefixes = [p for p in ('visual.', 'model.visual.') if any(k.startswith(p) for k in mapping)]
    if len(prefixes) != 1:
        raise ValueError('ambiguous or missing vision prefix')
    selected = {k: v for k, v in mapping.items() if k.startswith(prefixes[0])}
    if len(selected) != 333:
        raise ValueError('expected exactly 333 vision tensors')
    tensors = {}
    for shard in sorted(set(selected.values())):
        path = (source / shard).resolve(strict=True)
        if not path.is_relative_to(source) or path.suffix != '.safetensors':
            raise ValueError('vision shard outside checkpoint')
        with safe_open(path, framework='pt', device='cpu') as handle:
            for name in sorted(k for k, v in selected.items() if v == shard):
                tensor = handle.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError('vision tensors must remain BF16')
                tensors[name] = tensor.contiguous()
    tensor_bytes = sum(t.nbytes for t in tensors.values())
    if not 800 * 2**20 <= tensor_bytes <= 920 * 2**20:
        raise ValueError('vision tensor bytes outside expected 856 MiB range')
    files = {'vision.safetensors': save(tensors)}
    files['model.safetensors.index.json'] = json.dumps(dict(metadata=dict(total_size=tensor_bytes),
        weight_map={name: 'vision.safetensors' for name in sorted(tensors)}), sort_keys=True).encode()
    for name in CONFIG_FILES:
        path = source / name
        if path.is_file():
            if path.is_symlink() or path.stat().st_size > 64 * 2**20:
                raise ValueError('unsafe processor/config file: ' + name)
            files[name] = path.read_bytes()
    if set(REQUIRED) - files.keys():
        raise ValueError('required processor/tokenizer files missing')
    files['vision-provenance.json'] = json.dumps(dict(schema='flash-next-vision-weights/1',
        config_sha256=CONFIG_SHA, source_index_sha256=sha(index_bytes), tensor_count=len(tensors),
        tensor_bytes=tensor_bytes, tensor_prefix=prefixes[0], dtype='bfloat16',
        language_tensors_copied=0), sort_keys=True).encode()
    files['VISION-SHA256SUMS'] = ''.join(sha(data) + '  ' + name + '\n'
        for name, data in sorted(files.items())).encode()
    with tarfile.open(fileobj=stream, mode='w|') as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o444
            archive.addfile(info, io.BytesIO(data))
    return dict(schema='vision-extract/1', tensor_count=333, tensor_bytes=tensor_bytes,
                manifest_sha256=sha(files['VISION-SHA256SUMS']), language_tensors_copied=0)


def verify(root):
    manifest = (root / 'VISION-SHA256SUMS').read_bytes()
    expected = {}
    for line in manifest.decode().splitlines():
        digest, name = line.split('  ', 1)
        if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or Path(name).name != name or name in expected):
            raise ValueError('invalid vision manifest entry')
        expected[name] = digest
    actual = {p.name for p in root.iterdir()}
    if actual != set(expected) | {'VISION-SHA256SUMS'}:
        raise ValueError('unmanifested or missing bundle file')
    for name, digest in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('non-regular bundle file')
        with path.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != digest:
                raise ValueError('vision SHA mismatch: ' + name)
    if set(REQUIRED) - expected.keys() or 'vision.safetensors' not in expected:
        raise ValueError('incomplete vision bundle')
    info = json.loads((root / 'vision-provenance.json').read_text())
    if (info.get('config_sha256') != CONFIG_SHA or info.get('tensor_count') != 333
            or info.get('language_tensors_copied') != 0 or info.get('dtype') != 'bfloat16'):
        raise ValueError('vision provenance differs')
    if sha((root / 'config.json').read_bytes()) != CONFIG_SHA:
        raise ValueError('vision config differs')
    return sha(manifest)


def unpack(archive_path, parent):
    # A failed incoming directory is retained, never removed or overwritten.
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if (len(members) > 16 or sum(m.size for m in members) > 1100 * 2**20
                or len({m.name for m in members}) != len(members)
                or any(not m.isfile() or Path(m.name).name != m.name for m in members)):
            raise ValueError('unsafe vision archive')
        parent.mkdir(parents=True, exist_ok=True)
        digest = sha(archive.extractfile('VISION-SHA256SUMS').read())
        incoming = parent / ('incoming-' + digest[:12])
        incoming.mkdir(exist_ok=False)
        for member in members:
            with (incoming / member.name).open('xb') as output, archive.extractfile(member) as data:
                while block := data.read(2**20):
                    output.write(block)
        if verify(incoming) != digest:
            raise ValueError('staged manifest identity changed')
        final = parent / ('flash-next-vision-' + digest[:8])
        if final.exists():
            raise ValueError('content-addressed destination already exists')
        incoming.rename(final)
        return final, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('pack'); a.add_argument('--source', type=Path, required=True)
    a = sub.add_parser('unpack'); a.add_argument('--archive', type=Path, required=True)
    a.add_argument('--parent', type=Path, required=True)
    a = sub.add_parser('verify'); a.add_argument('--model', type=Path, required=True)
    args = p.parse_args()
    if args.action == 'pack':
        print(json.dumps(pack(args.source, sys.stdout.buffer)), file=sys.stderr)
    elif args.action == 'unpack':
        path, digest = unpack(args.archive, args.parent)
        print(json.dumps(dict(model=str(path), manifest_sha256=digest)))
    else:
        print(json.dumps(dict(manifest_sha256=verify(args.model), verdict='PASS')))


if __name__ == '__main__':
    main()
