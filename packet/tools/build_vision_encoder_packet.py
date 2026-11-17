#!/usr/bin/env python3
"""Freeze the VM encoder-only trial. Does not stage, arm, mint clearance or use a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import zlib

FILES = ['harness/' + name for name in (
    'vision_encoder_job.py', 'vision_encoder_backend.py', 'vision_encoder_service.py', 'vision_speech_guard.py',
    'vision_tower_probe.py', 'test_vision_encoder_protocol.py', 'test_vision_encoder_bundle.py', 'test_vision_encoder_stage.py')]
FILES += ['tools/' + name for name in ('vision_encoder_read.py', 'vision_encoder_stop.py',
    'vision_encoder_bundle.py', 'vision_encoder_start.sh', 'stage_vision_encoder_weights.sh', 'build_vision_encoder_packet.py')]

def red_png(size, rgb=(255,0,0)):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = (b'\0' + bytes(rgb) * size) * size
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))

def build(out):
    root = Path(__file__).resolve().parent.parent
    out.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        dest = out / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, dest)
    images = out / 'harness/vision_assets'
    images.mkdir()
    for name in ('table.png', 'diagram.png', 'screenshot.png'):
        shutil.copy2(root / 'harness/vision_assets' / name, images / name)
    (images / 'red.png').write_bytes(red_png(512))
    (images / 'red1024.png').write_bytes(red_png(1024))
    import sys
    sys.path.insert(0, str(root / 'harness'))
    from vision_speech_guard import ASR_EXPECTED
    (out / 'asr-expected.txt').write_text(ASR_EXPECTED + '\n')
    source = (json.loads((root / 'delivery.json').read_text())['source'] if (root / 'delivery.json').is_file()
        else subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
    (out / 'delivery.json').write_text(json.dumps(dict(schema='vision-encoder-delivery/1', source=source,
        scope='VM<vm-id> encoder-only trial; NOT GPU-qualified, NOT box integration', gpu_cleared=False,
        model_manifest_sha256='8b333e2380aac6ed98c6ed5700474972cefb919cf126949fc31f70b47a39ca93',
        image='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b',
        gpu_physical_limit_mib=3072, torch_allocator_limit_mib=2048, bind='<encoder-host>:9041'), indent=2)+'\n')
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file())
    (out / 'PACKET-SHA256SUMS').write_text(''.join(hashlib.sha256((out/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in names))
    print(json.dumps(dict(packet=str(out), manifest_sha256=hashlib.sha256((out/'PACKET-SHA256SUMS').read_bytes()).hexdigest())))

if __name__ == '__main__':
    import sys, traceback
    try:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out', type=Path, required=True)
        build(p.parse_args().out)
    except Exception as exc:
        frame=traceback.extract_tb(exc.__traceback__)[-1]
        print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}', file=sys.stderr)
        raise SystemExit(2)
