#!/usr/bin/env python3
"""Build a gateway-only bind overlay; the frozen engine packet is never rewritten."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('<hub-dir>/cargo-target/flash-next/vision-headroom-w28-20261118/build-6/production')
PK='/data/flash-next-packets/serve-vision-prod-w28-20261118b'
PATCH='/data/flash-next-packets/gateway-queue-w28-20261118-build-7'
PARENT_SHA='344cb0fd98c1520b386117bc80de20e8f07eacd5b2589c029e80e80d287b2345'
IMAGE='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
NAME='flashnext-vision-gateway-w28-queue7'
AUTH='/root/.flashnext-serve-vision-prod-w27-20261117a'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(out):
 out.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(ROOT/'harness/vision_public_gateway.py',out/'vision_public_gateway.py')
 (out/'gateway-defaults.env').write_text('VISION_MAX_IMAGES=4\nVISION_GATEWAY_MAX_INFLIGHT=32\nVISION_GATEWAY_QUEUE_TIMEOUT_S=3000\n')
 gateway_sha=sha(out/'vision_public_gateway.py')
 # Use the gateway source identity as the retained-container identity; no circular unit hash.
 inspect='{{index .Config.Labels "flashnext.vision.gateway-source"}} {{.State.Running}} {{.Image}}'
 pre=f'import subprocess,time; n="{NAME}"; r=subprocess.run(["docker","inspect","-f",{inspect!r},n],capture_output=True,text=True); assert r.returncode or r.stdout.strip()=="{gateway_sha} false {IMAGE}","foreign/running gateway container"; r.returncode==0 and subprocess.run(["docker","rename",n,n+"-retained-"+str(time.time_ns())],check=True)'
 stop=f'import subprocess; n="{NAME}"; r=subprocess.run(["docker","inspect","-f",{inspect!r},n],capture_output=True,text=True); assert r.returncode or r.stdout.strip() in ("{gateway_sha} true {IMAGE}","{gateway_sha} false {IMAGE}"),"foreign gateway container"; r.returncode==0 and subprocess.run(["docker","stop","--time","30",n],check=True)'
 # systemd's parser honors shell-style quoting but does not run a shell.
 import shlex
 unit=f'''[Unit]
Description=Flash-Next keyed image gateway; queued admission build-7 (CPU only)
After=network-online.target docker.service vllm-flash.service
Wants=network-online.target
Requires=docker.service vllm-flash.service
PartOf=vllm-flash.service
Conflicts=vllm-qwen38.service

[Service]
Type=simple
EnvironmentFile=/etc/default/flashnext-vision-gateway
WorkingDirectory={PK}
ExecStartPre=/usr/bin/sha256sum --strict --quiet -c PACKET-SHA256SUMS
ExecStartPre=/bin/sh -ec 'cd {PATCH}; exec sha256sum --strict --quiet -c GATEWAY-SHA256SUMS'
ExecStartPre=/usr/bin/python3 -B {PK}/harness/vision_production_ready.py --packet {PK} --state-file /data/flash-next-control/vision-prod-w28/readiness.json
ExecStartPre=/usr/bin/python3 -c {shlex.quote(pre)}
ExecStart=/usr/bin/docker run --pull never --runtime runc --name {NAME} --label flashnext.vision.gateway-source={gateway_sha} --label flashnext.vision.packet={PARENT_SHA} --network host --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --cpus 2 --memory 2g --memory-swap 2g --log-driver local --log-opt max-size=16m --log-opt max-file=2 --tmpfs /tmp:rw,nosuid,nodev,size=64m -v {PK}:/packet:ro -v {PATCH}/vision_public_gateway.py:/packet/harness/vision_public_gateway.py:ro -v /data/flash-next-control/vision-prod-w28:/readiness:ro -v {AUTH}/vllm-api.env:/key/vllm-api.env:ro -v {AUTH}/vision-encoder.env:/key/vision-encoder.env:ro -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 {IMAGE} -B /packet/harness/vision_public_gateway.py --bind 0.0.0.0 --port 8000 --backend http://127.0.0.1:8001 --encoder http://<encoder-host>:9041 --max-images ${{VISION_MAX_IMAGES}} --max-inflight ${{VISION_GATEWAY_MAX_INFLIGHT}} --queue-timeout-s ${{VISION_GATEWAY_QUEUE_TIMEOUT_S}} --ready-file /readiness/readiness.json --packet-sha256 {PARENT_SHA} --key-file /key/vllm-api.env --encoder-key-file /key/vision-encoder.env
ExecStop=/usr/bin/python3 -c {shlex.quote(stop)}
Restart=on-failure
RestartSec=5
TimeoutStopSec=45
TimeoutStartSec=1800
UMask=0077

[Install]
WantedBy=multi-user.target
'''
 (out/'flashnext-vision-gateway.service').write_text(unit)
 files=('vision_public_gateway.py','flashnext-vision-gateway.service','gateway-defaults.env')
 (out/'GATEWAY-SHA256SUMS').write_text(''.join(sha(out/f)+'  '+f+'\n' for f in files))
 (out/'delivery.json').write_text(json.dumps(dict(schema='gateway-queue-w28/1',engine_parent_packet_sha256=PARENT_SHA,engine_overlay_unchanged=True,engine_launch_unchanged=True,cases_readers_unchanged=True,live_files=list(files),gateway_sha256=gateway_sha,admission_limit=32,admission_timeout_s=3000,metadata_bypass=['/health','/metrics','/v1/models'],body_preparation_limit=8,production_max_images=4,container=NAME,ladder=dict(steps=[4,8,12,16,20,24,32],gateway_code_bound=32,frozen_engine_image_bound=8,engine_row_bound=8192,engine_cache_bytes=41943040,above_engine_bound='clean400 before encoder/model; overlay unchanged',encoder_peak='VM allocator allocated/reserved from encode metrics; physical-card peak not inferred')),indent=2)+'\n')
 shutil.copyfile(ROOT/'harness/vision_image_count_ladder.py',out/'vision_image_count_ladder.py')
 for filename in ('vision_gateway_queue_repin.py','vision_gateway_queue_swap.sh'):shutil.copyfile(ROOT/'tools'/filename,out/filename)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.out)
