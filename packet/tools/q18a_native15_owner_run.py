#!/usr/bin/env python3
"""Owner-only serialized candidate chain. Codex never executes it on the box.

Combined PRE-STAGE + ARM GO requires an already stopped engine. Reads the freshly
minted owner-private env file; its value is never in argv or output. This
runner keeps the harness immutable and creates only fresh RAW/FINAL outputs.
Operator order28Oct night requires queue hard end by29Oct08:00CET, owner restores by08:30.
"""
import argparse
import datetime
import time
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

sys.dont_write_bytecode=True
H=Path('/home/<service-user>/harness/6525c8058108')
S0=Path('/home/<service-user>/s0')
QNAME='q18a-native15'
OVERLAY=Path('/data/vllm/flash-next-tp4/prod/overlay-v16s5-q18a-native15')
BUNDLE='6525c805810822a095ac6011ef31a969dbeb6344ddd9dfd5e86c78d7c4366bc7'
BASE=Path('/data/vllm/flash-next-tp4/prod/overlay-v16e2f')
BASE_MANIFEST='da561411fadbb82922eeb8eafbe9424bc9703aaf8c2030bd813743c692b641a5'
IMAGE='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
REVIEW='50852eaebe7e0e7c6f143ed17190cde3a9f05fdf97e2ec996ff18483fea957c0'
LAUNCHER='83ca0128798fe184b87c006149800994c825979ecedae24a72387b4d7a7085a4'
EXPECTED_SOURCES={
    "E2-GENERATION.json": "822c670c45bcd49f52f72dd985699349caf14590e68042c7b9ba81f4eee4cef6",
    "core/SHA256SUMS": "895bcf0bcda5ea2d31146495b1f3d8cdaa7ffc54d77a809566b415370bb3bf21",
    "core/block_pool.py": "540bb7bec14b0dfc6106e3a02d1853fa842f951b70ccab79c77bb9e729f3fe39",
    "core/scheduler.py": "714137a08e97ee517c29c3c9befd79121711ddaaab4ebedda50d1b40250b97d7",
    "core/single_type_kv_cache_manager.py": "c49dcf97240537443c956c249b9c0e2baf3c5c77a6afeec28eb0a2a48de2ac18",
    "e2/__init__.py": "d050b91c6f73554fc3e86611fd008190c8e2b82cbd8eafe6cd5fb04fc93ff3a2",
    "e2/arena_contract.py": "7023d93b310444d32cbdc86a03576f8008eddca2ef2d22c5ee0db16e142e603a",
    "e2/arena_dma.py": "ed6339cb6aeb9a5dff3753d5e022cb44fbb4e5b9a9f5ea3f7c988b9bad2c94cd",
    "e2/block_mirror.py": "cd4245fe0d8629275c26aaf8f5b0e4bd4fba5df747de4397208981ee6292696d",
    "e2/exactness.py": "baf79d18af1b1c97f5b61472c7bac93d9d937d9f72adcbe517212718c1f10545",
    "e2/retention_contract.py": "65331a63dbbbf63d1c1c177add552911b28edf194bd695e203759fd8a4508cf3",
    "e2/runner_lifecycle.py": "4a0b521bb4c0581d46194738f47b96d8c7b870f32aadb98186c392879234cd87",
    "e2/serial_dma.py": "c4f9da7430bbe7ad869357b024196b046028ebb044488c4b0c225be02a5cb071",
    "e2/startup.py": "83e06f59b3fcbe3260efacf9ef9f92391201111a8177e4c4c4980fe1d266a87c",
    "e2_sources/block_table.py": "fc96b4c814d6f1a15f5e5f297689057e7bf05e5fa524b3771aac7fe61796bccb",
    "e2_sources/scheduler_output.py": "fa4937fdd0564a8d91a91f370d5a84f2e5c458d05442b545d9ad88cd88e36467",
    "e3/__init__.py": "f7337dd3a07c897b341f09ab17b7c04bd2b409f12dd6faa3fa16d38ea9f35910",
    "e3/activity_accounting.py": "3390bd7ace83abae039d7371947d79148899ae772a8ed75c6691b59aa5e9ffb3",
    "e3/admission_metadata.py": "cc66a1cbf13132cb624213fe4761d5c95f0755f56851f075d8110557bb665cdb",
    "e3/async_handoff.py": "0bf965d34f09cc3ea68de57195922c34409783b21c7893fae7882e5978e4f452",
    "e3/capture_streams.py": "46181a6f4ee4650244f0de9cfa54a663be95347083accaf82297f53de9dea4c2",
    "e3/decode-probe-evidence.json": "f71cde928752d7eb57fd7d1f9d0a61f2d70ebe75de4adbd17bbb6717fabd5976",
    "e3/decode_probe.py": "67415c45806f96297d82f6c0b2c9d746613b5f6931d976664d513ec560f1d1d8",
    "e3/e3_types.py": "8dda9d6b2fa181352a0bacd387a042790fe3c328796153b8e8f10a60ed835bcf",
    "e3/engine_driver.py": "c9fe60dcfbef8ec387182b7d865380411e11c73a3c4290bf6714f9589f962dd8",
    "e3/engine_offers.py": "c314ca1937b72ef4b8a53cef2cb13083c489881dafc6cb9efbbe6c675cdb27ec",
    "e3/executor_hooks.py": "00ba86e8537fbf072791f9de1363b2ddbe8380ac89e7fb1415d66ba68b202fcf",
    "e3/graph_hooks.py": "d635ef93713da22b274bb819dbe506f8992e5a9305bc8ef2300ea4498400c222",
    "e3/graph_router.py": "9545c5d7d9346872a89a5c89e0b12aac6de914b57affd6d71e641d0c2235dc8c",
    "e3/green_context.py": "e875f26b3a7d4b9e20e8e2b5ab4d1b7a4ebfce6d6b15e7301da1c831abee8847",
    "e3/lane_chunk.py": "8c349430c6401cbcd6c4aedc25b70918feef188d33437443aff70658f9deb062",
    "e3/lane_comm.py": "7f964d9d3771bfe3fbe99128293295d3b9f14581453009547338c4542882df26",
    "e3/lane_dispatch.py": "5ceae14accc4c0c06dbb696e7d9068bd994a278f0c874f674ad105d6dcc28901",
    "e3/lane_hooks.py": "0a0a03ef2784f15ca173c1a54a15900bdd9cc85e1b6fb5cede794ed956308d71",
    "e3/lane_private.py": "d76016655f5951670615efd1f6d6bb7491965587c4ef7b2b03c7ae95909d4da2",
    "e3/lane_state.py": "f31c755e47f0ad04a61a87deb9f5777f96e62bc4b4fa071e809782d8d24678c1",
    "e3/launch_diagnostic.py": "29925d2946217906d8a4134a69ad41d408d9c09de670a677df80534608e0fff7",
    "e3/model_hooks.py": "7b9c24ac3a2fcd220fbfa1b18393be89dcde9d79084b5de63c4aeaea1929b8d0",
    "e3/native-implementation-closure.json": "e2b83a67fb7427654503640aeb020a4ab95f635b5a1ec226cc66feb531725e68",
    "e3/native/build_green_helper.sh": "b47a0d466d676dc08d85d12458926f3e5dcd6a3cb28097136207f8ff1a979b73",
    "e3/native/build_q17_helpers.sh": "6e59366e0b057c1bcea8bc677d29d694111125f11b89edb3b15773801b710ec4",
    "e3/native/flash_s5_graph_bytes.cpp": "56f550ebc345ee96077bb473264c42cce92140218dc014f7555718bf95e36110",
    "e3/native/flash_s5_green.cu": "c2151ef6a805f620d6c57ef57c405fc0cdbce3c815b25574f9e0d2e7f1fa1907",
    "e3/native/flash_s5_replay_observer.cpp": "3713542b4165fb28e85d7099b3c3d91690856b2770b81da1aa179c6b4e605ece",
    "e3/native_backend.py": "9547ccea32e56e7266199a411a65067900cfed6d5697697b5fb03886145aaaa3",
    "e3/native_graph_placement.py": "51544880fee8c7b8a3fe9da65a25e609179911d7c77cc6a9fff3a0f7e7168435",
    "e3/native_measurement.py": "538556c7e9a13fef85738f0d5ca62f3aa06b9c258cb4380c07f848e768da51c0",
    "e3/native_observation.py": "e06c1a9a924bb17473668633e2aa865a2eb075131d32070db94140fc4550f659",
    "e3/native_overlap_control.py": "5edc4917f846cb530361c46a09f04ed375319e62076f30b0c17cc3838425de31",
    "e3/native_overlap_metrics.py": "17d3f5de2db7db8acc4cb5a130351ce169d7b5b76f052a03fd9a0315430827fa",
    "e3/native_overlap_session.py": "16fd8da27d3b2330c8d42828e789a8e421ac5f3e887812ea1e0d3315ef058999",
    "e3/native_overlap_worker.py": "ea9e38b1ffd515e0c59c2c7620e4a958ca70177b465a0a50622a04eae376d380",
    "e3/native_placement.py": "0f243688dcde5c72b8338fd2bbc1755b002050a76e5f044661cee146a8afe17e",
    "e3/native_prefill_offers.py": "711b175772276603985572e4ee2e7352c31e46ec9929620ef12b400e4a652e56",
    "e3/native_qualification.py": "e0d7f7d3651d8e1edc9b85e1662b0ac57c7d271f115a1b12288c72beb61f9272",
    "e3/native_qualification_frontend.py": "1a3cd4f9e4ce6756481cb7480381d553197d41d259942758cdcf47ccebb6b8bb",
    "e3/native_readiness.py": "e6fb9a28599a3206b14d2e9d35af68d108eeda02599f219b81bec5139d88f082",
    "e3/native_resources.py": "e6bd08679b74a004fee42b2a4ca37fe7fba73e3b8aa63216ff6a889381e054fd",
    "e3/native_source.py": "a5172076b061fa4dbbb473399cb8357372f8edf53676735ee60a972d1e2a5361",
    "e3/native_startup.py": "9cfe8c4f644bebf088a2ca353f3c1bace6dda6339b57931472eb098ec8b2caaa",
    "e3/observer_runtime.py": "277ad3ad74a8ba496482326cb09a457073dba192ab5d31e47e07481e8f5936e0",
    "e3/operation_source.py": "4dbcab39f7f207c519b436ea6308b304ffc46a70fe9090042ee20daf3a5ae8f6",
    "e3/probe_control.py": "5f4488aa793eade263aaa5e951c00e044c4148cff9746c52fd0820c464ea8818",
    "e3/probe_control_hooks.py": "4e3f298a67d013a74822d1e6a1c3e84c807a5e6eef8e2c8f861f19afbe154301",
    "e3/probe_session.py": "d5e25b0772b1f77b2a672df2b83df9c37078ed837c786f46bd82eb4427bf731c",
    "e3/qsa_driver.py": "b7f78c8830ab2a5a9617da4fc49da2a0c1178cda3af959cc651202c50a3a1bb8",
    "e3/qsa_operation.py": "f2801bfa53ef0855829998245d3bd4bb77db8e227c1a57deac30537372c68718",
    "e3/replay_observer.py": "214342376ed6fb054940a85e2363f1817f260b9ed26a921c0b8af40c4651edbd",
    "e3/reply_transport.py": "6c0978654e022605379a7b845ba5f592f3d200f7015024ec69f42936e509259b",
    "e3/route_invariant.py": "e20333e2a4e104cf2fe24aad2b3ef1d91a75effc2ee47b7a763f595a66ff6884",
    "e3/row1_exactness.py": "2b816cf781a9c25bc069afd25b26dc64412e77428a0889221610837a7be2f799",
    "e3/row1_oracle.py": "bd94ba04c0317c4cdb90e140bbe4f09fc79c6a4c73d64939197ffd4a4044dc04",
    "e3/s5_startup.py": "c8ab796b9ea9d627485962f6fc4895454b8c4504a6a90dc952b0e0791ae9f1f3",
    "e3/scheduler_hooks.py": "45a24a099f9e0669b927fa1294a3502a00691b593481e9e1817cca36d041681d",
    "e3/sparse_history_dma.py": "060e728aa519f88fdc39a43218dd553c3dd8d18be8b5758f5d9a9657a52777c2",
    "e3/split_bank.py": "f504a07ff2b56c1f9ba5ec3ea371ec0fdce51bc06a42ae9c9ea5ea8522b7c50d",
    "e3/teacher_decode.py": "8ab74b57de7ea90f4a94154d9bd609ab748400e989f58eb5bc553d427faff05c",
    "e3/wire_v1.py": "c97e245811560b00321837e0f882f7b88bf3d67904f4b9247bebdf101a879000",
    "e3/worker_bridge.py": "30ee0e84ea1be09eafc218ad5ca9b4545da7518d5c38a1fd9ae136619e6f4dc6",
    "e3/worker_hooks.py": "16d207a3a6c1c69a32417f3959218aada9412b24aa6a4fe9b5491414e49cde4e",
    "e3_sources/api_routers.py": "867116ae133e04ff70651d25d50aabdfe0ec70de875fbcef7326a83ef92814f1",
    "e3_sources/async_utils.py": "db92eb57f96117c502ca5a5d435ddc23e4541f378111745edd3fa6f5e1fb8a8f",
    "e3_sources/core.py": "cf54cfbd78890373a95644da8a2790077604a6e438e7f2a65a2a3d6e34be1df2",
    "e3_sources/cudagraph_utils.py": "c7ecc66d5763f0e54f0ef28ed3e11fb4a9f0a6378c47cf0139cf24c33ce70f54",
    "e3_sources/executor.py": "b24fff559f478ce28a1edeb22b0e4a0aeecb0fc6b78c6386c8fc78fd0f8bbcb7",
    "e3_sources/model.py": "14e2ccda5a050688c7cef8405689926f20914a3f35cc56d1477e819d3c907ffb",
    "e3_sources/sampler.py": "351b5d5d61d2d2ce3592895ca020785a05a008473bc793353ef09a2c153866aa",
    "e3_sources/worker.py": "16680f39373f36efdccf7d1a88ea4e6ebde4d9572b4d1eb260bfe77e94b0aa0f",
    "model_runner_v2.py": "fd387b20b819594694b9947c29a4cb6368082b53731e7a2512beb7cd5c25d187",
    "ops_qsa.py": "c98a74a33c22debc795d4d5b2f8abd3fee3b99595fd1a89de48d5c036a07deb4",
    "qsa.py": "4414b1a840b42f733c48720a27cb3581568802d810d3b9d9df8f0256dcd61454",
    "run-flash-e2-test.sh": "2630f86c961c3c5ecff9b1be87b1602fca182f48484991ea1744071dea5f7d8b",
    "run-flash-s5-test.sh": "83ca0128798fe184b87c006149800994c825979ecedae24a72387b4d7a7085a4",
    "vision/SHA256SUMS": "8c8b75d8821ad2db71377311226a8453778afcb426037665c596a786026a87e1",
    "vision/model.py": "a1a7712b188238c0ca2f29522ef0b65a8f17efe64181b54a7636990a7a1b518e"
}
RAW=OVERLAY.with_name(OVERLAY.name+'-build')
BUILD_LOGS=S0/(QNAME+'-build-logs')
GREEN=None  # bound only after source/image/build/finalizer checks
MANIFEST=None  # bound to this fresh verified CPU build
OBSERVER='05a4846d17699a692e923882fcf6c7e8eb4cf486f1888823ac1e4072695ac06d'
_REDACTIONS=[]  # credential values stay in this owner process's memory only


class Refused(ValueError):
    pass


def require(condition,reason):
    if not condition:raise Refused(reason)


def sha(path):
    require(path.is_file() and not path.is_symlink(),'regular-public-file-required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(key,model,cap,original=None):
    require(type(GREEN) is str and re.fullmatch('[0-9a-f]{64}',GREEN),'verified-build-required')
    require(type(key) is str and re.fullmatch('[0-9a-f]{64}',key) is not None,'fresh-64-hex-key-required')
    require(type(model) is str and model and len(model)<=200 and not any(x in model for x in '\r\n'),'model-required')
    require(type(cap) is int and 64*1024**2<=cap<=2*1024**3,'snapshot-cap-invalid')
    if key not in _REDACTIONS:_REDACTIONS.append(key)
    env=dict(os.environ if original is None else original)
    for name in list(env):
        if name.startswith('Q17_TEST_') or name in ('Q18A_OVERLAP_BOUNDED','S5_LAUNCHER_SHA','CONTROLLER_CMD'):
            env.pop(name)
    env.update(HARNESS_DIR=str(H),S0=str(S0),QNAME=QNAME,
        S5_OVERLAY=str(OVERLAY),S5_GREEN_LIB_SHA256=GREEN,S5_REPLAY_LIB_SHA256=OBSERVER,
        S5_LAUNCHER_NAME='run-flash-s5-test.sh',S5_REQUESTED_DECODE_SMS='44',
        S5_MODEL=model,Q18A_TARGET_LABEL='s5-overlap-80-80',
        Q18A_READINESS=str(OVERLAY/'e3/native-overlap-readiness.json'),
        S5_INVARIANT_MAX_BYTES=str(cap),PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join(map(str,(H/'e3_builder',H/'e2_builder',H))),
        PROBE_URL='http://127.0.0.1:8000',VLLM_API_KEY=key)
    return env


def available_memory():
    info={line.split(':')[0]:int(line.split()[1])*1024
          for line in Path('/proc/meminfo').read_text().splitlines()
          if len(line.split())>=2 and line.split()[1].isdigit()}
    available=info['MemAvailable']
    group=Path('/sys/fs/cgroup')
    if (group/'memory.current').is_file() and (group/'memory.max').is_file():
        current=(group/'memory.current').read_text().strip()
        maximum=(group/'memory.max').read_text().strip()
        if current.isdigit() and maximum.isdigit():available=min(available,int(maximum)-int(current))
    return available


def public_output(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    require(p.returncode==0,'public-preflight-command-failed')
    return p.stdout.strip()


def stopped_unit(raw):
    """Certify physically stopped unit state, including retained failure status.

    native2 cleanup intentionally leaves failed/NRestarts1. That historical
    flag is not an active engine; retain it until the unchanged owned queue's
    reset-failed/start sequence. Both main and control PIDs must be zero.
    """
    expected={'ActiveState','SubState','MainPID','ControlPID'}
    fields={}
    for line in raw.splitlines():
        key,sep,value=line.partition('=')
        require(sep and key in expected and key not in fields,'unit-state-proof-malformed')
        fields[key]=value
    require(set(fields)==expected,'unit-state-proof-incomplete')
    require((fields['ActiveState'],fields['SubState']) in
            (('inactive','dead'),('failed','failed'))
            and fields['MainPID']=='0' and fields['ControlPID']=='0',
            'owner-must-stop-production-first')
    return fields


def manifest(root):
    rows={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split(maxsplit=1)
        path=Path(name)
        require(re.fullmatch('[0-9a-f]{64}',digest) and name not in rows
                and not path.is_absolute() and '..' not in path.parts
                and sha(root/path)==digest,'manifest-payload-invalid')
        rows[name]=digest
    require(bool(rows),'empty-manifest')
    return rows


def inventory(root):
    require(root.is_dir() and not root.is_symlink(),'regular-package-directory-required')
    entries=list(root.rglob('*'))
    require(not any(p.is_symlink() or not (p.is_dir() or p.is_file()) for p in entries),
            'unsupported-package-entry')
    return {p.relative_to(root).as_posix() for p in entries if p.is_file()}


def check_slot():
    hard_end=datetime.datetime(2026,10,29,7,0,tzinfo=datetime.timezone.utc).timestamp()
    require(time.time()+14400+900<=hard_end,'night-window-worst-case-exceeds-restore-margin')


def live_guards(cap):
    check_slot()
    require(type(cap) is int and 64*1024**2<=cap<=2*1024**3,'snapshot-cap-invalid')
    require(os.geteuid()==0,'owner-root-required')
    stopped_unit(public_output(['systemctl','show',
        '--property=ActiveState,SubState,MainPID,ControlPID','vllm-flash']))
    raw=public_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'])
    devices=[tuple(int(part.strip()) for part in line.split(',')) for line in raw.splitlines()]
    require(len(devices)==4 and {i for i,_ in devices}==set(range(4)) and all(v==0 for _,v in devices),'four-idle-gpus-required')
    staged=Path('/etc/default/vllm-flash.next')
    require(not staged.exists() and not staged.is_symlink(),'defaults-next-present')
    for suffix in ('','-requests','.log','-declare.log','-declare-chain.log'):
        p=S0/(QNAME+suffix)
        require(not p.exists() and not p.is_symlink(),'fresh-output-path-required')
    available=available_memory();required=4*cap+5*1024**3
    require(available>=required,'stopped-host-headroom-insufficient')
    print('ARM_GUARD PASS available_bytes='+str(available)+' required_bytes='+str(required),flush=True)


def preflight(cap):
    live_guards(cap)
    require(sha(H/'SHA256SUMS')==BUNDLE,'harness-manifest-pin')
    rows=manifest(H)
    require(inventory(H)==set(rows)|{'SHA256SUMS'},'harness-inventory')
    require(sha(BASE/'SHA256SUMS')==BASE_MANIFEST,'base-manifest-pin')
    manifest(BASE)
    for p in (RAW,OVERLAY,BUILD_LOGS):
        require(not p.exists() and not p.is_symlink(),'fresh-build-path-required')
    sys.path[:0]=[str(H/'e3_builder'),str(H)]


def verify_sources(root,*,built=False,final=False):
    require(bool(EXPECTED_SOURCES),'predeclared-source-map-missing')
    extras={'e3/native/q17-helper-build.json',
            'e3/native/flash_s5_green.so','e3/native/flash_s5_replay_observer.so'} if built else set()
    allowed=set(EXPECTED_SOURCES)|{'S5-OVERLAP-GENERATION.json','SHA256SUMS'}|extras
    if final:allowed.add('e3/native-overlap-readiness.json')
    require(inventory(root)==allowed,'generated-inventory-differs-from-predeclared-source-map')
    for name,digest in EXPECTED_SOURCES.items():
        require(sha(root/name)==digest,'predeclared-source-pin:'+name)
    rows=manifest(root)
    require(set(rows)==allowed-{'SHA256SUMS'}- (extras if not final else set()),
            'generated-manifest-inventory-differs')
    generation=json.loads((root/'S5-OVERLAP-GENERATION.json').read_text())
    require(generation.get('candidate_profile')=='q18a-native-performance'
            and generation.get('implementation_review_sha256')==REVIEW
            and generation.get('base_manifest_sha256')==BASE_MANIFEST,
            'generation-source-identity-differs')
    require(sha(root/'run-flash-s5-test.sh')==LAUNCHER,'trusted-launcher-pin')
    if not final:require(generation.get('armable') is False,'raw-package-must-be-unarmed')


def build_step(args,stdout,stderr=None):
    with stdout.open('xb') as out:
        if stderr is None:
            rc=subprocess.run(args,env=build_environment(),stdout=out,stderr=subprocess.STDOUT,timeout=900).returncode
        else:
            with stderr.open('xb') as err:
                rc=subprocess.run(args,env=build_environment(),stdout=out,stderr=err,timeout=900).returncode
    require(rc==0,'CPU-build-step-refused:'+stdout.name+':rc='+str(rc))
    print('CPU_BUILD '+stdout.name+' rc=0',flush=True)


def build_environment():
    # Credentials and test/loader overrides do not enter CPU build children.
    env={k:v for k,v in os.environ.items() if k not in
         ('VLLM_API_KEY','ANTHROPIC_AUTH_TOKEN','ARM_AUTH_ENV_FILE','PYTHONPATH','PYTHONHOME','LD_PRELOAD')
         and not k.startswith(('Q17_TEST_','Q18A_','FLASH_S5_','S5_'))}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(map(str,(H/'e3_builder',H/'e2_builder',H))))
    return env


def cpu_build():
    BUILD_LOGS.mkdir(mode=0o700)
    build_step(['python3','-B',str(H/'e3_builder/make_s5_overlap.py'),str(BASE),str(RAW),
        '--reference',str(H/'e3_reference'),'--base-manifest-sha256',BASE_MANIFEST,'--native-candidate'],
        BUILD_LOGS/'step1.json',BUILD_LOGS/'step1.err')
    verify_sources(RAW)
    build_step(['bash',str(H/'e3_builder/native/build_q17_helpers.sh'),IMAGE,str(RAW/'e3/native')],
        BUILD_LOGS/'step2.log')
    verify_sources(RAW,built=True)
    build_step(['python3','-B',str(H/'e3_finalize_package.py'),str(RAW),str(OVERLAY)],
        BUILD_LOGS/'step3.json',BUILD_LOGS/'step3.err')
    return verify_final_build()


def verify_final_build():
    verify_sources(OVERLAY,built=True,final=True)
    from native_readiness import verify_finalized_package
    from decode_probe import GENERATED_SOURCES
    capability=verify_finalized_package(OVERLAY,GENERATED_SOURCES)
    require(capability.get('review_sha256')==REVIEW,'final-review-pin')
    require(sha(OVERLAY/'e3/native/flash_s5_replay_observer.so')==OBSERVER,'VM-qualified-observer-pin')
    receipt=json.loads((OVERLAY/'e3/native/q17-helper-build.json').read_text())
    require(receipt.get('image')==IMAGE and receipt.get('gpu_work') is False,'CPU-build-image-or-gpu-work')
    green=sha(OVERLAY/'e3/native/flash_s5_green.so')
    final=sha(OVERLAY/'SHA256SUMS')
    # This immutable public receipt pins the *produced* binary/manifest after
    # all predeclared source/build checks; it cannot approve arbitrary input.
    proof={'schema':'q18a-native15-owner-build/1','bundle_sha256':BUNDLE,
           'review_sha256':REVIEW,'manifest_sha256':final,'green_library_sha256':green,
           'observer_library_sha256':OBSERVER,'hardware_qualified':False,'release_qualified':False}
    return green,final,proof


def phase(script,log,env):
    with log.open('xb') as handle:
        return subprocess.run(['bash',str(H/'runs'/script)],env=env,
                              stdout=handle,stderr=subprocess.STDOUT).returncode


def diagnostic(exc):
    message=('input-encoding-invalid' if isinstance(exc,UnicodeError) else str(exc))
    for secret in (*_REDACTIONS,os.environ.get('VLLM_API_KEY')):
        if type(secret) is str and secret:message=message.replace(secret,'***')
    return {'error_type':type(exc).__name__,
            'error_message':message.replace('\n',' ').replace('\r',' ')[:512],
            'error_stack':[{'file':Path(frame.filename).name,'line':frame.lineno,
                            'function':frame.name}
                           for frame in traceback.extract_tb(exc.__traceback__)[-8:]]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--auth-env-file',type=Path,required=True)
    p.add_argument('--model',default='qwen38-w8a8-tp4')
    p.add_argument('--snapshot-max-bytes',type=int,default=2*1024**3)
    a=p.parse_args(argv)
    global GREEN,MANIFEST
    preflight(a.snapshot_max_bytes)
    GREEN,MANIFEST,proof=cpu_build()
    with (BUILD_LOGS/'owner-build-receipt.json').open('x') as f:json.dump(proof,f,sort_keys=True,indent=2);f.write('\n')
    print('PRE_STAGE PASS manifest_sha256='+MANIFEST+' green_library_sha256='+GREEN,flush=True)
    live_guards(a.snapshot_max_bytes)
    require(sha(OVERLAY/'SHA256SUMS')==MANIFEST,'built-manifest-changed-before-declare')
    # Rehash all declared payloads at the last pre-engine boundary.
    verify_sources(OVERLAY,built=True,final=True)
    from e3_private_auth import read_private
    env=environment(read_private(a.auth_env_file),a.model,a.snapshot_max_bytes)
    rc=phase('q18a_native15_declare.sh',S0/(QNAME+'-declare-chain.log'),env)
    print('DECLARE rc='+str(rc),flush=True)
    if rc:
        print('OWNER_CHAIN DONE rc='+str(rc)+' queue_started=false',flush=True)
        return rc
    require(sha(OVERLAY/'SHA256SUMS')==MANIFEST,'built-manifest-changed-before-queue')
    manifest(OVERLAY)
    env['REQUESTS']=str(S0/(QNAME+'-requests'))
    env['Q']=str(S0/QNAME)
    rc=phase('q18a_native15.draft.sh',S0/(QNAME+'.log'),env)
    print('QUEUE rc='+str(rc),flush=True)
    print('OWNER_CHAIN DONE rc='+str(rc)+' queue_started=true',flush=True)
    return rc


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        # Bounded location/message diagnostics, never key data or source/locals.
        print('OWNER_CHAIN INCONCLUSIVE '+json.dumps(diagnostic(exc),sort_keys=True),flush=True)
        raise SystemExit(2)
