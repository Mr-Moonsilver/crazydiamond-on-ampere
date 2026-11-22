#!/usr/bin/env bash
# CPU-only owner build. No GPU flags, network, engine control or cleanup.
# All outputs are new; a partial build is retained and needs a fresh package.
set -euo pipefail
image=${1:?pinned image required}
native=$(realpath "${2:?new generated native directory required}")
[[ $image == sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b ]]
for name in flash_s5_green.so flash_s5_replay_observer.so q17-helper-build.json; do
  [[ ! -e $native/$name && ! -L $native/$name ]]
done
for name in flash_s5_green.cu flash_s5_replay_observer.cpp flash_s5_graph_bytes.cpp; do
  [[ -f $native/$name && ! -L $native/$name ]]
done
name="q17-native-build-$(date -u +%Y%m%dT%H%M%SZ)-$$"
timeout --signal=TERM --kill-after=30s 660s docker run --name "$name" --network none \
  -v "$native:/native" --entrypoint timeout "$image" \
  --signal=TERM --kill-after=20s 600s /bin/bash -euc '
wheel=/usr/local/lib/python3.12/dist-packages/nvidia/cu13
expected=e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e
[[ $(sha256sum "$wheel/lib/libcupti.so.13" | cut -c1-64) == "$expected" ]]
stub=/usr/local/cuda-13.0/lib64/stubs
if [[ ! -f $stub/libcuda.so ]]; then
  stub=/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs
fi
[[ -f $stub/libcuda.so ]]
nvcc -O2 -std=c++17 -shared -Xcompiler=-fPIC -L"$stub" \
  --generate-code=arch=compute_86,code=sm_86 --cudart=shared -lcuda \
  /native/flash_s5_green.cu -o /native/flash_s5_green.so
nvcc -O2 -std=c++17 -shared -Xcompiler=-fPIC,-pthread -L"$stub" \
  -I"$wheel/include" -L"$wheel/lib" -Xlinker=-rpath -Xlinker="$wheel/lib" \
  /native/flash_s5_replay_observer.cpp /native/flash_s5_graph_bytes.cpp -l:libcupti.so.13 -lcuda \
  -o /native/flash_s5_replay_observer.so
python3 - <<"PY"
import hashlib, json, pathlib, re, subprocess
root=pathlib.Path("/native")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={name:sha(root/name) for name in ("flash_s5_green.cu", "flash_s5_green.so",
                                     "flash_s5_replay_observer.cpp", "flash_s5_graph_bytes.cpp", "flash_s5_replay_observer.so")}
needed={}
for name in ("flash_s5_green.so", "flash_s5_replay_observer.so"):
    output=subprocess.check_output(["readelf","-d",str(root/name)],text=True)
    needed[name]=re.findall(r"\(NEEDED\).*\[([^\]]+)\]",output)
assert "libcuda.so.1" in needed["flash_s5_green.so"]
assert "libcuda.so.1" in needed["flash_s5_replay_observer.so"]
assert "libcupti.so.13" in needed["flash_s5_replay_observer.so"]
assert not any("libcupti" in x and x!="libcupti.so.13" for xs in needed.values() for x in xs)
record={"schema":1,"image":"sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b",
        "files":files,"needed":needed,"declared_abi":{"green":1,"observer":1},
        "abi_basis":"source contract; loaded binary ABI rechecked at engine startup",
        "cupti_library_sha256":sha(pathlib.Path("/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13")),
        "nvcc":subprocess.check_output(["nvcc","--version"],text=True),
        "gpu_work":False}
with (root/"q17-helper-build.json").open("x") as f:json.dump(record,f,sort_keys=True,indent=2);f.write("\n")
PY
'
echo "Q17_NATIVE_BUILD PASS retained-container=$name"
