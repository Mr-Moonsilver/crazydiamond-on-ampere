#!/usr/bin/env python3
"""One memory ledger for the running engine (goal item 8): device and host side, read-only, no inference.

Device, per GPU / rank: total and used (nvidia-smi), the KV pool budget (kv_cache_memory_bytes), model weights and
CUDA-graph memory (engine log), the PyTorch allocator's reserved / allocated / peak (last QSA stage-timing line per
rank, if QSA_STAGE_TIMING=1), and the configured extras: QSA row cache (QSA_ROWCACHE_SLOTS), retained history
(QSA_RETAIN_HISTORY / QSA_RETAIN_PAGES x 6 QSA layers x 4.0 MiB per page -- 24 Oct: 40 pages = 0.93 GiB per rank),
staging arena, vision tower (0.42 GiB per stage-0 GPU when vision is on). "Unexplained" = used - the explained parts.
Host: total / used / available RAM, shared memory (the pinned host K/V pool lives in shm), /dev/shm.
Only allowlisted engine settings are printed (never the full environment).
usage: memory_ledger.py [CONTAINER] [OUT.json]   (default qwen-flash-prod)
"""
import json, re, subprocess, sys

C = sys.argv[1] if len(sys.argv) > 1 else "qwen-flash-prod"
OUT = sys.argv[2] if len(sys.argv) > 2 else None
GiB = 2**30
ALLOW = re.compile(r"^(QSA_[A-Z_]+|FLASH_[A-Z_]+|VLLM_QSA_[A-Z_]+|CUDA_VISIBLE_DEVICES|PREFILL_CHUNK)=")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


env = {}
for kv in json.loads(sh(f"docker inspect {C} --format '{{{{json .Config.Env}}}}'") or "[]"):
    if ALLOW.match(kv):
        k, v = kv.split("=", 1)
        env[k] = v
log = sh(f"docker logs {C} 2>&1")
args = re.search(r"non-default args: (\{.*\})", log)
argtxt = args.group(1) if args else ""
kv_bytes = re.search(r"'kv_cache_memory_bytes': (\d+)", argtxt)
gmu = re.search(r"'gpu_memory_utilization': ([0-9.]+)", argtxt)
vision = "language_model_only': True" not in argtxt

per_rank = {}
for m in re.finditer(r"(Worker_PP\d_TP\d)[^\n]*?Model loading took ([0-9.]+) GiB", log):
    per_rank.setdefault(m.group(1), {})["weights_gib"] = float(m.group(2))
for m in re.finditer(r"(Worker_PP\d_TP\d)[^\n]*?Graph capturing finished in [0-9.]+ secs?, took ([0-9.]+) GiB", log):
    per_rank.setdefault(m.group(1), {})["graphs_gib"] = float(m.group(2))
for m in re.finditer(r"(Worker_PP\d_TP\d)[^\n]*?alloc retries (\d+) ooms (\d+) reserved ([0-9.]+)G allocated ([0-9.]+)G "
                     r"peak ([0-9.]+)G", log):
    per_rank.setdefault(m.group(1), {}).update(alloc_retries=int(m.group(2)), ooms=int(m.group(3)),
                                               reserved_gib=float(m.group(4)), allocated_gib=float(m.group(5)),
                                               peak_gib=float(m.group(6)))
for m in re.finditer(r"(Worker_PP\d_TP\d)[^\n]*?QSA row cache [^\n]*?, ([0-9.]+) MiB VRAM", log):
    r = per_rank.setdefault(m.group(1), {})
    r["rowcache_gib"] = round(r.get("rowcache_gib", 0) + float(m.group(2)) / 1024, 3)
for m in re.finditer(r"(Worker_PP\d_TP\d)[^\n]*?Initial free memory ([0-9.]+) GiB", log):
    per_rank.setdefault(m.group(1), {})["initial_free_gib"] = float(m.group(2))
pool = re.search(r"GPU KV cache size: ([0-9,]+) tokens", log)

gpus = []
for line in sh("nvidia-smi --query-gpu=index,pci.bus_id,memory.total,memory.used --format=csv,noheader,nounits").splitlines():
    i, bus, tot, used = [x.strip() for x in line.split(",")]
    gpus.append({"index": int(i), "bus": bus, "total_gib": round(int(tot) / 1024, 2), "used_gib": round(int(used) / 1024, 2)})

retain = (int(env.get("QSA_RETAIN_PAGES", "40")) * 6 * 4.0 / 1024) if env.get("QSA_RETAIN_HISTORY") == "1" else 0.0
extras = {"kv_pool_gib": round(int(kv_bytes.group(1)) / GiB, 2) if kv_bytes else None,
          "retained_history_gib_per_rank": round(retain, 2),
          "vision_tower_gib_per_stage0_gpu": 0.42 if vision else 0.0,
          "rowcache_slots": env.get("QSA_ROWCACHE_SLOTS")}

mem = dict(re.findall(r"^(\w+):\s+(\d+) kB", open("/proc/meminfo").read(), re.M))
host = {k: round(int(mem[v]) / 2**20, 1) for k, v in (("total_gib", "MemTotal"), ("available_gib", "MemAvailable"),
                                                      ("shmem_gib", "Shmem"), ("hugepages_total", "HugePages_Total"))
        if v in mem}
host["dev_shm"] = " ".join(sh("df -h /dev/shm | tail -1").split()[1:5])  # size used avail use%
# Shmem is by design ~220 GiB in CT <ct-id> (24 Oct): the PLE table (~95.9 GiB) + the pinned host K/V pools (4 x <= 33 GiB)
host["host_kv_cap_gib_total"] = 4 * int(env.get("VLLM_QSA_KV_OFFLOAD_MAX_GIB", "0") or 0)

# GPU_ORDER maps rank -> physical GPU: ranks PP0_TP0, PP0_TP1, PP1_TP0, PP1_TP1 run on CUDA_VISIBLE_DEVICES[0..3]
order = [int(x) for x in env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")]
for k, r in per_rank.items():
    pp, tp = int(k[9]), int(k[13])
    g = next((x for x in gpus if x["index"] == order[pp * 2 + tp]), None)
    if g is None:
        continue
    r["gpu"] = g["index"]
    known = (r.get("weights_gib", 0) + (extras["kv_pool_gib"] or 0) + r.get("rowcache_gib", 0) + r.get("graphs_gib", 0)
             + extras["retained_history_gib_per_rank"] + (extras["vision_tower_gib_per_stage0_gpu"] if pp == 0 else 0))
    r["explained_gib"] = round(known, 2)
    r["unexplained_gib"] = round(g["used_gib"] - known, 2)  # activations, allocator cache, staging arena, NCCL, context
ledger = {"container": C, "settings": env, "gpu_memory_utilization": float(gmu.group(1)) if gmu else None,
          "kv_pool_tokens": int(pool.group(1).replace(",", "")) if pool else None, "extras": extras,
          "per_rank": per_rank, "gpus": gpus, "host": host}
print(json.dumps(ledger, indent=1))
if OUT:
    json.dump(ledger, open(OUT, "w"), indent=1)
