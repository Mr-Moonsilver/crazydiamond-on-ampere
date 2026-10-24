#!/usr/bin/env python3
"""Make an FP8 (e4m3fn) copy of the PLE n-gram table: ~95.4 GiB BF16 -> ~47.7 GiB, i.e. ~48 GiB of pinned host RAM
back in CT <ct-id> (operator decision 24 Oct: FP8 table instead of NVMe offload).

The engine selects its FP8 PLE path from the text config key ple_embedding_dtype = "float8_e4m3fn"
(Qwen4ExpPLEEmbeddingMethod.from_quant_config, checked before the AutoRound quantization config) and then requires
one global float32 scale, `<prefix>.ngram_embedding.weight_scale` (PerTensorScaleParameter, shape [1]); rows are
dequantized as fp8.to(bf16) * scale after the lookup.

What this does (SRC is never modified; DST must not exist):
  * DST/<every other file>: hard links to SRC (same filesystem; symlinks would not resolve inside the container's
    /model bind mount), except config.json (+ ple_embedding_dtype) and model.safetensors.index.json (+ weight_scale).
  * DST/<the file holding the 128 shards>: rewritten. Pass 1 streams every shard (ROWS rows at a time) for the
    global max |w| -> scale = amax / 448. Pass 2 writes q = (w / scale).to(float8_e4m3fn) per chunk, every other tensor
    of that file byte for byte, and the scale tensor. Error statistics are collected on a seeded 1 % row sample:
    relative RMS error of the dequantized rows, the share of nonzero values flushed to zero, max abs error.
  * DST/PLE_FP8.json: source/destination checksums of the rewritten file's tensors, scale, error statistics.
Peak memory is a few chunks (ROWS x 160 values); the file is read sequentially twice.
usage: ple_fp8_convert.py SRC_DIR DST_DIR   (env ROWS=262144)   run inside the vLLM image (torch, CPU only)
"""
import hashlib, json, math, os, struct, sys, time
import torch

SRC, DST = sys.argv[1], sys.argv[2]
ROWS = int(os.environ.get("ROWS", 262144))
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448
SHARD = "ple.ple_embedding.ngram_embedding.shard_"
if os.path.exists(DST) or os.path.exists(DST + ".tmp"):
    raise SystemExit(f"{DST} or {DST}.tmp exists -- refusing to touch it")
FINAL, DST = DST, DST + ".tmp"  # everything is written under DST.tmp and renamed at the end: no half-made DST


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


index = json.load(open(f"{SRC}/model.safetensors.index.json"))
wm = index["weight_map"]
shards = sorted((k for k in wm if SHARD in k and k.endswith(".weight")), key=lambda k: int(k.split(SHARD)[1][:-7]))
files = {wm[k] for k in shards}
if len(files) != 1:
    raise SystemExit(f"shards spread over {sorted(files)}; this tool handles one file")
table_file = files.pop()
prefix = shards[0].split(".shard_")[0]  # ...ple.ple_embedding.ngram_embedding
scale_name = f"{prefix}.weight_scale"
if scale_name in wm:
    raise SystemExit(f"{scale_name} already in the checkpoint")
hlen, hdr = header(f"{SRC}/{table_file}")
base = 8 + hlen
for k in shards:
    if hdr[k]["dtype"] != "BF16" or len(hdr[k]["shape"]) != 2:
        raise SystemExit(f"{k}: {hdr[k]['dtype']} {hdr[k]['shape']} (expected 2-D BF16)")
print(f"{len(shards)} shards in {table_file}, prefix {prefix}", flush=True)


def chunks(f, name):
    """Yield (row0, float32 tensor [rows, dim]) for one BF16 tensor, ROWS rows at a time."""
    rows, dim = hdr[name]["shape"]
    o0 = base + hdr[name]["data_offsets"][0]
    for r in range(0, rows, ROWS):
        n = min(ROWS, rows - r)
        f.seek(o0 + r * dim * 2)
        buf = bytearray(f.read(n * dim * 2))
        yield r, torch.frombuffer(buf, dtype=torch.bfloat16).view(n, dim).float()


# ---- pass 1: global amax
t0 = time.time()
amax = 0.0
with open(f"{SRC}/{table_file}", "rb") as f:
    for i, k in enumerate(shards):
        for _, x in chunks(f, k):
            amax = max(amax, float(x.abs().max()))
        if i % 16 == 15:
            print(f"pass 1: {i + 1}/{len(shards)} shards, amax {amax:.6g}, {time.time() - t0:.0f} s", flush=True)
if not math.isfinite(amax) or amax <= 0:
    raise SystemExit(f"bad amax {amax}")
# the engine dequantizes with weight_scale.to(bf16) (Qwen4ExpPLEFp8EmbeddingMethod.dequantize): use a scale that bf16
# represents exactly, rounded up (so |w| / scale <= 448 still holds), or every row would carry the rounding error
scale = amax / FP8_MAX
sb = torch.tensor([scale], dtype=torch.float32).to(torch.bfloat16).float().item()
if sb < scale:
    sb = torch.nextafter(torch.tensor([sb], dtype=torch.bfloat16), torch.tensor([math.inf], dtype=torch.bfloat16)).float().item()
scale = sb
assert torch.tensor([scale]).to(torch.bfloat16).float().item() == scale and scale * FP8_MAX >= amax
print(f"amax {amax:.6g} -> scale {scale:.6g}", flush=True)

# ---- new header: shards become F8_E4M3 (half the bytes), the scale is appended, everything else keeps its bytes
order = sorted((k for k in hdr if k != "__metadata__"), key=lambda k: hdr[k]["data_offsets"][0])
new_hdr, off = {}, 0
if "__metadata__" in hdr:
    new_hdr["__metadata__"] = hdr["__metadata__"]
for k in order:
    a, b = hdr[k]["data_offsets"]
    size = (b - a) // 2 if k in shards else b - a
    new_hdr[k] = {"dtype": "F8_E4M3" if k in shards else hdr[k]["dtype"], "shape": hdr[k]["shape"],
                  "data_offsets": [off, off + size]}
    off += size
new_hdr[scale_name] = {"dtype": "F32", "shape": [1], "data_offsets": [off, off + 4]}
hbytes = json.dumps(new_hdr, separators=(",", ":")).encode()
hbytes += b" " * (-len(hbytes) % 8)

# ---- DST: hard links, edited config/index, rewritten table file
os.makedirs(DST)
for name in os.listdir(SRC):
    if name not in (table_file, "config.json", "model.safetensors.index.json") and os.path.isfile(f"{SRC}/{name}"):
        os.link(f"{SRC}/{name}", f"{DST}/{name}")
cfg = json.load(open(f"{SRC}/config.json"))
(cfg["text_config"] if "text_config" in cfg else cfg)["ple_embedding_dtype"] = "float8_e4m3fn"
json.dump(cfg, open(f"{DST}/config.json", "w"), indent=2)
index["weight_map"][scale_name] = table_file
if "total_size" in index.get("metadata", {}):
    index["metadata"]["total_size"] += off + 4 - sum(hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0] for k in order)
json.dump(index, open(f"{DST}/model.safetensors.index.json", "w"), indent=2)

g = torch.Generator().manual_seed(0)
st = {"n": 0, "sq_err": 0.0, "sq_ref": 0.0, "nonzero": 0, "flushed": 0, "max_abs_err": 0.0}
digests = {}
t1 = time.time()
with open(f"{SRC}/{table_file}", "rb") as f, open(f"{DST}/{table_file}.part", "wb") as out:
    out.write(struct.pack("<Q", len(hbytes)))
    out.write(hbytes)
    for i, k in enumerate(order):
        a, b = hdr[k]["data_offsets"]
        h = hashlib.sha256()
        if k not in shards:
            f.seek(base + a)
            left = b - a
            while left:
                blk = f.read(min(left, 64 << 20)); out.write(blk); h.update(blk); left -= len(blk)
        else:
            for _, x in chunks(f, k):
                q = (x / scale).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                blk = q.view(torch.uint8).numpy().tobytes()
                out.write(blk); h.update(blk)
                sel = torch.rand(x.shape[0], generator=g) < 0.01
                if sel.any():
                    xs, ds = x[sel], q[sel].float() * scale
                    st["n"] += xs.numel(); st["sq_err"] += float(((ds - xs) ** 2).sum())
                    st["sq_ref"] += float((xs ** 2).sum()); nz = xs != 0
                    st["nonzero"] += int(nz.sum()); st["flushed"] += int((nz & (ds == 0)).sum())
                    st["max_abs_err"] = max(st["max_abs_err"], float((ds - xs).abs().max()))
        digests[k] = h.hexdigest()
        if k in shards and int(k.split(SHARD)[1][:-7]) % 16 == 15:
            print(f"pass 2: {k.split('.')[-2]}, {time.time() - t1:.0f} s", flush=True)
    out.write(struct.pack("<f", scale))
os.rename(f"{DST}/{table_file}.part", f"{DST}/{table_file}")
stats = {"rel_rms_err": math.sqrt(st["sq_err"] / max(st["sq_ref"], 1e-30)),
         "flushed_to_zero_share": st["flushed"] / max(st["nonzero"], 1), "max_abs_err": st["max_abs_err"],
         "sampled_values": st["n"]}
rep = {"src": SRC, "dst": DST, "table_file": table_file, "prefix": prefix, "shards": len(shards), "amax": amax,
       "scale": scale, "bytes_src": os.path.getsize(f"{SRC}/{table_file}"), "bytes_dst": os.path.getsize(f"{DST}/{table_file}"),
       "error_on_1pct_rows": stats, "tensor_sha256": digests, "seconds": round(time.time() - t0)}
rep["converter_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
rep["dst"] = FINAL
json.dump(rep, open(f"{DST}/PLE_FP8.json", "w"), indent=1)
os.rename(DST, FINAL)
print("ple_fp8_convert done", json.dumps({k: rep[k] for k in ("amax", "scale", "bytes_src", "bytes_dst", "seconds")}),
      json.dumps(stats), flush=True)
