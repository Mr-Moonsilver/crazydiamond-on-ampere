#!/usr/bin/env python3
"""Offline test of ple_fp8_convert.py on a tiny fake checkpoint (needs torch, numpy, safetensors; CPU only).

The fake SRC mirrors the real layout: a text_config config.json, an index, one file holding 4 BF16 PLE shards
(uneven last shard, one outlier row) next to two unrelated tensors, plus a second weight file and a side file.
Checks: SRC unchanged (bytes); DST refuses to exist twice; unchanged files are hard links; config has
ple_embedding_dtype in text_config; the index maps the scale to the table file; safetensors.safe_open (the loader's
reader) opens the rewritten file, shards are float8_e4m3fn with the right shapes, the other tensors are bit-identical,
weight_scale is [1] float32 = amax / 448; dequantized shards match the source within e4m3's relative step;
PLE_FP8.json digests match the written tensors. Exit 0 when every check holds.
usage: test_ple_fp8_convert.py [PYTHON]   (PYTHON runs the converter; default: this interpreter)
"""
import atexit, hashlib, json, os, shutil, subprocess, sys, tempfile
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.argv[1] if len(sys.argv) > 1 else sys.executable
P = "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_"
fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


tmp = tempfile.mkdtemp(prefix="plefp8-", dir=os.environ.get("TMPDIR"))
atexit.register(shutil.rmtree, tmp, True)
src, dst = f"{tmp}/src", f"{tmp}/dst"
os.makedirs(src)
torch.manual_seed(1)
shards = {f"{P}{i}.weight": (torch.randn(n, 160) * 0.02).to(torch.bfloat16) for i, n in enumerate((300, 300, 300, 77))}
shards[f"{P}1.weight"][5, 7] = 3.3  # outlier: sets amax (3.3 / 448 is not a bf16 value)
other = {"model.language_model.layers.2.ple.kv_proj.weight": torch.randn(64, 32).to(torch.bfloat16),
         "model.language_model.layers.2.ple.ple_embedding.layer_multipliers": torch.arange(10, dtype=torch.int64)}
save_file({**shards, **other}, f"{src}/model-00016-of-00017.safetensors")
save_file({"model.language_model.embed_tokens.weight": torch.randn(8, 4).to(torch.bfloat16)},
          f"{src}/model-00001-of-00017.safetensors")
json.dump({"model_type": "x", "text_config": {"hidden_size": 2560}, "quantization_config": {"q": 1}},
          open(f"{src}/config.json", "w"))
wm = {k: "model-00016-of-00017.safetensors" for k in {**shards, **other}}
wm["model.language_model.embed_tokens.weight"] = "model-00001-of-00017.safetensors"
json.dump({"metadata": {"total_size": 1000}, "weight_map": wm}, open(f"{src}/model.safetensors.index.json", "w"))
open(f"{src}/generation_config.json", "w").write("{}")
before = {n: digest(f"{src}/{n}") for n in os.listdir(src)}

r = subprocess.run([PY, os.path.join(HERE, "ple_fp8_convert.py"), src, dst], capture_output=True, text=True,
                   env=dict(os.environ, ROWS="128"))
check("converter exits 0", r.returncode == 0, r.stderr[-800:])
if r.returncode == 0:
    check("SRC unchanged", {n: digest(f"{src}/{n}") for n in os.listdir(src)} == before)
    r2 = subprocess.run([PY, os.path.join(HERE, "ple_fp8_convert.py"), src, dst], capture_output=True, text=True)
    check("refuses an existing DST", r2.returncode != 0 and "exists" in (r2.stdout + r2.stderr))
    for n in ("model-00001-of-00017.safetensors", "generation_config.json"):
        check(f"hard link {n}", os.stat(f"{src}/{n}").st_ino == os.stat(f"{dst}/{n}").st_ino)
    cfg = json.load(open(f"{dst}/config.json"))
    check("text_config.ple_embedding_dtype", cfg["text_config"].get("ple_embedding_dtype") == "float8_e4m3fn"
          and "ple_embedding_dtype" not in cfg)
    idx = json.load(open(f"{dst}/model.safetensors.index.json"))["weight_map"]
    scale_name = f"{P[:-len('.shard_')]}.weight_scale"
    meta = json.load(open(f"{dst}/model.safetensors.index.json"))["metadata"]
    shard_bytes = sum(t.numel() * 2 for t in shards.values())
    check("index total_size: shards halved, + 4 scale bytes", meta["total_size"] == 1000 - shard_bytes // 2 + 4, meta)
    check("no DST.tmp left", not os.path.exists(dst + ".tmp"))
    check("index maps the scale", idx.get(scale_name) == "model-00016-of-00017.safetensors")
    amax = max(float(t.float().abs().max()) for t in shards.values())
    rep = json.load(open(f"{dst}/PLE_FP8.json"))
    with safe_open(f"{dst}/model-00016-of-00017.safetensors", "pt") as f:
        keys = set(f.keys())
        check("tensor set", keys == set(shards) | set(other) | {scale_name}, sorted(keys))
        sc = f.get_tensor(scale_name)
        v = float(sc[0])
        check("scale [1] float32, bf16-exact, amax/448 rounded up by < 1 bf16 step",
              sc.dtype == torch.float32 and tuple(sc.shape) == (1,) and float(sc.to(torch.bfloat16)[0]) == v
              and amax / 448 <= v < amax / 448 * (1 + 2 ** -7), (sc.dtype, tuple(sc.shape), v, amax / 448))
        for k, t in other.items():
            check(f"unchanged {k.split('.')[-1]}", torch.equal(f.get_tensor(k), t))
        worst = 0.0
        for k, t in shards.items():
            q = f.get_tensor(k)
            check(f"fp8 {k.split('.')[-2]}", q.dtype == torch.float8_e4m3fn and q.shape == t.shape)
            d = q.float() * float(sc[0])
            ref = t.float()
            big = ref.abs() > 64 * float(sc[0]) * 2 ** -6  # normal range of e4m3 (min normal 2^-6) with margin
            if big.any():
                worst = max(worst, float(((d - ref).abs() / ref.abs())[big].max()))
            check(f"digest {k.split('.')[-2]}", hashlib.sha256(q.view(torch.uint8).numpy().tobytes()).hexdigest()
                  == rep["tensor_sha256"][k])
        check("relative error within e4m3 step (2^-4) on normal values", worst <= 2 ** -4 + 1e-6, worst)
print(f"test_ple_fp8_convert: {'PASS' if not fails else 'FAIL ' + str(fails)}  ({tmp})")
sys.exit(1 if fails else 0)
