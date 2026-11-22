# q18a real E3 overlap: public settings shared by declaration and queue.
# The existing API key lives only in the retained owner-only env file; it is
# never placed in EXTRA_ENV, argv or the declaration.
#
# The launcher pin is deliberately NOT a setting here.  q18a reads the trusted
# bundled identity_declare.S5_LAUNCHER_SHA and refuses any arbitrary
# S5_LAUNCHER_SHA environment pin that differs from it.
QNAME=${QNAME:-q18a-parity-n3s24q2024p-j2-cxfx42-b1}
S5_OVERLAY=${S5_OVERLAY:-}
S5_GREEN_LIB_SHA256=${S5_GREEN_LIB_SHA256:-}
S5_REPLAY_LIB_SHA256=${S5_REPLAY_LIB_SHA256:-}
S5_AUTH_ENV_FILE=${S5_AUTH_ENV_FILE:-}
# Public hub path only. The box never reads or transfers this file; the owner
# supplies the matching fresh key at arm time and retains cleanup ownership.
CANARY_CLIENT_AUTH_FILE='/home/<service-user>/.flashnext/canary-client-auth.env'
S5_REQUESTED_DECODE_SMS=${S5_REQUESTED_DECODE_SMS:-24}
Q18A_TARGET_LABEL=${Q18A_TARGET_LABEL:-}
# Client-side read bound only. The frozen W1/W2 request manifests and TTFT
# calculation remain unchanged; this permits a slow first token to be graded.
Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=1200
export Q18A_FIRST_ADVANCE_READ_TIMEOUT_S
Q18A_READINESS=${Q18A_READINESS:-${S5_OVERLAY:+$S5_OVERLAY/e3/native-overlap-readiness.json}}
S5_LAUNCHER_NAME=${S5_LAUNCHER_NAME:-run-flash-s5-test.sh}
[[ -n $S5_OVERLAY && $S5_AUTH_ENV_FILE == /* ]] || {
  echo 'overlay/auth file path required' >&2; return 2; }
[[ $S5_GREEN_LIB_SHA256 =~ ^[0-9a-f]{64}$ && $S5_REPLAY_LIB_SHA256 =~ ^[0-9a-f]{64}$ ]] || {
  echo 'compiled library pins required' >&2; return 2; }
[[ $S5_REQUESTED_DECODE_SMS == 24 ]] || {
  echo 'decode split remains declared 24' >&2; return 2; }
[[ $Q18A_TARGET_LABEL == s5-overlap-80-80 ]] || {
  echo 'operator target label required: s5-overlap-80-80' >&2; return 2; }
[[ -n $Q18A_READINESS ]] || {
  echo 'reviewed native-overlap readiness receipt required' >&2; return 2; }
[[ ${FLASH_NATIVE30_SYNC:-none} =~ ^(none|native|decode|both)$ ]] || { echo 'invalid native30 sync mode' >&2; return 2; }
# Native39 trunk: the API server binds 127.0.0.1.  The found production EXTRA_ARGS (checked at
# preflight) is carried forward with --host 127.0.0.1 appended; the launcher appends EXTRA_ARGS after
# its own --host 0.0.0.0 and the last --host wins (q18b-w1ab and q18b-w1ab2 on this box).
TRUNK_PROD_EXTRA_ARGS="--prefix-match-unit 176 --enable-prompt-tokens-details"
# Native41: FLASH_S5_AUTO_ARM=1 below, the only setting added to native40.
# compose-5: connections 32 and the two PLE launch switches below, the only settings added to a1-trunk's, for the whole run; each arm selects its levers by typed control.
# a1-trunk: the four FLASH_S5_APPEND_YIELD_* launch bounds below (installed, not selected), for the whole run, the only setting added to the parent queue; each arm selects the yield by typed control.
# compose5: FLASH_S5_INWAIT=1 below (installed, not selected), for the whole run, the only setting added to compose5; each arm selects it by typed control.
# compose5: FLASH_ATTN_META_GDN_BATCH=1 below, beside FLASH_S5_LAZY_RECV=1 FLASH_S5_DECODE_MERGE=1, for the whole run (installed, not selected), the only setting added to
# stage1-step; each arm selects each lever by typed control.
# stage1-step: FLASH_S5_LAZY_RECV=1 FLASH_S5_DECODE_MERGE=1 below (installed, not selected), for the whole run, the only settings
# added to alpha-next-c1; each arm selects its levers by the typed native_stage_local control.
# alpha-next-c1: FLASH_S5_STAGE_OVERLAP=1 (C1), FLASH_S5_STAGE_OVERLAP_DEFER_BIND=1 (the deferred PP bind installed)
# and FLASH_S5_STAGE_OVERLAP_OBSERVE=1 (the overlap observation lines) below, for the whole run, the only settings
# added to alpha-next; each arm selects C1 off, on or defer (with C0 local) and its defer_bind by typed control.
# alpha-next: FLASH_S5_STAGE_LOCAL=1 (C0) and FLASH_S5_HANDLER_ANATOMY=1 (stage-1 anatomy lines) below, the only
# settings added to Step B; each arm selects C0 on or off by typed control. FLASH_S5_UNION_RETIRE stays unset and no
# FLASH_S5_PROBE_* key is set.
# Step B: FLASH_S5_LAYER_GROUPS=1 below, the only setting added to Step A; each arm selects its size by typed control.
# Step A: FLASH_S5_QSA_FUSED=1 below, the only setting added to alpha-perf; each arm selects fused or stepped by typed control.
# Alpha-perf: FLASH_S5_NATIVE_QSA_POLICY=dense below, the dense policy as q18a-native43-lifecycle-f configures it (no trace,
# no stop frontier), the only setting added to held-f.
# Held-f: FLASH_S5_NATIVE_PROGRESS=burst below, the only setting added to held-e (receipts on
# auto-owned native chunks; each arm selects current or burst by typed control).
# Native39 speed engine: FLASH_S5_REPLAY_OBSERVER=0 and FLASH_S5_ENGINE_OFFERS=0 below (q18b-w1ab2
# measured ~5 % decode for the two together); row4 placement needs both and is declared NOT_RUN.
export TRUNK_PROD_EXTRA_ARGS
S5_SETTINGS=(
 "OVERLAY=$S5_OVERLAY" CORE_OVERLAY=1 PP_LAYER_PARTITION=24,24
 MODEL_DIR=- VISION=- VISION_ARGS=- BREAKABLE_CG=- COMPILATION_CONFIG=-
 QSA_COW_VERIFY=- SPARSE_PREFILL=- S5=1 FLASH_SCHED_STATS=200
 PREFILL_CHUNK=1024 FLASH_PD_ALTERNATE=1
 QSA_ROWCACHE_SLOTS=6144
 "FLASH_S5_AUTH_ENV_FILE=$S5_AUTH_ENV_FILE"
 "EXTRA_ARGS=$TRUNK_PROD_EXTRA_ARGS --host 127.0.0.1"
 "EXTRA_ENV=FLASH_S5_JUNCTION_YIELD=1 FLASH_S5_BULK_SHORTEST=1 FLASH_S5_OVERLAP=1 FLASH_S5_DECODE_ONLY_PROBE=0 FLASH_S5_DECODE_SMS=24 FLASH_S5_GREEN_LIB=/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3/native/flash_s5_green.so FLASH_S5_GREEN_LIB_SHA256=$S5_GREEN_LIB_SHA256 FLASH_S5_REPLAY_OBSERVER=0 FLASH_S5_ENGINE_OFFERS=0 FLASH_S5_AUTO_ARM=1 FLASH_S5_NATIVE_PROGRESS=burst FLASH_S5_NATIVE_QSA_POLICY=dense FLASH_S5_QSA_FUSED=1 FLASH_S5_LAYER_GROUPS=1 FLASH_S5_STAGE_LOCAL=1 FLASH_S5_HANDLER_ANATOMY=1 FLASH_S5_STAGE_OVERLAP=1 FLASH_S5_STAGE_OVERLAP_DEFER_BIND=1 FLASH_S5_STAGE_OVERLAP_OBSERVE=1 FLASH_S5_LAZY_RECV=1 FLASH_S5_DECODE_MERGE=1 FLASH_ATTN_META_GDN_BATCH=1 FLASH_S5_INWAIT=1 FLASH_S5_APPEND_YIELD_MAX_NEW=8192 FLASH_S5_APPEND_YIELD_PER_BULK=8 FLASH_S5_APPEND_YIELD_HANDBACK_STEPS=16 FLASH_S5_APPEND_YIELD_REENTRY_RETRIES=4 CUDA_DEVICE_MAX_CONNECTIONS=32 FLASH_S5_PLE_PRIVATE=1 FLASH_S5_PLE_WITNESS=1 FLASH_S5_CHUNK_TOKENS=2048 FLASH_S5_PREPARE_AHEAD=1 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 FLASH_S5_O3=1 FLASH_S5_AHEAD_NOSYNC=1 FLASH_S5_FUSED_PHASE_EVERY=16 FLASH_S5_NEXT_LAYER_PREFETCH=1 FLASH_S5_SMALL_PIECE_BATCH=0 FLASH_S5_YIELD_REENTRY_SETTLE=0 FLASH_S5_YIELD_NO_APPEND_OWNER=1 FLASH_S5_YIELD_WINDOW=1 FLASH_S5_INGRESS_INCREMENTAL=0 FLASH_S5_APPEND_NO_BLOCK_STOP=1 FLASH_S5_DECODE_GROUPS=chunk FLASH_S5_DECODE_LIVENESS=0 FLASH_S5_CONTROL_INWAIT=0 FLASH_S5_CONTROL_O3_INWAIT=0 FLASH_S5_DECODE_SAMPLE_FAST=0 FLASH_S5_DECODE_ANATOMY=1 FLASH_S5_DECODE_TIMELINE=1 FLASH_S5_DECODE_OVERLAP_SMS=42 FLASH_S5_YIELD_WINDOW_FIX=1 FLASH_S5_REPLAY_LIB=/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3/native/flash_s5_replay_observer.so FLASH_S5_REPLAY_LIB_SHA256=$S5_REPLAY_LIB_SHA256 Q18A_TARGET_LABEL=$Q18A_TARGET_LABEL FLASH_NATIVE30_LOCALIZE=0 FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0 FLASH_NATIVE30_SYNC=${FLASH_NATIVE30_SYNC:-none} PYTHONPATH=/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3/native30_plugins"
)

# Full active-model hashing took170.5s on native28; keep >2x that measured cost.
Q18A_IDENTITY_TIMEOUT_S=360
export Q18A_IDENTITY_TIMEOUT_S
