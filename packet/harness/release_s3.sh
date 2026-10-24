#!/bin/bash
# Matched-phase release qualification of the S3 candidate (retained history QSA_RETAIN_HISTORY=1 PAGES=24 on
# overlay-v16, chunk 1024, PD alternation), adapted from runs/final.sh (Codex round 4: q5 ran one verify-release on a
# non-verifying engine -- bytes NOT_RUN by construction -- and none of the cancel/full-pool lifetime cases).
# The phases differ only in the two diagnostic switches, both set explicitly:
#   V  QSA_COW_VERIFY=1 QSA_RETAIN_VERIFY=1: verify-release.sh skip-bench (bytes = CoW invariants on all ranks) and the
#      retained-value gate over this container's whole log (cov_gate retain: active + verified pages on all 4 workers,
#      0 bad, no MISMATCH, never disabled)
#   P  both off, the configuration that would ship: inside-gen200-cancel, long-fullpool, verify-release.sh with the
#      benchmark at BENCH_MIN, and retained reads used on every worker (cov_gate retain-used)
# Each restart must pass and install exactly the phase's configuration (qlib installed) or the phase is skipped.
# release_combine.py text-s3 (permitted settings difference: the QSA_RETAIN_VERIFY=1 word) -> exit 0 PASS, 1, 2.
# usage: HARNESS_DIR=<dir> release_s3.sh OUTDIR BENCH_MIN    "STEP name verdict ..."
set -u
O=${1:?outdir}; BMIN=${2:?bench min}; H=${HARNESS_DIR:?}; mkdir -p $O
U=http://127.0.0.1:8000; M=qwen38-w8a8-tp4; P=/data/vllm/flash-next-tp4/prod
source $H/qlib.sh
: > $O/steps.txt
rstep() { step "$@"; local v; v=$(case $2 in (0) echo PASS;; (2) echo INCONCLUSIVE;; (*) echo FAIL;; esac); echo "$1 $v" >> $O/steps.txt; }
S3="QSA_RETAIN_HISTORY=1 QSA_RETAIN_PAGES=24"
phase() {  # name, QSA_COW_VERIFY (1|-), extra EXTRA_ENV words
  python3 /tmp/setcfg.py OVERLAY=$P/overlay-v16 VISION=- VISION_ARGS=- BREAKABLE_CG=- COMPILATION_CONFIG=- \
    FLASH_SCHED_STATS=200 PREFILL_CHUNK=1024 FLASH_PD_ALTERNATE=1 QSA_COW_VERIFY=$2 \
    "EXTRA_ENV=QSA_STAGE_TIMING=1 $S3${3:+ $3}" > /dev/null
  systemctl reset-failed vllm-flash; /usr/local/sbin/flash-reconfig.sh > $O/reconfig-$1.log 2>&1; local rc=$? why
  why=$(installed overlay-v16 1024 QSA_STAGE_TIMING=1 FLASH_PD_ALTERNATE=1 $S3 ${3:-} $([[ $2 == 1 ]] && echo QSA_COW_VERIFY=1))
  [[ -z $why ]] || rc=1
  rstep restart-$1 $rc "$why"
  return $rc
}

# ---- phase V: verify on
if phase verify 1 QSA_RETAIN_VERIFY=1; then
  HARNESS_DIR=$H bash $H/verify-release.sh $O/release-verify skip-bench > $O/release-verify.log 2>&1
  step release-check-verify-raw $? "$(tail -1 $O/release-verify.log)"   # judged gate by gate by release_combine.py
  grep -E "^(PASS|FAIL|INCONCLUSIVE|NOT_RUN) " $O/release-verify.log | cut -c1-220
  docker logs qwen-flash-prod 2>&1 | python3 $H/cov_gate.py retain 4 > $O/retain-verify.txt
  rstep retain-verify $? "$(cut -c1-400 $O/retain-verify.txt)"
fi

# ---- phase P: verify off, the shipping configuration
if phase async -; then
  TIMEOUT=1800 GEN=200 CANCEL_FRAC=0.3 MODE=inside timeout 5400 python3 $H/leak_partial_tail.py $U $M 10 4 7568 $O/inside-gen200.json > $O/inside-gen200.log 2>&1
  rstep inside-gen200-cancel $? "$(tail -1 $O/inside-gen200.log | cut -c1-300)"
  TIMEOUT=3600 GEN=6000 UNIQUE_PREFIX=1 BLOCK=246928 FACT=247300 FACT2=247900 MODE=inside timeout 7200 python3 $H/leak_partial_tail.py $U $M 10 1 248160 $O/long-fullpool.json > $O/long-fullpool.log 2>&1
  rstep long-fullpool $? "$(tail -1 $O/long-fullpool.log | cut -c1-300)"
  BENCH_MIN=$BMIN HARNESS_DIR=$H bash $H/verify-release.sh $O/release-async > $O/release-async.log 2>&1
  step release-check-async-raw $? "$(tail -1 $O/release-async.log)"
  grep -E "^(PASS|FAIL|INCONCLUSIVE|NOT_RUN) " $O/release-async.log | cut -c1-220
  docker logs qwen-flash-prod 2>&1 | python3 $H/cov_gate.py retain-used 4 > $O/retain-used.txt
  rstep retain-used $? "$(cut -c1-400 $O/retain-used.txt)"
fi
HARNESS_DIR=$H bash $H/manifest.sh $O/manifest-release.txt > /dev/null
RELEASE_DIAG_WORDS=QSA_RETAIN_VERIFY=1 python3 $H/release_combine.py $O/release-verify $O/release-async $O/steps.txt \
  $O/release-verdict.txt text-s3; rc=$?
echo "RELEASE-S3 DONE rc=$rc"
exit $rc
