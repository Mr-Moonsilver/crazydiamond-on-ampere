# Queue engine identity

`identity` is read-only. It does not stop, start, reconfigure or query inference.
`identity_probe.py` runs via `docker exec` on CPU and reads the mounted files.
Existing queues and `ident`/`installed` retain their current behavior until a
queue explicitly adopts these helpers.

## Queue usage

Prepare `requested.json` **before changing the engine**, from the intended image,
launcher, overlay hashes and settings. It is an independent declaration, not an
automatic copy of whichever engine happens to be running. Pin it with the queue's
other inputs. Then:

```bash
source "$H/qlib.sh"
identity "$Q/before.json" "$Q/requested.json"
rc=$?; step identity-before "$rc"
if [[ $rc == 0 ]]; then
  # The queue's measurement goes here; retain its own exit code independently.
  identity "$Q/after.json" "$Q/requested.json"
  rc=$?; step identity-after "$rc"
  if [[ $rc == 0 ]]; then
    identity_check "$Q/before.json" "$Q/requested.json" "$Q/after.json"
    step identity-stable "$?"
  fi
fi
```

Capture: `identity OUT REQUEST [--container NAME] [--defaults PATH] [--timeout SEC]
[--max-bytes N] [--max-files N]`. Defaults are `qwen-flash-prod`,
`/etc/default/vllm-flash`, 300 seconds per Docker operation, 512 GiB of file reads,
100000 files/directory entries. A budget error is INCONCLUSIVE; no partial hash is
accepted. Capture fails on an existing output path rather than overwriting it.
Artifacts are mode 0600. Raw Docker output/defaults never go to disk or stdout.

`identity_check BEFORE REQUEST` rechecks the before manifest offline.
Adding AFTER also requires distinct, correctly ordered captures of the same
container ID, start time, PID and restart count, with identical identity and
redacted defaults snapshot. Capture itself inspects the container before and
after hashing, uses the immutable ID for `exec`, and rejects changes during the
capture. Routine healthcheck log updates are ignored. Both endpoints must match
the requested manifest; an unchanged wrong engine fails.

Exit codes: **0 PASS, 1 identity mismatch, 2 incomplete/unavailable evidence**.
Missing Docker/image/file, unsupported special file, hash budget/timeout, unstable
content, restart during capture, malformed artifact or incomplete request all
fail closed. The comparison ignores any saved `check.verdict` and regrades fields.
Diagnostic output identifies sections that differ without echoing values.

## Request schema

Top level is exactly `{"schema":"flash-engine-request/v1","identity":{...}}`.
`identity` has these five mandatory fields, compared exactly:

| Field | Content |
|---|---|
| `image` | `{"id":"sha256:<64 hex>","repo_digests":["registry/name@sha256:<64 hex>"]}`. Empty RepoDigests is allowed for locally built images; the immutable image ID is always required. |
| `command` | `{"path":"vllm","args":[...]}` from the intended executable and complete ordered argument vector, with redaction as below. No substring flag matching. |
| `env` | Complete expected Docker `Config.Env` as a map, including image defaults. Extra, missing or changed keys fail. Duplicate Docker environment keys are rejected as ambiguous. |
| `defaults` | Complete expected key/value map for the EnvironmentFile, after redaction/normalization. This is checked separately from the effective container settings: matching the file alone is insufficient. |
| `mounts` | Complete mount list sorted by container destination; no unexpected/missing mounts allowed. Each has the fields below. |

Every mount has exactly `type`, `source`, `destination`, `rw`, `propagation`,
`content`, `watch`. Type is `bind`, `volume` or `tmpfs`. Source and destination are
exact paths, and rw is a Boolean. Docker's missing propagation value is `""`.

For a regular file, `content` is
`{"kind":"file","bytes":123,"sha256":"<64 hex>"}` and `watch` is `[]`.
The hash is calculated from **the destination inside the container**, not the
host source. This catches an old bind-mounted inode after host-path replacement,
wrong destination, wrong overlay content, and extra runner/core/vision mounts.

For a directory, `content` is:

```json
{
  "kind": "directory",
  "entries": [
    {"path": "code.py", "kind": "file", "bytes": 123, "sha256": "<64 hex>"}
  ],
  "exclusions": [{"path": "cache", "reason": "Mutable JIT cache; outside source identity"}]
}
```

Entries are sorted relative paths. Every included directory has a
`{"path":"subdir","kind":"directory"}` entry. Symlinks to regular files
include `target` and a `referent` file record containing the target's actual hash;
directory symlinks, sockets/FIFOs/devices and dangling links are unsupported and
produce INCONCLUSIVE unless explicitly excluded. Exclusions are literal relative
paths/subtrees, not globs, and need a nonempty reason. Empty/dot/absolute/parent
paths are rejected. Nothing is excluded by default. Changed files or an inventory
change during hashing invalidate the capture.

`watch` on a directory holds separately checked relative paths, for example
`[{"path":"rowcache.off","kind":"absent"}]`. Existing watched files use the
same file/symlink format as entries. This can pin absent `rowcache.off`,
`rowcache.report` and `retain.off` controls even when other runtime paths are
excluded. A required code/model file must never be classified as mutable merely
to obtain a passing result.

This is deliberately **not** a cheap metadata-only weight check: a directory
with model shards hashes all included bytes again at each endpoint, including
when the same tree is mounted through `/model` and `/work`. Budget/time this
outside the measured interval. A full model hash can warm host page caches and
cause I/O/CPU pressure; allow the machine to settle before benchmarking. If the
queue chooses to exclude weights or runtime content, the manifest explicitly
narrows its claim to the remaining files. There is no size/mtime hash cache and
no silent exclusion of `/work`, weights, code or test switches.

The independent fixture in `test_identity.py` demonstrates a complete request
with a file mount, directory inventory, cache exclusion and watched absent file.
Replace fixture paths/hashes/args with intended deployment values; do not use it
as a live manifest.

## Redaction and limits of the claim

Sensitive keys/flags become `{"redacted_length":N}` (UTF-8 bytes), including
`--api-key VALUE`, `--api-key=VALUE`, nested `EXTRA_ENV`, `EXTRA_ARGS` and JSON
configuration secret keys. Unknown environment keys and unknown argument values
also default to length-only **and mark the comparison INCONCLUSIVE** until the
public allowlist is extended. A same-length change to an unknown performance
setting must not silently PASS. Recognized operational values remain visible; see
`PUBLIC`, `PUBLIC_PREFIX`, `PUBLIC_FLAGS` and `JSON_FLAGS` in `engine_identity.py`.
URL-like values are conservatively redacted. Add a newly needed public operational
key/flag explicitly in reviewed code rather than supplying raw secret values in
a requested manifest. Public environment keys use strings, JSON flags use parsed
JSON, and equals-form arguments use `{"option":"--flag","value":...}`.

Defaults are parsed without sourcing or executing shell text. The supported
EnvironmentFile subset is one `KEY=value` per line with unquoted, single-quoted or
double-quoted values; duplicate keys and unsupported continuations are rejected.
Snapshot lines preserve normalized assignments. Comments/blank lines retain only
their byte length, since comments can contain secrets. Unknown values are lengths.

Length-only fields **cannot prove equal secret values across captures**; unknown
nonsecret fields do not qualify for a PASS at all.
During a capture raw fields are compared in memory, so an intra-capture change is
detected even at equal length. No secret hashes are persisted for env/defaults.
File mounts still use content hashes as requested, not text dumps.

This is endpoint configuration/content evidence, not a signature or proof of
which Python bytecode was already loaded into an engine. It does not rule out a
change-and-revert between captures, runtime mutation outside declared mounts,
external proxy routing, altered driver/clocks, or a forged artifact. Those need
their own run controls. It does catch ordinary rollback, restart, stale mount,
wrong image/args/env/defaults and source-content drift without reading a GPU.

## Source-only declarations (25 Oct)

`engine_identity.py declare OUT --launcher REVIEWED_LAUNCHER --settings KEY=VALUE ...`
uses pinned image **metadata**, the reviewed launcher, defaults transformed by the
exact pinned `setcfg.py` algorithm, and bind-source bytes. Docker in the launcher
is an inert private-PATH stub; it neither launches nor removes anything. The only
real Docker operation is `docker image inspect`. `--image-inspect FILE` and
`--source-mirror DIR` support entirely offline fixtures. Unknown scripts, options,
environment keys, mutable contents and exhausted hash budgets fail closed.

For the grouping release, build a fresh bundle:

```sh
python3 tools/build_harness_bundle.py <hub-dir>/agents/flash-next/harness-release-pins
```

The queue owner copies that bundle to CT <ct-id> using their normal procedure, then
creates REQUESTS **from sources, before starting the candidate**:

```sh
H=/home/<service-user>/s0/harness-release-pins
R=/home/<service-user>/s0/requests-v16g-release
python3 "$H/engine_identity.py" declare-pair "$R" --harness "$H" \
  --launcher "$H/run-flash-prod.sh.q9"
```

This writes on.json, verify.json, launcher.found, declaration.json and SHA256SUMS.
The pair must differ only by `QSA_COW_VERIFY=1` in defaults and container env.
It pins the found production launcher (45cc5d81...) independently of the intended
grouping launcher (98d6d2a2...). It projects that exact planned launcher replacement
into the `/work` content manifest; the file remains hashed, not excluded.

The queue installs the intended launcher only for its test slot, stops its engine
on every exit, verifies the restored found launcher, restores the private defaults,
and leaves the engine STOPPED. A restore failure fails the queue.

The built bundle includes the generated `ppg` payload, checked against its source
manifest. Source hashing covers model weights and `/work` code. The explicit
release policy excludes only mutable `/work/cache` and `/work/logs`; three test
switches are watched and must be absent. Full reads may be expensive (model files
appear under both mounts). Run declaration after concurrent conversion/source
writes finish; an unstable tree is INCONCLUSIVE. No previously captured container
can bless its own configuration. `capture` still rehashes actual mounted bytes
inside the container, detecting stale bind inodes.
