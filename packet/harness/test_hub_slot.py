#!/usr/bin/env python3
"""tools/hub-slot.sh, tools/dryrun-record.sh, tools/require-dryrun.sh: real processes, real flock, no mocks.

Run: nice -n 19 python3 -m unittest harness/test_hub_slot.py  (CPU only; lock dirs and repos are temporary,
HUB_SLOT_DIR keeps the real lock dir untouched).
"""
import hashlib, json, os, re, shutil, signal, subprocess, sys, tempfile, time, unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"
SLOT, RECORD, REQUIRE = (str(TOOLS / n) for n in ("hub-slot.sh", "dryrun-record.sh", "require-dryrun.sh"))

# Child used as the slotted command: logs start/stop with a system-wide monotonic clock, and holds until at
# least `need` starts exist (default 2, or 4 s), so that many slotted commands overlap deterministically.
WORKER = r"""
import os, sys, time
ev, name = sys.argv[1], sys.argv[2]
need = int(sys.argv[3]) if len(sys.argv) > 3 else 2
def put(kind):
    fd = os.open(ev, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, f"{time.monotonic_ns()} {kind} {name}\n".encode()); os.close(fd)
put("start")
t0 = time.monotonic()
def starts():
    with open(ev) as f:
        return sum(l.split()[1] == "start" for l in f)
while time.monotonic() - t0 < 4 and starts() < need:
    time.sleep(0.02)
time.sleep(0.3)
put("stop")
"""


def max_overlap(events_path):
    with open(events_path) as f:
        rows = sorted((int(t), 0 if k == "start" else 1) for t, k, _ in (l.split() for l in f))
    cur = best = 0
    for _, kind in rows:  # start sorts before stop on a tie: ties count as overlap
        cur += 1 if kind == 0 else -1
        best = max(best, cur)
    return best


class SlotBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hub-slot-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.locks = self.tmp / "locks" / "nested"  # missing on purpose: the tool creates it
        self.log = self.tmp / "evidence" / "slot.log"
        self.procs = []
        self.addCleanup(self.reap)

    def reap(self):
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            p.wait()

    def env(self, **extra):
        e = dict(os.environ, HUB_SLOT_DIR=str(self.locks), HUB_SLOT_POLL_S="0.05", HUB_SLOT_LOG_EVERY_S="1")
        e.update(extra)
        return e

    def spawn(self, cls, n, cmd, **kw):
        p = subprocess.Popen([SLOT, cls, str(n), str(self.log), "--", *cmd], env=self.env(), start_new_session=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
        self.procs.append(p)
        return p

    def run_slot(self, cls, n, cmd, **kw):
        return subprocess.run([SLOT, cls, str(n), str(self.log), "--", *cmd], env=self.env(), capture_output=True,
                              text=True, timeout=60, **kw)

    def log_text(self):
        return self.log.read_text() if self.log.exists() else ""

    def wait_for(self, pred, what, timeout=15):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return
            time.sleep(0.05)
        self.fail(f"timed out waiting for {what}; log:\n{self.log_text()}")


class Slots(SlotBase):
    def test_three_commands_n2_overlap_exactly_two(self):
        ev = self.tmp / "events"
        ps = [self.spawn("strict", 2, [sys.executable, "-c", WORKER, str(ev), f"c{i}"]) for i in range(3)]
        for p in ps:
            out, _ = p.communicate(timeout=60)
            self.assertEqual(p.returncode, 0, out)
        self.assertEqual(max_overlap(ev), 2)  # never 3 (lock works), and 2 do run together (n is honoured)
        self.assertEqual(ev.read_text().count(" start "), 3)
        lines = self.log_text().splitlines()
        acq = [l for l in lines if " acquired " in l]
        self.assertEqual(len(acq), 3)
        self.assertEqual({re.search(r"strict (\d)/2 ", l).group(1) for l in acq}, {"1", "2"})
        self.assertEqual(len([l for l in lines if " released rc=0 " in l]), 3)

    def test_n3_uses_three_slots(self):
        ev = self.tmp / "events"
        ps = [self.spawn("strict", 3, [sys.executable, "-c", WORKER, str(ev), f"c{i}", "3"]) for i in range(3)]
        for p in ps:
            out, _ = p.communicate(timeout=60)
            self.assertEqual(p.returncode, 0, out)
        self.assertEqual(max_overlap(ev), 3)
        self.assertEqual({re.search(r"strict (\d)/3 ", l).group(1) for l in self.log_text().splitlines() if " acquired " in l}, {"1", "2", "3"})

    def test_classes_are_independent(self):
        h = self.spawn("strict", 1, ["sleep", "30"])
        self.wait_for(lambda: " acquired " in self.log_text(), "holder")
        r = self.run_slot("other", 1, ["true"])  # different class: must not wait
        self.assertEqual(r.returncode, 0)
        self.assertIn("other 1/1 acquired waited 0s", self.log_text())

    def test_rc_passes_through(self):
        for rc in (0, 1, 7, 42):
            self.assertEqual(self.run_slot("strict", 1, ["bash", "-c", f"exit {rc}"]).returncode, rc)
            self.assertRegex(self.log_text().splitlines()[-1], rf"strict 1/1 released rc={rc} \d{{4}}-\d\d-\d\dT[\d:]+Z$")
        self.assertEqual(self.run_slot("strict", 1, ["definitely-not-a-command-xyz"]).returncode, 127)

    def test_args_stdin_stdout_pass_through(self):
        r = self.run_slot("strict", 1, ["bash", "-c", 'echo "[$1] [$2]"; cat', "name", "a b  c", "d"], input="piped in\n")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout, "[a b  c] [d]\npiped in\n")

    def test_log_line_formats(self):
        self.assertEqual(self.run_slot("strict", 2, ["true"]).returncode, 0)
        acq, rel = self.log_text().splitlines()
        stamp = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"
        self.assertRegex(acq, rf"^slot strict 1/2 acquired waited 0s load \d+\.\d\d \d+\.\d\d \d+\.\d\d {stamp}$")
        self.assertRegex(rel, rf"^slot strict 1/2 released rc=0 {stamp}$")
        self.assertTrue(self.locks.is_dir() and (self.locks / "strict.1.lock").exists())

    def test_waiting_lines_with_load_then_acquire(self):
        h = self.spawn("strict", 1, ["sleep", "30"])
        self.wait_for(lambda: " acquired " in self.log_text(), "holder")
        w = self.spawn("strict", 1, ["true"])
        self.wait_for(lambda: len([l for l in self.log_text().splitlines() if " waiting " in l]) >= 2, "two waiting lines")
        w_lines = [l for l in self.log_text().splitlines() if " waiting " in l]
        for l in w_lines:
            self.assertRegex(l, r"^slot strict waiting \d+s load \d+\.\d\d \d+\.\d\d \d+\.\d\d \d{4}-\d\d-\d\dT[\d:]+Z$")
        self.assertIsNone(w.poll())  # still blocked, no timeout
        os.killpg(h.pid, signal.SIGKILL)
        w.communicate(timeout=30)
        self.assertEqual(w.returncode, 0)
        acq = [l for l in self.log_text().splitlines() if " acquired " in l][-1]
        self.assertRegex(acq, r"acquired waited ([2-9]|\d\d)s ")

    def test_killed_holder_releases_slot(self):
        h = self.spawn("strict", 1, ["sleep", "60"])
        self.wait_for(lambda: " acquired " in self.log_text(), "holder")
        w = self.spawn("strict", 1, ["true"])
        time.sleep(0.8)
        self.assertIsNone(w.poll(), "second command must wait while the slot is held")
        self.assertEqual(len([l for l in self.log_text().splitlines() if " acquired " in l]), 1)
        os.killpg(h.pid, signal.SIGKILL)  # wrapper and its command die without any cleanup
        w.communicate(timeout=30)
        self.assertEqual(w.returncode, 0)
        self.assertEqual(len([l for l in self.log_text().splitlines() if " acquired " in l]), 2)

    def test_sigterm_to_wrapper_is_forwarded_and_slot_released(self):
        h = self.spawn("strict", 1, ["sleep", "60"])
        self.wait_for(lambda: " acquired " in self.log_text(), "holder")
        h.send_signal(signal.SIGTERM)  # wrapper only, not the group
        h.communicate(timeout=30)
        self.assertEqual(h.returncode, 143)
        self.assertRegex(self.log_text(), r"strict 1/1 released rc=143 ")
        self.assertEqual(self.run_slot("strict", 1, ["true"]).returncode, 0)

    def test_wrapper_kill9_keeps_slot_until_command_ends(self):
        # The command inherits the lock fd: a dead wrapper with a live command still counts as a heavy step.
        pidfile = self.tmp / "cmd.pid"
        h = self.spawn("strict", 1, ["bash", "-c", f"echo $$ > {pidfile}; exec sleep 60"])
        self.wait_for(pidfile.exists, "command pid")
        self.wait_for(lambda: pidfile.read_text().strip() != "", "command pid content")
        cmd_pid = int(pidfile.read_text())
        w = self.spawn("strict", 1, ["true"])
        os.kill(h.pid, signal.SIGKILL)  # wrapper only
        h.wait()
        time.sleep(0.8)
        self.assertIsNone(w.poll(), "slot must stay taken while the orphaned command lives")
        os.kill(cmd_pid, signal.SIGKILL)
        w.communicate(timeout=30)
        self.assertEqual(w.returncode, 0)

    def test_usage_errors_refuse_and_run_nothing(self):
        marker = self.tmp / "ran"
        cmd = ["touch", str(marker)]
        for cls, n in (("Bad", 2), ("a_b", 2), ("a/b", 2), ("", 2), ("strict", 0), ("strict", 9), ("strict", "x"), ("strict", "")):
            self.assertEqual(self.run_slot(cls, n, cmd).returncode, 64, (cls, n))
        for argv in ([SLOT], [SLOT, "strict", "2", str(self.log)], [SLOT, "strict", "2", str(self.log), "--"],
                     [SLOT, "strict", "2", str(self.log), "nodashes", "true"]):
            self.assertEqual(subprocess.run(argv, env=self.env(), capture_output=True).returncode, 64, argv)
        self.assertFalse(marker.exists())
        self.assertFalse(self.log.exists())
        self.assertFalse(self.locks.exists(), "usage errors must not create lock files")


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


class DryRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dryrun-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.wt = self.tmp / "wt"
        self.wt.mkdir()
        git(self.wt, "init", "-q")
        (self.wt / "a.txt").write_text("a\n")
        git(self.wt, "add", "a.txt")
        git(self.wt, "commit", "-q", "-m", "one")
        self.log = self.tmp / "dryrun.log"
        self.log.write_text("dry run output\nPASS\n")
        self.rec = self.tmp / "rec.json"

    def record(self, rc=0, out=None, log=None):
        return subprocess.run([RECORD, str(self.wt), str(rc), str(log or self.log), str(out or self.rec)], capture_output=True, text=True)

    def require(self, rec=None):
        return subprocess.run([REQUIRE, str(self.wt), str(rec or self.rec)], capture_output=True, text=True)

    def assertRefused(self, r, reason_re):
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        lines = r.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1, r.stderr)
        self.assertRegex(lines[0], reason_re)

    def test_record_fields(self):
        r = self.record()
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(self.rec.read_text())
        self.assertEqual(set(d), {"schema", "head", "dirty", "rc", "log", "log_sha256", "utc"})
        self.assertEqual(d["schema"], "dryrun-record/1")
        self.assertEqual(d["head"], git(self.wt, "rev-parse", "HEAD"))
        self.assertEqual((d["dirty"], d["rc"]), (0, 0))
        self.assertEqual(d["log"], str(self.log.resolve()))
        self.assertEqual(d["log_sha256"], hashlib.sha256(self.log.read_bytes()).hexdigest())
        self.assertRegex(d["utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")

    def test_record_counts_status_lines(self):
        (self.wt / "b.txt").write_text("b")
        (self.wt / "a.txt").write_text("changed\n")
        self.assertEqual(self.record().returncode, 0)
        self.assertEqual(json.loads(self.rec.read_text())["dirty"], 2)

    def test_record_refuses_overwrite(self):
        self.assertEqual(self.record().returncode, 0)
        before = self.rec.read_bytes()
        r = self.record(rc=1)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refusing to overwrite", r.stderr)
        self.assertEqual(self.rec.read_bytes(), before)

    def test_record_refuses_bad_inputs(self):
        self.assertNotEqual(self.record(rc="x").returncode, 0)
        self.assertNotEqual(self.record(log=self.tmp / "nope.log").returncode, 0)
        r = subprocess.run([RECORD, str(self.tmp), "0", str(self.log), str(self.tmp / "o.json")], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)  # not a git worktree
        self.assertFalse((self.tmp / "o.json").exists())

    def test_accepts_clean_passing_dryrun_of_this_head(self):
        self.assertEqual(self.record().returncode, 0)
        r = self.require()
        self.assertEqual((r.returncode, r.stderr), (0, ""), r.stdout)

    def test_refuses_missing_record(self):
        self.assertRefused(self.require(), r"does not exist")

    def test_refuses_garbage_record(self):
        self.rec.write_text("{not json")
        self.assertRefused(self.require(), r"unreadable")

    def test_refuses_wrong_schema(self):
        self.assertEqual(self.record().returncode, 0)
        d = json.loads(self.rec.read_text()); d["schema"] = "dryrun-record/2"; self.rec.write_text(json.dumps(d))
        self.assertRefused(self.require(), r"schema")

    def test_refuses_nonzero_rc(self):
        self.assertEqual(self.record(rc=1).returncode, 0)
        self.assertRefused(self.require(), r"rc is 1")

    def test_refuses_dirty_dryrun(self):
        (self.wt / "stray.txt").write_text("x")
        self.assertEqual(self.record().returncode, 0)
        (self.wt / "stray.txt").unlink()  # clean again: the record still says it was dirty
        self.assertRefused(self.require(), r"dirty")

    def test_refuses_other_head(self):
        self.assertEqual(self.record().returncode, 0)
        (self.wt / "a.txt").write_text("two\n")
        git(self.wt, "commit", "-q", "-am", "two")
        self.assertRefused(self.require(), r"head mismatch")

    def test_refuses_changed_log(self):
        self.assertEqual(self.record().returncode, 0)
        self.log.write_text("dry run output\nPASS (edited)\n")
        self.assertRefused(self.require(), r"sha256 mismatch")

    def test_refuses_missing_log(self):
        self.assertEqual(self.record().returncode, 0)
        self.log.unlink()
        self.assertRefused(self.require(), r"missing or unreadable")

    def test_refuses_bad_worktree_and_usage(self):
        self.assertEqual(self.record().returncode, 0)
        r = subprocess.run([REQUIRE, str(self.tmp / "no-such-dir"), str(self.rec)], capture_output=True, text=True)
        self.assertRefused(r, r"cannot read HEAD")
        r = subprocess.run([REQUIRE], capture_output=True, text=True)
        self.assertEqual(r.returncode, 3)


if __name__ == "__main__":
    unittest.main()
