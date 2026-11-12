"""decode-merge probe (12 Nov 2026): the arm's decode liveness guard. Observational on the engine; it ends only the client.

Inputs: the engine's ``native_stage_local.overlap.decode_step.liveness`` block (``FLASH_S5_DECODE_LIVENESS=1``; the
engine counts a running decoder whose output has not grown for 10 s as one trip) on

* the arm's own status replies (entry/before/after), kept on the status rows by ``persist`` (one marked line in the
  bundle's ``n3j2_box.status_row``), and
* the arm's 15 s status polls, which the guard sees by wrapping the poll client (the poller's rows are unchanged).

Rules, every replay arm alike (chunk, off, off-small and stock):
* **ENGINE** ``engine-trip``: the engine's trip counter moved during the arm (polls or after − before);
* **ENGINE** ``status-unanswered``: POLL_FAILS consecutive polls raised (the engine loop or its control is wedged);
* **HARNESS** ``guard-blind``: a status that answered has no liveness block (the launch lacks the setting, or the path
  changed), so the guard cannot see; recorded, the replay is not ended for it.
An ENGINE trip seen by a poll ends the replay within about a second: the agents' clock jumps past the window, so no new
turn starts, and an agent whose turn is still streaming leaves it (each turn runs in its own daemon thread, so its socket
read is not waited for); the join returns. Open streams stay open in the engine (the next arm's idle-FULL entry then
refuses, as for any crash). The arm gets
the reason ``decode-liveness-guard:<CLASS>:<detail>`` and ``complete`` False, so the inherited grade FAILs it. The guard
never sends a control action and never changes what the engine runs.
"""
import threading
import time

STAGE_CONTROL = 'native_stage_local'
POLL_FAILS = 4                       # 4 x 15 s polls: a minute with no status answer
JUMP_S = 10.0 ** 7                   # past any replay window
REASON = 'decode-liveness-guard:%s:%s'


def liveness_of(reply):
    """The engine's liveness block from a status reply, or None."""
    try:
        block = reply[STAGE_CONTROL]['overlap']['decode_step'].get('liveness')
    except (KeyError, TypeError, AttributeError):
        return None
    return dict(block) if type(block) is dict else None


def persist(row, reply):
    """Marked observational line: keep the engine's liveness block on an arm status row. No key without a block."""
    if isinstance(reply, dict):
        block = liveness_of(reply)
        if block is not None:
            row['decode_liveness'] = block
    return row


class Guard:
    def __init__(self, label, *, clock=time.monotonic, poll_fails=POLL_FAILS):
        self.label, self._clock, self.poll_fails = label, clock, poll_fails
        self.offset = 0.0
        self.lock = threading.Lock()
        self.base_trips = None
        self.polls = self.failures = self.consecutive = self.blind = 0
        self.max_live_stall_ms = 0
        self.trip = None                  # (class, detail, wall_s)
        self.harness = None

    def clock(self):
        return self._clock() + self.offset

    def _end(self, cls, detail):
        if self.trip is None:
            self.trip = (cls, detail, time.time())
            self.offset = JUMP_S

    def seen(self, reply):
        with self.lock:
            self.polls += 1
            self.consecutive = 0
            block = liveness_of(reply)
            if block is None:
                self.blind += 1
                if self.harness is None:
                    self.harness = 'guard-blind'
                return
            trips = block.get('trips')
            if type(trips) is not int:
                self.blind += 1
                self.harness = self.harness or 'guard-blind'
                return
            if self.base_trips is None:
                self.base_trips = trips
            live = block.get('live_max_stall_ms')
            if type(live) is int:
                self.max_live_stall_ms = max(self.max_live_stall_ms, live)
            if trips > self.base_trips:
                self._end('ENGINE', 'engine-trip')

    def failed(self):
        with self.lock:
            self.failures += 1
            self.consecutive += 1
            if self.consecutive >= self.poll_fails:
                self._end('ENGINE', 'status-unanswered')

    def settle(self, record):
        """After the arm: the status rows' after − before, the reason, the record's view."""
        rows = {r.get('tag') or r.get('label'): r for r in record.get('statuses') or () if isinstance(r, dict)}
        before, after = (rows.get(k, {}).get('decode_liveness') for k in ('before', 'after'))
        delta = None
        if isinstance(before, dict) and isinstance(after, dict) and type(before.get('trips')) is int and type(after.get('trips')) is int:
            delta = after['trips'] - before['trips']
            if delta > 0 and self.trip is None:
                self.trip = ('ENGINE', 'engine-trip', None)
        elif self.harness is None and 'before' in rows and 'after' in rows:
            self.harness = 'guard-blind'
        view = dict(polls=self.polls, poll_failures=self.failures, blind_polls=self.blind, base_trips=self.base_trips,
                    trips_delta=delta, max_live_stall_ms=self.max_live_stall_ms,
                    trip=None if self.trip is None else dict(classification=self.trip[0], detail=self.trip[1], wall_s=self.trip[2]),
                    harness=self.harness, ended_replay=self.offset > 0,
                    before=before, after=after)
        record['decode_liveness_guard'] = view
        reasons = record.setdefault('reasons', [])
        if self.trip is not None:
            reasons.append(REASON % self.trip[:2])
        if self.harness is not None:
            reasons.append(REASON % ('HARNESS', self.harness))
        if reasons:
            record['complete'] = False
        return view


class WatchedClient:
    """The poll client, every call passed through unchanged; a status reply or failure is shown to the guard."""

    def __init__(self, client, guard):
        self._client, self._guard = client, guard

    def __getattr__(self, name):
        return getattr(self._client, name)

    def call(self, action, **fields):
        try:
            reply = self._client.call(action, **fields)
        except Exception:
            if action == 'status':
                self._guard.failed()
            raise
        if action == 'status':
            self._guard.seen(reply)
        return reply


class GuardEnded(RuntimeError):
    """Raised in an agent thread whose turn is still streaming when the guard ends the replay."""


def watched_stream(inner, guard, *, wait_s=1.0):
    """The turn runs in its own daemon thread; the agent waits for it and leaves when the guard has ended the replay.
    The turn's own timing (arrivals, ttft) is taken inside ``inner`` as before."""
    def stream(*args, **kwargs):
        box = {}

        def turn():
            try:
                box['result'] = inner(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - handed back to the agent below
                box['error'] = exc
        # The agent thread's own name: compose5's in-flight tracker (inside ``inner``) keys open requests by it.
        worker = threading.Thread(target=turn, name=threading.current_thread().name, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(wait_s)
            if guard.trip is not None and worker.is_alive():
                raise GuardEnded(REASON % guard.trip[:2])
        if 'error' in box:
            raise box['error']
        return box['result']
    return stream


def guarded(replay_arm, replay_run):
    """compose5's replay_arm with the guard: the poll client watched, the agents on the guard's clock and stream."""
    def arm(a, row, client, poll_client, *, run=None):
        guard = Guard(row.get('label'))

        def run_on_guard(*args, **kwargs):
            kwargs['clock'] = guard.clock
            kwargs['stream'] = watched_stream(kwargs.get('stream') or _default_stream(), guard)
            return (run or replay_run())(*args, **kwargs)
        record = replay_arm(a, row, client, WatchedClient(poll_client, guard), run=run_on_guard)
        guard.settle(record)
        return record
    setattr(arm, 'decode_liveness_guard', True)
    return arm


def _default_stream():
    import compose4_replay
    return compose4_replay.stream_turn


def install(box):
    """Bind the guard into the compose5 box module (``e3_compose5_box``): its CLI looks replay_arm up at main()."""
    if getattr(box.replay_arm, 'decode_liveness_guard', False):
        return box.replay_arm
    box.replay_arm = guarded(box.replay_arm, lambda: box._box.replay.run)
    return box.replay_arm
