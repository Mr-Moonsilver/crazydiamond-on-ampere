"""decode-merge probe (12 Nov 2026): the arm's decode liveness guard. Observational on the engine; it ends only the client.

Inputs: the engine's ``native_stage_local.overlap.decode_step.liveness`` block (``FLASH_S5_DECODE_LIVENESS=1``; the
engine counts a running decoder whose output has not grown for 10 s as one trip) on

* the arm's own status replies (entry/before/after), kept on the status rows by ``persist`` (one marked line in the
  bundle's ``n3j2_box.status_row``), and
* the arm's 15 s status polls, which the guard sees by wrapping the poll client (the poller's rows are unchanged).

Rules, every replay arm alike (chunk, off, off-small and stock):
* **ENGINE** ``engine-stuck``: the engine's trip counter moved during the arm AND, in that same reply, a running decoder
  is still stalled (``live_max_stall_ms >= stall_ms``): polls, or the after row for after − before;
* a trip that has recovered by the reply that shows it (the counter moved, the live stall is back under the threshold)
  is **recorded only**: counted in ``recovered_trips``, one ``decode-liveness-guard recovered`` line on stderr (the arm
  log); the replay goes on and the arm is not failed for it. Stock arms on the j2 records show recovering client gaps of
  up to 8.0 s (v1b4 sp.g2-live-0), so a recovered stall must not break the ABBA;
* **ENGINE** ``status-unanswered``: POLL_FAILS consecutive polls raised (the engine loop or its control is wedged);
* **HARNESS** ``guard-blind``: a status that answered has no liveness block although the observer is declared on (the
  launch lacks the setting, or the path changed), so the guard cannot see; recorded, the replay is not ended for it.

The observer declaration (brief 11, 14 Nov 2026): the binding states whether the packet launches the engine's liveness
observer (``install(box, observer='on'|'off')``; the cx builder passes its own FLASH_S5_DECODE_LIVENESS, the same value
the settings line and the draft assert). Default ``'on'``, the dmp packets' binding, unchanged.
* ``observer='on'``: as above; a reply without a block is blind and the arm is INC.
* ``observer='off'``: no block is the expected state. The liveness reading is NOT_APPLICABLE (``liveness`` in the
  record's view), polls without a block are counted as ``absent_polls``, not ``blind_polls``, and no reason is added.
  The poll watch stays bound: ``status-unanswered`` still ends a wedged arm and the trip rules still apply to any block.
  A block seen while the observer is declared off is **HARNESS** ``observer-declared-off`` (the declaration and the
  engine disagree): recorded, the arm is INC.
Why bound rather than unbound when off: unbinding would also drop the status-unanswered end and the poll counts, and an
engine that runs the observer against an off declaration would go unseen.
An ENGINE trip seen by a poll ends the replay within about a second: the agents' clock jumps past the window, so no new
turn starts, and an agent whose turn is still streaming leaves it (each turn runs in its own daemon thread, so its socket
read is not waited for); the join returns. Open streams stay open in the engine (the next arm's idle-FULL entry then
refuses, as for any crash). The arm gets
the reason ``decode-liveness-guard:<CLASS>:<detail>`` and ``complete`` False, so the inherited grade FAILs it. The guard
never sends a control action and never changes what the engine runs.
"""
import sys
import threading
import time

STAGE_CONTROL = 'native_stage_local'
POLL_FAILS = 4                       # 4 x 15 s polls: a minute with no status answer
STALL_MS = 10_000                    # the engine's threshold when its block does not state it (stage_overlap.LIVENESS_STALL_NS)
JUMP_S = 10.0 ** 7                   # past any replay window
REASON = 'decode-liveness-guard:%s:%s'
OBSERVER = ('on', 'off')
NOT_APPLICABLE = 'NOT_APPLICABLE'


def stall_of(block):
    value = block.get('stall_ms')
    return value if type(value) is int and value > 0 else STALL_MS


def stuck(block):
    """A running decoder is still stalled at the engine's threshold in this reply."""
    live = block.get('live_max_stall_ms')
    return type(live) is int and live >= stall_of(block)


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
    def __init__(self, label, *, clock=time.monotonic, poll_fails=POLL_FAILS, observer='on'):
        if observer not in OBSERVER:
            raise ValueError('decode-liveness-guard: observer must be one of %s, not %r' % ('|'.join(OBSERVER), observer))
        self.label, self._clock, self.poll_fails, self.observer = label, clock, poll_fails, observer
        self.absent = 0
        self.offset = 0.0
        self.lock = threading.Lock()
        self.base_trips = None
        self.polls = self.failures = self.consecutive = self.blind = 0
        self.max_live_stall_ms = 0
        self.seen_trips = None            # the trip count last accounted for
        self.recovered_trips = 0
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
                if self.observer == 'off':
                    self.absent += 1
                    return
                self.blind += 1
                if self.harness is None:
                    self.harness = 'guard-blind'
                return
            if self.observer == 'off':
                self.harness = self.harness or 'observer-declared-off'
            trips = block.get('trips')
            if type(trips) is not int:
                self.blind += 1
                self.harness = self.harness or 'guard-blind'
                return
            if self.base_trips is None:
                self.base_trips = self.seen_trips = trips
            live = block.get('live_max_stall_ms')
            if type(live) is int:
                self.max_live_stall_ms = max(self.max_live_stall_ms, live)
            if trips > self.seen_trips:
                new, self.seen_trips = trips - self.seen_trips, trips
                if stuck(block):
                    self._end('ENGINE', 'engine-stuck')
                else:
                    self._recovered(new, block)

    def _recovered(self, new, block):
        self.recovered_trips += new
        try:
            sys.stderr.write('decode-liveness-guard recovered %s trips=%s live_max_stall_ms=%s stall_ms=%s\n'
                             % (self.label, block.get('trips'), block.get('live_max_stall_ms'), stall_of(block)))
        except Exception:  # noqa: BLE001 - a log line never ends the arm
            pass

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
            unseen = after['trips'] - max(before['trips'], self.seen_trips if self.seen_trips is not None else before['trips'])
            if delta > 0 and self.trip is None and stuck(after):
                self.trip = ('ENGINE', 'engine-stuck', None)
            elif unseen > 0 and self.trip is None:
                self._recovered(unseen, after)
        elif self.harness is None and 'before' in rows and 'after' in rows and self.observer == 'on':
            self.harness = 'guard-blind'
        if self.observer == 'off' and self.harness is None and (isinstance(before, dict) or isinstance(after, dict)):
            self.harness = 'observer-declared-off'
        applicable = self.observer == 'on' or self.harness is not None or self.base_trips is not None
        view = dict(observer=self.observer, liveness='read' if applicable else NOT_APPLICABLE, absent_polls=self.absent,
                    polls=self.polls, poll_failures=self.failures, blind_polls=self.blind, base_trips=self.base_trips,
                    trips_delta=delta, recovered_trips=self.recovered_trips, max_live_stall_ms=self.max_live_stall_ms,
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


def guarded(replay_arm, replay_run, observer='on'):
    """compose5's replay_arm with the guard: the poll client watched, the agents on the guard's clock and stream."""
    if observer not in OBSERVER:
        raise ValueError('decode-liveness-guard: observer must be one of %s, not %r' % ('|'.join(OBSERVER), observer))

    def arm(a, row, client, poll_client, *, run=None):
        guard = Guard(row.get('label'), observer=observer)

        def run_on_guard(*args, **kwargs):
            kwargs['clock'] = guard.clock
            kwargs['stream'] = watched_stream(kwargs.get('stream') or _default_stream(), guard)
            return (run or replay_run())(*args, **kwargs)
        record = replay_arm(a, row, client, WatchedClient(poll_client, guard), run=run_on_guard)
        guard.settle(record)
        return record
    setattr(arm, 'decode_liveness_guard', True)
    setattr(arm, 'decode_liveness_observer', observer)
    return arm


def _default_stream():
    import compose4_replay
    return compose4_replay.stream_turn


def install(box, observer='on'):
    """Bind the guard into the compose5 box module (``e3_compose5_box``): its CLI looks replay_arm up at main().
    A second install with another observer declaration is refused (one packet, one declaration)."""
    if getattr(box.replay_arm, 'decode_liveness_guard', False):
        bound = getattr(box.replay_arm, 'decode_liveness_observer', 'on')
        if bound != observer:
            raise ValueError('decode-liveness-guard: already bound with observer=%s, not %s' % (bound, observer))
        return box.replay_arm
    box.replay_arm = guarded(box.replay_arm, lambda: box._box.replay.run, observer)
    return box.replay_arm
