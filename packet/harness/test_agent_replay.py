#!/usr/bin/env python3
"""agent_replay v2 summarizer: cut-off streams are not completed turns; per-turn full reuse (CPU only)."""
import importlib.util, unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("agent_replay", Path(__file__).with_name("agent_replay.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def rec(t, kind, prompt, cached, complete=True, agent=0):
    return {"agent": agent, "session": 1, "kind": kind, "t": t, "history_est": prompt, "reply_target": 300,
            "ttft": 1.0, "seconds": 5.0, "completion_tokens": 300 if complete else None, "prompt_tokens": prompt,
            "cached_tokens": cached, "finish": "length" if complete else None, "complete": complete}


class Summary(unittest.TestCase):
    def test_cut_off_stream_is_not_a_turn(self):
        s = m.summarize([rec(10, "fresh", 100000, 0), rec(20, "append", 102000, 99000, complete=False)], 0, 100)
        self.assertEqual((s["warm_turns"], s["incomplete"]), (1, 1))

    def test_full_reuse_counts_whole_history_hits_only(self):
        rs = [rec(10, "fresh", 100000, 0), rec(20, "append", 102000, 99000), rec(30, "append", 104000, 0)]
        s = m.summarize(rs, 0, 100)
        self.assertEqual((s["append_full_reuse"], s["appends_judged"], s["incomplete"]), (0.5, 2, 0))



class TurnComplete(unittest.TestCase):
    def test_endings(self):
        base = {"ttft": 1.0, "completion_tokens": 430, "prompt_tokens": 128564, "cached_tokens": 126192}
        for finish, want in (("stop", True), ("length", True), ("tool_calls", True), (None, False), ("abort", False)):
            self.assertEqual(m.turn_complete(dict(base, finish=finish), 430), want, finish)
        self.assertFalse(m.turn_complete(dict(base, finish="tool_calls", completion_tokens=200), 430))
        self.assertFalse(m.turn_complete(dict(base, finish="stop", cached_tokens=None), 430))

if __name__ == "__main__":
    unittest.main()
