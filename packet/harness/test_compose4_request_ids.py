"""Opt-in replay identity evidence; CPU-only scripted SSE and replay turns."""
import io
import json
import threading
import unittest
from unittest import mock

import compose4_replay as replay


class RequestIds(unittest.TestCase):
    def stream(self, events, *, request_ids=True):
        body = replay.request_body('m', [], 2, 's')
        if request_ids:
            body['request_id'] = 'test-turn'
        wire = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)
        wire += 'data: [DONE]\n\n'
        with mock.patch.object(replay.urllib.request, 'urlopen', return_value=io.BytesIO(wire.encode())):
            return replay.stream_turn('http://unused', body, None, 0, clock=lambda: 0)

    def events(self, response_id='chatcmpl-test-turn'):
        events = [dict(choices=[dict(delta=dict(content='x'), finish_reason='length')]),
                  dict(choices=[], usage=dict(prompt_tokens=20, completion_tokens=2,
                                             prompt_tokens_details=dict(cached_tokens=0)))]
        if response_id is not None:
            for event in events:
                event['id'] = response_id
        return events

    def test_default_stream_shape_and_no_id_fixtures_unchanged(self):
        expected = {'ttft', 'seconds', 'text', 'finish', 'arrivals_ms', 'completion_tokens',
                    'prompt_tokens', 'cached_tokens'}
        for response_id in (None, 'chatcmpl-server-chosen'):
            with self.subTest(response_id=response_id):
                got = self.stream(self.events(response_id), request_ids=False)
                self.assertEqual(set(got), expected)
                self.assertEqual(got['text'], 'x')

    def test_candidate_returns_consistent_sse_id_including_usage(self):
        self.assertEqual(self.stream(self.events())['response_id'], 'chatcmpl-test-turn')

    def test_candidate_rejects_missing_malformed_mismatched_or_changing_ids(self):
        for value in (None, '', 17, 'x' * 129, 'chatcmpl-other-turn'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.stream(self.events(value))
        events = self.events()
        events[-1]['id'] = 'chatcmpl-other-turn'
        with self.assertRaises(ValueError):
            self.stream(events)
        del events[-1]['id']
        with self.assertRaises(ValueError):
            self.stream(events)
        with self.assertRaises(ValueError):
            self.stream([])

    def test_ids_are_deterministic_distinct_and_bounded(self):
        args = ['run', 'arm', 0, 1, 1]
        baseline = replay.request_id_of(*args)
        self.assertEqual(replay.request_id_of(*args), baseline)
        ids = {baseline}
        for index, value in enumerate(('other-run', 'other-arm', 1, 2, 2)):
            changed = args.copy()
            changed[index] = value
            ids.add(replay.request_id_of(*changed))
        self.assertEqual(len(ids), 6)
        self.assertRegex(baseline, r'^q18a-[0-9a-f]{32}$')
        self.assertLess(len('chatcmpl-' + baseline + '-01234567'), 64)
        self.assertNotEqual(replay.request_id_of('a|b', 'c', 0, 1, 1),
                            replay.request_id_of('a', 'b|c', 0, 1, 1))

    def play(self, enabled=None, response='valid'):
        cfg = dict(replay.T.REPLAY, AGENTS=1, START_MIN=20, START_MAX=20,
                   APPEND_MIN=2, APPEND_MAX=2, REPLY_MIN=2, REPLY_MAX=2,
                   TOOL_MIN=0, TOOL_MAX=0)
        if enabled is not None:
            cfg['REQUEST_IDS'] = enabled
        now, bodies, records, journal = [0.0], [], [], []

        def stream(_url, body, _key, _origin):
            bodies.append(json.loads(json.dumps(body)))
            now[0] += 1
            result = dict(ttft=0.5, seconds=1, text='x', finish='length', arrivals_ms=[0],
                          completion_tokens=2, prompt_tokens=20, cached_tokens=0)
            if response == 'valid' and 'request_id' in body:
                result['response_id'] = 'chatcmpl-' + body['request_id']
            elif response == 'mismatch':
                result['response_id'] = 'chatcmpl-wrong'
            return result

        replay.agent_loop(0, 'http://unused', 'm', None, cfg, 'run', 'arm', 0, 4,
                          records, threading.Lock(), clock=lambda: now[0], sleep=lambda _: None,
                          stream=stream, sink=mock.Mock(write=journal.append))
        self.assertEqual(records, journal)
        return bodies, records

    def test_opt_in_journals_both_ids_without_changing_draws_or_cache_salts(self):
        plain_bodies, plain_records = self.play()
        off_bodies, off_records = self.play(False)
        self.assertEqual((plain_bodies, plain_records), (off_bodies, off_records))
        bodies, records = self.play(True)
        self.assertEqual(len({body['request_id'] for body in bodies}), 4)
        self.assertEqual(len({row['client_request_id'] for row in records}), 4)
        for plain_body, body, plain_row, row in zip(plain_bodies, bodies, plain_records, records):
            self.assertEqual(body['request_id'], row['client_request_id'])
            self.assertEqual(row['response_id'], 'chatcmpl-' + body['request_id'])
            self.assertEqual({k: v for k, v in body.items() if k != 'request_id'}, plain_body)
            self.assertEqual({k: v for k, v in row.items() if k not in ('client_request_id', 'response_id')}, plain_row)

    def test_candidate_injected_stream_cannot_silently_omit_or_invent_evidence(self):
        for response in ('missing', 'mismatch'):
            with self.subTest(response=response):
                bodies, records = self.play(True, response)
                self.assertEqual(len({body['request_id'] for body in bodies}), 4)
                for row in records:
                    self.assertIn('client_request_id', row)
                    self.assertIsNone(row['response_id'])
                    self.assertIn('error', row)
                    self.assertNotIn('complete', row)
                self.assertEqual([row['session'] for row in records], [1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
