"""exposure_compare: realized exposure of the matched steps (CPU only)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exposure_compare as x


def leak(n=8, cached=7568, retries=0, cancelled=0, p1=7000, gen=300):
    """retries = number of sessions whose SUSPECT verdict needed path-recreating retries (3 records each)"""
    res = []
    for i in range(n):
        r = dict(round=0, session=i, p1_ids=[1] * p1, gen1_ids=[2] * gen, p2_len=p1 + gen + 20, p3=[3] * (p1 + gen + 60), cached=cached, cached3=cached + 48)
        if i < retries: r['answer_retries'] = [dict(warm='x')] * 3
        res.append(r)
    return dict(results=res, run=dict(cancelled=cancelled), errors=[])   # leak_partial_tail.py: run['cancelled'] = len(cancelled)


def pressure(issued=(40, 41), completed=(39, 40), dur=900.0, occ=(0.55, 0.80), pre=3, hits=10, errors=0):
    # pressure2.py keeps vllm:kv_cache_usage_perc raw: a 0..1 fraction (q15b metrics_end 0.132935)
    return dict(issued=list(issued), completed_requests=list(completed), started_wall=1000.0, ended_wall=1000.0 + dur,
                occupancy=dict(kv_cache_usage_perc=dict(samples=10, min=1.0, max=occ[1], mean=occ[0])), preemptions=pre,
                prefix_cache_hits_delta=hits, errors=['e'] * errors)


class ExposureTests(unittest.TestCase):
    def test_identical_protocol_and_realization_is_matched(self):
        rep = x.compare([('leak', leak(), leak())], [('lifetime', pressure(), pressure())])
        self.assertEqual(rep['verdict'], 'MATCHED', rep)
        self.assertEqual(rep['leak'][0]['sessions'], (8, 8)); self.assertEqual(rep['pressure'][0]['completed'], (79, 79))

    def test_lengths_or_hits_differ_is_differs(self):
        self.assertEqual(x.compare([('l', leak(), leak(gen=301))], [])['verdict'], 'DIFFERS')
        rep = x.compare([('l', leak(), leak(cached=4048))], [])
        self.assertEqual(rep['verdict'], 'DIFFERS'); self.assertIn('l: prefix hits differ', rep['differences'])
        self.assertEqual(rep['leak'][0]['hit_mismatches'][0], [[0, 0], (7568, 7616), (4048, 4096)])

    def test_small_retry_and_cancel_differences_tolerated_large_ones_not(self):
        rep = x.compare([('l', leak(retries=1), leak(retries=3))], [])
        self.assertEqual(rep['verdict'], 'MATCHED'); self.assertEqual(rep['leak'][0]['retry_records'], (1, 3))
        self.assertEqual(x.compare([('l', leak(retries=0), leak(retries=4))], [])['verdict'], 'DIFFERS')
        self.assertEqual(x.compare([('l', leak(cancelled=2), leak(cancelled=4))], [])['verdict'], 'MATCHED')
        self.assertEqual(x.compare([('l', leak(cancelled=0), leak(cancelled=6))], [])['verdict'], 'DIFFERS')

    def test_pressure_exposure_tolerance(self):
        ok = lambda **kw: x.compare([], [('p', pressure(), pressure(**kw))])
        self.assertEqual(ok(completed=(42, 42))['verdict'], 'MATCHED')          # 84 vs 79: 6 %
        self.assertEqual(ok(completed=(50, 50))['verdict'], 'DIFFERS')          # 100 vs 79
        self.assertEqual(ok(dur=1100.0)['verdict'], 'DIFFERS')                   # 22 % longer
        self.assertEqual(ok(pre=5)['verdict'], 'MATCHED'); self.assertEqual(ok(pre=9)['verdict'], 'DIFFERS')
        self.assertEqual(ok(occ=(0.70, 0.90))['verdict'], 'DIFFERS')             # 15 points of mean occupancy (fractions)
        self.assertEqual(ok(occ=(0.60, 0.90))['verdict'], 'MATCHED')             # 5 points
        self.assertEqual(ok(occ=(0.90, 0.95))['verdict'], 'DIFFERS')             # Codex counterexample: 0.10 vs 0.90 must not match
        self.assertEqual(x.compare([], [('p', pressure(occ=(0.10, 0.2)), pressure(occ=(0.90, 0.95)))])['verdict'], 'DIFFERS')
        self.assertEqual(ok(errors=2)['verdict'], 'DIFFERS')
        rep = ok(completed=(50, 50)); self.assertIn('p: completed differ beyond tolerance', rep['differences'])

    def test_real_schemas_cancelled_count_and_list_both_read(self):
        a = leak(cancelled=3); b = leak(); b['run']['cancelled'] = [[0, 0], [0, 1], [0, 2]]   # count (the tool) and a list both read
        rep = x.compare([('l', a, b)], []); self.assertEqual(rep['leak'][0]['cancelled'], (3, 3)); self.assertEqual(rep['verdict'], 'MATCHED')

    def test_nothing_compared_or_missing_fields_is_not_matched(self):
        self.assertEqual(x.compare([], [])['verdict'], 'DIFFERS')
        bare = dict(issued=[1], completed_requests=[1])
        rep = x.compare([], [('p', bare, bare)])
        self.assertEqual(rep['verdict'], 'DIFFERS'); self.assertTrue(any('missing' in d for d in rep['differences']))


if __name__ == '__main__':
    unittest.main()
