"""Pure actual-planner contract evidence; no claim about GPU indexer equality."""
import math
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import sparse_history_dma as dma

def plan(count):
    selected=tuple(range(0,2*count,2))
    result=dma.make_sparse_plan(selected,tuple(range(1,51)),tuple(range(1,52)),1,
                                valid_tokens=202400,null_page=0)
    if result.refused:result=dma.stream_sparse_plan(result)
    if result.refused:raise AssertionError(result.fallback)
    return result

def profile(result):
    return dict(rows=len(result.selected),buckets=len(result.buckets),
                waves=[math.ceil(b.descriptors/dma.MAX_DESCRIPTORS) for b in result.buckets])

class PlanContract(unittest.TestCase):
    def test_identical_inputs_produce_identical_plans(self):
        for count in (4097,97153):
            a,b=plan(count),plan(count)
            self.assertEqual(a,b)
            self.assertEqual(profile(a),profile(b))

    def test_legal_rank_local_inputs_need_not_have_equal_schedule_profiles(self):
        a,b=plan(4097),plan(97153)
        self.assertEqual(a.mapping,b.mapping)
        self.assertEqual(a.versions,b.versions)
        self.assertEqual(a.epoch,b.epoch)
        self.assertEqual(profile(a),dict(rows=4097,buckets=1,waves=[17]))
        self.assertEqual(profile(b),dict(rows=97153,buckets=2,waves=[380,1]))
        # Both choices are inside the sparse route with the same50-page
        # frontier. They are legal planner inputs, not full-model selections.
        self.assertTrue(len(a.selected)*5<202400*4)
        self.assertTrue(len(b.selected)*5<202400*4)

if __name__=='__main__':unittest.main()
