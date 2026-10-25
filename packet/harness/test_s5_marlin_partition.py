"""Integer-layout checks only; never tests numerical Marlin correctness."""
import unittest
from s5_marlin_partition import Layout,example


class Partition(unittest.TestCase):
    def test_each_tile_covers_k_exactly_and_lock_order_is_complete(self):
        for blocks in (1,17,82,129,256,257,300):
            for grid in (32,82,164):
                for tn,tk in ((256,64),(128,64),(64,128)):
                    c=Layout(blocks*32,32,2560,640,tn,tk,grid)
                    self.assertEqual(c.prefix%c.grid,0)
                    for m in range(blocks):
                        for n in range(c.nt):
                            parts=c.slices(m,n)['partials']
                            self.assertEqual([x['slice_idx'] for x in parts],list(range(len(parts))))
                            ordered=sorted(parts,key=lambda x:x['k_begin'])
                            self.assertEqual(ordered[0]['k_begin'],0)
                            self.assertEqual(ordered[-1]['k_end'],c.k)
                            self.assertTrue(all(a['k_end']==b['k_begin'] for a,b in zip(ordered,ordered[1:])))

    def test_same_expert_adjacent_m_blocks_can_change_reduction_partition(self):
        e=example()
        self.assertEqual(e['before']['path'],'DP')
        self.assertEqual([(s['k_begin'],s['k_end']) for s in e['before']['partials']],[(0,640)])
        self.assertEqual(e['after']['path'],'Stream-K')
        self.assertGreater(len(e['after']['partials']),1)
        self.assertEqual([s['k_begin'] for s in e['after']['partials']],
                         sorted([s['k_begin'] for s in e['after']['partials']],reverse=True))

    def test_row_shuffle_inside_same_block_does_not_change_k_partition(self):
        c=Layout(257*32,32,2560,640,128,64,164)
        self.assertEqual(c.token(254*32)['columns'],c.token(254*32+31)['columns'])
        self.assertNotEqual(c.token(253*32)['columns'],c.token(254*32)['columns'])

    def test_zero_remainder_uses_stream_k_wave_and_group_alignment(self):
        c=Layout(82*32,32,2560,640,128,64,82)
        self.assertEqual(c.tail,82)
        self.assertEqual(c.iters%2,0)
        with self.assertRaises(ValueError):Layout(3,32,2560,640,128,64,82)
        with self.assertRaises(ValueError):c.token(c.padded_tokens)


if __name__=='__main__':unittest.main()
