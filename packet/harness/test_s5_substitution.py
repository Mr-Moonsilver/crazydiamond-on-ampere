import copy
import json
import struct
import unittest
from s5_substitution import TensorBytes, assess, tensor_diff


def tensor(name, value, dtype='float32'):
    fmt={'float32':'f','bfloat16':'H','uint8':'B'}[dtype]
    return TensorBytes(name,dtype,(1,),struct.pack('<'+fmt,value))


def sequence(output=10):
    return [[tensor('output',output+i),tensor('state-0',i),tensor('conv-0',2*i),
             tensor('ple_raw',i,'uint8'),tensor('ple_output',i+1)] for i in range(3)]


class Substitution(unittest.TestCase):
    def test_observed_whole_solo_sequence_only(self):
        a,b=sequence(10),sequence(20)
        r=assess([a,b,a],[b,a,b])
        self.assertTrue(r['passed']);self.assertEqual(len(r['comparisons']),15)
        self.assertEqual(r['resident_matches_host_sequences']['resident-0'],['host-1'])
        # Every chunk separately matches, but this whole trajectory was never
        # observed in a host solo run. A per-chunk tolerance/oracle must not pass.
        mosaic=[a[0],b[1],a[2]]
        r=assess([a,b,a],[a,mosaic,b]);self.assertFalse(r['passed'])
        self.assertEqual(r['resident_matches_host_sequences']['resident-1'],[])

    def test_all_state_conv_and_ple_bits_always_exact(self):
        for path in ('host','resident'):
            for slot in (1,2,3,4):
                host=[sequence() for _ in range(3)];candidate=copy.deepcopy(host)
                xs=host if path=='host' else candidate
                old=xs[1][1][slot]
                xs[1][1][slot]=tensor(old.name,7,old.dtype)
                r=assess(host,candidate)
                self.assertFalse(r['passed']);self.assertFalse(r['all_states_and_ple_bitwise'])

    def test_bit_exact_even_for_signed_zero_and_no_nonfinite(self):
        host=[sequence() for _ in range(3)];candidate=copy.deepcopy(host)
        candidate[0][0][1]=tensor('state-0',-0.)
        self.assertFalse(assess(host,candidate)['passed'])
        host[0][0][0]=tensor('output',float('nan'))
        r=assess(host,host);self.assertFalse(r['passed']);self.assertFalse(r['finite'])
        json.dumps(r,allow_nan=False)

    def test_difference_metrics_and_payload_free_reports(self):
        r=tensor_diff(tensor('output',0x3f80,'bfloat16'),tensor('output',0x3f81,'bfloat16'))
        self.assertEqual(r['differing_elements'],1);self.assertEqual(r['max_ulp'],1)
        self.assertEqual(r['max_abs'],1/128)
        r=tensor_diff(tensor('output',0xbf80,'bfloat16'),tensor('output',0xbf82,'bfloat16'))
        self.assertEqual(r['max_ulp'],2);self.assertEqual(r['max_abs'],1/64)
        r=tensor_diff(tensor('ple_raw',0,'uint8'),tensor('ple_raw',255,'uint8'))
        self.assertEqual(r['max_abs'],255);self.assertIsNone(r['max_ulp'])
        host=[sequence() for _ in range(3)];candidate=[sequence(11) for _ in range(3)]
        report=assess(host,candidate)
        self.assertFalse(report['passed'])
        def visit(x):
            if isinstance(x,dict):
                self.assertNotIn('data',x)
                for v in x.values():visit(v)
            elif isinstance(x,list):
                for v in x:visit(v)
            else:self.assertNotIsInstance(x,bytes)
        visit(report);json.dumps(report,allow_nan=False)
        hh=[p for p in report['comparisons'] if p['left'].startswith('host') and p['right'].startswith('host')]
        self.assertEqual(len(hh),3);self.assertTrue(all(p['output_equal'] for p in hh))

    def test_missing_or_misaligned_snapshots_fail_closed(self):
        with self.assertRaises(ValueError):assess([sequence()]*2,[sequence()]*3)
        host=[sequence() for _ in range(3)];candidate=copy.deepcopy(host)
        candidate[1][0].pop()
        with self.assertRaises(ValueError):assess(host,candidate)
        bad=TensorBytes('x','float32',(2,),b'1234')
        with self.assertRaises(ValueError):tensor_diff(bad,bad)


if __name__=='__main__':unittest.main()
