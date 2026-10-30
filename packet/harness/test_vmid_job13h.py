"""VM13h CPU acceptance; real GPU contexts and stock classes require the VM."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'patches/s5_engine'), str(ROOT/'harness')]
import vmid_job13h as job
import vmid_mamba_align_proof as proof
import vmid_support as support


class MambaProofTests(unittest.TestCase):
    def positive(self):
        return dict.fromkeys(proof.REQUIRED, True) | {
            'families': ['GDN_ATTN', 'SHORT_CONV'], 'copy_prefix_lengths': [2, 1]}

    def test_all_claims_are_required(self):
        row = self.positive()
        self.assertEqual(proof.grade(row)['verdict'], 'PASS')
        for key in proof.REQUIRED:
            bad = copy.deepcopy(row)
            bad[key] = False
            self.assertEqual(proof.grade(bad)['verdict'], 'FAIL', key)
            bad.pop(key)
            self.assertEqual(proof.grade(bad)['verdict'], 'INCONCLUSIVE', key)

    def test_missing_or_wrong_family_is_not_success(self):
        self.assertEqual(proof.grade(None)['verdict'], 'INCONCLUSIVE')
        row = self.positive()
        row['families'] = ['GDN_ATTN']
        self.assertEqual(proof.grade(row)['verdict'], 'INCONCLUSIVE')
        row = self.positive()
        row['copy_prefix_lengths'] = [2, 2]
        self.assertEqual(proof.grade(row)['verdict'], 'INCONCLUSIVE')

    def test_whole_job_cannot_omit_startup_proof(self):
        grade = job.grade({'failures': [], 'missing': [], 'cases': []})
        self.assertIn('mamba_align:proof_missing', grade['missing'])

    def test_imported_pins_match_transferred_reference_and_builder(self):
        builder = (ROOT/'tools/build_harness_bundle.py').read_text()
        for rel, digest in proof.SOURCE_PINS.items():
            path = ROOT/'reference/vllm-src'/rel
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, rel)
            if rel != 'models/qwen4_exp/nvidia/model.py':
                self.assertIn(digest, builder)

    def test_probe_uses_real_factory_then_stock_initializer(self):
        text = (ROOT/'harness/vmid_mamba_align_proof.py').read_text()
        self.assertIn('lane_chunk.LaneChunkFactory(runner)', text)
        self.assertIn('state._ensure_align_ctx(config, gids, ordinary_tables)', text)
        self.assertIn('private.compute_aligned_state_indices(seq_lens, 1)', text)
        self.assertIn('ordinary.compute_aligned_state_indices(seq_lens, 1)', text)
        self.assertIn('MambaHybridModelState.__new__', text)

    def test_owned_names_fraction_sidecar_and_result(self):
        runner = (ROOT/'harness/runs/vmid_job13h.sh').read_text()
        for text in ('/opt/flashnext/runs/job13h', 'flashnext-job13h-build',
                     'flashnext-job13h', '/opt/flashnext/.job.lock',
                     'PYTORCH_CUDA_ALLOC_CONF', '--mode monitor'):
            self.assertIn(text, runner)
        self.assertIn('flashnext-job13h', support.OWNED_CONTAINERS)
        self.assertIn('flashnext-job13h', support.LONG_RUN_CONTAINERS)
        probe = (ROOT/'harness/vmid_job13h.py').read_text()
        self.assertIn('set_per_process_memory_fraction(.125, 0)', probe)
        self.assertIn("'mamba-align-startup.json'", probe)
        self.assertIn('mamba_align_startup=raw.get', probe)


if __name__ == '__main__':
    unittest.main()
