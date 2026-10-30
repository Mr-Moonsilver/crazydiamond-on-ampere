import unittest
from e3_native_speculation_gate import grade


class SpeculationGate(unittest.TestCase):
    def line(self, config):
        return '2026-10-29Z (EngineCore pid=86) INFO x [core.py:125] Initializing a V1 LLM engine (v0.29) with config: model=x, ' + config + ', seed=0'

    def test_actual_none_and_enabled_and_missing(self):
        self.assertEqual(grade(self.line('speculative_config=None'))['rc'], 0)
        self.assertEqual(grade(self.line("speculative_config=SpeculativeConfig(method='mtp')"))['rc'], 1)
        self.assertEqual(grade(self.line('seed=0'))['rc'], 2)
        self.assertEqual(grade('MTP_TOKENS=0\nspeculative_config=None')['rc'], 2)

    def test_contradictory_or_incomplete_incarnations_never_pass(self):
        good = self.line('speculative_config=None')
        self.assertEqual(grade(good + '\n' + self.line('speculative_config=enabled'))['rc'], 1)
        self.assertEqual(grade(good + '\n' + self.line('seed=0'))['rc'], 2)
        self.assertEqual(grade(self.line('speculative_config=NoneX'))['rc'], 1)


if __name__ == '__main__':
    unittest.main()
