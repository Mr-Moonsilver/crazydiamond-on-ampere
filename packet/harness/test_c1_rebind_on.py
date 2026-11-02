"""C1's connected suite (``test_e3_stage_overlap``) rerun with the deferred PP rebind selected.

Every class of the C1 suite, unchanged, with each stack's driver given ``defer_bind = True``
(``FLASH_S5_STAGE_OVERLAP_DEFER_BIND=1``; leaf ``selection-by-attribute`` of
``stage_overlap_fixture_leaves.json``, as ``overlap_selected``).  Identity against the frozen
serial driver, admission, lanes, stop at every tick and at every internal boundary of the pair,
stale and duplicate replies: all must hold with the bind moved in front of the PP_SEND.
CPU only; leaves in ``stage_overlap_fixture_leaves.json`` and ``c1_rebind_fixture_leaves.json``.
"""
import unittest

import test_e3_stage_overlap as c1


class DeferOn:
    def network(self, **kwargs):
        network = super().network(**kwargs)
        network.driver.defer_bind = True
        return network


class IdentityOn(DeferOn, c1.Identity):
    pass


class AdmissionOn(DeferOn, c1.Admission):
    pass


class LanesOn(DeferOn, c1.Lanes):
    pass


class StopSweepOn(DeferOn, c1.StopSweep):
    pass


class StopStatesOn(DeferOn, c1.StopStates):
    pass


class ControlOn(DeferOn, c1.Control):
    pass


class StaleAndDuplicateOn(DeferOn, c1.StaleAndDuplicate):
    pass


class Engaged(DeferOn, c1.OverlapBase):
    def test_every_early_successor_defers_its_bind(self):
        network = self.network()
        network.run_request()
        driver = network.driver
        self.assertGreater(driver.overlap_counters['early_opens'], 0)
        self.assertEqual(driver.defer_counters['deferred_opens'], driver.overlap_counters['early_opens'])
        self.assertEqual(driver.defer_counters['deferred_binds'], driver.overlap_counters['early_opens'])


if __name__ == '__main__':
    unittest.main()
