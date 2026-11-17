#!/usr/bin/env python3
"""alpha-next-c1b: supersedes ``test_alpha_next_c1_defer_control.Mutants`` by name (that file stays as it ran in chains).

READY (alpha-next-c1, "Known test gaps" 2, HARNESS): ``Mutants.test_install_check_removed`` and
``Mutants.test_overlap_check_removed`` compile the mutated ``OverlapSegmentDriver._stage_control`` as a method of an
exec'd subclass ``_M``; its zero-argument ``super()`` binds to ``_M``, so on a real ``OverlapSegmentDriver`` it raises
``TypeError`` before any rule runs. The refusal tests then "catch" the mutant by that TypeError, not by their own
assertions: the mutant is never really exercised.

``Mutants`` here: the same five mutants as the original class, each run against the same scenario test(s), with the two
``_stage_control`` mutants rebuilt as module-level functions whose ``super()`` is the explicit
``super(OverlapSegmentDriver, self)`` (rebind_mutants_3's construction). Each test asserts the mutant is caught by the
scenario's assertion failures (``AssertionError`` only), with no error at all and no ``TypeError`` anywhere in the
outcome. ``Defect`` records the superseded construction precisely: the original's ``overlap_control`` mutant, run against
the same scenario, yields only ``TypeError`` errors and no assertion failure.

``mutant_outcomes()`` returns each mutant's scenario outcome (ids and exception kinds) for the evidence runner.
CPU only.
"""
import inspect
import textwrap
import unittest
from unittest import mock

import test_alpha_next_c1_defer_control as dc   # first: its imports put the engine sources on the path
import native_overlap_session as nos
import s5_startup
import stage_overlap as so
import stage_progress as sp

INSTALL_OLD = ("        if not self.defer_bind_installed:\n"
               "            raise DriverError('native-stage-overlap-defer-bind-not-installed')\n")
OVERLAP_OLD = ("        if defer and overlap is not True:\n"
               "            raise DriverError('native-stage-overlap-defer-bind-needs-overlap')\n")


# A scenario written ``with self.subTest(...), self.assertRaises(X) as caught:`` followed by ``caught.exception``: when
# the mutant lets the call through, the subTest records the assertion failure ("X not raised") and swallows it, then
# reading ``caught.exception`` raises this AttributeError. It is a knock-on of a recorded assertion failure, not a catch.
KNOCK_ON = "'_AssertRaisesContext' object has no attribute 'exception'"


class KindResult(unittest.TestResult):
    """A TestResult that keeps each failure's and error's exception class and first message line."""

    def __init__(self):
        super().__init__()
        self.kinds = dict(failures=[], errors=[])

    def _keep(self, test, err):
        exc_type, value = err[0], err[1]
        chain, seen = [], value
        while seen is not None:
            chain.append(type(seen).__name__)
            seen = seen.__cause__ or seen.__context__
        bucket = 'failures' if issubclass(exc_type, unittest.TestCase.failureException) else 'errors'
        self.kinds[bucket].append((test.id(), exc_type.__name__, str(value).splitlines()[0] if str(value) else '',
                                   chain))

    def addFailure(self, test, err):
        self._keep(test, err)
        super().addFailure(test, err)

    def addError(self, test, err):
        self._keep(test, err)
        super().addError(test, err)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self._keep(subtest, err)
        super().addSubTest(test, subtest, err)


def run_scenario(target, name, replacement, scenario):
    with mock.patch.object(target, name, replacement):
        result = KindResult()
        scenario.run(result)
    rows = result.kinds['failures'] + result.kinds['errors']
    return dict(tests=result.testsRun,
                failures=[row[:3] for row in result.kinds['failures']],
                errors=[row[:3] for row in result.kinds['errors']],
                type_error=any('TypeError' in row[3] for row in rows)
                or any('TypeError' in text for _test, text in result.failures + result.errors))


# ----------------------------------------------------------------------------------------- the five mutants
def no_idle(self, command):
    """C0's control without its paused idle FULL rule (the original's ``no_idle``, unchanged: no super())."""
    previous = self.stage_selected
    self.stage_selected = command['local']
    return {'schema': 1, 'action': sp.STAGE_CONTROL, 'local': command['local'], 'previous': previous}


def stage_control_mutant(old, new):
    """``OverlapSegmentDriver._stage_control`` with one exact-once mutation, compiled as a module-level function in a
    copy of stage_overlap's namespace, its zero-argument ``super()`` made explicit first."""
    source = textwrap.dedent(inspect.getsource(so.OverlapSegmentDriver._stage_control))
    assert source.count(old) == 1, old
    # Two calls (the non-dict pass-through and the C0 call); a comment also names "super()", harmless to replace.
    assert source.count('super()._stage_control(') == 2, 'two super() calls expected in _stage_control'
    source = source.replace(old, new).replace('super()', 'super(OverlapSegmentDriver, self)')
    namespace = dict(vars(so))
    exec(compile(source, '<mutant _stage_control>', 'exec'), namespace)
    return namespace['_stage_control']


def startup_mutant():
    source = textwrap.dedent(inspect.getsource(s5_startup.install_engine))
    old = "        if value == '1' and env.get('FLASH_S5_STAGE_OVERLAP', '0') != '1':\n"
    assert source.count(old) == 1 and 'super()' not in source
    namespace = dict(vars(s5_startup))
    exec(compile(source.replace(old, "        if False:\n"), '<mutant install_engine>', 'exec'), namespace)
    return namespace['install_engine']


def session_mutant():
    source = textwrap.dedent(inspect.getsource(nos.NativeOverlapSession._action_native_stage_local))
    old = "        raise NativeOverlapSessionError('native-stage-overlap-defer-bind-not-installed')\n"
    assert source.count(old) == 1 and 'super()' not in source
    namespace = dict(vars(nos))
    # An async method compiles as a module-level async function as well; no class cell is needed (no super()).
    exec(compile(source.replace(old, '        pass\n'), '<mutant session>', 'exec'), namespace)
    return namespace['_action_native_stage_local']


def mutants():
    """name -> (target, attribute, replacement factory, scenario factories): the original class's five, rebuilt."""
    return {
        'idle-rule-skipped': (sp.StagedSegmentDriver, '_stage_control', lambda: no_idle,
                              (lambda: dc.Idle('test_never_accepted_while_a_chunk_or_a_pp_ticket_is_live'),
                               lambda: dc.Idle('test_refused_mid_transaction_and_with_a_retained_row'))),
        'install-check-removed': (so.OverlapSegmentDriver, '_stage_control',
                                  lambda: stage_control_mutant(INSTALL_OLD, ''),
                                  (lambda: dc.Install('test_refused_where_not_installed'),)),
        'overlap-check-removed': (so.OverlapSegmentDriver, '_stage_control',
                                  lambda: stage_control_mutant(OVERLAP_OLD, ''),
                                  (lambda: dc.Install('test_refused_with_overlap_false_or_absent_and_nothing_applied'),)),
        'startup-guard-removed': (s5_startup, 'install_engine', startup_mutant,
                                  (lambda: dc.Startup('test_rebind_switches_need_the_overlap_switch'),)),
        'session-refusal-removed': (nos.NativeOverlapSession, '_action_native_stage_local', session_mutant,
                                    (lambda: dc.RouterNotInstalled('test_field_refused_where_not_installed'),)),
    }


def mutant_outcomes():
    out = {}
    for name, (target, attribute, factory, scenarios) in mutants().items():
        out[name] = [run_scenario(target, attribute, factory(), scenario()) for scenario in scenarios]
    return out


class Mutants(unittest.TestCase):
    """Each mutant must fail its scenario by the scenario's own assertions: failures only, all AssertionError."""

    def caught_by_assertions(self, name):
        target, attribute, factory, scenarios = mutants()[name]
        for scenario in scenarios:
            case = scenario()
            with self.subTest(scenario=case.id()):
                # The scenario passes unmutated (the catch below is the mutant's doing).
                baseline = unittest.TestResult()
                scenario().run(baseline)
                self.assertEqual((baseline.testsRun, baseline.failures, baseline.errors), (1, [], []))
                outcome = run_scenario(target, attribute, factory(), case)
                self.assertEqual(outcome['tests'], 1, outcome)
                self.assertTrue(outcome['failures'], 'mutant survived: %s %r' % (name, outcome))
                self.assertEqual({kind for _id, kind, _msg in outcome['failures']}, {'AssertionError'}, outcome)
                self.assertFalse(outcome['type_error'], outcome)
                # No error but the knock-on of a recorded "not raised" failure (see KNOCK_ON).
                for _id, kind, message in outcome['errors']:
                    self.assertEqual((kind, message), ('AttributeError', KNOCK_ON), outcome)
                if outcome['errors']:
                    self.assertTrue(any('not raised' in message for _id, _kind, message in outcome['failures']), outcome)

    def test_idle_rule_skipped(self):
        self.caught_by_assertions('idle-rule-skipped')

    def test_install_check_removed(self):
        self.caught_by_assertions('install-check-removed')

    def test_overlap_check_removed(self):
        self.caught_by_assertions('overlap-check-removed')

    def test_startup_guard_removed(self):
        self.caught_by_assertions('startup-guard-removed')

    def test_session_refusal_removed(self):
        self.caught_by_assertions('session-refusal-removed')

    def test_rebuilt_mutants_really_mutate(self):
        """The rebuilt _stage_control mutants run on a real driver (no TypeError) and apply the refused selection."""
        case = dc.Install('test_refused_where_not_installed')
        case.setUp()
        try:
            network = case.network(installed=False)
            driver = network.driver
            with mock.patch.object(so.OverlapSegmentDriver, '_stage_control', stage_control_mutant(INSTALL_OLD, '')):
                reply = case.control(network, local=True, overlap=True, defer_bind=True)
            self.assertIs(reply['overlap'], True)
            self.assertNotIn('defer_bind', reply)          # not installed: the mutant accepted the field and dropped it
            network = case.network()
            driver = network.driver
            with mock.patch.object(so.OverlapSegmentDriver, '_stage_control', stage_control_mutant(OVERLAP_OLD, '')):
                reply = case.control(network, local=True, overlap=False, defer_bind=True)
            self.assertEqual((reply['overlap'], reply['defer_bind'], driver.defer_bind), (False, False, False))
        finally:
            case.doCleanups()


class Defect(unittest.TestCase):
    """The superseded construction, recorded: dc.Mutants.overlap_control's mutant is caught by TypeError only."""

    def test_original_construction_is_caught_only_by_type_error(self):
        original = dc.Mutants('test_install_check_removed')
        for old, scenario in ((INSTALL_OLD, 'test_refused_where_not_installed'),
                              (OVERLAP_OLD, 'test_refused_with_overlap_false_or_absent_and_nothing_applied')):
            with self.subTest(scenario=scenario):
                mutant = original.overlap_control(old, '')
                outcome = run_scenario(so.OverlapSegmentDriver, '_stage_control', mutant, dc.Install(scenario))
                # No assertion failure at all: every catch is the zero-argument super() TypeError (plus, where the
                # scenario reads caught.exception after a swallowed subTest, its knock-on AttributeError).
                self.assertEqual(outcome['failures'], [], outcome)
                errors = outcome['errors']
                type_errors = [row for row in errors if row[1] == 'TypeError']
                self.assertTrue(type_errors, outcome)
                self.assertTrue(all('super(type, obj)' in message for _id, _kind, message in type_errors), outcome)
                for _id, kind, message in errors:
                    self.assertTrue((kind == 'TypeError' and 'super(type, obj)' in message)
                                    or (kind == 'AttributeError' and message == KNOCK_ON), outcome)


if __name__ == '__main__':
    unittest.main()
