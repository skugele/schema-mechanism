from random import sample
from time import time
from unittest import TestCase

import numpy as np

import test_share
from schema_mechanism.data_structures import DelegatedValueHelper
from schema_mechanism.data_structures import GlobalParams
from schema_mechanism.data_structures import GlobalStats
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import State
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.func_api import sym_state
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestSymbolicItem(TestCase):

    def setUp(self) -> None:
        self.item = SymbolicItem(state_element='1234', primitive_value=1.0)

    def test_init(self):
        # both the state element and the primitive value should be set properly
        self.assertEqual('1234', self.item.state_element)
        self.assertEqual(1.0, self.item.primitive_value)

        # default primitive value should be 0.0
        i = SymbolicItem(state_element='1234')
        self.assertEqual(0.0, i.primitive_value)

    def test_primitive_value(self):
        # an item's primitive value should be settable
        self.item.primitive_value = -2.0
        self.assertEqual(-2.0, self.item.primitive_value)

    def test_is_on(self):
        # item expected to be ON for these states
        self.assertTrue(self.item.is_on(state=['1234']))
        self.assertTrue(self.item.is_on(state=['123', '1234']))

        # item expected to be OFF for these states
        self.assertFalse(self.item.is_on(state=[]))
        self.assertFalse(self.item.is_on(state=['123']))
        self.assertFalse(self.item.is_on(state=['123', '4321']))

    def test_copy(self):
        copy = self.item.copy()

        self.assertEqual(self.item, copy)
        self.assertIsNot(self.item, copy)

    def test_equal(self):
        copy = self.item.copy()
        other = SymbolicItem('123')

        self.assertEqual(self.item, self.item)
        self.assertEqual(self.item, copy)
        self.assertNotEqual(self.item, other)

        self.assertTrue(is_eq_reflexive(self.item))
        self.assertTrue(is_eq_symmetric(x=self.item, y=copy))
        self.assertTrue(is_eq_transitive(x=self.item, y=copy, z=copy.copy()))
        self.assertTrue(is_eq_consistent(x=self.item, y=copy))
        self.assertTrue(is_eq_with_null_is_false(self.item))

    def test_hash(self):
        self.assertIsInstance(hash(self.item), int)
        self.assertTrue(is_hash_consistent(self.item))
        self.assertTrue(is_hash_same_for_equal_objects(x=self.item, y=self.item.copy()))

    @test_share.performance_test
    def test_performance(self):
        n_items = 100_000
        n_runs = 100_000
        n_distinct_states = 100_000_000_000
        n_state_elements = 25

        items = [SymbolicItem(str(value)) for value in range(n_items)]
        state = State(sample(range(n_distinct_states), k=n_state_elements))

        start = time()
        for run in range(n_runs):
            _ = map(lambda i: self.item.is_on(state), items)
        end = time()

        # TODO: Need to add a test that includes an upper bound on the elapsed time
        print(f'Time for {n_runs * n_items:,} SymbolicItem.is_on calls: {end - start}s')


class TestDelegatedValueHelper(TestCase):
    def setUp(self) -> None:

        self.items = {

            # Item used for testing delegated value functionality
            'A': ItemPool().get('A', primitive_value=0.0, item_type=SymbolicItem),

            # Mock Items that have static, pre-defined, avg accessible values
            'B': ItemPool().get('B', primitive_value=1.0, avg_accessible_value=2.0, item_type=MockSymbolicItem),
            'C': ItemPool().get('C', primitive_value=5.0, avg_accessible_value=-1.0, item_type=MockSymbolicItem),
            'D': ItemPool().get('D', primitive_value=1.0, avg_accessible_value=6.0, item_type=MockSymbolicItem),
            'E': ItemPool().get('E', primitive_value=-2.0, avg_accessible_value=4.0, item_type=MockSymbolicItem),
            'F': ItemPool().get('F', primitive_value=-5.0, avg_accessible_value=0.0, item_type=MockSymbolicItem),
            'G': ItemPool().get('G', primitive_value=9.0, avg_accessible_value=1.0, item_type=MockSymbolicItem),
            'H': ItemPool().get('H', primitive_value=-3.0, avg_accessible_value=12.0, item_type=MockSymbolicItem),
            'Z': ItemPool().get('Z', primitive_value=0.0, avg_accessible_value=0.0, item_type=MockSymbolicItem),
        }

        self.dvh = DelegatedValueHelper(item=self.items['A'])

        # initialize a static baseline value to simplify testing
        GlobalStats(baseline_value=2.0)

        # explicitly set max trace length to facilitate testing (e.g., guarantee trace termination)
        DelegatedValueHelper.DV_TRACE_MAX_LEN = 3

        self.states = [
            # trace 1 states (len = 3 [MAX])
            sym_state('A'),  # selection state (item On) that starts trace
            sym_state('B,C'),  # 1st result state
            sym_state('C'),  # 2nd result state
            sym_state('H'),  # 3rd result state
            sym_state('E'),  # 4th result state

            # trace 2 states (len = 3 [MAX])
            sym_state('A'),  # selection state (item On) that starts trace
            sym_state('D,E,F'),  # 1st result state
            sym_state('D,G'),  # 2nd result state
            sym_state('B,F,G'),  # 3rd result state

            # trace 3 states (len = 1)
            sym_state('A'),  # selection state (item On) that starts trace (and terminates previous trace)

            # trace 4 states (len = 3 [MAX])
            sym_state('A,C'),  # selection state (item On) that (re-)starts trace; result state that terminates trace 3
            sym_state('Z'),  # 1st result state (no primitive or delegated value)
            sym_state('Z'),  # 2nd result state (no primitive or delegated value)
            sym_state('Z'),  # 3rd result state (no primitive or delegated value)
            sym_state('G'),  # 4th result state (no primitive or delegated value)
        ]

        self.trace_states = [
            self.states[0:5],
            self.states[5:10],
            self.states[9:11],
            self.states[10:15],
        ]

        self.s_select = self.states[0:-1]
        self.s_result = self.states[1:]

    def test_init(self):
        self.assertEqual(0.0 - GlobalStats().baseline_value, self.dvh.delegated_value)
        self.assertEqual(0.0, self.dvh.avg_accessible_value)
        self.assertEqual(0, self.dvh.trace_updates_remaining)
        self.assertIs(self.items['A'], self.dvh.item)

    def test_update_1(self):
        # test trace updates remaining
        ##############################

        for ts in self.trace_states:

            # test: verify that dv helper's item is on is initial state
            self.assertTrue(self.dvh.item.is_on(ts[0]))

            # select and result states for this trace trajectory
            s_select = ts[0:-1]
            s_result = ts[1:]

            # used to track the number of times the dv helper's item is On vs Off in the traces' state trajectories
            item_status_counts = {'on': 0, 'off': 0}

            updates_remaining = self.dvh.trace_updates_remaining
            for ss, sr in zip(s_select, s_result):
                early_term = (
                        (self.dvh.item.is_on(ss) and self.dvh.item.is_on(sr)) or
                        (self.dvh.trace_updates_remaining > 1 and self.dvh.item.is_on(sr))
                )

                # must call update first to start the dv trace BEFORE checking updates remaining
                self.dvh.update(selection_state=ss, result_state=sr)

                # test: dvh SHOULD initialize a NEW trace if item is On in selection state
                item_status_counts['on' if self.dvh.item.is_on(ss) else 'off'] += 1

                updates_remaining = (
                    0 if early_term else
                    DelegatedValueHelper.DV_TRACE_MAX_LEN - 1 if self.dvh.item.is_on(ss) else
                    max(0, updates_remaining - 1)
                )

                # test: update should always reduce the current updates remaining by one (unless already zero)
                self.assertEqual(updates_remaining, self.dvh.trace_updates_remaining)

            self.assertEqual(1, item_status_counts['on'])
            self.assertEqual(len(s_select) - 1, item_status_counts['off'])

    def test_update_2(self):
        # test update of trace, average accessibility, and delegated values
        ###################################################################

        # simplify the tests by setting learning rate to 1.0
        GlobalParams().learn_rate = 1.0

        for ts in self.trace_states:
            old_trace_value = self.dvh.trace_value
            old_avg_accessible_value = self.dvh.avg_accessible_value
            old_delegated_value = self.dvh.delegated_value

            # select and result states for this trace trajectory
            s_select = ts[0:-1]
            s_result = ts[1:]

            for ss, sr in zip(s_select, s_result):
                terminating = (
                        self.dvh.trace_updates_remaining > 1 and self.dvh.item.is_on(sr) or
                        self.dvh.trace_updates_remaining == 1 or

                        # needed because this scenario terminates in a single update
                        self.dvh.item.is_on(ss) and self.dvh.item.is_on(sr)
                )
                updating = self.dvh.trace_updates_remaining > 0 or self.dvh.item.is_on(ss)

                s_pv = sr.primitive_value
                s_aav = sr.avg_accessible_value

                self.dvh.update(selection_state=ss, result_state=sr)

                # trace value should be max of:
                #       (1) old trace value,
                #       (2) state primitive value, and
                #       (3) state average accessible value
                trace_value = max(old_trace_value, s_pv, s_aav)

                expected_trace_value = (
                    trace_value
                    if updating and not terminating
                    else -np.inf
                )

                expected_avg_access_value = (
                    trace_value
                    if updating and terminating
                    else old_avg_accessible_value
                )

                expected_delegated_value = (
                    expected_avg_access_value - GlobalStats().baseline_value
                    if updating and terminating
                    else old_delegated_value
                )

                self.assertEqual(expected_trace_value, self.dvh.trace_value)
                self.assertEqual(expected_avg_access_value, self.dvh.avg_accessible_value)
                self.assertEqual(expected_delegated_value, self.dvh.delegated_value)

                old_trace_value = expected_trace_value
                old_avg_accessible_value = self.dvh.avg_accessible_value
                old_delegated_value = self.dvh.delegated_value

    def test_update_3(self):
        # test dv for full, multi-trace trajectory with learning rate < 1.0
        ###################################################################
        GlobalParams().learn_rate = 0.5

        # select and result states from full trajectory
        s_select = self.states[0:-1]
        s_result = self.states[1:]

        for ss, sr in zip(s_select, s_result):
            self.dvh.update(selection_state=ss, result_state=sr)

        self.assertEqual(3.25, self.dvh.avg_accessible_value)
        self.assertEqual(self.dvh.avg_accessible_value - GlobalStats().baseline_value, self.dvh.delegated_value)
