import unittest

import numpy as np

from schema_mechanism.core import EligibilityTraceDelegatedValueHelper
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.share import GlobalParams
from schema_mechanism.util import pairwise
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup


class TestEligibilityTraceDelegatedValueHelper(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.item_a = sym_item('A', primitive_value=0.0)
        self.item_b = sym_item('B', primitive_value=0.0)

        self.dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=0.5, trace_decay=0.5)

        # mocks
        self.i_no_pv_no_dv = sym_item('1', item_type=MockSymbolicItem, primitive_value=0.0, delegated_value=0.0)
        self.i_no_pv_neg_dv = sym_item('2', item_type=MockSymbolicItem, primitive_value=0.0, delegated_value=-1.0)
        self.i_no_pv_pos_dv = sym_item('3', item_type=MockSymbolicItem, primitive_value=0.0, delegated_value=1.0)

        self.i_neg_pv_no_dv = sym_item('4', item_type=MockSymbolicItem, primitive_value=-1.0, delegated_value=0.0)
        self.i_neg_pv_neg_dv = sym_item('5', item_type=MockSymbolicItem, primitive_value=-1.0, delegated_value=-1.0)
        self.i_neg_pv_pos_dv = sym_item('6', item_type=MockSymbolicItem, primitive_value=-1.0, delegated_value=1.0)

        self.i_pos_pv_no_dv = sym_item('7', item_type=MockSymbolicItem, primitive_value=1.0, delegated_value=0.0)
        self.i_pos_pv_neg_dv = sym_item('8', item_type=MockSymbolicItem, primitive_value=1.0, delegated_value=-1.0)
        self.i_pos_pv_pos_dv = sym_item('9', item_type=MockSymbolicItem, primitive_value=1.0, delegated_value=1.0)

    def test_init(self):
        discount_factor = 0.75
        trace_decay = 0.25

        dvh = EligibilityTraceDelegatedValueHelper(discount_factor=discount_factor, trace_decay=trace_decay)

        # test: discount_factor should have been set properly by initializer
        self.assertEqual(discount_factor, dvh.discount_factor)

        # test: trace_decay should have been set properly by initializer
        self.assertEqual(trace_decay, dvh.eligibility_trace.decay_rate)

    def test_item_never_on(self):
        # test: delegated value of On item in previous state should be updated towards target

        # self.item_a was On in previous state
        selection_state = sym_state('A')
        result_state = sym_state('7,9')

        # simplify testing by setting learning rate to 1.0
        GlobalParams().set('learning_rate', 1.0)

        self.dv_helper.update(selection_state=selection_state, result_state=result_state)

        target = self.dv_helper.effective_state_value(selection_state=selection_state, result_state=result_state)
        self.assertEqual(target, self.dv_helper.delegated_value(self.item_a))

    def test_item_on_in_last_selection_state(self):
        # test: an item that was on in the most recent selection state should receive higher delegated value than
        #     : those on in earlier selection states

        states = [
            sym_state('B'),
            sym_state('A'),
            sym_state('7,9')
        ]

        for selection_state, result_state in pairwise(states):
            self.dv_helper.update(selection_state=selection_state, result_state=result_state)

        dv_a = self.dv_helper.delegated_value(self.item_a)
        dv_b = self.dv_helper.delegated_value(self.item_b)

        self.assertGreater(dv_a, dv_b)

    def test_item_on_in_multiple_states_leading_up_to_result(self):
        # test: items on multiple times in recent selection states should receive more delegated value than items
        #     : that were only on once (ceteris paribus)

        states = [
            sym_state('A'),
            sym_state('A'),
            sym_state('A,B'),
            sym_state('7,9')
        ]

        for selection_state, result_state in pairwise(states):
            self.dv_helper.update(selection_state=selection_state, result_state=result_state)

        dv_a = self.dv_helper.delegated_value(self.item_a)
        dv_b = self.dv_helper.delegated_value(self.item_b)

        self.assertGreater(dv_a, dv_b)

    def test_item_maximum_delegated_value(self):
        # test: delegated value should not accumulate without limit

        GlobalParams().set('learning_rate', 0.1)

        self.dv_helper.update(selection_state=sym_state('A'), result_state=sym_state('A,3'))

        last_diff = np.inf
        for _ in range(100):
            before = self.dv_helper.delegated_value(self.item_a)
            self.dv_helper.update(selection_state=sym_state('A,7'), result_state=sym_state('A,9'))
            after = self.dv_helper.delegated_value(self.item_a)

            # test: the difference in dv before and after update should gradually decrease
            diff = after - before
            self.assertLess(diff, last_diff)
            last_diff = diff

        # test: the difference should approach zero as dv approaches target values
        self.assertAlmostEqual(0.0, float(last_diff))

    def test_pos_primitive_value_states_produce_pos_delegated_value(self):
        GlobalParams().set('learning_rate', 0.1)

        for _ in range(100):
            self.dv_helper.update(selection_state=sym_state('A'), result_state=sym_state('7'))

        self.assertGreater(self.dv_helper.delegated_value(self.item_a), 0.0)

    def test_pos_delegated_value_states_produce_pos_delegated_value(self):
        GlobalParams().set('learning_rate', 0.1)

        for _ in range(100):
            self.dv_helper.update(selection_state=sym_state('A'), result_state=sym_state('3'))

        self.assertGreater(self.dv_helper.delegated_value(self.item_a), 0.0)

    def test_neg_primitive_value_states_produce_neg_delegated_value(self):
        GlobalParams().set('learning_rate', 0.1)

        for _ in range(100):
            self.dv_helper.update(selection_state=sym_state('A'), result_state=sym_state('4'))

        self.assertLess(self.dv_helper.delegated_value(self.item_a), 0.0)

    def test_neg_delegated_value_states_produce_neg_delegated_value(self):
        GlobalParams().set('learning_rate', 0.1)

        for _ in range(100):
            self.dv_helper.update(selection_state=sym_state('A'), result_state=sym_state('2'))

        self.assertLess(self.dv_helper.delegated_value(self.item_a), 0.0)

    def test_zero_value_states_produce_zero_delegated_value(self):
        GlobalParams().set('learning_rate', 0.1)

        for _ in range(100):
            self.dv_helper.update(selection_state=sym_state('A'), result_state=sym_state('1'))

        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_a))

    def test_pv_results_in_greater_target_than_dv_of_same_magnitude(self):
        # 3 -> dv only
        # 7 -> pv only

        # update A from delegated value only
        for _ in range(100):
            self.dv_helper.update(selection_state=sym_state('A'), result_state=sym_state('3'))

        # update B from primitive value only
        for _ in range(100):
            self.dv_helper.update(selection_state=sym_state('B'), result_state=sym_state('7'))

        # test: item B should have greater delegated value than A
        dv_a = self.dv_helper.delegated_value(self.item_a)
        dv_b = self.dv_helper.delegated_value(self.item_b)

        self.assertLess(dv_a, dv_b)
