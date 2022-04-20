import unittest

import numpy as np

from schema_mechanism.core import EligibilityTraceDelegatedValueHelper
from schema_mechanism.core import ItemPool
from schema_mechanism.core import calc_value
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.share import GlobalParams
from schema_mechanism.util import pairwise
from test_share import disable_test
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
        # noinspection PyBroadException
        try:
            # test: discount factors and trace decays between 0.0 and 1.0 (inclusive) should be allowed
            _ = EligibilityTraceDelegatedValueHelper(discount_factor=0.0, trace_decay=0.0)
            _ = EligibilityTraceDelegatedValueHelper(discount_factor=1.0, trace_decay=1.0)

            discount_factor = 0.5
            trace_decay = 0.5

            dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=discount_factor, trace_decay=trace_decay)

            self.assertIsNotNone(dv_helper.eligibility_trace)

            # test: discount factor should have been set to the value passed to initializer
            self.assertTrue(dv_helper.discount_factor, discount_factor)

            # test: trace decay should have been set in eligibility trace to the value passed to initializer
            self.assertTrue(dv_helper.eligibility_trace.decay_rate, trace_decay)

            # test: delegated values should be initialized to zero
            for item in [sym_item(str(v)) for v in range(100)]:
                self.assertEqual(0.0, dv_helper.delegated_value(item))

        except Exception as e:
            self.fail(f'Unexpected exception raised from dv helper initializer: {str(e)}')

        # test: discount factors less than 0.0 should raise a ValueError
        self.assertRaises(ValueError,
                          lambda: EligibilityTraceDelegatedValueHelper(discount_factor=-0.1, trace_decay=0.5))

        # test: discount factors greater than 1.0 should raise a ValueError
        self.assertRaises(ValueError,
                          lambda: EligibilityTraceDelegatedValueHelper(discount_factor=1.1, trace_decay=0.5))

        # test: trace decay less than 0.0 should raise a ValueError
        self.assertRaises(ValueError,
                          lambda: EligibilityTraceDelegatedValueHelper(discount_factor=0.5, trace_decay=-0.1))

        # test: trace decay greater than 1.0 should raise a ValueError
        self.assertRaises(ValueError,
                          lambda: EligibilityTraceDelegatedValueHelper(discount_factor=0.5, trace_decay=1.1))

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

    # TODO: this is true of accumulating traces, not replacing traces. This test case needs to be moved into a
    # TODO: separate test suite for accumulating traces.
    # def test_item_on_in_multiple_states_leading_up_to_result(self):
    #     # test: items on multiple times in recent selection states should receive more delegated value than items
    #     #     : that were only on once (ceteris paribus)
    #
    #     states = [
    #         sym_state('A'),
    #         sym_state('A'),
    #         sym_state('A,B'),
    #         sym_state('7,9')
    #     ]
    #
    #     for selection_state, result_state in pairwise(states):
    #         self.dv_helper.update(selection_state=selection_state, result_state=result_state)
    #
    #     dv_a = self.dv_helper.delegated_value(self.item_a)
    #     dv_b = self.dv_helper.delegated_value(self.item_b)
    #
    #     self.assertGreater(dv_a, dv_b)

    def test_item_maximum_undiscounted_delegated_value(self):
        # undiscounted dv helper
        dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=1.0, trace_decay=0.5)

        GlobalParams().set('learning_rate', 0.1)

        dv_helper.update(selection_state=sym_state('A'), result_state=sym_state('A,3'))

        last_diff = np.inf
        for _ in range(1000):
            before = dv_helper.delegated_value(self.item_a)
            dv_helper.update(selection_state=sym_state('A,7'), result_state=sym_state('9'))
            after = dv_helper.delegated_value(self.item_a)

            # test: the difference in dv before and after update should gradually decrease
            diff = after - before
            if not np.isclose(0, diff):
                self.assertLess(diff, last_diff)
            last_diff = diff

        # test: the difference should approach zero as dv approaches target values
        self.assertAlmostEqual(0.0, float(last_diff))

        # test: undiscounted delegated value should converge to target state value
        self.assertAlmostEqual(calc_value(sym_state('9')), dv_helper.delegated_value(self.item_a))

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

    def test_effective_state_value_no_change(self):
        # test: state SHOULD have zero value if no new elements between selection and result states
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1,2,3'),
                                                         result_state=sym_state('1,2,3'))
        self.assertEqual(0.0, eff_value)

    def test_effective_state_value_with_no_discount(self):
        # simplifying this test by making discount factor and trace decay 1.0. (Only testing basic value calculations.)
        self.dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=1.0, trace_decay=1.0)

        # test: result state with primitive and delegated value of 0.0 SHOULD result in eff. value of 1.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('2'), result_state=sym_state('1'))
        self.assertEqual(0.0, eff_value)

        # test: result state with primitive value of 1.0 and delegated value of 0.0 SHOULD result in eff. value of 1.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('7'))
        self.assertEqual(1.0, eff_value)

        # test: result state with delegated value of 1.0 and primitive value of 0.0 SHOULD result in eff. value of 1.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('3'))
        self.assertEqual(1.0, eff_value)

        # test: result state with primitive and delegated value of 1.0 SHOULD result in eff. value of 2.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('9'))
        self.assertEqual(2.0, eff_value)

        # test: result state with primitive value of -1.0 and delegated value of 0.0 SHOULD result in eff. value of -1.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('4'))
        self.assertEqual(-1.0, eff_value)

        # test: result state with delegated value of -1.0 SHOULD result in eff. value of -1.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('2'))
        self.assertEqual(-1.0, eff_value)

        # test: result state with primitive and delegated values of -1.0 SHOULD result in eff. value of -2.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('5'))
        self.assertEqual(-2.0, eff_value)

    def test_effective_state_value_multiple_state_elements(self):
        # simplifying this test by making discount factor and trace decay 1.0. (Only testing basic value calculations.)
        self.dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=1.0, trace_decay=1.0)

        _ = sym_item('P1', item_type=MockSymbolicItem, primitive_value=1.0, delegated_value=1.0)
        _ = sym_item('P2', item_type=MockSymbolicItem, primitive_value=1.0, delegated_value=1.0)
        _ = sym_item('P3', item_type=MockSymbolicItem, primitive_value=1.0, delegated_value=1.0)
        _ = sym_item('P4', item_type=MockSymbolicItem, primitive_value=-1.0, delegated_value=-1.0)

        # test: result state's value should be the SUM of the item primitive and delegated values of those states
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('P4'),
                                                         result_state=sym_state('P1,P2,P3'))
        self.assertEqual(6.0, eff_value)

        # test: negative and positive item values be summed properly
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('P2'),
                                                         result_state=sym_state('P1,P4'))
        self.assertEqual(0.0, eff_value)

    def test_effective_state_value_with_discount(self):
        discount_factor = 0.5
        self.dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=discount_factor, trace_decay=1.0)

        # test: state SHOULD have zero value if no new elements between selection and result states
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('2'), result_state=sym_state('1'))
        self.assertEqual(0.0, eff_value)

        # test: result state with primitive value of 1.0 SHOULD result in eff. value of 1.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('7'))
        self.assertEqual(1.0 * discount_factor, eff_value)

        # test: result state with delegated value of 1.0 SHOULD result in eff. value of 1.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('3'))
        self.assertEqual(1.0 * discount_factor ** 2, eff_value)

        # test: result state with primitive and delegated value of 1.0 SHOULD result in eff. value of 2.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('9'))
        self.assertEqual(1.0 * discount_factor + 1.0 * discount_factor ** 2, eff_value)

        # test: result state with primitive value of -1.0 SHOULD result in eff. value of -1.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('4'))
        self.assertEqual(-1.0 * discount_factor, eff_value)

        # test: result state with delegated value of -1.0 SHOULD result in eff. value of -1.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('2'))
        self.assertEqual(-1.0 * discount_factor ** 2, eff_value)

        # test: result state with primitive and delegated values of -1.0 SHOULD result in eff. value of -2.0
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('5'))
        self.assertEqual(-1.0 * discount_factor - 1.0 * discount_factor ** 2, eff_value)

    def test_effective_state_value_with_zero_discount_factor(self):
        # zero discount factor means that the agent only values the present
        self.dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=0.0, trace_decay=1.0)

        # test: all result states should have zero value due to 0.0 discount factor
        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('2'), result_state=sym_state('1'))
        self.assertEqual(0.0, eff_value)

        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('7'))
        self.assertEqual(0.0, eff_value)

        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('3'))
        self.assertEqual(0.0, eff_value)

        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('9'))
        self.assertEqual(0.0, eff_value)

        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('4'))
        self.assertEqual(0.0, eff_value)

        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('2'))
        self.assertEqual(0.0, eff_value)

        eff_value = self.dv_helper.effective_state_value(selection_state=sym_state('1'), result_state=sym_state('5'))
        self.assertEqual(0.0, eff_value)

    def test_update_no_change_if_no_new_items_in_result_state(self):
        # sanity check: item should have no delegated value initially
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_b))

        # test: items' delegated values should not change if the state elements did not change
        self.dv_helper.update(selection_state=sym_state('A'), result_state=sym_state('A'))

        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_b))

        self.dv_helper.update(selection_state=sym_state('B,C'), result_state=sym_state('B,C'))

        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_b))

    def test_update_item_not_on(self):
        # test: items that were not On in a recent selection state should not have their delegated value updated

        # sanity check: item should have no delegated value initially
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_b))

        self.dv_helper.update(selection_state=sym_state('A'), result_state=sym_state('B,C'))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_b))

        self.dv_helper.update(selection_state=sym_state('B'), result_state=sym_state('1,C'))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_b))

    def test_update_undiscounted_full_trace_decay(self):
        # simplifying this test. discount factor = 1.0 => present and future values equal;
        #                        trace decay = 0.0 => full trace decay between updates;
        #                        learning rate = 1.0 => new value replaces old values
        self.dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=1.0, trace_decay=0.0)
        GlobalParams().set('learning_rate', 1.0)

        # sanity check: item should have no delegated value initially
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_b))

        selection_state = sym_state('1,2')
        result_state = sym_state('B,C')

        eff_value = self.dv_helper.effective_state_value(selection_state=selection_state, result_state=result_state)

        # test: items' delegated values after single update should be the effective state value of result state
        self.dv_helper.update(selection_state=selection_state, result_state=result_state)

        self.assertEqual(eff_value, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(eff_value, self.dv_helper.delegated_value(self.item_b))

        selection_state = sym_state('1,2')
        result_state = sym_state('D,E')

        eff_value = self.dv_helper.effective_state_value(selection_state=selection_state, result_state=result_state)

        # test: delegated values should be replaced by new state's eff. value on 2nd update (w/ learning rate  = 1.0)
        self.dv_helper.update(selection_state=selection_state, result_state=result_state)

        self.assertEqual(eff_value, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(eff_value, self.dv_helper.delegated_value(self.item_b))

    def test_update_discounted_full_trace_decay(self):
        # simplifying this test. trace decay = 0.0 => full trace decay between updates;
        #                        learning rate = 1.0 => new value replaces old values
        self.dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=0.5, trace_decay=0.0)
        GlobalParams().set('learning_rate', 1.0)

        # sanity check: item should have no delegated value initially
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_b))

        selection_state = sym_state('1,2')
        result_state = sym_state('B,C')

        eff_value = self.dv_helper.effective_state_value(selection_state=selection_state, result_state=result_state)

        # test: items' delegated values after single update should be the effective state value of result state
        self.dv_helper.update(selection_state=selection_state, result_state=result_state)

        self.assertEqual(eff_value, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(eff_value, self.dv_helper.delegated_value(self.item_b))

        selection_state = sym_state('1,2')
        result_state = sym_state('D,E')

        eff_value = self.dv_helper.effective_state_value(selection_state=selection_state, result_state=result_state)

        # test: delegated values should be replaced by new state's eff. value on 2nd update (w/ learning rate  = 1.0)
        self.dv_helper.update(selection_state=selection_state, result_state=result_state)

        self.assertEqual(eff_value, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(eff_value, self.dv_helper.delegated_value(self.item_b))

    def test_update_undiscounted_partial_trace_decay(self):
        # simplifying this test. trace decay = 0.5 => full trace decay between updates;
        #                        learning rate = 1.0 => new value replaces old values
        self.dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=1.0, trace_decay=0.5)
        GlobalParams().set('learning_rate', 1.0)

        # sanity check: item should have no delegated value initially
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(0.0, self.dv_helper.delegated_value(self.item_b))

        selection_state = sym_state('1,2')
        result_state = sym_state('B,C')

        eff_value = self.dv_helper.effective_state_value(selection_state=selection_state, result_state=result_state)

        # test: items' delegated values after single update should be the effective state value of result state
        self.dv_helper.update(selection_state=selection_state, result_state=result_state)

        self.assertEqual(eff_value, self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(eff_value, self.dv_helper.delegated_value(self.item_b))

        selection_state = sym_state('1,2')
        result_state = sym_state('D,E')

        eff_value = self.dv_helper.effective_state_value(selection_state=selection_state, result_state=result_state)

        # test: delegated values after second update should be proportional to the trace value
        self.dv_helper.update(selection_state=selection_state, result_state=result_state)

        e_trace = self.dv_helper.eligibility_trace

        indexes = e_trace.indexes([self.item_a, self.item_b])
        trace_values = e_trace.values[indexes]

        self.assertEqual(eff_value * trace_values[0], self.dv_helper.delegated_value(self.item_a))
        self.assertEqual(eff_value * trace_values[1], self.dv_helper.delegated_value(self.item_b))

    def test_update_items_own_value_should_not_be_included(self):
        pass

    @disable_test
    def test_value_blowing_up(self):
        ItemPool().clear()
        self.dv_helper = EligibilityTraceDelegatedValueHelper(discount_factor=1.0, trace_decay=0.2)

        items = [
            self.item_a,
            sym_item('AA', primitive_value=0.0),
            sym_item('AB', primitive_value=0.0),
            sym_item('AC', primitive_value=10.0),
            sym_item('AD', primitive_value=10.0),
            sym_item('AE', primitive_value=10.0),
            sym_item('(AA,AB,AC)'),
            sym_item('(AD,AE)'),
        ]

        trajectory = [
            (items[0].source,),  # A
            (items[1].source,),  # AA
            (items[2].source,),  # AB
            (items[3].source,),  # AC
            (items[5].source,),  # AE
            (items[4].source,),  # AD
            (items[6].source,),  # AA,AB,AC
            (items[7].source,),  # AD,AE
            (items[0].source,),  # A
        ]

        GlobalParams().set('learning_rate', 0.01)

        pairwise_trajectory = list(pairwise(trajectory))
        for n in range(1, 100_000):
            for selection_state, result_state in pairwise_trajectory:
                self.dv_helper.update(selection_state=selection_state, result_state=result_state)
                # print(self.dv_helper.eligibility_trace.keys())

        for item in items:
            print(f'dv [{item}]: {self.dv_helper.delegated_value(item)}')
