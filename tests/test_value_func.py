import itertools
import unittest

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.core import Chain
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ItemPool
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import SymbolicItem
from schema_mechanism.core import calc_delegated_value
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_item_assert
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import habituation_exploratory_value
from schema_mechanism.modules import instrumental_values
from schema_mechanism.modules import pending_focus_values
from schema_mechanism.modules import reliability_values
from schema_mechanism.share import GlobalParams
from schema_mechanism.util import AccumulatingTrace
from test_share.test_classes import MockSchema
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup


class TestPrimitiveValueFunctions(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.i1 = ItemPool().get('A', item_type=MockSymbolicItem, primitive_value=-100.0)
        self.i2 = ItemPool().get('B', item_type=MockSymbolicItem, primitive_value=0.0)
        self.i3 = ItemPool().get('C', item_type=MockSymbolicItem, primitive_value=100.0)

        self.known_items: list[SymbolicItem] = [self.i1, self.i2, self.i3]
        self.known_state_elements = list(itertools.chain.from_iterable([i.state_elements for i in self.known_items]))

    def test_pv_other(self):
        self.assertRaises(TypeError, lambda: calc_primitive_value(set()))
        self.assertRaises(TypeError, lambda: calc_primitive_value(dict()))

    def test_pv_item(self):
        # test: primitive_value(item) should return the primitive value of the item
        for item in self.known_items:
            self.assertEqual(item.primitive_value, calc_primitive_value(item))

    def test_pv_state_element(self):
        # test: primitive_value(state element) should return the primitive value of the corresponding item in pool
        for se in self.known_state_elements:
            item = ReadOnlyItemPool().get(se)
            self.assertEqual(item.primitive_value, calc_primitive_value(se))

        # test: primitive_value(state element) should return a ValueError if item does not exist in pool
        self.assertEqual(0.0, calc_primitive_value('UNK'))

    def test_pv_state(self):
        # test: primitive_value(state) should return the sum of state elements' primitive values
        self.assertEqual(-100.0, calc_primitive_value(sym_state('A')))
        self.assertEqual(0.0, calc_primitive_value(sym_state('B')))
        self.assertEqual(100.0, calc_primitive_value(sym_state('C')))
        self.assertEqual(-100.0, calc_primitive_value(sym_state('A,B')))
        self.assertEqual(0.0, calc_primitive_value(sym_state('A,C')))
        self.assertEqual(100.0, calc_primitive_value(sym_state('B,C')))

        # test: empty state should return 0.0
        self.assertEqual(0.0, calc_primitive_value(sym_state('')))

        # test: state containing only unknown state elements (not in item pool) should return 0.0
        self.assertEqual(0.0, calc_primitive_value(sym_state('UNK,N/A,NOPE,NADA')))

    def test_pv_item_assertion(self):
        # test: primitive_value(item assert) should return the item's primitive value
        self.assertEqual(-100.0, calc_primitive_value(sym_item_assert('A')))
        self.assertEqual(0.0, calc_primitive_value(sym_item_assert('B')))
        self.assertEqual(100.0, calc_primitive_value(sym_item_assert('C')))

        # test: primitive_value(item assert) should return 0.0 for negated assertions
        self.assertEqual(0.0, calc_primitive_value(sym_item_assert('~A')))
        self.assertEqual(0.0, calc_primitive_value(sym_item_assert('~B')))
        self.assertEqual(0.0, calc_primitive_value(sym_item_assert('~C')))

        # test: composite item assertions should function like state assertions
        self.assertEqual(-100.0, calc_primitive_value(sym_item_assert('(A,B)')))
        self.assertEqual(0.0, calc_primitive_value(sym_item_assert('(A,C)')))
        self.assertEqual(100.0, calc_primitive_value(sym_item_assert('(B,C)')))
        self.assertEqual(0.0, calc_primitive_value(sym_item_assert('(~A,B)')))
        self.assertEqual(100.0, calc_primitive_value(sym_item_assert('(~A,C)')))
        self.assertEqual(0.0, calc_primitive_value(sym_item_assert('(B,~C)')))

    def test_pv_state_assertion(self):
        # test: primitive_value(state assert) for non-negated asserts should return the sum of the primitive values
        self.assertEqual(-100.0, calc_primitive_value(sym_state_assert('A')))
        self.assertEqual(0.0, calc_primitive_value(sym_state_assert('B')))
        self.assertEqual(100.0, calc_primitive_value(sym_state_assert('C')))

        self.assertEqual(-100.0, calc_primitive_value(sym_state_assert('A,B')))
        self.assertEqual(0.0, calc_primitive_value(sym_state_assert('A,C')))
        self.assertEqual(100.0, calc_primitive_value(sym_state_assert('B,C')))
        self.assertEqual(0.0, calc_primitive_value(sym_state_assert('~A,B')))
        self.assertEqual(100.0, calc_primitive_value(sym_state_assert('~A,C')))
        self.assertEqual(0.0, calc_primitive_value(sym_state_assert('B,~C')))


class TestDelegatedValueFunctions(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.i1 = ItemPool().get('A', item_type=MockSymbolicItem, avg_accessible_value=-100.0)
        self.i2 = ItemPool().get('B', item_type=MockSymbolicItem, avg_accessible_value=0.0)
        self.i3 = ItemPool().get('C', item_type=MockSymbolicItem, avg_accessible_value=100.0)

        self.known_items: list[SymbolicItem] = [self.i1, self.i2, self.i3]
        self.known_state_elements = list(itertools.chain.from_iterable([i.state_elements for i in self.known_items]))

    def test_dv_other(self):
        self.assertRaises(TypeError, lambda: calc_delegated_value(set()))
        self.assertRaises(TypeError, lambda: calc_delegated_value(dict()))

    def test_dv_item(self):
        # test: delegated_value(item) should return the primitive value of the item
        for item in self.known_items:
            self.assertEqual(item.delegated_value, calc_delegated_value(item))

    def test_dv_state_element(self):
        # test: delegated_value(state element) should return the delegated value of the corresponding item in pool
        for se in self.known_state_elements:
            item = ReadOnlyItemPool().get(se)
            self.assertEqual(item.delegated_value, calc_delegated_value(se))

        # test: delegated_value(state element) should return a ValueError if item does not exist in pool
        self.assertRaises(ValueError, lambda: calc_delegated_value('UNK'))

    def test_dv_state(self):
        # test: delegated_value(state) should return the max of state elements' delegated values
        self.assertEqual(-100.0, calc_delegated_value(sym_state('A')))
        self.assertEqual(0.0, calc_delegated_value(sym_state('B')))
        self.assertEqual(100.0, calc_delegated_value(sym_state('C')))
        self.assertEqual(0.0, calc_delegated_value(sym_state('A,B')))
        self.assertEqual(100.0, calc_delegated_value(sym_state('A,C')))
        self.assertEqual(100.0, calc_delegated_value(sym_state('B,C')))

        # test: empty state should return 0.0
        self.assertEqual(0.0, calc_delegated_value(sym_state('')))

        # test: state containing unknown state elements should return ValueError
        self.assertRaises(ValueError, lambda: calc_delegated_value(sym_state('UNK,N/A,NOPE,NADA')))

    def test_dv_item_assertion(self):
        # test: delegated_value(item assert) should return the item's delegated value
        self.assertEqual(-100.0, calc_delegated_value(sym_item_assert('A')))
        self.assertEqual(0.0, calc_delegated_value(sym_item_assert('B')))
        self.assertEqual(100.0, calc_delegated_value(sym_item_assert('C')))

        # test: delegated_value(item assert) should return 0.0 for negated assertions
        self.assertEqual(0.0, calc_delegated_value(sym_item_assert('~A')))
        self.assertEqual(0.0, calc_delegated_value(sym_item_assert('~B')))
        self.assertEqual(0.0, calc_delegated_value(sym_item_assert('~C')))

        # test: composite item assertions should function like state assertions
        self.assertEqual(0.0, calc_delegated_value(sym_item_assert('(A,B)')))
        self.assertEqual(100.0, calc_delegated_value(sym_item_assert('(A,C)')))
        self.assertEqual(100.0, calc_delegated_value(sym_item_assert('(B,C)')))
        self.assertEqual(0.0, calc_delegated_value(sym_item_assert('(~A,B)')))
        self.assertEqual(100.0, calc_delegated_value(sym_item_assert('(~A,C)')))
        self.assertEqual(0.0, calc_delegated_value(sym_item_assert('(B,~C)')))

    def test_dv_state_assertion(self):
        # test: delegated_value(state assert) for non-negated asserts should return the max of the delegated values
        self.assertEqual(-100.0, calc_delegated_value(sym_state_assert('A')))
        self.assertEqual(0.0, calc_delegated_value(sym_state_assert('B')))
        self.assertEqual(100.0, calc_delegated_value(sym_state_assert('C')))

        self.assertEqual(0.0, calc_delegated_value(sym_state_assert('A,B')))
        self.assertEqual(100.0, calc_delegated_value(sym_state_assert('A,C')))
        self.assertEqual(100.0, calc_delegated_value(sym_state_assert('B,C')))
        self.assertEqual(0.0, calc_delegated_value(sym_state_assert('~A,B')))
        self.assertEqual(100.0, calc_delegated_value(sym_state_assert('~A,C')))
        self.assertEqual(0.0, calc_delegated_value(sym_state_assert('B,~C')))

    def test_dv_with_nonzero_baseline(self):
        GlobalStats().baseline_value = 10.0

        self.assertEqual(0.0 - GlobalStats().baseline_value, calc_delegated_value(sym_state('')))
        self.assertEqual(0.0 - GlobalStats().baseline_value, calc_delegated_value(sym_item_assert('~A')))
        self.assertEqual(100.0 - GlobalStats().baseline_value, calc_delegated_value(sym_item_assert('C')))

        self.assertEqual(0.0 - GlobalStats().baseline_value, calc_delegated_value(sym_state_assert('A,B')))
        self.assertEqual(100.0 - GlobalStats().baseline_value, calc_delegated_value(sym_state_assert('~A,C')))

        for se in self.known_state_elements:
            item = ReadOnlyItemPool().get(se)
            self.assertEqual(calc_delegated_value(item), calc_delegated_value(se))


class TestInstrumentalValues(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_no_schemas(self):
        # test: should return an empty numpy array if no schemas provided and no pending
        result = instrumental_values(schemas=[], pending=None)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.array([])))

    def test_no_pending(self):
        # test: should return a numpy array of zeros with length == 1 if len(schemas) == 1 and no pending
        schemas = [sym_schema('/A1/')]
        result = instrumental_values(schemas=schemas, pending=None)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

        # test: should return a numpy array of zeros with length == 100 if len(schemas) == 100 and no pending
        schemas = [sym_schema(f'/A{i}/') for i in range(100)]
        result = instrumental_values(schemas=schemas, pending=None)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

    # noinspection PyTypeChecker
    def test_pending_but_no_schemas(self):
        # test: should return a ValueError if pending provided but no schemas
        self.assertRaises(ValueError, lambda: instrumental_values(schemas=[], pending=sym_schema('1,/2,3/4,')))
        self.assertRaises(ValueError, lambda: instrumental_values(schemas=None, pending=sym_schema('1,/2,3/4,')))

    def test_non_composite_pending(self):
        # test: should return a ValueError if pending schema has a non-composite action
        schemas = [sym_schema(f'/A{i}/') for i in range(5)]

        non_composite_pending = sym_schema('1,/A2/3,')
        self.assertFalse(non_composite_pending.action.is_composite())  # sanity check

        self.assertRaises(ValueError, lambda: instrumental_values(schemas=schemas, pending=non_composite_pending))

    def test_pending_with_non_positive_goal_state_value(self):
        _i_neg = sym_item('1', primitive_value=-100.0)
        _i_zero = sym_item('2', primitive_value=0.0)
        _i_pos = sym_item('3', primitive_value=100.0)

        schemas = [sym_schema('X,/A1/1,'), sym_schema('1,/A2/2,'), sym_schema('2,/A3/3,')]

        neg_value_goal_state = sym_schema('/1,/')
        neg_value_goal_state.action.controller.update([Chain([sym_schema('X,/A1/1,')])])

        zero_value_goal_state = sym_schema('/2,/')
        zero_value_goal_state.action.controller.update([Chain([sym_schema('1,/A2/2,')])])

        pos_value_goal_state = sym_schema('/3,/')
        pos_value_goal_state.action.controller.update([Chain([sym_schema('2,/A3/3,')])])

        # test: instrumental values should be zeros if pending schema's goal state has zero value (pv + dv)
        result = instrumental_values(schemas=schemas, pending=neg_value_goal_state)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

        result = instrumental_values(schemas=schemas, pending=zero_value_goal_state)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

        # test: instrumental values should be non-zeros if pending schema's goal state has non-zero value (pv + dv)
        #    assumes: (1) cost is less than goal value (2) proximity is not close to zero
        result = instrumental_values(schemas=schemas, pending=pos_value_goal_state)
        self.assertFalse(np.array_equal(result, np.zeros_like(schemas)))

    def test_proximity_scaling(self):
        GlobalParams().set('learning_rate', 1.0)

        _ = [sym_item(str(i), primitive_value=0.0) for i in range(1, 6)]
        goal = sym_item('6', item_type=MockSymbolicItem, primitive_value=100.0, avg_accessible_value=10.0)

        chain = Chain([
            sym_schema('1,/A1/2,', schema_type=MockSchema, cost=0.0, avg_duration=1.0),  # proximity = 1.0/5.0 = 0.2
            sym_schema('2,/A2/3,', schema_type=MockSchema, cost=0.0, avg_duration=1.0),  # proximity = 1.0/4.0 = 0.25
            sym_schema('3,/A3/4,', schema_type=MockSchema, cost=0.0, avg_duration=1.0),  # proximity = 1.0/3.0 = 0.33
            sym_schema('4,/A2/5,', schema_type=MockSchema, cost=0.0, avg_duration=1.0),  # proximity = 1.0/2.0 = 0.5
            sym_schema('5,/A4/6,', schema_type=MockSchema, cost=0.0, avg_duration=1.0),  # proximity = 1.0
        ])

        pending = sym_schema('/6,/')
        pending.action.controller.update([chain])

        non_components = [
            sym_schema('/A1/', schema_type=MockSchema, cost=0.0),
            sym_schema('/A2/', schema_type=MockSchema, cost=0.0),
        ]
        schemas = [*chain, *non_components]

        proximities = [pending.action.controller.proximity(s) for s in chain]

        # test: when cost is zero, instrumental value should be the goal value scaled by a schema's goal proximity
        component_ivs = np.array(proximities, dtype=np.float64) * (
                calc_primitive_value(goal) + calc_delegated_value(goal))
        non_component_ivs = np.zeros_like(non_components, dtype=np.float64)

        expected_result = np.concatenate([component_ivs, non_component_ivs])
        actual_result = instrumental_values(schemas, pending)

        self.assertTrue(np.allclose(expected_result, actual_result))

    def test_cost_deduction(self):
        GlobalParams().set('learning_rate', 1.0)

        _ = [sym_item(str(i), primitive_value=0.0) for i in range(1, 6)]
        goal = sym_item('6', item_type=MockSymbolicItem, primitive_value=100.0, avg_accessible_value=10.0)

        chain = Chain([
            sym_schema('1,/A1/2,', schema_type=MockSchema, cost=10.0, avg_duration=1.0),  # proximity = 1.0/5.0 = 0.2
            sym_schema('2,/A2/3,', schema_type=MockSchema, cost=10.0, avg_duration=1.0),  # proximity = 1.0/4.0 = 0.25
            sym_schema('3,/A3/4,', schema_type=MockSchema, cost=10.0, avg_duration=1.0),  # proximity = 1.0/3.0 = 0.33
            sym_schema('4,/A2/5,', schema_type=MockSchema, cost=10.0, avg_duration=1.0),  # proximity = 1.0/2.0 = 0.5
            sym_schema('5,/A4/6,', schema_type=MockSchema, cost=10.0, avg_duration=1.0),  # proximity = 1.0
        ])

        pending = sym_schema('/6,/')
        pending.action.controller.update([chain])

        non_components = [
            sym_schema('/A1/', schema_type=MockSchema, cost=0.0),
            sym_schema('/A2/', schema_type=MockSchema, cost=0.0),
        ]
        schemas = [*chain, *non_components]

        proximities = [pending.action.controller.proximity(s) for s in chain]
        costs = [pending.action.controller.total_cost(s) for s in chain]

        # test: instrumental value should always be non-negative
        component_ivs = np.maximum(
            np.array(proximities, dtype=np.float64) * (calc_primitive_value(goal) + calc_delegated_value(goal)) - costs,
            np.zeros_like(chain, dtype=np.float64)
        )

        non_component_ivs = np.zeros_like(non_components, dtype=np.float64)

        expected_result = np.concatenate([component_ivs, non_component_ivs])
        actual_result = instrumental_values(schemas, pending)

        self.assertTrue(np.allclose(expected_result, actual_result))


class TestReliabilityValues(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_no_schemas(self):
        # test: should return an empty numpy array if no schemas provided and no pending
        result = reliability_values(schemas=[], pending=None)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.array([])))

    def test_values(self):
        max_penalty = 10.0

        # test: a reliability of 1.0 should result in penalty of 0.0
        schemas = [sym_schema('A,/A1/B,', schema_type=MockSchema, reliability=1.0)]
        rvs = reliability_values(schemas, max_penalty=max_penalty)
        self.assertTrue(np.array_equal(np.zeros_like(schemas), rvs))

        # test: a reliability of 0.0 should result in max penalty
        schemas = [sym_schema('A,/A1/C,', schema_type=MockSchema, reliability=0.0)]
        rvs = reliability_values(schemas, max_penalty=max_penalty)
        self.assertTrue(np.array_equal(-max_penalty * np.ones_like(schemas), rvs))

        # test: a reliability of nan should result in max penalty
        schemas = [sym_schema('A,/A1/D,', schema_type=MockSchema, reliability=np.nan)]
        rvs = reliability_values(schemas, max_penalty=max_penalty)
        self.assertTrue(np.array_equal(-max_penalty * np.ones_like(schemas), rvs))

        # test: a reliability less than 1.0 should result in penalty greater than 0.0
        schemas = [sym_schema('A,/A1/E,', schema_type=MockSchema, reliability=rel)
                   for rel in np.linspace(0.01, 1.0, endpoint=False)]
        rvs = reliability_values(schemas, max_penalty=max_penalty)
        self.assertTrue(all({-max_penalty < rv < 0.0 for rv in rvs}))

        # test: max penalty <= 0.0 should raise a ValueError
        self.assertRaises(ValueError, lambda: reliability_values(schemas, max_penalty=0.0))
        self.assertRaises(ValueError, lambda: reliability_values(schemas, max_penalty=-1.0))


class TestHabituationExploratoryValue(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.actions = [Action(f'A{i}') for i in range(11)]
        self.schemas = [sym_schema(f'/A{i}/') for i in range(11)]

        self.tr: AccumulatingTrace[Action] = AccumulatingTrace()

        # elements 0-4 should have value of 0.25
        self.tr.update(self.actions)

        # element 5 should have value value of 0.75
        self.tr.update(self.actions[5:])

        # elements 6-10 should have value of 1.75
        self.tr.update(self.actions[6:])

        # sanity checks
        expected = 0.25 * np.ones_like(self.actions[0:5])
        actual = self.tr.values[self.tr.indexes(self.actions[0:5])]

        self.assertTrue(np.array_equal(expected, actual))

        expected = np.array([0.75])
        actual = self.tr.values[self.tr.indexes([self.actions[5]])]

        self.assertTrue(np.array_equal(expected, actual))

        expected = 1.75 * np.ones_like(self.actions[6:])
        actual = self.tr.values[self.tr.indexes(self.actions[6:])]

        self.assertTrue(np.array_equal(expected, actual))

    # noinspection PyTypeChecker
    def test_no_trace(self):
        # test: no trace or trace with
        self.assertRaises(ValueError, lambda: habituation_exploratory_value(schemas=[], trace=None))

    def test_empty_schemas_list(self):
        # test: empty schemas list should return empty numpy array
        expected = np.array([])
        actual = habituation_exploratory_value(schemas=[], trace=self.tr)

        self.assertTrue(np.array_equal(expected, actual))

    def test_single_action(self):
        # test: single action should have zero value
        expected = np.zeros(1)
        actual = habituation_exploratory_value(schemas=[self.schemas[0]], trace=self.tr)

        self.assertTrue(np.array_equal(expected, actual))

    def test_multiple_actions_same_value(self):
        # test: multiple actions with same trace value should return zeros

        schemas = self.schemas[:5]

        expected = np.zeros_like(schemas)
        actual = habituation_exploratory_value(schemas=schemas, trace=self.tr)

        self.assertTrue(np.array_equal(expected, actual))

    def test_multiple_actions_different_values_contains_median(self):
        values = habituation_exploratory_value(schemas=self.schemas, trace=self.tr)

        # sanity check
        self.assertTrue(np.median(values) in values)

        # test: actions with trace values equal to median should have zero value
        self.assertEqual(np.zeros(1), values[5])

        # test: actions with trace values below median should have positive value
        self.assertTrue(np.alltrue(values[:5] > 0.0))

        # test: actions with trace values above median should have negative value
        self.assertTrue(np.alltrue(values[6:] < 0.0))

    def test_multiple_actions_different_values_does_not_contain_median(self):
        values = habituation_exploratory_value(schemas=self.schemas[1:], trace=self.tr)

        # sanity check
        self.assertTrue(np.median(values) not in values)

        # test: actions with trace values below median should have positive value
        self.assertTrue(np.alltrue(values[:5] > 0.0))

        # test: actions with trace values above median should have negative value
        self.assertTrue(np.alltrue(values[5:] < 0.0))

    def test_unknown_values(self):
        # test: schemas with unknown actions should raise ValueError
        schemas = [*self.schemas, sym_schema('/UNK/')]
        self.assertRaises(ValueError, lambda: habituation_exploratory_value(schemas=schemas, trace=self.tr))

    def test_multiplier(self):
        mult = 8.0

        values = habituation_exploratory_value(schemas=self.schemas,
                                               trace=self.tr,
                                               multiplier=None)

        values_with_mult = habituation_exploratory_value(schemas=self.schemas,
                                                         trace=self.tr,
                                                         multiplier=mult)

        self.assertTrue(np.array_equal(values * mult, values_with_mult))


class TestPendingFocusValue(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.sa1_a_b = sym_schema('A,/A1/B,')
        self.sa2_b_c = sym_schema('B,/A2/C,')
        self.sa3_c_s1 = sym_schema('C,/A3/S1,')
        self.sa4_c_s2 = sym_schema('C,/A4/S2,')

        # schemas with composite actions
        self.s_s1 = sym_schema('/S1,/')
        self.s_s2 = sym_schema('/S2,/')

        self.s_s1.action.controller.update([
            Chain([self.sa1_a_b, self.sa2_b_c, self.sa3_c_s1])
        ])

        self.s_s2.action.controller.update([
            Chain([self.sa1_a_b, self.sa2_b_c, self.sa4_c_s2])
        ])

        self.schemas = [
            self.sa1_a_b,
            self.sa2_b_c,
            self.sa3_c_s1,
            self.sa4_c_s2,
            self.s_s1,
            self.s_s2,
        ]

    def test_no_schemas(self):
        # test: an empty array should be returned if not schemas supplied
        self.assertTrue(np.array_equal(np.array([]), pending_focus_values(schemas=[])))

    def test_no_pending(self):
        # test: schemas should have no focus value if there is no active pending schema
        expected = np.zeros_like(self.schemas)
        actual = pending_focus_values(schemas=self.schemas, pending=None)

        self.assertTrue(np.array_equal(expected, actual))

    def test_initial_component_values(self):
        max_value = pending_focus_values.max_focus
        pending = self.s_s1
        pending_components = pending.action.controller.components

        values = pending_focus_values(schemas=self.schemas, pending=pending)
        for schema, value in zip(self.schemas, values):
            # test: components of the active pending schema should initially have max focus
            if schema in pending_components:
                self.assertEqual(max_value, value)

            # test: other schemas should have zero focus value
            else:
                self.assertEqual(0.0, value)

    def test_unbounded_reduction_in_value(self):
        pending = self.s_s1
        schemas = pending.action.controller.components

        values = pending_focus_values(schemas=schemas, pending=pending)
        diff = 0.0
        for n in range(1, 20):
            new_values = pending_focus_values(schemas=schemas, pending=pending)
            new_diff = values - new_values

            # test: new values should be strictly less than previous values
            self.assertTrue(np.alltrue(new_values < values))

            # test: the differences between subsequent values should increase
            self.assertTrue(np.alltrue(new_diff > diff))

            values = new_values
            diff = new_diff
