import itertools
import unittest

import numpy as np

from schema_mechanism.core import Chain
from schema_mechanism.core import ItemPool
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import SymbolicItem
from schema_mechanism.core import delegated_value
from schema_mechanism.core import primitive_value
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_item_assert
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import instrumental_values
from schema_mechanism.share import GlobalParams
from test_share.test_classes import MockCompositeItem
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
        self.assertRaises(TypeError, lambda: primitive_value(set()))
        self.assertRaises(TypeError, lambda: primitive_value(dict()))

    def test_pv_item(self):
        # test: primitive_value(item) should return the primitive value of the item
        for item in self.known_items:
            self.assertEqual(item.primitive_value, primitive_value(item))

    def test_pv_state_element(self):
        # test: primitive_value(state element) should return the primitive value of the corresponding item in pool
        for se in self.known_state_elements:
            item = ReadOnlyItemPool().get(se)
            self.assertEqual(item.primitive_value, primitive_value(se))

        # test: primitive_value(state element) should return a ValueError if item does not exist in pool
        self.assertEqual(0.0, primitive_value('UNK'))

    def test_pv_state(self):
        # test: primitive_value(state) should return the sum of state elements' primitive values
        self.assertEqual(-100.0, primitive_value(sym_state('A')))
        self.assertEqual(0.0, primitive_value(sym_state('B')))
        self.assertEqual(100.0, primitive_value(sym_state('C')))
        self.assertEqual(-100.0, primitive_value(sym_state('A,B')))
        self.assertEqual(0.0, primitive_value(sym_state('A,C')))
        self.assertEqual(100.0, primitive_value(sym_state('B,C')))

        # test: empty state should return 0.0
        self.assertEqual(0.0, primitive_value(sym_state('')))

        # test: state containing only unknown state elements (not in item pool) should return 0.0
        self.assertEqual(0.0, primitive_value(sym_state('UNK,N/A,NOPE,NADA')))

    def test_pv_item_assertion(self):
        # test: primitive_value(item assert) should return the item's primitive value
        self.assertEqual(-100.0, primitive_value(sym_item_assert('A')))
        self.assertEqual(0.0, primitive_value(sym_item_assert('B')))
        self.assertEqual(100.0, primitive_value(sym_item_assert('C')))

        # test: primitive_value(item assert) should return 0.0 for negated assertions
        self.assertEqual(0.0, primitive_value(sym_item_assert('~A')))
        self.assertEqual(0.0, primitive_value(sym_item_assert('~B')))
        self.assertEqual(0.0, primitive_value(sym_item_assert('~C')))

        # test: composite item assertions should function like state assertions
        self.assertEqual(-100.0, primitive_value(sym_item_assert('(A,B)')))
        self.assertEqual(0.0, primitive_value(sym_item_assert('(A,C)')))
        self.assertEqual(100.0, primitive_value(sym_item_assert('(B,C)')))
        self.assertEqual(0.0, primitive_value(sym_item_assert('(~A,B)')))
        self.assertEqual(100.0, primitive_value(sym_item_assert('(~A,C)')))
        self.assertEqual(0.0, primitive_value(sym_item_assert('(B,~C)')))

    def test_pv_state_assertion(self):
        # test: primitive_value(state assert) for non-negated asserts should return the sum of the primitive values
        self.assertEqual(-100.0, primitive_value(sym_state_assert('A')))
        self.assertEqual(0.0, primitive_value(sym_state_assert('B')))
        self.assertEqual(100.0, primitive_value(sym_state_assert('C')))

        self.assertEqual(-100.0, primitive_value(sym_state_assert('A,B')))
        self.assertEqual(0.0, primitive_value(sym_state_assert('A,C')))
        self.assertEqual(100.0, primitive_value(sym_state_assert('B,C')))
        self.assertEqual(0.0, primitive_value(sym_state_assert('~A,B')))
        self.assertEqual(100.0, primitive_value(sym_state_assert('~A,C')))
        self.assertEqual(0.0, primitive_value(sym_state_assert('B,~C')))


class TestDelegatedValueFunctions(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.i1 = ItemPool().get('A', item_type=MockSymbolicItem, avg_accessible_value=-100.0)
        self.i2 = ItemPool().get('B', item_type=MockSymbolicItem, avg_accessible_value=0.0)
        self.i3 = ItemPool().get('C', item_type=MockSymbolicItem, avg_accessible_value=100.0)

        self.known_items: list[SymbolicItem] = [self.i1, self.i2, self.i3]
        self.known_state_elements = list(itertools.chain.from_iterable([i.state_elements for i in self.known_items]))

    def test_dv_other(self):
        self.assertRaises(TypeError, lambda: delegated_value(set()))
        self.assertRaises(TypeError, lambda: delegated_value(dict()))

    def test_dv_item(self):
        # test: delegated_value(item) should return the primitive value of the item
        for item in self.known_items:
            self.assertEqual(item.delegated_value, delegated_value(item))

    def test_dv_state_element(self):
        # test: delegated_value(state element) should return the primitive value of the corresponding item in pool
        for se in self.known_state_elements:
            item = ReadOnlyItemPool().get(se)
            self.assertEqual(item.delegated_value, delegated_value(se))

        # test: delegated_value(state element) should return a ValueError if item does not exist in pool
        self.assertEqual(0.0, delegated_value('UNK'))

    # def test_dv_state(self):
    #     # test: delegated_value(state) should return the sum of state elements' primitive values
    #     self.assertEqual(-100.0, delegated_value(sym_state('A')))
    #     self.assertEqual(0.0, delegated_value(sym_state('B')))
    #     self.assertEqual(100.0, delegated_value(sym_state('C')))
    #     self.assertEqual(-100.0, delegated_value(sym_state('A,B')))
    #     self.assertEqual(0.0, delegated_value(sym_state('A,C')))
    #     self.assertEqual(100.0, delegated_value(sym_state('B,C')))
    #
    #     # test: empty state should return 0.0
    #     self.assertEqual(0.0, delegated_value(sym_state('')))
    #
    #     # test: state containing only unknown state elements (not in item pool) should return 0.0
    #     self.assertEqual(0.0, delegated_value(sym_state('UNK,N/A,NOPE,NADA')))

    def test_dv_item_assertion(self):
        # test: delegated_value(item assert) should return the item's primitive value
        self.assertEqual(-100.0, delegated_value(sym_item_assert('A')))
        self.assertEqual(0.0, delegated_value(sym_item_assert('B')))
        self.assertEqual(100.0, delegated_value(sym_item_assert('C')))

        # test: delegated_value(item assert) should return 0.0 for negated assertions
        self.assertEqual(0.0, delegated_value(sym_item_assert('~A')))
        self.assertEqual(0.0, delegated_value(sym_item_assert('~B')))
        self.assertEqual(0.0, delegated_value(sym_item_assert('~C')))

        # # test: composite item assertions should function like state assertions
        # self.assertEqual(-100.0, delegated_value(sym_item_assert('(A,B)')))
        # self.assertEqual(0.0, delegated_value(sym_item_assert('(A,C)')))
        # self.assertEqual(100.0, delegated_value(sym_item_assert('(B,C)')))
        # self.assertEqual(0.0, delegated_value(sym_item_assert('(~A,B)')))
        # self.assertEqual(100.0, delegated_value(sym_item_assert('(~A,C)')))
        # self.assertEqual(0.0, delegated_value(sym_item_assert('(B,~C)')))

    # def test_dv_state_assertion(self):
    #     # test: delegated_value(state assert) for non-negated asserts should return the sum of the primitive values
    #     self.assertEqual(-100.0, delegated_value(sym_state_assert('A')))
    #     self.assertEqual(0.0, delegated_value(sym_state_assert('B')))
    #     self.assertEqual(100.0, delegated_value(sym_state_assert('C')))
    #
    #     self.assertEqual(-100.0, delegated_value(sym_state_assert('A,B')))
    #     self.assertEqual(0.0, delegated_value(sym_state_assert('A,C')))
    #     self.assertEqual(100.0, delegated_value(sym_state_assert('B,C')))
    #     self.assertEqual(0.0, delegated_value(sym_state_assert('~A,B')))
    #     self.assertEqual(100.0, delegated_value(sym_state_assert('~A,C')))
    #     self.assertEqual(0.0, delegated_value(sym_state_assert('B,~C')))

    # def test_dv_with_nonzero_baseline(self):
    #     GlobalStats.baseline_value = 10.0
    #
    #     self.assertEqual(0.0, delegated_value(sym_state('')))
    #
    #     self.assertEqual(0.0, delegated_value(sym_item_assert('~A')))
    #     self.assertEqual(90.0, delegated_value(sym_item_assert('C')))
    #
    #     self.assertEqual(-120.0, delegated_value(sym_state_assert('A,B')))
    #     self.assertEqual(90.0, delegated_value(sym_state_assert('~A,C')))
    #
    #     for se in self.known_state_elements:
    #         item = ReadOnlyItemPool().get(se)
    #         self.assertEqual(item.delegated_value, delegated_value(se))


class TestInstrumentalValues(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        GlobalParams().set('item_type', MockSymbolicItem)
        GlobalParams().set('composite_item_type', MockCompositeItem)
        GlobalParams().set('schema_type', MockSchema)

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
        i_neg = sym_item('1', primitive_value=-100.0)
        i_zero = sym_item('2', primitive_value=0.0)
        i_pos = sym_item('3', primitive_value=100.0)

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
        goal = sym_item('6', primitive_value=100.0, avg_accessible_value=10.0)

        chain = Chain([
            sym_schema('1,/A1/2,', cost=0.0, avg_duration=1.0),  # proximity = 1.0/5.0 = 0.2
            sym_schema('2,/A2/3,', cost=0.0, avg_duration=1.0),  # proximity = 1.0/4.0 = 0.25
            sym_schema('3,/A3/4,', cost=0.0, avg_duration=1.0),  # proximity = 1.0/3.0 = 0.333333
            sym_schema('4,/A2/5,', cost=0.0, avg_duration=1.0),  # proximity = 1.0/2.0 = 0.5
            sym_schema('5,/A4/6,', cost=0.0, avg_duration=1.0),  # proximity = 1.0
        ])

        pending = sym_schema('/6,/')
        pending.action.controller.update([chain])

        non_components = [
            sym_schema('/A1/', cost=0.0),
            sym_schema('/A2/', cost=0.0),
        ]
        schemas = [*chain, *non_components]

        proximities = [pending.action.controller.proximity(s) for s in chain]

        # test: when cost is zero, instrumental value should be the goal value scaled by a schema's goal proximity
        component_ivs = np.array(proximities, dtype=np.float64) * (primitive_value(goal) + delegated_value(goal))
        non_component_ivs = np.zeros_like(non_components, dtype=np.float64)

        expected_result = np.concatenate([component_ivs, non_component_ivs])
        actual_result = instrumental_values(schemas, pending)

        self.assertTrue(np.allclose(expected_result, actual_result))

    def test_cost_deduction(self):
        GlobalParams().set('learning_rate', 1.0)

        _ = [sym_item(str(i), primitive_value=0.0) for i in range(1, 6)]
        goal = sym_item('6', primitive_value=100.0, avg_accessible_value=10.0)

        chain = Chain([
            sym_schema('1,/A1/2,', cost=10.0, avg_duration=1.0),  # proximity = 1.0/5.0 = 0.2
            sym_schema('2,/A2/3,', cost=10.0, avg_duration=1.0),  # proximity = 1.0/4.0 = 0.25
            sym_schema('3,/A3/4,', cost=10.0, avg_duration=1.0),  # proximity = 1.0/3.0 = 0.333333
            sym_schema('4,/A2/5,', cost=10.0, avg_duration=1.0),  # proximity = 1.0/2.0 = 0.5
            sym_schema('5,/A4/6,', cost=10.0, avg_duration=1.0),  # proximity = 1.0
        ])

        pending = sym_schema('/6,/')
        pending.action.controller.update([chain])

        non_components = [
            sym_schema('/A1/', cost=0.0),
            sym_schema('/A2/', cost=0.0),
        ]
        schemas = [*chain, *non_components]

        proximities = [pending.action.controller.proximity(s) for s in chain]
        costs = [pending.action.controller.total_cost(s) for s in chain]

        # test: instrumental value should always be non-negative
        component_ivs = np.maximum(
            np.array(proximities, dtype=np.float64) * (primitive_value(goal) + delegated_value(goal)) - costs,
            np.zeros_like(chain, dtype=np.float64)
        )

        non_component_ivs = np.zeros_like(non_components, dtype=np.float64)

        expected_result = np.concatenate([component_ivs, non_component_ivs])
        actual_result = instrumental_values(schemas, pending)

        self.assertTrue(np.allclose(expected_result, actual_result))
