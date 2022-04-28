import itertools
from unittest import TestCase

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.core import Chain
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import Item
from schema_mechanism.core import ItemAssertion
from schema_mechanism.core import ItemPool
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import calc_delegated_value
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_item_assert
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.share import GlobalParams
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.evaluation import EpsilonGreedyEvaluationStrategy
from schema_mechanism.strategies.evaluation import EvaluationStrategy
from schema_mechanism.strategies.evaluation import HabituationEvaluationStrategy
from schema_mechanism.strategies.evaluation import InstrumentalValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import NoOpEvaluationStrategy
from schema_mechanism.strategies.evaluation import PendingFocusEvaluationStrategy
from schema_mechanism.strategies.evaluation import ReliabilityEvaluationStrategy
from schema_mechanism.util import AccumulatingTrace
from test_share.test_classes import MockCompositeItem
from test_share.test_classes import MockSchema
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup


class TestNoOpEvaluationStrategy(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.strategy = NoOpEvaluationStrategy()

        self.schemas = [
            sym_schema(f'C{i},C{i + 1},/A{i}/R{i},R{i + 1},', schema_type=MockSchema) for i in range(10)
        ]

        self.composite_action_schema = sym_schema('C1,/S1,/R1,', schema_type=MockSchema)

        # sanity check
        self.assertTrue(self.composite_action_schema.action.is_composite())

    def test_all_implement_evaluation_strategy_protocol(self):
        # test: strategy should implement the EvaluationStrategy protocol
        self.assertTrue(isinstance(self.strategy, EvaluationStrategy))

    def test_all_return_array_of_correct_length(self):
        # test: strategy should return a numpy array equal in length to the given schemas
        for length in range(len(self.schemas) + 1):
            self.assertIsInstance(self.strategy(schemas=self.schemas[:length], pending=None), np.ndarray)
            self.assertEqual(length, len(self.strategy(schemas=self.schemas[:length], pending=None)))

    def test_call(self):
        # test: no op strategy should return zero for all supplied schemas (check with no pending)
        self.assertTrue(
            np.array_equal(
                np.zeros_like(self.schemas, dtype=np.float64),
                self.strategy(self.schemas, pending=None)))

        # test: no op strategy should return zero for all supplied schemas (check with pending)
        self.assertTrue(
            np.array_equal(
                np.zeros_like(self.schemas, dtype=np.float64),
                self.strategy(self.schemas, pending=self.composite_action_schema)))


class TestPrimitiveValueEvaluationStrategy(TestCase):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        common_test_setup()

        self.i1 = ItemPool().get('A', item_type=MockSymbolicItem, primitive_value=-100.0)
        self.i2 = ItemPool().get('B', item_type=MockSymbolicItem, primitive_value=0.0)
        self.i3 = ItemPool().get('C', item_type=MockSymbolicItem, primitive_value=100.0)

        self.ci1: CompositeItem = sym_item('(A,B)')
        self.ci2: CompositeItem = sym_item('(B,C)')
        self.ci3: CompositeItem = sym_item('(~A,C)')
        self.ci4: CompositeItem = sym_item('(A,~C)')

        self.items: list[Item] = [self.i1, self.i2, self.i3]
        self.composite_items: list[CompositeItem] = [self.ci1, self.ci2, self.ci3, self.ci4]
        self.state_elements = list(itertools.chain.from_iterable([i.state_elements for i in self.items]))

    def test_pv_other(self):
        self.assertRaises(TypeError, lambda: calc_primitive_value(set()))
        self.assertRaises(TypeError, lambda: calc_primitive_value(dict()))

    def test_pv_item(self):
        # test: primitive_value(item) should equal the primitive value associated with a non-composite item
        for item in self.items:
            self.assertEqual(item.primitive_value, calc_primitive_value(item))

    def test_pv_composite_item(self):
        for item in self.composite_items:
            value = calc_primitive_value(item)

            # test: should return the sum of the primitive values of their non-negated item assertions
            expected_value = sum(calc_primitive_value(ia) for ia in item.asserts)
            self.assertEqual(expected_value, value)

            # test: the returned value should also equal CompositeItem.primitive_value
            self.assertEqual(item.primitive_value, value)

    def test_pv_state_element(self):
        # test: primitive_value(state element) should return the primitive value of the corresponding item in pool
        for se in self.state_elements:
            item = ReadOnlyItemPool().get(se)
            self.assertEqual(item.primitive_value, calc_primitive_value(se))

        # test: primitive_value(state element) should return a ValueError if item does not exist in pool
        self.assertEqual(0.0, calc_primitive_value('UNK'))

    def test_pv_state(self):
        s_empty = sym_state('')
        s_none = None

        # test: empty state should have zero primitive value
        self.assertEqual(0.0, calc_primitive_value(s_empty))
        self.assertEqual(0.0, calc_primitive_value(s_none))

        # test: state with unknown state elements should have zero primitive value
        self.assertEqual(0.0, calc_primitive_value(sym_state('UNK,NOPE,NADA')))

        # test: single element states should have that element's corresponding non-composite item's primitive value
        for state_element in 'ABC':
            state = sym_state(state_element)
            item = sym_item(state_element)

            self.assertEqual(item.primitive_value, calc_primitive_value(state))

        # test: primitive_value(state) should return the SUM of state elements' primitive values
        for elements in itertools.combinations('ABC', 2):
            state_str = ','.join(elements)
            state = sym_state(state_str)

            items = [ReadOnlyItemPool().get(e) for e in elements]
            item_sum = sum(item.primitive_value for item in items if item)

            self.assertEqual(item_sum, calc_primitive_value(state),
                             msg=f'Primitive value of items differs from state\'s {state}\'s primitive value!')

    # noinspection PyTypeChecker
    def test_pv_item_assertion(self):
        # test: non-negated, non-composite item assertions should have values equal to the item's primitive value
        for state_element in 'ABC':
            item_assertion = sym_item_assert(state_element)
            item: Item = sym_item(state_element)

            self.assertEqual(item.primitive_value, calc_primitive_value(item_assertion))

        # test: negated, non-composite item assertions should have values equal to 0.0
        for state_element in ('~A', '~B', '~C'):
            item_assertion = sym_item_assert(state_element)

            self.assertEqual(0.0, calc_primitive_value(item_assertion))

        # test: composite item assertions should have values equal to the sum of their non-negated component assertions
        for state_str in ('(A,B)', '(B,C)', '(A,C)', '(~A,B)', '(~B,C)', '(~B,~C)',):
            item_assertion = sym_item_assert(state_str)

            item: CompositeItem = item_assertion.item

            # sum of non-negated component item assertions
            item_sum = sum(ia.item.primitive_value for ia in item.asserts)

            self.assertEqual(
                item_sum, calc_primitive_value(item_assertion),
                msg=f'Primitive value of items differs from item assertion\'s {item_assertion}\'s primitive value!')

    def test_pv_state_assertion(self):
        # test: state assertions about previously unknown state elements should have zero primitive value
        self.assertEqual(0.0, calc_primitive_value(sym_state_assert('UNK,NOPE,NADA')))

        # test: negated state assertions should have zero primitive value
        item_asserts = [sym_item_assert(item_assert_str) for item_assert_str in ('A', 'B', 'C')]
        self.assertEqual(0.0, calc_primitive_value(StateAssertion(asserts=item_asserts, negated=True)))

        # test: non-negated state assertions with all negated item assertions should have zero primitive value
        item_asserts = [sym_item_assert(item_assert_str) for item_assert_str in ('~A', '~B', '~C')]
        self.assertEqual(0.0, calc_primitive_value(StateAssertion(asserts=item_asserts, negated=False)))

        # test: state assertions with multiple item assertions should have primitive values equal to the SUM of their
        #       non-negated item assertions' primitive values
        for state_str in ('A', 'B', 'C', '~A', 'A,B', 'B,C', 'A,C', '~A,B', 'B,~C', '~B,~C'):
            state_assert = sym_state_assert(state_str)

            item_sum = sum(item_assert.item.primitive_value for item_assert in state_assert.asserts)

            self.assertEqual(
                item_sum, calc_primitive_value(state_assert),
                msg=f'Unexpected primitive value for state assertions: {state_assert}!')


class TestDelegatedValueEvaluationStrategy(TestCase):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        common_test_setup()

        self.i1 = ItemPool().get('A', item_type=MockSymbolicItem, delegated_value=-100.0)
        self.i2 = ItemPool().get('B', item_type=MockSymbolicItem, delegated_value=0.0)
        self.i3 = ItemPool().get('C', item_type=MockSymbolicItem, delegated_value=100.0)

        self.ci1: CompositeItem = sym_item('(A,B)', item_type=MockCompositeItem, delegated_value=-100.0)
        self.ci2: CompositeItem = sym_item('(B,C)', item_type=MockCompositeItem, delegated_value=0.0)
        self.ci3: CompositeItem = sym_item('(~A,C)', item_type=MockCompositeItem, delegated_value=-25.0)
        self.ci4: CompositeItem = sym_item('(A,~C)', item_type=MockCompositeItem, delegated_value=50.0)

        self.items: list[Item] = [self.i1, self.i2, self.i3]
        self.composite_items: list[CompositeItem] = [self.ci1, self.ci2, self.ci3, self.ci4]
        self.state_elements = list(itertools.chain.from_iterable([i.state_elements for i in self.items]))

    def test_dv_other(self):
        self.assertRaises(TypeError, lambda: calc_delegated_value(set()))
        self.assertRaises(TypeError, lambda: calc_delegated_value(dict()))

    def test_dv_item(self):
        # test: delegated_value(item) should equal the delegated value associated with a non-composite item
        for item in self.items:
            self.assertEqual(item.delegated_value, calc_delegated_value(item))

    def test_dv_composite_item(self):
        # test: delegated_value(item) should equal the delegated value associated with the composite item
        for item in self.composite_items:
            self.assertEqual(item.delegated_value, calc_delegated_value(item))

    def test_dv_state_element(self):
        # test: delegated_value(state element) should return the delegated value of the corresponding item in pool
        for se in self.state_elements:
            item = ReadOnlyItemPool().get(se)
            self.assertEqual(item.delegated_value, calc_delegated_value(se))

        # test: delegated_value(state element) should return a ValueError if item does not exist in pool
        self.assertEqual(0.0, calc_delegated_value('UNK'))

    def test_dv_state(self):
        s_empty = sym_state('')
        s_none = None

        # test: empty state should have zero delegated value
        self.assertEqual(0.0, calc_delegated_value(s_empty))
        self.assertEqual(0.0, calc_delegated_value(s_none))

        # test: state with unknown state elements should have zero delegated value
        self.assertEqual(0.0, calc_delegated_value(sym_state('UNK,NOPE,NADA')))

        # test: single element states should have that element's corresponding non-composite item's delegated value
        for state_element in 'ABC':
            state = sym_state(state_element)
            item = sym_item(state_element)

            self.assertEqual(item.delegated_value, calc_delegated_value(state))

        # test: delegated_value(state) should return the SUM of state elements' delegated values
        for elements in itertools.combinations('ABC', 2):
            state_str = ','.join(elements)
            state = sym_state(state_str)

            items = [ReadOnlyItemPool().get(e) for e in elements]
            item_sum = sum(item.delegated_value for item in items if item)

            self.assertEqual(item_sum, calc_delegated_value(state),
                             msg=f'delegated value of items differs from state\'s {state}\'s delegated value!')

    def test_dv_item_assertion(self):
        # test: non-negated, non-composite item assertions should have values equal to the item's delegated value
        for state_element in 'ABC':
            item_assertion = sym_item_assert(state_element)
            item: Item = sym_item(state_element)

            self.assertEqual(item.delegated_value, calc_delegated_value(item_assertion))

        # test: negated, non-composite item assertions should have values equal to 0.0
        for state_element in ('~A', '~B', '~C'):
            item_assertion = sym_item_assert(state_element)

            self.assertEqual(0.0, calc_delegated_value(item_assertion))

        # test: non-negated, composite item assertions should have values equal to the item's delegated value
        for item in self.composite_items:
            item_assertion = ItemAssertion(item)

            self.assertEqual(item.delegated_value, calc_delegated_value(item_assertion))

    def test_dv_state_assertion(self):
        # test: state assertions about previously unknown state elements should have zero delegated value
        self.assertEqual(0.0, calc_delegated_value(sym_state_assert('UNK,NOPE,NADA')))

        # test: negated state assertions should have zero delegated value
        item_asserts = [sym_item_assert(item_assert_str) for item_assert_str in ('A', 'B', 'C')]
        self.assertEqual(0.0, calc_delegated_value(StateAssertion(asserts=item_asserts, negated=True)))

        # test: non-negated state assertions with all negated item assertions should have zero delegated value
        item_asserts = [sym_item_assert(item_assert_str) for item_assert_str in ('~A', '~B', '~C')]
        self.assertEqual(0.0, calc_delegated_value(StateAssertion(asserts=item_asserts, negated=False)))

        # test: state assertions with multiple item assertions should have delegated values equal to the SUM of their
        #       non-negated item assertions' delegated values
        for state_str in ('A', 'B', 'C', '~A', 'A,B', 'B,C', 'A,C', '~A,B', 'B,~C', '~B,~C'):
            state_assert = sym_state_assert(state_str)

            item_sum = sum(item_assert.item.delegated_value for item_assert in state_assert.asserts)

            self.assertEqual(
                item_sum, calc_delegated_value(state_assert),
                msg=f'Unexpected delegated value for state assertions: {state_assert}!')


class TestInstrumentalValueEvaluationStrategy(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.instrumental_values = InstrumentalValueEvaluationStrategy()

    def test_no_schemas(self):
        # test: should return an empty numpy array if no schemas provided and no pending
        result = self.instrumental_values(schemas=[], pending=None)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.array([])))

    def test_no_pending(self):
        # test: should return a numpy array of zeros with length == 1 if len(schemas) == 1 and no pending
        schemas = [sym_schema('/A1/')]
        result = self.instrumental_values(schemas=schemas, pending=None)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

        # test: should return a numpy array of zeros with length == 100 if len(schemas) == 100 and no pending
        schemas = [sym_schema(f'/A{i}/') for i in range(100)]
        result = self.instrumental_values(schemas=schemas, pending=None)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

    # noinspection PyTypeChecker
    def test_pending_but_no_schemas(self):
        # test: should return a ValueError if pending provided but no schemas
        self.assertRaises(ValueError, lambda: self.instrumental_values(schemas=[], pending=sym_schema('1,/2,3/4,')))
        self.assertRaises(ValueError, lambda: self.instrumental_values(schemas=None, pending=sym_schema('1,/2,3/4,')))

    def test_non_composite_pending(self):
        # test: should return a ValueError if pending schema has a non-composite action
        schemas = [sym_schema(f'/A{i}/') for i in range(5)]

        non_composite_pending = sym_schema('1,/A2/3,')
        self.assertFalse(non_composite_pending.action.is_composite())  # sanity check

        self.assertRaises(ValueError, lambda: self.instrumental_values(schemas=schemas, pending=non_composite_pending))

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
        result = self.instrumental_values(schemas=schemas, pending=neg_value_goal_state)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

        result = self.instrumental_values(schemas=schemas, pending=zero_value_goal_state)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

        # test: instrumental values should be non-zeros if pending schema's goal state has non-zero value (pv + dv)
        #    assumes: (1) cost is less than goal value (2) proximity is not close to zero
        result = self.instrumental_values(schemas=schemas, pending=pos_value_goal_state)
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
        actual_result = self.instrumental_values(schemas, pending)

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
        actual_result = self.instrumental_values(schemas, pending)

        self.assertTrue(np.allclose(expected_result, actual_result))


class TestPendingFocusEvaluationStrategy(TestCase):
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

        self.pending_focus_values = PendingFocusEvaluationStrategy()

    def test_no_schemas(self):
        # test: an empty array should be returned if not schemas supplied
        self.assertTrue(np.array_equal(np.array([]), self.pending_focus_values(schemas=[])))

    def test_no_pending(self):
        # test: schemas should have no focus value if there is no active pending schema
        expected = np.zeros_like(self.schemas)
        actual = self.pending_focus_values(schemas=self.schemas, pending=None)

        self.assertTrue(np.array_equal(expected, actual))

    def test_initial_component_values(self):
        max_value = self.pending_focus_values.max_value
        pending = self.s_s1
        pending_components = pending.action.controller.components

        values = self.pending_focus_values(schemas=self.schemas, pending=pending)
        for schema, value in zip(self.schemas, values):
            # test: components of the active pending schema should initially have max focus
            if schema in pending_components:
                self.assertEqual(max_value, value)

            # test: other schemas should have zero focus value
            else:
                self.assertEqual(0.0, value)

    def test_unbounded_reduction_in_value(self):
        pending = self.s_s1
        schemas = list(pending.action.controller.components)

        values = self.pending_focus_values(schemas=schemas, pending=pending)
        diff = 0.0
        for n in range(1, 20):
            new_values = self.pending_focus_values(schemas=schemas, pending=pending)
            new_diff = values - new_values

            # test: new values should be strictly less than previous values
            self.assertTrue(np.alltrue(new_values < values))

            # test: the differences between subsequent values should increase
            self.assertTrue(np.alltrue(new_diff > diff))

            values = new_values
            diff = new_diff


class TestReliabilityEvaluationStrategy(TestCase):

    def setUp(self) -> None:
        common_test_setup()

        self.reliability_values = ReliabilityEvaluationStrategy()

    def test_no_schemas(self):
        # test: should return an empty numpy array if no schemas provided and no pending
        result = self.reliability_values(schemas=[], pending=None)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.array([])))

    def test_values(self):
        max_penalty = 1.0
        self.reliability_values.max_penalty = max_penalty

        # test: a reliability of 1.0 should result in penalty of 0.0
        schemas = [sym_schema('A,/A1/B,', schema_type=MockSchema, reliability=1.0)]
        rvs = self.reliability_values(schemas, max_penalty=max_penalty)
        self.assertTrue(np.array_equal(np.zeros_like(schemas), rvs))

        # test: a reliability of 0.0 should result in max penalty
        schemas = [sym_schema('A,/A1/C,', schema_type=MockSchema, reliability=0.0)]
        rvs = self.reliability_values(schemas, max_penalty=max_penalty)
        self.assertTrue(np.array_equal(-max_penalty * np.ones_like(schemas), rvs))

        # test: a reliability of nan should result in max penalty
        schemas = [sym_schema('A,/A1/D,', schema_type=MockSchema, reliability=np.nan)]
        rvs = self.reliability_values(schemas, max_penalty=max_penalty)
        self.assertTrue(np.array_equal(-max_penalty * np.ones_like(schemas), rvs))

        # test: a reliability less than 1.0 should result in penalty greater than 0.0
        schemas = [sym_schema('A,/A1/E,', schema_type=MockSchema, reliability=rel)
                   for rel in np.linspace(0.01, 1.0, endpoint=False)]
        rvs = self.reliability_values(schemas, max_penalty=max_penalty)
        self.assertTrue(all({-max_penalty < rv < 0.0 for rv in rvs}))

    def test_max_penalty(self):

        # test: max penalty values > 0.0 should be accepted
        try:
            # test via property setter
            self.reliability_values.max_penalty = 0.0001
            self.reliability_values.max_penalty = 1.0

            # test via initializer argument
            _ = ReliabilityEvaluationStrategy(max_penalty=0.0001)
            _ = ReliabilityEvaluationStrategy(max_penalty=1.0)
        except ValueError as e:
            self.fail(e)

        # test: max penalty <= 0.0 should raise a ValueError
        with self.assertRaises(ValueError):

            # test via property setter
            self.reliability_values.max_penalty = 0.0
            self.reliability_values.max_penalty = -1.0

            # test via initializer argument
            _ = ReliabilityEvaluationStrategy(max_penalty=0.0)
            _ = ReliabilityEvaluationStrategy(max_penalty=-1.0)


class TestEpsilonGreedyEvaluationStrategy(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.eps_always_explore = EpsilonGreedyEvaluationStrategy(epsilon=1.0)
        self.eps_never_explore = EpsilonGreedyEvaluationStrategy(epsilon=0.0)
        self.eps_even_chance = EpsilonGreedyEvaluationStrategy(epsilon=0.5)

        self.schema = sym_schema('1,2/A/3,4')

        # schema with a composite action
        self.ca_schema = sym_schema('1,2/S1,/5,7')

        self.schemas = [self.schema,
                        self.ca_schema,
                        sym_schema('1,2/A/3,5'),
                        sym_schema('1,2/A/3,6')]

    def test_init(self):
        epsilon = 0.1
        eps_greedy = EpsilonGreedyEvaluationStrategy(epsilon)
        self.assertEqual(epsilon, eps_greedy.epsilon)

    def test_epsilon_setter(self):
        eps_greedy = EpsilonGreedyEvaluationStrategy(0.5)

        eps_greedy.epsilon = 0.0
        self.assertEqual(0.0, eps_greedy.epsilon)

        eps_greedy.epsilon = 1.0
        self.assertEqual(1.0, eps_greedy.epsilon)

        try:
            eps_greedy.epsilon = -1.00001
            eps_greedy.epsilon = 1.00001
            self.fail('ValueError expected on illegal assignment')
        except ValueError:
            pass

    def test_call(self):
        # test: should return empty np.array given empty ndarray
        values = self.eps_never_explore([])
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(0, len(values))

        values = self.eps_always_explore([])
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(0, len(values))

        # test: non-empty schema array should return an np.array with a single np.inf on exploratory choice
        values = self.eps_always_explore([self.schema])
        self.assertEqual(1, len(values))
        self.assertEqual(1, np.count_nonzero(values == np.inf))
        self.assertIsInstance(values, np.ndarray)

        values = self.eps_always_explore(self.schemas)
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(len(self.schemas), len(values))
        self.assertEqual(1, np.count_nonzero(values == np.inf))

        # test: non-empty schema array should return an np.array with same length of zeros non-exploratory choice
        values = self.eps_never_explore([self.schema])
        self.assertIsInstance(values, np.ndarray)
        self.assertListEqual(list(np.array([0])), list(values))

        values = self.eps_never_explore(self.schemas)
        self.assertIsInstance(values, np.ndarray)
        self.assertListEqual(list(np.zeros_like(self.schemas)), list(values))

    def test_epsilon_decay(self):
        epsilon = 1.0
        rate = 0.99

        decay_strategy = GeometricDecayStrategy(rate=rate)
        eps_greedy = EpsilonGreedyEvaluationStrategy(epsilon=epsilon, decay_strategy=decay_strategy)

        prev_epsilon = epsilon
        for _ in range(100):
            _ = eps_greedy(schemas=self.schemas)
            self.assertEqual(decay_strategy.decay(prev_epsilon), eps_greedy.epsilon)
            prev_epsilon = eps_greedy.epsilon

    def test_epsilon_decay_to_minimum(self):
        epsilon = 1.0
        minimum = 0.5

        decay_strategy = GeometricDecayStrategy(rate=0.99, minimum=minimum)
        eps_greedy = EpsilonGreedyEvaluationStrategy(epsilon=epsilon, decay_strategy=decay_strategy)

        for _ in range(100):
            _ = eps_greedy(schemas=self.schemas)

        self.assertEqual(minimum, eps_greedy.epsilon)

    def test_pending_bypass(self):
        # test: epsilon greedy values SHOULD all be zero when a pending schema is provided
        expected = np.zeros_like(self.schemas)
        actual = self.eps_always_explore(self.schemas, pending=self.schema)

        self.assertTrue(np.array_equal(expected, actual))


class TestHabituationEvaluationStrategy(TestCase):
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

        self.habituation_strategy = HabituationEvaluationStrategy(trace=self.tr)

    def test_no_trace(self):
        # test: no trace or trace with
        self.assertRaises(ValueError, lambda: HabituationEvaluationStrategy(trace=None, multiplier=1.0))

    def test_empty_schemas_list(self):
        # test: empty schemas list should return empty numpy array
        expected = np.array([])
        actual = self.habituation_strategy(schemas=[])

        self.assertTrue(np.array_equal(expected, actual))

    def test_single_action(self):
        # test: single action should have zero value
        expected = np.zeros(1)
        actual = self.habituation_strategy(schemas=[self.schemas[0]])

        self.assertTrue(np.array_equal(expected, actual))

    def test_multiple_actions_same_value(self):
        # test: multiple actions with same trace value should return zeros

        schemas = self.schemas[:5]

        expected = np.zeros_like(schemas)
        actual = self.habituation_strategy(schemas=schemas)

        self.assertTrue(np.array_equal(expected, actual))

    def test_multiple_actions_different_values_contains_median(self):
        values = self.habituation_strategy(schemas=self.schemas)

        # sanity check
        self.assertTrue(np.median(values) in values)

        # test: actions with trace values equal to median should have zero value
        self.assertEqual(np.zeros(1), values[5])

        # test: actions with trace values below median should have positive value
        self.assertTrue(np.alltrue(values[:5] > 0.0))

        # test: actions with trace values above median should have negative value
        self.assertTrue(np.alltrue(values[6:] < 0.0))

    def test_multiple_actions_different_values_does_not_contain_median(self):
        values = self.habituation_strategy(schemas=self.schemas[1:])

        # sanity check
        self.assertTrue(np.median(values) not in values)

        # test: actions with trace values below median should have positive value
        self.assertTrue(np.alltrue(values[:5] > 0.0))

        # test: actions with trace values above median should have negative value
        self.assertTrue(np.alltrue(values[5:] < 0.0))

    def test_unknown_values(self):
        # test: schemas with unknown actions should raise ValueError
        schemas = [*self.schemas, sym_schema('/UNK/')]
        self.assertRaises(ValueError, lambda: self.habituation_strategy(schemas=schemas))

    def test_multiplier(self):
        multiplier = 8.0

        habituation_with_default_multiplier = HabituationEvaluationStrategy(trace=self.tr)
        habituation_with_non_default_multiplier = HabituationEvaluationStrategy(trace=self.tr, multiplier=multiplier)

        values = habituation_with_default_multiplier(schemas=self.schemas)
        values_with_mult = habituation_with_non_default_multiplier(schemas=self.schemas)

        self.assertTrue(np.array_equal(values * multiplier, values_with_mult))


class TestCompositeEvaluationStrategy(TestCase):
    def setUp(self) -> None:
        common_test_setup()


class TestDefaultExploratoryEvaluationStrategy(TestCase):
    def setUp(self) -> None:
        common_test_setup()


class TestDefaultGoalPursuitEvaluationStrategy(TestCase):
    def setUp(self) -> None:
        common_test_setup()
