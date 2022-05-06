import itertools
from typing import Sequence
from unittest import TestCase
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import call

import numpy as np

from schema_mechanism.core import Chain
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import Item
from schema_mechanism.core import Schema
from schema_mechanism.core import calc_delegated_value
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.share import GlobalParams
from schema_mechanism.strategies.evaluation import EvaluationStrategy
from schema_mechanism.strategies.evaluation import InstrumentalValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import MaxDelegatedValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import NoOpEvaluationStrategy
from schema_mechanism.strategies.evaluation import TotalDelegatedValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import TotalPrimitiveValueEvaluationStrategy
from test_share.test_classes import MockCompositeItem
from test_share.test_classes import MockSchema
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup


class TestCommon(TestCase):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        common_test_setup()

        self.item_a = sym_item('A', item_type=MockSymbolicItem, primitive_value=-100.5, delegated_value=0.0)
        self.item_b = sym_item('B', item_type=MockSymbolicItem, primitive_value=0.0, delegated_value=25.65)
        self.item_c = sym_item('C', item_type=MockSymbolicItem, primitive_value=101.27, delegated_value=-75.8)

        # default primitive values for CompositeItems is the sum of the primitive values over their state elements;
        # however, this can be overridden with an initializer argument.
        self.composite_item_ab: CompositeItem = (
            sym_item('(A,B)', item_type=MockCompositeItem, delegated_value=0.0)
        )
        self.composite_item_bc: CompositeItem = (
            sym_item('(B,C)', item_type=MockCompositeItem, primitive_value=25.1, delegated_value=-20.1)
        )
        self.composite_item_abc: CompositeItem = (
            sym_item('(A,B,C)', item_type=MockCompositeItem, delegated_value=55.1)
        )

        # sanity check: primitive values should have been set as expected
        self.assertAlmostEqual(-100.5, self.item_a.primitive_value)
        self.assertAlmostEqual(0.0, self.item_b.primitive_value)
        self.assertAlmostEqual(101.27, self.item_c.primitive_value)
        self.assertAlmostEqual(-100.5, self.composite_item_ab.primitive_value)
        self.assertAlmostEqual(25.1, self.composite_item_bc.primitive_value)
        self.assertAlmostEqual(0.77, self.composite_item_abc.primitive_value)

        # sanity check: delegated values should have been set as expected
        self.assertAlmostEqual(0.0, self.item_a.delegated_value)
        self.assertAlmostEqual(25.65, self.item_b.delegated_value)
        self.assertAlmostEqual(-75.8, self.item_c.delegated_value)
        self.assertAlmostEqual(0.0, self.composite_item_ab.delegated_value)
        self.assertAlmostEqual(-20.1, self.composite_item_bc.delegated_value)
        self.assertAlmostEqual(55.1, self.composite_item_abc.delegated_value)

        self.simple_result_schemas = [
            sym_schema('X,/A1/A,'),
            sym_schema('Y,/A1/B,'),
            sym_schema('Z,/A1/C,'),
        ]

        self.composite_result_schemas = [
            sym_schema('X,/A1/(A,B),'),
            sym_schema('Y,/A1/(B,C),'),
            sym_schema('Z,/A1/(A,B,C),'),
        ]

        self.composite_action_schema = sym_schema('X,/X,Y/A,B')

        # sanity check: composite action schema should have a composite action
        self.assertTrue(self.composite_action_schema.action.is_composite())

        self.schemas = [
            *self.simple_result_schemas,
            *self.composite_result_schemas,
            self.composite_action_schema
        ]

        self.simple_items: list[Item] = [self.item_a, self.item_b, self.item_c]

        self.composite_items: list[CompositeItem] = [
            self.composite_item_ab,
            self.composite_item_bc,
            self.composite_item_abc,
        ]
        self.state_elements = list(itertools.chain.from_iterable([i.state_elements for i in self.simple_items]))

    def assert_implements_evaluation_strategy_protocol(self, strategy: EvaluationStrategy):
        # test: strategy should implement the EvaluationStrategy protocol
        self.assertTrue(isinstance(strategy, EvaluationStrategy))

    def assert_returns_array_of_correct_length(self, strategy: EvaluationStrategy):
        # test: strategy should return a numpy array equal in length to the given schemas
        for length in range(len(self.schemas) + 1):
            self.assertIsInstance(strategy(schemas=self.schemas[:length], pending=None), np.ndarray)
            self.assertEqual(length, len(strategy(schemas=self.schemas[:length], pending=None)))

    def assert_correctly_invokes_post_process_callables(self, strategy: EvaluationStrategy):
        values = strategy(schemas=self.schemas, pending=self.composite_action_schema)

        # mocking post-process callables
        mock = MagicMock()
        mock.post_process_1 = MagicMock()
        mock.post_process_1.return_value = values
        mock.post_process_2 = MagicMock()
        mock.post_process_2.return_value = values

        post_process_list = [mock.post_process_1, mock.post_process_2]
        _ = strategy(
            schemas=self.schemas,
            pending=None,
            post_process=post_process_list
        )

        # test: post-process callables should be invoked in order and called with the correct arguments
        mock.assert_has_calls([call.post_process_1(schemas=ANY, values=ANY),
                               call.post_process_2(schemas=ANY, values=ANY)])

        # test: strategy should invoke post-process methods with the correct arguments
        for mock in post_process_list:
            self.assertListEqual(self.schemas, mock.call_args.kwargs['schemas'])
            np.testing.assert_array_equal(values, mock.call_args.kwargs['values'])

        # noinspection PyUnusedLocal, PyShadowingNames
        # test: post-process callables should be capable of changing the values returned from strategy
        def add_one_post_process(schemas: Sequence[Schema], values: np.ndarray) -> np.ndarray:
            values += 1.0
            return values

        expected_values = add_one_post_process(self.schemas, values)
        actual_values = strategy(schemas=self.schemas, pending=None, post_process=[add_one_post_process])

        self.assertTrue(np.array_equal(expected_values, actual_values))

    def assert_values_consistent_with_call(self, strategy: EvaluationStrategy) -> None:
        np.testing.assert_array_equal(
            strategy(self.schemas, pending=None),
            strategy.values(self.schemas, pending=None)
        )
        np.testing.assert_array_equal(
            strategy(self.schemas, pending=self.composite_action_schema),
            strategy.values(self.schemas, pending=self.composite_action_schema)
        )

    def assert_all_common_functionality(self, strategy: EvaluationStrategy) -> None:
        self.assert_implements_evaluation_strategy_protocol(strategy)
        self.assert_returns_array_of_correct_length(strategy)
        self.assert_correctly_invokes_post_process_callables(strategy)
        self.assert_values_consistent_with_call(strategy)


class TestNoOpEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.strategy = NoOpEvaluationStrategy()

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_values(self):
        # test: strategy should return zero for all supplied schemas (check with no pending)
        self.assertTrue(
            np.array_equal(
                np.zeros_like(self.schemas, dtype=np.float64),
                self.strategy.values(self.schemas, pending=None)))

        # test: strategy should return zero for all supplied schemas (check with pending)
        self.assertTrue(
            np.array_equal(
                np.zeros_like(self.schemas, dtype=np.float64),
                self.strategy.values(self.schemas, pending=self.composite_action_schema)))


class TestTotalPrimitiveValueEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()
        common_test_setup()

        self.strategy = TotalPrimitiveValueEvaluationStrategy()

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_values(self):
        # test: bare schemas should have zero primitive value
        self.assertEqual(np.zeros(shape=1), self.strategy.values(schemas=[sym_schema('/A1/')]))

        # test: schemas with results containing previously unknown state elements should have zero primitive value
        for schema in [sym_schema('A,/A1/UNK,'), sym_schema('A,/A1/UNK,NOPE,NADA')]:
            self.assertEqual(np.zeros(shape=1), self.strategy.values(schemas=[schema]))

        # test: the primitive value of a schema should be the sum over the primitive values of its result's items
        actual_values = self.strategy.values(self.schemas)
        expected_values = (
            np.array([sum(item.primitive_value for item in schema.result.items) for schema in self.schemas])
        )
        np.testing.assert_array_equal(expected_values, actual_values)


class TestTotalDelegatedValueEvaluationStrategy(TestCommon):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        super().setUp()
        common_test_setup()

        self.strategy = TotalDelegatedValueEvaluationStrategy()

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_values(self):
        # test: bare schemas should have zero delegated value
        self.assertEqual(np.zeros(shape=1), self.strategy.values(schemas=[sym_schema('/A1/')]))

        # test: schemas with results containing previously unknown state elements should have zero delegated value
        for schema in [sym_schema('A,/A1/UNK,'), sym_schema('A,/A1/UNK,NOPE,NADA')]:
            self.assertEqual(np.zeros(shape=1), self.strategy.values(schemas=[schema]))

        # test: the delegated value of a schema should be the sum over the delegated values of its result's items
        actual_values = self.strategy.values(self.schemas)
        expected_values = (
            np.array([sum(item.delegated_value for item in schema.result.items) for schema in self.schemas])
        )
        np.testing.assert_array_equal(expected_values, actual_values)


class TestMaxDelegatedValueEvaluationStrategy(TestCommon):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        super().setUp()
        common_test_setup()

        self.strategy = MaxDelegatedValueEvaluationStrategy()

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_values(self):
        # test: bare schemas should have zero delegated value
        self.assertEqual(np.zeros(shape=1), self.strategy.values(schemas=[sym_schema('/A1/')]))

        # test: schemas with results containing previously unknown state elements should have zero delegated value
        for schema in [sym_schema('A,/A1/UNK,'), sym_schema('A,/A1/UNK,NOPE,NADA')]:
            self.assertEqual(np.zeros(shape=1), self.strategy.values(schemas=[schema]))

        # test: the delegated value of a schema should be the sum over the delegated values of its result's items
        actual_values = self.strategy.values(self.schemas)
        expected_values = (
            np.array([
                max([item.delegated_value for item in schema.result.items], default=0.0)
                for schema in self.schemas
            ])
        )
        np.testing.assert_array_equal(expected_values, actual_values)


class TestInstrumentalValueEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()
        common_test_setup()

        self.strategy = InstrumentalValueEvaluationStrategy()

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_no_pending(self):
        # test: should return a numpy array of zeros with length == 1 if len(schemas) == 1 and no pending
        schemas = [sym_schema('/A1/')]
        result = self.strategy(schemas=schemas, pending=None)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

        # test: should return a numpy array of zeros with length == 100 if len(schemas) == 100 and no pending
        schemas = [sym_schema(f'/A{i}/') for i in range(100)]
        result = self.strategy(schemas=schemas, pending=None)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

    # noinspection PyTypeChecker
    def test_pending_but_no_schemas(self):
        # test: should return a ValueError if pending provided but no schemas
        self.assertRaises(ValueError, lambda: self.strategy.values(schemas=[], pending=self.composite_action_schema))
        self.assertRaises(ValueError, lambda: self.strategy.values(schemas=None, pending=self.composite_action_schema))

    def test_non_composite_pending(self):
        # test: should return a ValueError if pending schema has a non-composite action
        non_composite_action_schema = self.simple_result_schemas[0]
        self.assertFalse(non_composite_action_schema.action.is_composite())

        self.assertRaises(
            ValueError,
            lambda: self.strategy.values(schemas=self.schemas, pending=non_composite_action_schema))

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
        result = self.strategy.values(schemas=schemas, pending=neg_value_goal_state)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

        result = self.strategy.values(schemas=schemas, pending=zero_value_goal_state)
        self.assertTrue(np.array_equal(result, np.zeros_like(schemas)))

        # test: instrumental values should be non-zeros if pending schema's goal state has non-zero value (pv + dv)
        #    assumes: (1) cost is less than goal value (2) proximity is not close to zero
        result = self.strategy.values(schemas=schemas, pending=pos_value_goal_state)
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
        actual_result = self.strategy.values(schemas, pending)

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
        actual_result = self.strategy.values(schemas, pending)

        self.assertTrue(np.allclose(expected_result, actual_result))

# class TestPendingFocusEvaluationStrategy(TestCase):
#     def setUp(self) -> None:
#         common_test_setup()
#
#         self.sa1_a_b = sym_schema('A,/A1/B,')
#         self.sa2_b_c = sym_schema('B,/A2/C,')
#         self.sa3_c_s1 = sym_schema('C,/A3/S1,')
#         self.sa4_c_s2 = sym_schema('C,/A4/S2,')
#
#         # schemas with composite actions
#         self.s_s1 = sym_schema('/S1,/')
#         self.s_s2 = sym_schema('/S2,/')
#
#         self.s_s1.action.controller.update([
#             Chain([self.sa1_a_b, self.sa2_b_c, self.sa3_c_s1])
#         ])
#
#         self.s_s2.action.controller.update([
#             Chain([self.sa1_a_b, self.sa2_b_c, self.sa4_c_s2])
#         ])
#
#         self.schemas = [
#             self.sa1_a_b,
#             self.sa2_b_c,
#             self.sa3_c_s1,
#             self.sa4_c_s2,
#             self.s_s1,
#             self.s_s2,
#         ]
#
#         self.pending_focus_values = PendingFocusEvaluationStrategy()
#
#     def test_no_schemas(self):
#         # test: an empty array should be returned if not schemas supplied
#         self.assertTrue(np.array_equal(np.array([]), self.pending_focus_values(schemas=[])))
#
#     def test_no_pending(self):
#         # test: schemas should have no focus value if there is no active pending schema
#         expected = np.zeros_like(self.schemas)
#         actual = self.pending_focus_values(schemas=self.schemas, pending=None)
#
#         self.assertTrue(np.array_equal(expected, actual))
#
#     def test_initial_component_values(self):
#         max_value = self.pending_focus_values.max_value
#         pending = self.s_s1
#         pending_components = pending.action.controller.components
#
#         values = self.pending_focus_values(schemas=self.schemas, pending=pending)
#         for schema, value in zip(self.schemas, values):
#             # test: components of the active pending schema should initially have max focus
#             if schema in pending_components:
#                 self.assertEqual(max_value, value)
#
#             # test: other schemas should have zero focus value
#             else:
#                 self.assertEqual(0.0, value)
#
#     def test_unbounded_reduction_in_value(self):
#         pending = self.s_s1
#         schemas = list(pending.action.controller.components)
#
#         values = self.pending_focus_values(schemas=schemas, pending=pending)
#         diff = 0.0
#         for n in range(1, 20):
#             new_values = self.pending_focus_values(schemas=schemas, pending=pending)
#             new_diff = values - new_values
#
#             # test: new values should be strictly less than previous values
#             self.assertTrue(np.alltrue(new_values < values))
#
#             # test: the differences between subsequent values should increase
#             self.assertTrue(np.alltrue(new_diff > diff))
#
#             values = new_values
#             diff = new_diff
#
#
# class TestReliabilityEvaluationStrategy(TestCase):
#
#     def setUp(self) -> None:
#         common_test_setup()
#
#         self.reliability_values = ReliabilityEvaluationStrategy()
#
#     def test_no_schemas(self):
#         # test: should return an empty numpy array if no schemas provided and no pending
#         result = self.reliability_values(schemas=[], pending=None)
#         self.assertIsInstance(result, np.ndarray)
#         self.assertTrue(np.array_equal(result, np.array([])))
#
#     def test_values(self):
#         max_penalty = 1.0
#         self.reliability_values.max_penalty = max_penalty
#
#         # test: a reliability of 1.0 should result in penalty of 0.0
#         schemas = [sym_schema('A,/A1/B,', schema_type=MockSchema, reliability=1.0)]
#         rvs = self.reliability_values(schemas, max_penalty=max_penalty)
#         self.assertTrue(np.array_equal(np.zeros_like(schemas), rvs))
#
#         # test: a reliability of 0.0 should result in max penalty
#         schemas = [sym_schema('A,/A1/C,', schema_type=MockSchema, reliability=0.0)]
#         rvs = self.reliability_values(schemas, max_penalty=max_penalty)
#         self.assertTrue(np.array_equal(-max_penalty * np.ones_like(schemas), rvs))
#
#         # test: a reliability of nan should result in max penalty
#         schemas = [sym_schema('A,/A1/D,', schema_type=MockSchema, reliability=np.nan)]
#         rvs = self.reliability_values(schemas, max_penalty=max_penalty)
#         self.assertTrue(np.array_equal(-max_penalty * np.ones_like(schemas), rvs))
#
#         # test: a reliability less than 1.0 should result in penalty greater than 0.0
#         schemas = [sym_schema('A,/A1/E,', schema_type=MockSchema, reliability=rel)
#                    for rel in np.linspace(0.01, 1.0, endpoint=False)]
#         rvs = self.reliability_values(schemas, max_penalty=max_penalty)
#         self.assertTrue(all({-max_penalty < rv < 0.0 for rv in rvs}))
#
#     def test_max_penalty(self):
#
#         # test: max penalty values > 0.0 should be accepted
#         try:
#             # test via property setter
#             self.reliability_values.max_penalty = 0.0001
#             self.reliability_values.max_penalty = 1.0
#
#             # test via initializer argument
#             _ = ReliabilityEvaluationStrategy(max_penalty=0.0001)
#             _ = ReliabilityEvaluationStrategy(max_penalty=1.0)
#         except ValueError as e:
#             self.fail(e)
#
#         # test: max penalty <= 0.0 should raise a ValueError
#         with self.assertRaises(ValueError):
#
#             # test via property setter
#             self.reliability_values.max_penalty = 0.0
#             self.reliability_values.max_penalty = -1.0
#
#             # test via initializer argument
#             _ = ReliabilityEvaluationStrategy(max_penalty=0.0)
#             _ = ReliabilityEvaluationStrategy(max_penalty=-1.0)
#
#
# class TestEpsilonGreedyEvaluationStrategy(TestCase):
#     def setUp(self) -> None:
#         common_test_setup()
#
#         self.eps_always_explore = EpsilonGreedyEvaluationStrategy(epsilon=1.0)
#         self.eps_never_explore = EpsilonGreedyEvaluationStrategy(epsilon=0.0)
#         self.eps_even_chance = EpsilonGreedyEvaluationStrategy(epsilon=0.5)
#
#         self.schema = sym_schema('1,2/A/3,4')
#
#         # schema with a composite action
#         self.ca_schema = sym_schema('1,2/S1,/5,7')
#
#         self.schemas = [self.schema,
#                         self.ca_schema,
#                         sym_schema('1,2/A/3,5'),
#                         sym_schema('1,2/A/3,6')]
#
#     def test_init(self):
#         epsilon = 0.1
#         eps_greedy = EpsilonGreedyEvaluationStrategy(epsilon)
#         self.assertEqual(epsilon, eps_greedy.epsilon)
#
#     def test_epsilon_setter(self):
#         eps_greedy = EpsilonGreedyEvaluationStrategy(0.5)
#
#         eps_greedy.epsilon = 0.0
#         self.assertEqual(0.0, eps_greedy.epsilon)
#
#         eps_greedy.epsilon = 1.0
#         self.assertEqual(1.0, eps_greedy.epsilon)
#
#         try:
#             eps_greedy.epsilon = -1.00001
#             eps_greedy.epsilon = 1.00001
#             self.fail('ValueError expected on illegal assignment')
#         except ValueError:
#             pass
#
#     def test_call(self):
#         # test: should return empty np.array given empty ndarray
#         values = self.eps_never_explore([])
#         self.assertIsInstance(values, np.ndarray)
#         self.assertEqual(0, len(values))
#
#         values = self.eps_always_explore([])
#         self.assertIsInstance(values, np.ndarray)
#         self.assertEqual(0, len(values))
#
#         # test: non-empty schema array should return an np.array with a single np.inf on exploratory choice
#         values = self.eps_always_explore([self.schema])
#         self.assertEqual(1, len(values))
#         self.assertEqual(1, np.count_nonzero(values == np.inf))
#         self.assertIsInstance(values, np.ndarray)
#
#         values = self.eps_always_explore(self.schemas)
#         self.assertIsInstance(values, np.ndarray)
#         self.assertEqual(len(self.schemas), len(values))
#         self.assertEqual(1, np.count_nonzero(values == np.inf))
#
#         # test: non-empty schema array should return an np.array with same length of zeros non-exploratory choice
#         values = self.eps_never_explore([self.schema])
#         self.assertIsInstance(values, np.ndarray)
#         self.assertListEqual(list(np.array([0])), list(values))
#
#         values = self.eps_never_explore(self.schemas)
#         self.assertIsInstance(values, np.ndarray)
#         self.assertListEqual(list(np.zeros_like(self.schemas)), list(values))
#
#     def test_epsilon_decay(self):
#         epsilon = 1.0
#         rate = 0.99
#
#         decay_strategy = GeometricDecayStrategy(rate=rate)
#         eps_greedy = EpsilonGreedyEvaluationStrategy(epsilon=epsilon, decay_strategy=decay_strategy)
#
#         prev_epsilon = epsilon
#         for _ in range(100):
#             _ = eps_greedy(schemas=self.schemas)
#             self.assertEqual(decay_strategy.decay(prev_epsilon), eps_greedy.epsilon)
#             prev_epsilon = eps_greedy.epsilon
#
#     def test_epsilon_decay_to_minimum(self):
#         epsilon = 1.0
#         minimum = 0.5
#
#         decay_strategy = GeometricDecayStrategy(rate=0.99, minimum=minimum)
#         eps_greedy = EpsilonGreedyEvaluationStrategy(epsilon=epsilon, decay_strategy=decay_strategy)
#
#         for _ in range(100):
#             _ = eps_greedy(schemas=self.schemas)
#
#         self.assertEqual(minimum, eps_greedy.epsilon)
#
#     def test_pending_bypass(self):
#         # test: epsilon greedy values SHOULD all be zero when a pending schema is provided
#         expected = np.zeros_like(self.schemas)
#         actual = self.eps_always_explore(self.schemas, pending=self.schema)
#
#         self.assertTrue(np.array_equal(expected, actual))
#
#
# class TestHabituationEvaluationStrategy(TestCase):
#     def setUp(self) -> None:
#         common_test_setup()
#
#         self.actions = [Action(f'A{i}') for i in range(11)]
#         self.schemas = [sym_schema(f'/A{i}/') for i in range(11)]
#
#         self.tr: AccumulatingTrace[Action] = AccumulatingTrace()
#
#         # elements 0-4 should have value of 0.25
#         self.tr.update(self.actions)
#
#         # element 5 should have value value of 0.75
#         self.tr.update(self.actions[5:])
#
#         # elements 6-10 should have value of 1.75
#         self.tr.update(self.actions[6:])
#
#         # sanity checks
#         expected = 0.25 * np.ones_like(self.actions[0:5])
#         actual = self.tr.values[self.tr.indexes(self.actions[0:5])]
#
#         self.assertTrue(np.array_equal(expected, actual))
#
#         expected = np.array([0.75])
#         actual = self.tr.values[self.tr.indexes([self.actions[5]])]
#
#         self.assertTrue(np.array_equal(expected, actual))
#
#         expected = 1.75 * np.ones_like(self.actions[6:])
#         actual = self.tr.values[self.tr.indexes(self.actions[6:])]
#
#         self.assertTrue(np.array_equal(expected, actual))
#
#         self.habituation_strategy = HabituationEvaluationStrategy(trace=self.tr)
#
#     def test_no_trace(self):
#         # test: no trace or trace with
#         self.assertRaises(ValueError, lambda: HabituationEvaluationStrategy(trace=None, multiplier=1.0))
#
#     def test_empty_schemas_list(self):
#         # test: empty schemas list should return empty numpy array
#         expected = np.array([])
#         actual = self.habituation_strategy.values(schemas=[])
#
#         self.assertTrue(np.array_equal(expected, actual))
#
#     def test_single_action(self):
#         # test: single action should have zero value
#         expected = np.zeros(1)
#         actual = self.habituation_strategy.values(schemas=[self.schemas[0]])
#
#         self.assertTrue(np.array_equal(expected, actual))
#
#     def test_multiple_actions_same_value(self):
#         # test: multiple actions with same trace value should return zeros
#
#         schemas = self.schemas[:5]
#
#         expected = np.zeros_like(schemas)
#         actual = self.habituation_strategy.values(schemas=schemas)
#
#         self.assertTrue(np.array_equal(expected, actual))
#
#     def test_multiple_actions_different_values_contains_median(self):
#         values = self.habituation_strategy.values(schemas=self.schemas)
#
#         # sanity check
#         self.assertTrue(np.median(values) in values)
#
#         # test: actions with trace values equal to median should have zero value
#         self.assertEqual(np.zeros(1), values[5])
#
#         # test: actions with trace values below median should have positive value
#         self.assertTrue(np.alltrue(values[:5] > 0.0))
#
#         # test: actions with trace values above median should have negative value
#         self.assertTrue(np.alltrue(values[6:] < 0.0))
#
#     def test_multiple_actions_different_values_does_not_contain_median(self):
#         values = self.habituation_strategy.values(schemas=self.schemas[1:])
#
#         # sanity check
#         self.assertTrue(np.median(values) not in values)
#
#         # test: actions with trace values below median should have positive value
#         self.assertTrue(np.alltrue(values[:5] > 0.0))
#
#         # test: actions with trace values above median should have negative value
#         self.assertTrue(np.alltrue(values[5:] < 0.0))
#
#     def test_unknown_values(self):
#         # test: schemas with unknown actions should raise ValueError
#         schemas = [*self.schemas, sym_schema('/UNK/')]
#         self.assertRaises(ValueError, lambda: self.habituation_strategy.values(schemas=schemas))
#
#     def test_multiplier(self):
#         multiplier = 8.0
#
#         habituation_with_default_multiplier = HabituationEvaluationStrategy(trace=self.tr)
#         habituation_with_non_default_multiplier = HabituationEvaluationStrategy(trace=self.tr, multiplier=multiplier)
#
#         values = habituation_with_default_multiplier(schemas=self.schemas)
#         values_with_mult = habituation_with_non_default_multiplier(schemas=self.schemas)
#
#         self.assertTrue(np.array_equal(values * multiplier, values_with_mult))
#
#
# class TestCompositeEvaluationStrategy(TestCase):
#     def setUp(self) -> None:
#         common_test_setup()
#
#     # def test_weights(self):
#     #     ss = SchemaSelection(
#     #         select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1)),
#     #         value_strategies=[primitive_value_evaluation_strategy, delegated_value_evaluation_strategy],
#     #     )
#     #
#     #     # test: initializer should set weights to the requested values
#     #     weights = [0.4, 0.6]
#     #     ss.weights = weights
#     #     self.assertListEqual(weights, list(ss.weights))
#     #
#     #     # test: should raise a ValueError if initializer given an invalid number of weights
#     #     with self.assertRaises(ValueError):
#     #         ss.weights = [1.0]
#     #
#     #     # test: should raise a ValueError if the weights do not sum to 1.0
#     #     with self.assertRaises(ValueError):
#     #         ss.weights = [0.1, 0.2]
#     #
#     #     # test: should raise a ValueError if any of the weights are non-positive
#     #     with self.assertRaises(ValueError):
#     #         ss.weights = [1.8, -0.8]
#     #
#     #     # test: weights of 0.0 and 1.0 should be allowed
#     #     try:
#     #         ss.weights = [0.0, 1.0]
#     #     except ValueError as e:
#     #         self.fail(f'Unexpected ValueError occurred: {str(e)}')
#
#     # def test_call(self):
#     #     s1 = sym_schema('/A1/1,')  # pv = 0.0
#     #     s2 = sym_schema('/A2/2,')  # pv = 0.0
#     #     s3 = sym_schema('/A1/3,')  # pv = 0.95
#     #     s4 = sym_schema('/A3/4,')  # pv = -1.0
#     #     s5 = sym_schema('/A1/5,')  # pv = 2.0
#     #     s6 = sym_schema('/A2/6,')  # pv = -3.0
#     #
#     #     self.s1_c12_r34 = sym_schema('1,2/A1/(3,4),', reliability=1.0)  # total pv = -0.05; total dv = 0.0
#     #     self.s1_c12_r345 = sym_schema('1,2/A1/(3,4,5),', reliability=1.0)  # total pv = 1.95; total dv = 0.0
#     #     self.s1_c12_r3456 = sym_schema('1,2/A1/(3,4,5,6),', reliability=1.0)  # total pv = -1.05; total dv = 0.0
#     #     self.s1_c12_r345_not6 = sym_schema('1,2/A1/(3,4,5,~6),', reliability=1.0)  # total pv = 1.95; total dv = 0.0
#     #
#     #     schemas = [s1, s2, s3, s4, s5, s6, self.s1_c12_r34, self.s1_c12_r345, self.s1_c12_r3456, self.s1_c12_r345_not6]
#     #
#     #     # testing with no evaluation strategies
#     #     ss = SchemaSelection(
#     #         select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1)),
#     #         value_strategies=[],
#     #     )
#     #
#     #     # test: should returns array of zeros if no value strategies specified
#     #     self.assertTrue(np.array_equal(np.zeros_like(schemas), ss.calc_effective_values(schemas)))
#     #
#     #     # testing single evaluation strategy
#     #     ss = SchemaSelection(
#     #         select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1)),
#     #         value_strategies=[primitive_value_evaluation_strategy],
#     #     )
#     #
#     #     expected_values = primitive_value_evaluation_strategy.values(schemas)
#     #
#     #     # test: primitive-only value strategy should return primitive values for each schema
#     #     self.assertTrue(np.array_equal(expected_values, ss.calc_effective_values(schemas, pending=None)))
#     #
#     #     # testing multiple evaluation strategies (equal weighting)
#     #     ss = SchemaSelection(
#     #         select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1)),
#     #         value_strategies=[primitive_value_evaluation_strategy, delegated_value_evaluation_strategy],
#     #     )
#     #
#     #     pvs = primitive_value_evaluation_strategy.values(schemas)
#     #     dvs = delegated_value_evaluation_strategy.values(schemas)
#     #     expected_values = (pvs + dvs) / 2.0
#     #     actual_values = ss.calc_effective_values(schemas, pending=None)
#     #
#     #     # test: should return weighted sum of evaluation strategy values
#     #     self.assertTrue(np.array_equal(expected_values, actual_values))
#     #
#     #     # testing multiple evaluation strategies (uneven weighting)
#     #     ss.weights = np.array([0.95, 0.05])
#     #
#     #     expected_values = 0.95 * pvs + 0.05 * dvs
#     #     actual_values = ss.calc_effective_values(schemas, pending=None)
#     #
#     #     # test: should return weighted sum of evaluation strategy values
#     #     self.assertTrue(np.array_equal(expected_values, actual_values))
#
#
# class TestDefaultExploratoryEvaluationStrategy(TestCase):
#     def setUp(self) -> None:
#         common_test_setup()
#
#
# class TestDefaultGoalPursuitEvaluationStrategy(TestCase):
#     def setUp(self) -> None:
#         common_test_setup()

# if __name__ == '__main__':
#     unittest.main()
