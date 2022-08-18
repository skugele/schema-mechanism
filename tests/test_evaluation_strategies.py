import itertools
import logging
import random
from collections import Counter
from copy import copy
from typing import Optional
from typing import Sequence
from unittest import TestCase
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import call

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.core import Chain
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import Item
from schema_mechanism.core import ItemPool
from schema_mechanism.core import Schema
from schema_mechanism.core import calc_delegated_value
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.core import get_action_trace
from schema_mechanism.core import get_global_params
from schema_mechanism.core import set_action_trace
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.strategies.decay import ExponentialDecayStrategy
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.evaluation import CompositeEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultExploratoryEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultGoalPursuitEvaluationStrategy
from schema_mechanism.strategies.evaluation import EpsilonRandomEvaluationStrategy
from schema_mechanism.strategies.evaluation import EvaluationStrategy
from schema_mechanism.strategies.evaluation import HabituationEvaluationStrategy
from schema_mechanism.strategies.evaluation import InstrumentalValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import MaxDelegatedValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import NoOpEvaluationStrategy
from schema_mechanism.strategies.evaluation import PendingFocusEvaluationStrategy
from schema_mechanism.strategies.evaluation import ReliabilityEvaluationStrategy
from schema_mechanism.strategies.evaluation import TotalDelegatedValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import TotalPrimitiveValueEvaluationStrategy
from schema_mechanism.strategies.evaluation import display_minmax
from schema_mechanism.strategies.evaluation import display_values
from schema_mechanism.strategies.scaling import SigmoidScalingStrategy
from schema_mechanism.strategies.trace import AccumulatingTrace
from schema_mechanism.strategies.trace import Trace
from schema_mechanism.strategies.weight_update import CyclicWeightUpdateStrategy
from schema_mechanism.util import equal_weights
from schema_mechanism.util import repr_str
from test_share import disable_test
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
            sym_schema('X,/A1/A,', schema_type=MockSchema, reliability=0.0),
            sym_schema('Y,/A1/B,', schema_type=MockSchema, reliability=0.5),
            sym_schema('Z,/A1/C,', schema_type=MockSchema, reliability=1.0),
        ]

        self.composite_result_schemas = [
            sym_schema('X,/A1/(A,B),', schema_type=MockSchema, reliability=0.0),
            sym_schema('Y,/A1/(B,C),', schema_type=MockSchema, reliability=0.5),
            sym_schema('Z,/A1/(A,B,C),', schema_type=MockSchema, reliability=1.0),
        ]

        self.composite_action_schema = sym_schema('X,/X,Y/A,B', schema_type=MockSchema, reliability=0.75)
        self.composite_action_schema.action.controller.update([
            Chain([sym_schema('A,/CA1/B,'), sym_schema('B,/CA2/C,'), sym_schema('C,/CA3/X,Y')])
        ])

        # sanity check: composite action schema should have a composite action
        self.assertTrue(self.composite_action_schema.action.is_composite())

        self.schemas = [
            *self.simple_result_schemas,
            *self.composite_result_schemas,
            self.composite_action_schema,
            *self.composite_action_schema.action.controller.components,
        ]

        self.actions = [schema.action for schema in self.schemas]
        self.action_trace = get_action_trace()
        self.action_trace.clear()
        self.action_trace.update(self.actions)

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

    def assert_call_correctly_invokes_post_process_callables(self, strategy: EvaluationStrategy):
        values = strategy.values(schemas=self.schemas, pending=self.composite_action_schema)

        # mocking post-process callables
        mock = MagicMock()
        mock.post_process_1 = MagicMock()
        mock.post_process_1.return_value = values
        mock.post_process_2 = MagicMock()
        mock.post_process_2.return_value = values

        post_process_list = [mock.post_process_1, mock.post_process_2]
        _ = strategy(
            schemas=self.schemas,
            pending=self.composite_action_schema,
            post_process=post_process_list
        )

        # test: post-process callables should be invoked in order and called with the correct arguments
        mock.assert_has_calls([call.post_process_1(schemas=ANY, pending=ANY, values=ANY),
                               call.post_process_2(schemas=ANY, pending=ANY, values=ANY)])

        # test: strategy should invoke post-process methods with the correct arguments
        for mock in post_process_list:
            self.assertListEqual(self.schemas, mock.call_args.kwargs['schemas'])
            self.assertEqual(self.composite_action_schema, mock.call_args.kwargs['pending'])
            np.testing.assert_array_equal(values, mock.call_args.kwargs['values'])

    def assert_post_process_can_update_values(self, strategy):
        values = strategy.values(schemas=self.schemas, pending=self.composite_action_schema)

        # post-process callables used for this test
        # noinspection PyShadowingNames,PyUnusedLocal
        def add_one_post_process(values: np.ndarray, **kwargs) -> np.ndarray:
            new_values = values + 1.0
            return new_values

        # noinspection PyShadowingNames,PyUnusedLocal
        def divide_by_two_post_process(values: np.ndarray, **kwargs) -> np.ndarray:
            new_values = values / 2.0
            return new_values

        # note that this example is also testing that post process callables are being called sequentially in the
        # proper order
        expected_values = values
        expected_values = add_one_post_process(values=expected_values)
        expected_values = divide_by_two_post_process(values=expected_values)

        actual_values = strategy(
            schemas=self.schemas,
            pending=self.composite_action_schema,
            post_process=[
                add_one_post_process,
                divide_by_two_post_process
            ]
        )

        # test: post-process callables should be capable of changing the values returned from strategy and other
        #       post-process callables (order matters!)
        np.testing.assert_array_equal(expected_values, actual_values)

    def assert_values_consistent_with_call(self, strategy: EvaluationStrategy) -> None:
        # the call to values MUST appear first because __call__ may update the strategy's internal state, changing
        # the values
        np.testing.assert_array_equal(
            strategy.values(self.schemas, pending=None),
            strategy(self.schemas, pending=None)
        )
        np.testing.assert_array_equal(
            strategy.values(self.schemas, pending=self.composite_action_schema),
            strategy(self.schemas, pending=self.composite_action_schema)
        )

    def assert_values_method_is_idempotent(self, strategy: EvaluationStrategy) -> None:
        n_calls_to_check = 5

        # test: values method should be idempotent when pending is None
        values = strategy.values(self.schemas, pending=None)
        for _ in range(n_calls_to_check):
            new_values = strategy.values(self.schemas, pending=None)
            np.testing.assert_array_equal(values, new_values)

            values = new_values

        # test: values method should be idempotent when pending is not None
        values = strategy.values(self.schemas, pending=self.composite_action_schema)
        for _ in range(n_calls_to_check):
            new_values = strategy.values(self.schemas, pending=self.composite_action_schema)
            np.testing.assert_array_equal(values, new_values)

            values = new_values

    def assert_all_common_functionality(self, strategy: EvaluationStrategy) -> None:
        self.assert_implements_evaluation_strategy_protocol(strategy)
        self.assert_returns_array_of_correct_length(strategy)
        self.assert_values_method_is_idempotent(strategy)
        self.assert_call_correctly_invokes_post_process_callables(strategy)
        self.assert_post_process_can_update_values(strategy)
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

    def test_repr(self):
        expected_str = repr_str(obj=self.strategy, attr_values=dict())
        self.assertEqual(expected_str, repr(self.strategy))


class TestTotalPrimitiveValueEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

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

    def test_repr(self):
        expected_str = repr_str(obj=self.strategy, attr_values=dict())
        self.assertEqual(expected_str, repr(self.strategy))


class TestTotalDelegatedValueEvaluationStrategy(TestCommon):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        super().setUp()

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

    def test_repr(self):
        expected_str = repr_str(obj=self.strategy, attr_values=dict())
        self.assertEqual(expected_str, repr(self.strategy))


class TestMaxDelegatedValueEvaluationStrategy(TestCommon):
    # noinspection PyTypeChecker
    def setUp(self) -> None:
        super().setUp()

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

    def test_repr(self):
        expected_str = repr_str(obj=self.strategy, attr_values=dict())
        self.assertEqual(expected_str, repr(self.strategy))


class TestInstrumentalValueEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

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
        get_global_params().set('learning_rate', 1.0)

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
        get_global_params().set('learning_rate', 1.0)

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

    def test_repr(self):
        expected_str = repr_str(obj=self.strategy, attr_values=dict())
        self.assertEqual(expected_str, repr(self.strategy))


class TestPendingFocusEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.strategy = PendingFocusEvaluationStrategy(
            max_value=1.0,
            decay_strategy=ExponentialDecayStrategy(rate=0.1, minimum=-np.inf)
        )

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

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_init(self):
        # test: attribute values should be set properly if values were given to initializer
        max_value = 1.0
        decay_strategy = ExponentialDecayStrategy(rate=0.1, initial=max_value)
        strategy = PendingFocusEvaluationStrategy(max_value=max_value, decay_strategy=decay_strategy)

        self.assertEqual(max_value, strategy.max_value)
        self.assertEqual(decay_strategy, strategy.decay_strategy)

        # test: default attribute values should be set if values not given to initializer
        default_max_value = 1.0
        default_decay_strategy = None

        strategy = PendingFocusEvaluationStrategy()
        self.assertEqual(default_max_value, strategy.max_value)
        self.assertEqual(default_decay_strategy, strategy.decay_strategy)

    def test_no_pending(self):
        # test: schemas should have zero value if there is no active pending schema
        expected = np.zeros_like(self.schemas)
        actual = self.strategy.values(schemas=self.schemas, pending=None)

        self.assertTrue(np.array_equal(expected, actual))

    def test_initial_component_values(self):
        max_value = self.strategy.max_value
        pending = self.s_s1
        pending_components = pending.action.controller.components

        values = self.strategy.values(schemas=self.schemas, pending=pending)
        for schema, value in zip(self.schemas, values):
            # test: components of the active pending schema should initially have max focus
            if schema in pending_components:
                self.assertEqual(max_value, value)

            # test: other schemas should have zero focus value
            else:
                self.assertEqual(0.0, value)

    def test_value_decay(self):
        # test: values should decay after each subsequent call with the same pending schema
        max_value = self.strategy.max_value * np.ones_like(self.schemas, dtype=np.float64)
        pending = self.s_s1
        pending_components = pending.action.controller.components

        expected_value_arrays = [
            self.strategy.decay_strategy.decay(values=max_value, step_size=i) for i in range(0, 10)
        ]

        non_component_indexes = [
            self.schemas.index(schema) for schema in self.schemas if schema not in pending_components
        ]

        for array in expected_value_arrays:
            array[non_component_indexes] = 0.0

        for expected_values in expected_value_arrays:
            values = self.strategy(schemas=self.schemas, pending=pending)
            np.testing.assert_array_equal(expected_values, values)

    def test_unbounded_reduction_in_value(self):
        pending = self.s_s1
        pending_components = list(pending.action.controller.components)

        old_values = self.strategy(schemas=pending_components, pending=pending)
        for _ in range(100):
            new_values = self.strategy(schemas=pending_components, pending=pending)
            np.testing.assert_array_less(new_values, old_values)

    def test_repr(self):
        attr_values = {
            'max_value': self.strategy.max_value,
            'decay_strategy': self.strategy.decay_strategy
        }

        expected_str = repr_str(obj=self.strategy, attr_values=attr_values)
        self.assertEqual(expected_str, repr(self.strategy))

    @disable_test
    def test_playground(self):
        logging.getLogger('schema_mechanism.strategies.evaluation').setLevel(logging.DEBUG)

        strategy = PendingFocusEvaluationStrategy(
            max_value=1.0,
            decay_strategy=ExponentialDecayStrategy(rate=0.1, minimum=-1.0)
        )

        for i in range(1000):
            values = strategy(schemas=self.schemas, pending=self.s_s1, post_process=[display_values])
            print(values)


class TestReliabilityEvaluationStrategy(TestCommon):

    def setUp(self) -> None:
        super().setUp()

        self.strategy = ReliabilityEvaluationStrategy()

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_init(self):
        # test: attribute values should be set properly if values were given to initializer
        max_penalty = 10.75
        severity = 2.5
        threshold = 0.65

        strategy = ReliabilityEvaluationStrategy(max_penalty=max_penalty, severity=severity, threshold=threshold)

        self.assertEqual(max_penalty, strategy.max_penalty)
        self.assertEqual(severity, strategy.severity)
        self.assertEqual(threshold, strategy.threshold)

        # test: default attribute values should be set if values not given to initializer
        default_max_penalty = 1.0
        default_severity = 2.0
        default_threshold = 0.0

        strategy = ReliabilityEvaluationStrategy()

        self.assertEqual(default_max_penalty, strategy.max_penalty)
        self.assertEqual(default_severity, strategy.severity)
        self.assertEqual(default_threshold, strategy.threshold)

    def test_values(self):
        max_penalty = 1.0
        self.strategy.max_penalty = max_penalty

        # test: a reliability of 1.0 should result in penalty of 0.0
        schemas = [sym_schema('J,/A1/B,', schema_type=MockSchema, reliability=1.0)]
        rvs = self.strategy(schemas)
        self.assertTrue(np.array_equal(np.zeros_like(schemas), rvs))

        # test: a reliability of 0.0 should result in max penalty
        schemas = [sym_schema('K,/A1/C,', schema_type=MockSchema, reliability=0.0)]
        rvs = self.strategy(schemas)
        self.assertTrue(np.array_equal(-max_penalty * np.ones_like(schemas), rvs))

        # test: a reliability of nan should result in max penalty
        schemas = [sym_schema('L,/A1/D,', schema_type=MockSchema, reliability=np.nan)]
        rvs = self.strategy(schemas)
        self.assertTrue(np.array_equal(-max_penalty * np.ones_like(schemas), rvs))

        # test: testing expected reliability values over a range of reliabilities between 0.0 and 1.0
        schemas = [
            sym_schema('A,/A1/U,', schema_type=MockSchema, reliability=0.01),
            sym_schema('A,/A1/V,', schema_type=MockSchema, reliability=0.1),
            sym_schema('A,/A1/W,', schema_type=MockSchema, reliability=0.25),
            sym_schema('A,/A2/X,', schema_type=MockSchema, reliability=0.5),
            sym_schema('A,/A3/Y,', schema_type=MockSchema, reliability=0.75),
            sym_schema('A,/A4/Z,', schema_type=MockSchema, reliability=0.9),
            sym_schema('A,/A5/ZZ,', schema_type=MockSchema, reliability=0.99),
        ]

        # these values assume that severity is 2.0 (the default)
        values = np.array([-0.9999, -0.99, -0.9375, -0.75, -0.4375, -0.19, -0.0199])

        np.testing.assert_array_almost_equal(self.strategy(schemas=schemas), values)

    def test_max_penalty(self):

        # test: max penalty values > 0.0 should be accepted
        try:
            # test via property setter
            self.strategy.max_penalty = 0.0001
            self.strategy.max_penalty = 1.0

            # test via initializer argument
            _ = ReliabilityEvaluationStrategy(max_penalty=0.0001)
            _ = ReliabilityEvaluationStrategy(max_penalty=1.0)
        except ValueError as e:
            self.fail(e)

        # test: max penalty <= 0.0 should raise a ValueError
        with self.assertRaises(ValueError):

            # test via property setter
            self.strategy.max_penalty = 0.0
            self.strategy.max_penalty = -1.0

            # test via initializer argument
            _ = ReliabilityEvaluationStrategy(max_penalty=0.0)
            _ = ReliabilityEvaluationStrategy(max_penalty=-1.0)

    def test_threshold(self):
        max_penalty = 1000
        strategy = ReliabilityEvaluationStrategy(
            max_penalty=max_penalty,
            severity=1.1,
        )

        try:
            # test: threshold values between 0.0 (inclusive) and 1.0 (exclusive) should be allowed
            for threshold in [0.0, 0.25, 0.5, 0.75, 0.999]:
                strategy.threshold = threshold

            # test: reliabilities <= threshold should have a value of max penalty
        except ValueError as e:
            self.fail(f'Unexpected ValueError: {e}')

        # test: threshold values less than 0.0 or greater than or equal to 1.0 should raise a ValueError
        with self.assertRaises(ValueError):
            for threshold in [-100.0, 0.0 - 1e-10, 1.0, 10.0]:
                strategy.threshold = threshold

        threshold = 0.5
        strategy.threshold = threshold

        schemas = [
            # reliability below threshold
            sym_schema('1,/AThreshold/2,', schema_type=MockSchema, reliability=threshold - 1e-8),

            # reliability at threshold
            sym_schema('2,/AThreshold/3,', schema_type=MockSchema, reliability=threshold),

            # reliability above threshold
            sym_schema('3,/AThreshold/4,', schema_type=MockSchema, reliability=threshold + 1e-8),
        ]

        values = strategy(schemas)

        # test: schemas with reliabilities under threshold should have values equal to -max_penalty
        self.assertEqual(values[0], -max_penalty)

        # test: schemas with reliabilities equal to threshold should have values equal to -max_penalty
        self.assertEqual(values[1], -max_penalty)

        # test: schemas with reliabilities greater than threshold should have values greater than -max_penalty
        self.assertGreater(values[2], -max_penalty)

    def test_repr(self):
        attr_values = {
            'max_penalty': self.strategy.max_penalty,
            'severity': self.strategy.severity,
            'threshold': self.strategy.threshold
        }

        expected_str = repr_str(obj=self.strategy, attr_values=attr_values)
        self.assertEqual(expected_str, repr(self.strategy))


class TestEpsilonRandomEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.strategy = EpsilonRandomEvaluationStrategy()

        self.eps_always_explore = EpsilonRandomEvaluationStrategy(epsilon=1.0)
        self.eps_never_explore = EpsilonRandomEvaluationStrategy(epsilon=0.0)
        self.eps_even_chance = EpsilonRandomEvaluationStrategy(epsilon=0.5)

        self.schema = sym_schema('1,2/A/3,4')

        # schema with a composite action
        self.ca_schema = sym_schema('1,2/S1,/5,7')

        self.schemas = [self.schema,
                        self.ca_schema,
                        sym_schema('1,2/A/3,5'),
                        sym_schema('1,2/A/3,6')]

    def test_init(self):
        # test: attribute values should be set properly if values were given to initializer
        epsilon = 0.991
        epsilon_min = 0.025
        decay_strategy = ExponentialDecayStrategy(rate=0.7)
        max_value = 1.72

        strategy = EpsilonRandomEvaluationStrategy(
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            decay_strategy=decay_strategy,
            max_value=max_value
        )

        self.assertEqual(epsilon, strategy.epsilon)
        self.assertEqual(epsilon_min, strategy.epsilon_min)
        self.assertEqual(decay_strategy, strategy.decay_strategy)
        self.assertEqual(max_value, strategy.max_value)

        # test: default attribute values should be set if values not given to initializer
        default_epsilon = 0.99
        default_epsilon_min = 0.0
        default_decay_strategy = None
        default_max_value = 1.0

        strategy = EpsilonRandomEvaluationStrategy()

        self.assertEqual(default_epsilon, strategy.epsilon)
        self.assertEqual(default_epsilon_min, strategy.epsilon_min)
        self.assertEqual(default_decay_strategy, strategy.decay_strategy)
        self.assertEqual(default_max_value, strategy.max_value)

        # test: epsilon values between 0.0 and 1.0 (inclusion) should be allowed
        for epsilon in np.linspace(0.0, 1.0, endpoint=True):
            try:
                _ = EpsilonRandomEvaluationStrategy(epsilon=epsilon)
            except ValueError:
                self.fail(f'Raised unexpected value error for valid epsilon value: {epsilon}')

        # test: initializer should raise a ValueError if epsilon values are not between 0.0 and 1.0 (inclusion)
        self.assertRaises(ValueError, lambda: EpsilonRandomEvaluationStrategy(epsilon=-1e-5))
        self.assertRaises(ValueError, lambda: EpsilonRandomEvaluationStrategy(epsilon=1 + 1e-5))

        # test: epsilon min values between 0.0 and 1.0 (inclusion) should be allowed
        for epsilon_min in np.linspace(0.0, 1.0, endpoint=True):
            try:
                _ = EpsilonRandomEvaluationStrategy(epsilon=1.0, epsilon_min=epsilon_min)
            except ValueError:
                self.fail(f'Raised unexpected value error for valid epsilon min value: {epsilon_min}')

        # test: initializer should raise a ValueError if epsilon values are not between 0.0 and 1.0 (inclusion)
        self.assertRaises(ValueError, lambda: EpsilonRandomEvaluationStrategy(epsilon=1.0, epsilon_min=-1e-5))
        self.assertRaises(ValueError, lambda: EpsilonRandomEvaluationStrategy(epsilon=1.0, epsilon_min=1 + 1e-5))

        # test: a ValueError should be raised if epsilon is less than epsilon min
        self.assertRaises(ValueError, lambda: EpsilonRandomEvaluationStrategy(epsilon=0.1, epsilon_min=0.3))

    def test_common(self):
        self.assert_all_common_functionality(self.eps_never_explore)

    def test_epsilon_setter(self):
        epsilon_random = EpsilonRandomEvaluationStrategy(0.5)

        epsilon_random.epsilon = 0.0
        self.assertEqual(0.0, epsilon_random.epsilon)

        epsilon_random.epsilon = 1.0
        self.assertEqual(1.0, epsilon_random.epsilon)

        try:
            epsilon_random.epsilon = -1.00001
            epsilon_random.epsilon = 1.00001
            self.fail('ValueError expected on illegal assignment')
        except ValueError:
            pass

    def test_values(self):
        # test: non-empty schema array should return an np.array with a single max_value value on exploratory choice
        values = self.eps_always_explore([self.schema])
        self.assertEqual(1, len(values))
        self.assertEqual(1, np.count_nonzero(values == self.eps_always_explore.max_value))
        self.assertIsInstance(values, np.ndarray)

        values = self.eps_always_explore(self.schemas)
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(len(self.schemas), len(values))
        self.assertEqual(1, np.count_nonzero(values == self.eps_always_explore.max_value))

        # test: non-empty schema array should return an np.array with same length of zeros on non-exploratory choice
        values = self.eps_never_explore([self.schema])
        self.assertIsInstance(values, np.ndarray)
        self.assertListEqual(list(np.array([0])), list(values))

        values = self.eps_never_explore(self.schemas)
        self.assertIsInstance(values, np.ndarray)
        self.assertListEqual(list(np.zeros_like(self.schemas)), list(values))

    def test_empirical_exploration_probability_consistent_with_epsilon(self):
        n = 10000

        for epsilon in [0.1, 0.9]:
            n_exploration = 0
            for n_calls in range(0, n):
                strategy = EpsilonRandomEvaluationStrategy(epsilon=epsilon)
                values = strategy(self.schemas)
                if strategy.max_value in values:
                    n_exploration += 1

            expected_probability = epsilon
            actual_probability = n_exploration / n

            self.assertAlmostEqual(
                expected_probability,
                actual_probability,
                delta=1e-1,
                msg=f'exploration probability {actual_probability} does not match expected probability '
                    f'{expected_probability}')

    def test_epsilon_decay(self):
        epsilon = 1.0
        rate = 0.99

        decay_strategy = GeometricDecayStrategy(rate=rate)
        epsilon_random = EpsilonRandomEvaluationStrategy(epsilon=epsilon, decay_strategy=decay_strategy)

        expected_epsilon_value = epsilon
        for i in range(100):
            # apply strategy to advance decay epsilon
            _ = epsilon_random(schemas=self.schemas)

            expected_epsilon_value = decay_strategy.decay(values=np.array([expected_epsilon_value]))[0]
            actual_epsilon_value = epsilon_random.epsilon

            self.assertAlmostEqual(expected_epsilon_value, actual_epsilon_value)

    def test_epsilon_decay_to_minimum(self):
        epsilon = 1.0
        epsilon_min = 0.5

        decay_strategy = GeometricDecayStrategy(rate=0.25)
        epsilon_random = EpsilonRandomEvaluationStrategy(epsilon=epsilon, epsilon_min=epsilon_min,
                                                         decay_strategy=decay_strategy)

        for _ in range(100):
            _ = epsilon_random(schemas=self.schemas)

            if epsilon <= epsilon_min:
                break

        self.assertEqual(epsilon_min, epsilon_random.epsilon)

    def test_pending_schema_component_selection(self):
        # test: greedy selection should be limited to pending schema components when a pending schema is provided

        pending_schema = sym_schema('A,/S1,/B,')
        pending_component_schemas = [
            sym_schema('A,/A1/B,', schema_type=MockSchema, reliability=1.0),
            sym_schema('B,/A1/C,', schema_type=MockSchema, reliability=1.0),
            sym_schema('C,/A1/D,', schema_type=MockSchema, reliability=1.0),
            sym_schema('D,/A1/S1,', schema_type=MockSchema, reliability=1.0),
        ]

        # registers component schemas with controller
        chains = [Chain(pending_component_schemas)]
        pending_schema.action.controller.update(chains)

        # sanity check: make sure components have been added to pending schema
        for schema in pending_component_schemas:
            self.assertIn(schema, pending_schema.action.controller.components)

        other_schemas = [
            sym_schema('X1,/A2/Y1,', schema_type=MockSchema, reliability=1.0),
            sym_schema('X2,/A2/Y2,', schema_type=MockSchema, reliability=1.0),
            sym_schema('X3,/A2/Y3,', schema_type=MockSchema, reliability=1.0),
            sym_schema('X4,/A2/Y4,', schema_type=MockSchema, reliability=1.0),
        ]

        all_schemas = [
            *pending_component_schemas,
            *other_schemas
        ]

        component_indexes = [all_schemas.index(schema) for schema in pending_component_schemas]
        non_component_indexes = [all_schemas.index(schema) for schema in other_schemas]

        counter = Counter()

        for _ in range(1000):
            values = self.eps_always_explore(schemas=all_schemas, pending=pending_schema)
            selection_index = np.argwhere(values == self.eps_always_explore.max_value)[0][0]
            counter[selection_index] += 1

        for index in non_component_indexes:
            self.assertEqual(0, counter[index], msg=f'Non-component schema {all_schemas[index]} was selected')

        for index in component_indexes:
            self.assertNotEqual(0, counter[index], msg=f'Component schema {all_schemas[index]} was never selected')

    def test_repr(self):
        attr_values = {
            'epsilon': self.strategy.epsilon,
            'epsilon_min': self.strategy.epsilon_min,
            'decay_strategy': self.strategy.decay_strategy,
            'max_value': self.strategy.max_value,

        }

        expected_str = repr_str(obj=self.strategy, attr_values=attr_values)
        self.assertEqual(expected_str, repr(self.strategy))


class TestHabituationEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.actions = [Action(f'A{i}') for i in range(11)]
        self.schemas = [sym_schema(f'/A{i}/') for i in range(11)]

        self.trace: Trace[Action] = AccumulatingTrace(
            decay_strategy=GeometricDecayStrategy(rate=0.5),
            active_increment=1.0
        )

        # elements 0-4 should have value of 0.25
        self.trace.update(self.actions)

        # element 5 should have value value of 0.75
        self.trace.update(self.actions[5:])

        # elements 6-10 should have value of 1.75
        self.trace.update(self.actions[6:])

        # sanity checks
        expected = 0.25 * np.ones_like(self.actions[0:5])
        actual = self.trace.values[self.trace.indexes(self.actions[0:5])]

        self.assertTrue(np.array_equal(expected, actual))

        expected = np.array([0.75])
        actual = self.trace.values[self.trace.indexes([self.actions[5]])]

        self.assertTrue(np.array_equal(expected, actual))

        expected = 1.75 * np.ones_like(self.actions[6:])
        actual = self.trace.values[self.trace.indexes(self.actions[6:])]

        self.assertTrue(np.array_equal(expected, actual))

        set_action_trace(self.trace)

        self.strategy = HabituationEvaluationStrategy(
            scaling_strategy=SigmoidScalingStrategy()
        )

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_init(self):
        scaling_strategy = SigmoidScalingStrategy(
            range_scale=1.0,
            vertical_shift=0.75
        )

        habituation_strategy = HabituationEvaluationStrategy(
            scaling_strategy=scaling_strategy
        )

        # test: attributes should have been set properly
        self.assertEqual(scaling_strategy, habituation_strategy.scaling_strategy)

    def test_single_action(self):
        # test: single action should have zero value
        expected = np.zeros(1)
        actual = self.strategy.values(schemas=[self.schemas[0]])

        self.assertTrue(np.array_equal(expected, actual))

    def test_multiple_actions_same_value(self):
        # test: multiple actions with same trace value should return zeros
        schemas = self.schemas[:5]

        expected = np.zeros_like(schemas)
        actual = self.strategy.values(schemas=schemas)

        self.assertTrue(np.array_equal(expected, actual))

    def test_multiple_actions_different_values_contains_median(self):
        values = self.strategy.values(schemas=self.schemas)

        # sanity check
        self.assertTrue(np.median(values) in values)

        # test: actions with trace values equal to median should have zero value
        self.assertEqual(np.zeros(1), values[5])

        # test: actions with trace values below median should have positive value
        self.assertTrue(np.alltrue(values[:5] > 0.0))

        # test: actions with trace values above median should have negative value
        self.assertTrue(np.alltrue(values[6:] < 0.0))

    def test_multiple_actions_different_values_does_not_contain_median(self):
        values = self.strategy.values(schemas=self.schemas[1:])

        # sanity check
        self.assertTrue(np.median(values) not in values)

        # test: actions with trace values below median should have positive value
        self.assertTrue(np.alltrue(values[:5] > 0.0))

        # test: actions with trace values above median should have negative value
        self.assertTrue(np.alltrue(values[5:] < 0.0))

    def test_unknown_values(self):
        # test: schemas with unknown actions should raise ValueError
        schemas = [*self.schemas, sym_schema('/UNK/')]
        self.assertRaises(ValueError, lambda: self.strategy.values(schemas=schemas))

    def test_repr(self):
        attr_values = {
            'scaling_strategy': self.strategy.scaling_strategy,
        }

        expected_str = repr_str(obj=self.strategy, attr_values=attr_values)
        self.assertEqual(expected_str, repr(self.strategy))


class TestCompositeEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.strategy = CompositeEvaluationStrategy(
            strategies=[
                TotalPrimitiveValueEvaluationStrategy(),
                TotalDelegatedValueEvaluationStrategy(),
            ],
            strategy_alias='TestEvaluationStrategy',
            post_process=[display_minmax]
        )

    def test_init(self):
        strategies = [
            TotalPrimitiveValueEvaluationStrategy(),
            TotalDelegatedValueEvaluationStrategy(),
            NoOpEvaluationStrategy()
        ]
        weights = [0.7, 0.2, 0.1]
        post_process = [
            display_minmax,
            display_values
        ]
        strategy_alias = 'test_strategy'

        # test: attributes should be set to given values when provided to initializer
        strategy = CompositeEvaluationStrategy(
            strategies=strategies,
            weights=weights,
            post_process=post_process,
            strategy_alias=strategy_alias
        )

        self.assertListEqual(strategies, list(strategy.strategies))
        self.assertListEqual(weights, list(strategy.weights))
        self.assertListEqual(post_process, list(strategy.post_process))
        self.assertEqual(strategy_alias, strategy.strategy_alias)

        # test: weights should be equal by default
        strategy = CompositeEvaluationStrategy(
            strategies=strategies,
        )
        np.testing.assert_array_equal(equal_weights(len(strategies)), strategy.weights)

        # test: the number of weights should equal the number of strategies
        self.assertRaises(ValueError, lambda: CompositeEvaluationStrategy(strategies=strategies, weights=[]))
        self.assertRaises(ValueError, lambda: CompositeEvaluationStrategy(strategies=strategies, weights=[1.0]))
        self.assertRaises(ValueError, lambda: CompositeEvaluationStrategy(strategies=strategies, weights=[0.5, 0.5]))

        # test: weights should sum to 1.0
        self.assertRaises(ValueError, lambda: CompositeEvaluationStrategy(strategies=strategies,
                                                                          weights=[0.1, 0.1, 0.1]))
        self.assertRaises(ValueError, lambda: CompositeEvaluationStrategy(strategies=strategies,
                                                                          weights=[0.8, 0.2, 0.1]))

        # test: individual weights should be between between 0.0 and 1.0 (inclusive)
        self.assertRaises(ValueError, lambda: CompositeEvaluationStrategy(strategies=strategies,
                                                                          weights=[1.1, -0.2, 0.1]))

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_post_process(self):
        post_process = [
            MagicMock(return_value=np.zeros_like(self.schemas)),
            MagicMock(return_value=np.zeros_like(self.schemas)),
            MagicMock(return_value=np.zeros_like(self.schemas)),
        ]

        strategy = CompositeEvaluationStrategy(
            strategies=[NoOpEvaluationStrategy()],
            post_process=post_process,
        )

        values = strategy(schemas=self.schemas, pending=self.composite_action_schema)
        for mock in post_process:
            mock.assert_called_once()
            kwargs_to_mock = mock.call_args.kwargs

            self.assertEqual(3, len(kwargs_to_mock))
            self.assertListEqual(self.schemas, kwargs_to_mock['schemas'])
            self.assertEqual(self.composite_action_schema, kwargs_to_mock['pending'])
            np.testing.assert_array_equal(values, kwargs_to_mock['values'])

    def test_values(self):
        schemas: list[Schema] = []

        for source in range(1, 11):
            _ = ItemPool().get(
                str(source),
                item_type=MockSymbolicItem,
                primitive_value=random.uniform(-1.0, 1.0),
                delegated_value=random.uniform(-1.0, 1.0),
            )

            schemas.append(sym_schema(f'/A1/{source},'))

        strategy = CompositeEvaluationStrategy(
            strategies=[
                TotalPrimitiveValueEvaluationStrategy(),
                TotalDelegatedValueEvaluationStrategy()
            ],
            weights=[0.1, 0.9]
        )

        values = strategy.values(schemas)
        expected_values = sum(weight * strategy.values(schemas)
                              for weight, strategy in zip(strategy.weights, strategy.strategies))

        np.testing.assert_array_equal(expected_values, values)

    def test_value_weighting(self):
        class AllOnesEvaluationStrategy(EvaluationStrategy):
            def values(self, schemas: Sequence[Schema], pending: Optional[Schema] = None, **kwargs) -> np.ndarray:
                return np.ones_like(schemas, dtype=np.float64)

        for weights in [(1.0, 0.0, 0.0), (0.6, 0.3, 0.1), (0.4, 0.4, 0.2), (0.1, 0.15, 0.75)]:
            composite_strategy = CompositeEvaluationStrategy(
                strategies=[
                    AllOnesEvaluationStrategy(),
                    AllOnesEvaluationStrategy(),
                    AllOnesEvaluationStrategy(),
                ],
                weights=weights
            )
            values = composite_strategy(self.schemas)
            np.testing.assert_allclose(np.ones_like(self.schemas, dtype=np.float64), values)

            for component_strategy, weight in zip(composite_strategy.strategies, weights):
                values = component_strategy(self.schemas)

                scaled_values = weight * values
                expected_values = weight * np.ones_like(self.schemas, dtype=np.float64)

                np.testing.assert_allclose(expected_values, scaled_values)

    def test_repr(self):
        attr_values = {
            'post_process': self.strategy.post_process,
            'strategy_alias': self.strategy.strategy_alias,
            'strategies': self.strategy.strategies,
            'weights': self.strategy.weights,
        }

        expected_str = repr_str(obj=self.strategy, attr_values=attr_values)
        self.assertEqual(expected_str, repr(self.strategy))

    @disable_test
    def test_playground(self):
        logging.getLogger('schema_mechanism.strategies.evaluation').setLevel(logging.DEBUG)

        strategy = CompositeEvaluationStrategy(
            strategies=[
                TotalPrimitiveValueEvaluationStrategy(),
                TotalDelegatedValueEvaluationStrategy()
            ],
            weights=[0.1, 0.9],
            post_process=[
                display_values,
                display_minmax
            ]
        )

        strategy(self.schemas)


class TestDefaultExploratoryEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        # some tests will fail unless the strategy provided to them is idempotent. Currently, the
        # DefaultExploratoryEvaluationStrategy is not idempotent in general, therefore a special, highly contrived
        # set of parameters is currently used to ensure idempotence. This seems to be a weakness of the current
        # implementation of this strategy rather than of the test cases.
        self.strategy = DefaultExploratoryEvaluationStrategy(epsilon=0.0, epsilon_min=0.0)

    def test_init(self):
        epsilon = 0.765
        epsilon_min = 0.12
        epsilon_decay_rate = 0.002
        post_process = [display_minmax, display_values]

        # test: attributes should be set to given values when provided to initializer
        strategy = DefaultExploratoryEvaluationStrategy(
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay_rate=epsilon_decay_rate,
            post_process=post_process
        )

        self.assertEqual(epsilon, strategy.epsilon)
        self.assertEqual(epsilon_min, strategy.epsilon_min)
        self.assertEqual(epsilon_decay_rate, strategy.epsilon_decay_rate)
        self.assertListEqual(list(post_process), list(strategy.post_process))

        # test: weights should be equal
        np.testing.assert_array_equal(equal_weights(len(strategy.strategies)), strategy.weights)

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_post_process(self):
        post_process = [
            MagicMock(return_value=np.zeros_like(self.schemas)),
            MagicMock(return_value=np.zeros_like(self.schemas)),
            MagicMock(return_value=np.zeros_like(self.schemas)),
        ]

        strategy = CompositeEvaluationStrategy(
            strategies=[NoOpEvaluationStrategy()],
            post_process=post_process,
        )

        values = strategy(schemas=self.schemas, pending=self.composite_action_schema)
        for mock in post_process:
            mock.assert_called_once()
            kwargs_to_mock = mock.call_args.kwargs

            self.assertEqual(3, len(kwargs_to_mock))
            self.assertListEqual(self.schemas, kwargs_to_mock['schemas'])
            self.assertEqual(self.composite_action_schema, kwargs_to_mock['pending'])
            np.testing.assert_array_equal(values, kwargs_to_mock['values'])

    def test_values(self):
        # test: verify that component strategies are called
        for strategy in self.strategy.strategies:
            strategy.values = MagicMock(return_value=np.ones_like(self.schemas))

        self.strategy.values(schemas=self.schemas, pending=self.composite_action_schema)

        for strategy in self.strategy.strategies:
            strategy.values.assert_called_once()
            kwargs = strategy.values.call_args.kwargs

            called_with_schemas: set[Schema] = set(kwargs['schemas'])
            called_with_pending: Schema = kwargs['pending']

            self.assertSetEqual(set(self.schemas), called_with_schemas)
            self.assertEqual(self.composite_action_schema, called_with_pending)

    def test_value_weighting(self):

        for weights in [(1.0, 0.0), (0.5, 0.5), (0.25, 0.75)]:
            self.strategy.weights = weights

            expected_values = sum(
                weight * strategy.values(self.schemas)
                for weight, strategy in zip(weights, self.strategy.strategies)
            )
            actual_values = self.strategy.values(self.schemas)

            np.testing.assert_array_almost_equal(np.array(expected_values), np.array(actual_values))

    def test_repr(self):
        attr_values = {
            'post_process': self.strategy.post_process,
            'strategy_alias': self.strategy.strategy_alias,
            'strategies': self.strategy.strategies,
            'weights': self.strategy.weights,
        }

        expected_str = repr_str(obj=self.strategy, attr_values=attr_values)
        self.assertEqual(expected_str, repr(self.strategy))


class TestDefaultGoalPursuitEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        # these shared tests will fail unless the strategy provided to them is idempotent. This is not the case for the
        # DefaultExploratoryEvaluationStrategy in general, therefore a special, highly contrived set of parameters
        # must be used. This seems to be a weakness of the current implementation of this strategy rather than of the
        # test cases in my opinion.
        self.strategy = DefaultGoalPursuitEvaluationStrategy(
            pending_focus_max_value=0.0,
            pending_focus_decay_rate=1e-24
        )

    def test_init(self):
        reliability_max_penalty = 0.765
        reliability_threshold = 0.652
        pending_focus_max_value = 0.12
        pending_focus_decay_rate = 0.002
        post_process = [display_minmax, display_values]

        # test: attributes should be set to given values when provided to initializer
        strategy = DefaultGoalPursuitEvaluationStrategy(
            reliability_max_penalty=reliability_max_penalty,
            reliability_threshold=reliability_threshold,
            pending_focus_max_value=pending_focus_max_value,
            pending_focus_decay_rate=pending_focus_decay_rate,
            post_process=post_process
        )

        self.assertEqual(reliability_max_penalty, strategy.reliability_max_penalty)
        self.assertEqual(reliability_threshold, strategy.reliability_threshold)
        self.assertEqual(pending_focus_max_value, strategy.pending_focus_max_value)
        self.assertEqual(pending_focus_decay_rate, strategy.pending_focus_decay_rate)
        self.assertListEqual(list(post_process), list(strategy.post_process))

        # test: default values should be set properly when other parameter values not provided
        strategy = DefaultGoalPursuitEvaluationStrategy()

        self.assertEqual(1.0, strategy.reliability_max_penalty)
        self.assertEqual(0.0, strategy.reliability_threshold)
        self.assertEqual(1.0, strategy.pending_focus_max_value)
        self.assertEqual(0.5, strategy.pending_focus_decay_rate)
        self.assertListEqual(list(), list(strategy.post_process))

        # test: weights should be equal
        np.testing.assert_array_equal(equal_weights(len(strategy.strategies)), strategy.weights)

    def test_mutators(self):
        reliability_max_penalty = 0.765
        reliability_threshold = 0.127
        pending_focus_max_value = 0.12
        pending_focus_decay_rate = 0.002
        post_process = [display_minmax, display_values]

        # test: attributes should be set to given values when provided to initializer
        strategy = DefaultGoalPursuitEvaluationStrategy()

        strategy.reliability_max_penalty = reliability_max_penalty
        strategy.reliability_threshold = reliability_threshold
        strategy.pending_focus_max_value = pending_focus_max_value
        strategy.pending_focus_decay_rate = pending_focus_decay_rate
        strategy.post_process = post_process

        self.assertEqual(reliability_max_penalty, strategy.reliability_max_penalty)
        self.assertEqual(reliability_threshold, strategy.reliability_threshold)
        self.assertEqual(pending_focus_max_value, strategy.pending_focus_max_value)
        self.assertEqual(pending_focus_decay_rate, strategy.pending_focus_decay_rate)
        self.assertListEqual(list(post_process), list(strategy.post_process))

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_post_process(self):
        post_process = [
            MagicMock(return_value=np.zeros_like(self.schemas)),
            MagicMock(return_value=np.zeros_like(self.schemas)),
            MagicMock(return_value=np.zeros_like(self.schemas)),
        ]

        strategy = CompositeEvaluationStrategy(
            strategies=[NoOpEvaluationStrategy()],
            post_process=post_process,
        )

        values = strategy(schemas=self.schemas, pending=self.composite_action_schema)
        for mock in post_process:
            mock.assert_called_once()
            kwargs_to_mock = mock.call_args.kwargs

            self.assertEqual(3, len(kwargs_to_mock))
            self.assertListEqual(self.schemas, kwargs_to_mock['schemas'])
            self.assertEqual(self.composite_action_schema, kwargs_to_mock['pending'])
            np.testing.assert_array_equal(values, kwargs_to_mock['values'])

    def test_values(self):
        # test: verify that component strategies are called
        for strategy in self.strategy.strategies:
            strategy.values = MagicMock(return_value=np.ones_like(self.schemas))

        self.strategy.values(schemas=self.schemas, pending=self.composite_action_schema)

        for strategy in self.strategy.strategies:
            strategy.values.assert_called_once()
            kwargs = strategy.values.call_args.kwargs

            called_with_schemas: set[Schema] = set(kwargs['schemas'])
            called_with_pending: Schema = kwargs['pending']

            self.assertSetEqual(set(self.schemas), called_with_schemas)
            self.assertEqual(self.composite_action_schema, called_with_pending)

    def test_value_weighting(self):

        for weights in [(1.0, 0.0, 0.0, 0.0, 0.0), (0.2, 0.2, 0.2, 0.2, 0.2), (0.25, 0.05, 0.0, 0.01, 0.69)]:
            self.strategy.weights = weights

            expected_values = sum(
                weight * strategy.values(self.schemas)
                for weight, strategy in zip(weights, self.strategy.strategies)
            )
            actual_values = self.strategy.values(self.schemas)

            np.testing.assert_array_almost_equal(np.array(expected_values), np.array(actual_values))

    def test_repr(self):
        attr_values = {
            'post_process': self.strategy.post_process,
            'strategy_alias': self.strategy.strategy_alias,
            'strategies': self.strategy.strategies,
            'weights': self.strategy.weights,
        }

        expected_str = repr_str(obj=self.strategy, attr_values=attr_values)
        self.assertEqual(expected_str, repr(self.strategy))


class TestDefaultEvaluationStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        # In order to make testing deterministic, the epsilon-random component strategy must be suppressed.
        self.strategy = DefaultEvaluationStrategy(
            exploratory_strategy=DefaultExploratoryEvaluationStrategy(
                epsilon_min=0.0,
                epsilon=0.0,
            )
        )

    def test_init(self):
        goal_pursuit_strategy = TotalPrimitiveValueEvaluationStrategy()
        exploratory_strategy = EpsilonRandomEvaluationStrategy()
        weights = [0.15, 0.85]
        weight_update_strategy = CyclicWeightUpdateStrategy(step_size=1e-7)
        post_process = [display_minmax]

        strategy = DefaultEvaluationStrategy(
            goal_pursuit_strategy=goal_pursuit_strategy,
            exploratory_strategy=exploratory_strategy,
            weights=weights,
            weight_update_strategy=weight_update_strategy,
            post_process=post_process,
        )

        # test: values should be set properly if they were explicitly provided to initializer
        self.assertEqual(goal_pursuit_strategy, strategy.goal_pursuit_strategy)
        self.assertEqual(exploratory_strategy, strategy.exploratory_strategy)
        self.assertListEqual(list(weights), list(strategy.weights))
        self.assertEqual(weight_update_strategy, strategy.weight_update_strategy)
        self.assertEqual(post_process, strategy.post_process)

        # test: default values should be properly set if they values were not explicitly provided to initializer
        strategy = DefaultEvaluationStrategy()

        self.assertEqual(DefaultGoalPursuitEvaluationStrategy(), strategy.goal_pursuit_strategy)
        self.assertEqual(DefaultExploratoryEvaluationStrategy(), strategy.exploratory_strategy)
        self.assertListEqual(list(equal_weights(2)), list(strategy.weights))
        self.assertEqual(CyclicWeightUpdateStrategy(), strategy.weight_update_strategy)
        self.assertListEqual(list(), list(strategy.post_process))

    def test_mutators(self):
        new_goal_pursuit_strategy = TotalDelegatedValueEvaluationStrategy()
        new_exploratory_strategy = EpsilonRandomEvaluationStrategy(
            epsilon=0.78,
            epsilon_min=0.12,
            decay_strategy=ExponentialDecayStrategy(rate=0.1)
        )
        new_weight_update_strategy = CyclicWeightUpdateStrategy(step_size=1e-4)
        new_weights = [0.35, 0.65]

        self.strategy.goal_pursuit_strategy = new_goal_pursuit_strategy
        self.strategy.exploratory_strategy = new_exploratory_strategy
        self.strategy.weight_update_strategy = new_weight_update_strategy
        self.strategy.weights = new_weights

        self.assertEqual(new_goal_pursuit_strategy, self.strategy.goal_pursuit_strategy)
        self.assertEqual(new_exploratory_strategy, self.strategy.exploratory_strategy)
        self.assertEqual(new_weight_update_strategy, self.strategy.weight_update_strategy)
        self.assertEqual(new_weights, self.strategy.weights)

    def test_values(self):
        # test: values should be the weighted sum of the goal-pursuit and exploratory strategies
        strategies = [
            self.strategy.goal_pursuit_strategy,
            self.strategy.exploratory_strategy,
        ]

        expected_values = sum(
            [
                weight * strategy.values(self.schemas)
                for weight, strategy in zip(self.strategy.weights, strategies)
            ]
        )

        actual_values = self.strategy.values(self.schemas)
        np.testing.assert_array_equal(expected_values, actual_values)

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_weight_updates(self):
        for _ in range(100):
            before_weights = copy(self.strategy.weights)
            self.strategy(schemas=self.schemas)
            after_weights = copy(self.strategy.weights)

            np.testing.assert_array_equal(
                self.strategy.weight_update_strategy.update(before_weights),
                after_weights
            )

    def test_repr(self):
        pass
