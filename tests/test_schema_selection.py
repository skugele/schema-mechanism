import unittest
from collections import defaultdict
from collections import deque
from collections.abc import Sequence
from typing import Optional
from unittest import TestCase

import numpy as np

from schema_mechanism.core import Chain
from schema_mechanism.core import ItemPool
from schema_mechanism.core import Schema
from schema_mechanism.core import calc_delegated_value
from schema_mechanism.core import calc_primitive_value
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import PendingDetails
from schema_mechanism.modules import PendingStatus
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.share import GlobalParams
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.evaluation import EpsilonGreedyExploratoryStrategy
from schema_mechanism.strategies.evaluation import GoalPursuitEvaluationStrategy
from schema_mechanism.strategies.match import AbsoluteDiffMatchStrategy
from schema_mechanism.strategies.selection import RandomizeBestSelectionStrategy
from test_share.test_classes import MockSchema
from test_share.test_classes import MockSchemaSelection
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup


def primitive_value_evaluation_strategy(schemas: Sequence[Schema], _pending: Optional[Schema] = None) -> np.ndarray:
    values = list([calc_primitive_value(s.result) for s in schemas])
    return np.array(values)


def delegated_value_evaluation_strategy(schemas: Sequence[Schema], _pending: Optional[Schema] = None) -> np.ndarray:
    values = list([calc_delegated_value(s.result) for s in schemas])
    return np.array(values)


class TestSchemaSelection(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        # allows direct setting of reliability (only reliable schemas are eligible for chaining)
        GlobalParams().set('schema_type', MockSchema)

        pool = ItemPool()

        self.i1 = pool.get('1', item_type=MockSymbolicItem, primitive_value=0.0, avg_accessible_value=-3.0)
        self.i2 = pool.get('2', item_type=MockSymbolicItem, primitive_value=0.0, avg_accessible_value=2.0)
        self.i3 = pool.get('3', item_type=MockSymbolicItem, primitive_value=0.95, avg_accessible_value=-1.0)
        self.i4 = pool.get('4', item_type=MockSymbolicItem, primitive_value=-1.0, avg_accessible_value=0.95)
        self.i5 = pool.get('5', item_type=MockSymbolicItem, primitive_value=2.0, avg_accessible_value=0.0)
        self.i6 = pool.get('6', item_type=MockSymbolicItem, primitive_value=-3.0, avg_accessible_value=0.0)

        # note: negated items have zero pv
        self.s1 = sym_schema('/A1/', reliability=0.0)  # total pv = 0.0; total dv = 0.0
        self.s2 = sym_schema('/A2/', reliability=0.0)  # total pv = 0.0; total dv = 0.0
        self.s3 = sym_schema('/A3/', reliability=0.0)  # total pv = 0.0; total dv = 0.0

        self.s1_c12_r3 = sym_schema('1,2/A1/3,', reliability=1.0)  # total pv = 0.95; total dv = -1.0
        self.s1_c12_r4 = sym_schema('1,2/A1/4,', reliability=1.0)  # total pv = -1.0; total dv = 0.95
        self.s1_c12_r5 = sym_schema('1,2/A1/5,', reliability=1.0)  # total pv = 2.0; total dv = 0.0
        self.s1_c12_r34 = sym_schema('1,2/A1/(3,4),', reliability=1.0)  # total pv = -0.05; total dv = 0.95
        self.s1_c12_r345 = sym_schema('1,2/A1/(3,4,5),', reliability=1.0)  # total pv = 1.95; total dv = 0.95
        self.s1_c12_r3456 = sym_schema('1,2/A1/(3,4,5,6),', reliability=1.0)  # total pv = -1.05; total dv = 0.95
        self.s1_c12_r345_not6 = sym_schema('1,2/A1/(3,4,5,~6),', reliability=1.0)  # total pv = 1.95; total dv = 0.95

        # chained schemas
        self.s1_c12_r2 = sym_schema('1,2/A1/2,', reliability=1.0)  # total pv = 0.0; total dv = 2.0
        self.s2_c2_r3 = sym_schema('2,/A2/3,', reliability=1.0)  # total pv = 0.95; total dv = -1.0
        self.s3_c3_r4 = sym_schema('3,/A3/4,', reliability=1.0)  # total pv = -1.0; total dv = 0.95
        self.s1_c4_r5 = sym_schema('4,/A1/5,', reliability=1.0)  # total pv = 2.0; total dv = 0.95
        self.s3_c5_r24 = sym_schema('5,/A3/(2,4),', reliability=1.0)  # total pv = -1.0; total dv = 2.0

        # composite action schemas
        self.sca24_c12_r35 = sym_schema('1,2/2,4/(3,5),', reliability=1.0)  # total pv = 2.95; total dv = 0.95
        self.sca24_c12_r136 = sym_schema('1,2/2,4/(1,3,6),', reliability=1.0)  # total pv = -2.05; total dv = 0.0
        self.sca5_c2_r35 = sym_schema('2,/5,/(3,5),', reliability=1.0)  # total pv = 2.95; total dv = 0.95

        self.sca24_chains = [Chain([self.s1_c12_r2, self.s2_c2_r3, self.s3_c3_r4, self.s1_c4_r5, self.s3_c5_r24])]
        self.sca24_c12_r35.action.controller.update(self.sca24_chains)
        self.sca24_c12_r136.action.controller.update(self.sca24_chains)

        self.sca5_chains = [Chain([self.s1_c12_r2, self.s2_c2_r3, self.s3_c3_r4, self.s1_c4_r5])]
        self.sca5_c2_r35.action.controller.update(self.sca5_chains)

        self.selection_state = sym_state('1,2')

        self.ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(1.0)),
            value_strategies=[
                GoalPursuitEvaluationStrategy(),
                EpsilonGreedyExploratoryStrategy(0.9)
            ],
        )

    def test_init(self):
        # test defaults
        ss = SchemaSelection()

        # test: a default selection strategy should be assigned if not explicitly requested
        self.assertIsNotNone(ss.select_strategy)

        # test: value strategies should be an empty collection
        self.assertEqual(0, len(ss.value_strategies))

        # test: weights should be an empty numpy array
        self.assertTrue(np.array_equal(np.array([]), ss.weights))

        select_strategy = RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1))
        value_strategies = [primitive_value_evaluation_strategy, delegated_value_evaluation_strategy]
        weights = [0.4, 0.6]

        ss = SchemaSelection(
            select_strategy=select_strategy,
            value_strategies=value_strategies,
            weights=weights
        )

        # test: initializer should set select strategy to the requested strategy
        self.assertEqual(select_strategy, ss.select_strategy)

        # test: initializer should set value strategies to the requested strategies
        self.assertListEqual(value_strategies, list(ss.value_strategies))

        # test: initializer should set weights to the requested values
        self.assertListEqual(weights, list(ss.weights))

        # test: should raise a ValueError if initializer given an invalid number of weights
        self.assertRaises(ValueError, lambda: SchemaSelection(select_strategy, value_strategies, weights=[0.1]))

        # test: should raise a ValueError if the weights do not sum to 1.0
        self.assertRaises(ValueError, lambda: SchemaSelection(select_strategy, value_strategies, weights=[0.1, 0.2]))

        # test: should raise a ValueError if any of the weights are non-positive
        self.assertRaises(ValueError, lambda: SchemaSelection(select_strategy, value_strategies, weights=[1.8, -0.8]))

        # test: weights of 0.0 and 1.0 should be allowed
        try:
            SchemaSelection(select_strategy, value_strategies, weights=[0.0, 1.0])
        except ValueError as e:
            self.fail(f'Unexpected ValueError occurred: {str(e)}')

    def test_select_1(self):
        # sanity checks
        ###############

        # Empty or None list of applicable schemas should return a ValueError
        self.assertRaises(ValueError, lambda: self.ss.select(schemas=[], state=self.selection_state))

        # noinspection PyTypeChecker
        self.assertRaises(ValueError, lambda: self.ss.select(schemas=None, state=self.selection_state))

        # Given a single applicable schema, select should return that schema
        schema = sym_schema('1,2/A1/3,4')
        sd = self.ss.select(schemas=[schema], state=self.selection_state)
        self.assertEqual(schema, sd.selected)

    def test_select_2(self):
        # primitive value-based selections
        ##################################
        ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(1.0)),
            value_strategies=[primitive_value_evaluation_strategy]
        )

        # selection between uneven schemas, all with non-negated items (s1_c12_r5 should win)
        sd = ss.select(schemas=[self.s1_c12_r3, self.s1_c12_r4, self.s1_c12_r5, self.s1_c12_r34],
                       state=self.selection_state)
        self.assertEqual(self.s1_c12_r5, sd.selected)

        # selection between uneven schemas, some with negated items (s1_c12_r345_not6 should win)
        sd = ss.select(schemas=[self.s1_c12_r34, self.s1_c12_r3456, self.s1_c12_r345_not6], state=self.selection_state)
        self.assertEqual(self.s1_c12_r345_not6, sd.selected)

        # selection between multiple schemas with close values should be randomized
        selections = defaultdict(lambda: 0.0)
        applicable_schemas = [self.s1_c12_r5, self.s1_c12_r345, self.s1_c12_r345_not6]
        for _ in range(100):
            sd = ss.select(applicable_schemas, state=self.selection_state)
            selections[sd.selected] += 1

        self.assertEqual(len(applicable_schemas), len(selections.keys()))

    def test_calc_effective_values(self):
        s1 = sym_schema('/A1/1,')  # pv = 0.0
        s2 = sym_schema('/A2/2,')  # pv = 0.0
        s3 = sym_schema('/A1/3,')  # pv = 0.95
        s4 = sym_schema('/A3/4,')  # pv = -1.0
        s5 = sym_schema('/A1/5,')  # pv = 2.0
        s6 = sym_schema('/A2/6,')  # pv = -3.0

        self.s1_c12_r34 = sym_schema('1,2/A1/(3,4),', reliability=1.0)  # total pv = -0.05; total dv = 0.0
        self.s1_c12_r345 = sym_schema('1,2/A1/(3,4,5),', reliability=1.0)  # total pv = 1.95; total dv = 0.0
        self.s1_c12_r3456 = sym_schema('1,2/A1/(3,4,5,6),', reliability=1.0)  # total pv = -1.05; total dv = 0.0
        self.s1_c12_r345_not6 = sym_schema('1,2/A1/(3,4,5,~6),', reliability=1.0)  # total pv = 1.95; total dv = 0.0

        schemas = [s1, s2, s3, s4, s5, s6, self.s1_c12_r34, self.s1_c12_r345, self.s1_c12_r3456, self.s1_c12_r345_not6]

        # testing with no evaluation strategies
        ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1)),
            value_strategies=[],
        )

        # test: should returns array of zeros if no value strategies specified
        self.assertTrue(np.array_equal(np.zeros_like(schemas), ss.calc_effective_values(schemas)))

        # testing single evaluation strategy
        ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1)),
            value_strategies=[primitive_value_evaluation_strategy],
        )

        expected_values = primitive_value_evaluation_strategy(schemas)

        # test: primitive-only value strategy should return primitive values for each schema
        self.assertTrue(np.array_equal(expected_values, ss.calc_effective_values(schemas, pending=None)))

        # testing multiple evaluation strategies (equal weighting)
        ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1)),
            value_strategies=[primitive_value_evaluation_strategy, delegated_value_evaluation_strategy],
        )

        pvs = primitive_value_evaluation_strategy(schemas)
        dvs = delegated_value_evaluation_strategy(schemas)
        expected_values = (pvs + dvs) / 2.0
        actual_values = ss.calc_effective_values(schemas, pending=None)

        # test: should return weighted sum of evaluation strategy values
        self.assertTrue(np.array_equal(expected_values, actual_values))

        # testing multiple evaluation strategies (uneven weighting)
        ss.weights = np.array([0.95, 0.05])

        expected_values = 0.95 * pvs + 0.05 * dvs
        actual_values = ss.calc_effective_values(schemas, pending=None)

        # test: should return weighted sum of evaluation strategy values
        self.assertTrue(np.array_equal(expected_values, actual_values))

    def test_weights(self):
        ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1)),
            value_strategies=[primitive_value_evaluation_strategy, delegated_value_evaluation_strategy],
        )

        # test: initializer should set weights to the requested values
        weights = [0.4, 0.6]
        ss.weights = weights
        self.assertListEqual(weights, list(ss.weights))

        # test: should raise a ValueError if initializer given an invalid number of weights
        with self.assertRaises(ValueError):
            ss.weights = [1.0]

        # test: should raise a ValueError if the weights do not sum to 1.0
        with self.assertRaises(ValueError):
            ss.weights = [0.1, 0.2]

        # test: should raise a ValueError if any of the weights are non-positive
        with self.assertRaises(ValueError):
            ss.weights = [1.8, -0.8]

        # test: weights of 0.0 and 1.0 should be allowed
        try:
            ss.weights = [0.0, 1.0]
        except ValueError as e:
            self.fail(f'Unexpected ValueError occurred: {str(e)}')

    def test_select_with_pending_schema_1a(self):
        # testing selection with composite action schema (scenario: composite action schema wins)
        ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1)),
            value_strategies=[primitive_value_evaluation_strategy],
        )

        self.assertIs(None, ss.pending_schema)

        schema_with_ca = self.sca24_c12_r35

        applicable = [
            self.s1,  # total pv = 0.0
            self.s1_c12_r3,  # total pv = 0.95
            self.s1_c12_r4,  # total pv = -1.0
            self.s1_c12_r5,  # total pv = 2.0
            self.s1_c12_r34,  # total pv = -0.05
            self.s1_c12_r345,  # total pv = 1.95
            self.s1_c12_r3456,  # total pv = -1.05
            self.s1_c12_r345_not6,  # total pv = 1.95
            self.s1_c12_r2,  # total pv = 0.0  [also a component of schema_with_ca]

            schema_with_ca  # total pv = 2.95
        ]

        expected_values = [0.0, 0.95, -1.0, 2.0, -0.05, 1.95, -1.05, 1.95, 0.0, 2.95]
        actual_values = primitive_value_evaluation_strategy(applicable)

        # sanity check:
        for exp, act in zip(expected_values, actual_values):
            self.assertAlmostEqual(exp, act)

        # test: pending schema should be selected
        sd = ss.select(applicable, state=sym_state('1,2'))

        # test: the pending schema should be set as the composite action schema
        self.assertIs(schema_with_ca, ss.pending_schema)

        # test: a schema should have been selected from the composite action's component schemas
        self.assertIn(sd.selected, ss.pending_schema.action.controller.components)

    def test_select_with_pending_schema_1b(self):
        # testing selection with composite action schema (scenario: composite action schema loses)
        ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            value_strategies=[primitive_value_evaluation_strategy],
        )

        self.assertIs(None, ss.pending_schema)

        schema_with_ca = self.sca24_c12_r136

        applicable = [
            self.s1,  # total pv = 0.0
            self.s1_c12_r3,  # total pv = 0.95
            self.s1_c12_r4,  # total pv = -1.0
            self.s1_c12_r5,  # total pv = 2.0
            self.s1_c12_r34,  # total pv = -0.05
            self.s1_c12_r345,  # total pv = 1.95
            self.s1_c12_r3456,  # total pv = -1.05
            self.s1_c12_r345_not6,  # total pv = 1.95
            self.s1_c12_r2,  # total pv = 0.0  [also a component of schema_with_ca]

            schema_with_ca  # total pv = -2.05
        ]

        expected_values = [0.0, 0.95, -1.0, 2.0, -0.05, 1.95, -1.05, 1.95, 0.0, -2.05]
        actual_values = primitive_value_evaluation_strategy(applicable)

        # sanity check:
        for exp, act in zip(expected_values, actual_values):
            self.assertAlmostEqual(exp, act)

        # test: pending schema should NOT be selected
        sd = ss.select(applicable, state=sym_state('1,2'))

        # test: terminated pending list in selection details should be empty
        self.assertEqual(0, len(sd.terminated_pending))

        self.assertIs(self.s1_c12_r5, sd.selected)
        self.assertIs(None, ss.pending_schema)

    def test_select_with_pending_schema_2(self):
        # testing the selection of a pending schema's components to completion

        # mock is used to directly set the pending schema
        mock_ss = MockSchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            value_strategies=[primitive_value_evaluation_strategy],
            pending_schemas=deque([PendingDetails(schema=self.sca24_c12_r35, selection_state=sym_state('2'))])
        )

        # sanity check: pending schema set
        self.assertIs(mock_ss.pending_schema, self.sca24_c12_r35)

        selection_states = [sym_state('2'), sym_state('3'), sym_state('4'), sym_state('5')]

        applicable_schemas = [
            [self.s2_c2_r3, self.s1_c12_r2],
            [self.s3_c3_r4],
            [self.s1_c4_r5],
            [self.s3_c5_r24],
        ]

        # test: selection should iterate through controller chain to goal state
        for state, applicable in zip(selection_states, applicable_schemas):
            sd = mock_ss.select(applicable, state)
            self.assertIs(applicable[0], sd.selected)
            self.assertIs(mock_ss.pending_schema, self.sca24_c12_r35)
            self.assertIn(sd.selected, self.sca24_c12_r35.action.controller.components)

        applicable = [sym_schema('2,4/A1/5,')]
        state = sym_state('2,4')
        mock_ss.select(applicable, state)

        # test: pending schema SHOULD be None since selection state was pending schema's controller's goal state
        self.assertEqual(None, mock_ss.pending_schema)

    def test_select_with_pending_schema_3(self):
        # testing the abortion of a pending schema (selection state leads to no applicable components)

        # mock is used to directly set the pending schema
        mock_ss = MockSchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            value_strategies=[primitive_value_evaluation_strategy],
            pending_schemas=deque([PendingDetails(schema=self.sca24_c12_r35, selection_state=sym_state('2'))])
        )

        # sanity check: pending schema set
        self.assertIs(mock_ss.pending_schema, self.sca24_c12_r35)

        selection_states = [
            sym_state('2'),  # satisfies context of component self.s2_c2_r3
            sym_state('3'),  # satisfies context of component self.s3_c3_r4
            sym_state('6'),  # no applicable components (should terminate pending schema)
        ]

        applicable_schemas = [
            [self.s2_c2_r3],
            [self.s3_c3_r4],
            [self.s3],
        ]

        for state, applicable in zip(selection_states, applicable_schemas):
            sd = mock_ss.select(applicable, state)

            self.assertIs(applicable[0], sd.selected)

            # current state does NOT satisfy a controller component
            if state == selection_states[-1]:

                self.assertEqual(1, len(sd.terminated_pending))

                # test: most recent entry in pending list should contain this pending schema with ABORTED status
                self.assertEqual(self.sca24_c12_r35, sd.terminated_pending[-1].schema)
                self.assertEqual(PendingStatus.ABORTED, sd.terminated_pending[-1].status)

                # test: pending schema should be set to None if no applicable controller components
                self.assertEqual(None, mock_ss.pending_schema)

            # current state satisfies a controller component
            else:
                # test: pending schema should remain the same
                self.assertIs(mock_ss.pending_schema, self.sca24_c12_r35)

                # test: selected schema should be a controller component
                self.assertIn(sd.selected, self.sca24_c12_r35.action.controller.components)

    def test_select_with_pending_schema_4(self):
        # testing the interruption of a pending schema (alternative schema with much higher value)
        pending = self.sca24_c12_r35

        # mock is used to directly set the pending schema
        mock_ss = MockSchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            value_strategies=[primitive_value_evaluation_strategy],
            pending_schemas=deque([PendingDetails(schema=pending, selection_state=sym_state('2'))])
        )

        # sanity check: pending schema set
        self.assertIs(mock_ss.pending_schema, pending)

        _ = ItemPool().get('HIGH', item_type=MockSymbolicItem, primitive_value=np.inf, avg_accessible_value=np.inf)
        s_high = sym_schema('2,/A2/HIGH,', reliability=1.0)

        # satisfies context of component self.s2_c2_r3 and high-value alternative
        sd = mock_ss.select([self.s2_c2_r3, s_high], sym_state('2'))

        # test: selected schema should not be a component of the pending schema
        self.assertNotIn(sd.selected, pending.action.controller.components)

        # test: there should be no pending schema set in SchemaSelection after losing the last selection
        self.assertIs(None, mock_ss.pending_schema)

        # test: most recent entry in pending list should contain this pending schema with INTERRUPTED status
        self.assertEqual(pending, sd.terminated_pending[-1].schema)
        self.assertEqual(PendingStatus.INTERRUPTED, sd.terminated_pending[-1].status)

    def test_select_with_pending_schema_5(self):
        # testing the interruption of a pending schema for another composite action (alternative has higher value)
        pending = self.sca24_c12_r136

        # mock is used to directly set the pending schema
        mock_ss = MockSchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            value_strategies=[primitive_value_evaluation_strategy],
            pending_schemas=deque([PendingDetails(schema=pending, selection_state=sym_state('2'))])
        )

        # sanity check: pending schema set
        self.assertIs(mock_ss.pending_schema, pending)

        # satisfies context pending schema, higher-value alternative sca24_c12_r35, and non-composite component s2_c2_r3
        sd = mock_ss.select([pending, self.s2_c2_r3, self.sca24_c12_r35], sym_state('1,2'))

        # test: there should be new pending schema set in SchemaSelection
        self.assertIs(self.sca24_c12_r35, mock_ss.pending_schema)

        # test: there should be 1 interrupted schema in terminated list
        self.assertEqual(1, len(sd.terminated_pending))
        self.assertEqual(PendingStatus.INTERRUPTED, sd.terminated_pending[0].status)

    def test_select_with_pending_schema_6(self):
        # testing completed pending schema followed by immediate selection of another composite action schema
        pending = self.sca24_c12_r136

        # mock is used to directly set the pending schema
        mock_ss = MockSchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            value_strategies=[primitive_value_evaluation_strategy],
            pending_schemas=deque([PendingDetails(schema=pending, selection_state=sym_state('2'))])
        )

        # sanity check: pending schema set
        self.assertIs(mock_ss.pending_schema, pending)

        # satisfies context of component self.s2_c2_r3 and high-value alternative
        sd = mock_ss.select([pending, self.s2_c2_r3, self.sca5_c2_r35], sym_state('2,4'))

        # test: there should be a new pending schema set in SchemaSelection
        self.assertIs(self.sca5_c2_r35, mock_ss.pending_schema)

        # test: there should be a single entry in selection detail's terminated pending list for completed schema
        self.assertEqual(1, len(sd.terminated_pending))

        # test: oldest entry in pending list should contain previous pending schema with INTERRUPTED status
        self.assertEqual(pending, sd.terminated_pending[0].schema)
        self.assertEqual(PendingStatus.COMPLETED, sd.terminated_pending[0].status)

    # FIXME: Uncomment this if/when components with composite actions are supported
    # def test_select_with_nested_pending_schemas(self):
    #     # testing selection from nested pending schemas with composite actions
    #
    #     # Items
    #     #######
    #     pool = ItemPool()
    #
    #     _ = pool.get('1', item_type=MockSymbolicItem, primitive_value=0.0)
    #     _ = pool.get('2', item_type=MockSymbolicItem, primitive_value=0.0)
    #     _ = pool.get('3', item_type=MockSymbolicItem, primitive_value=0.95)
    #     _ = pool.get('4', item_type=MockSymbolicItem, primitive_value=-1.0)
    #     _ = pool.get('5', item_type=MockSymbolicItem, primitive_value=2.0)
    #     _ = pool.get('6', item_type=MockSymbolicItem, primitive_value=-3.0)
    #
    #     _ = pool.get('J', item_type=MockSymbolicItem, primitive_value=10.0)
    #     _ = pool.get('K', item_type=MockSymbolicItem, primitive_value=15.0)
    #     _ = pool.get('L', item_type=MockSymbolicItem, primitive_value=20.0)
    #
    #     _ = pool.get('M', item_type=MockSymbolicItem, primitive_value=50.0)
    #     _ = pool.get('N', item_type=MockSymbolicItem, primitive_value=100.0)
    #
    #     _ = pool.get('Z', item_type=MockSymbolicItem, primitive_value=1000.0)
    #
    #     # L0 Schemas
    #     ############
    #
    #     # 1>2
    #     s_1_2 = sym_schema('1,/A1/2,', reliability=1.0)
    #
    #     # 2>3
    #     s_2_3 = sym_schema('2,/A2/3,', reliability=1.0)
    #
    #     # 3>4
    #     s_3_4 = sym_schema('3,/A3/4,', reliability=1.0)
    #
    #     # 4>5
    #     s_4_5 = sym_schema('4,/A1/5,', reliability=1.0)
    #
    #     # 5>6
    #     s_5_6 = sym_schema('5,/A2/6,', reliability=1.0)
    #
    #     # L1 Schemas
    #     ############
    #
    #     # 1>3j
    #     c_1_3j = sym_schema('1,/3,/(3,J),', reliability=1.0)
    #
    #     chains = [Chain([s_1_2, s_2_3])]
    #     c_1_3j.action.controller.update(chains)
    #
    #     # 2>4k
    #     c_2_4k = sym_schema('2,/4,/(4,K),', reliability=1.0)
    #
    #     chains = [Chain([s_2_3, s_3_4])]
    #     c_2_4k.action.controller.update(chains)
    #
    #     # 3>6l
    #     c_3_6l = sym_schema('3,/6,/(6,L),', reliability=1.0)
    #
    #     chains = [Chain([s_3_4, s_4_5, s_5_6])]
    #     c_3_6l.action.controller.update(chains)
    #
    #     # L2 Schemas
    #     ############
    #     # c1_4km: s_1_2 > c_2_4k
    #     c1_4km = sym_schema('1,/(4,K),/(4,K,M),', reliability=1.0)
    #
    #     chains = [Chain([s_1_2, c_2_4k])]
    #     c1_4km.action.controller.update(chains)
    #
    #     # C23_35: c_2_4k > C3_5
    #     c2_6ln = sym_schema('2,/(6,L),/(6,L,N),', reliability=1.0)
    #
    #     chains = [Chain([c_1_3j, c_3_6l])]
    #     c2_6ln.action.controller.update(chains)
    #
    #     # L3 Schemas
    #     ############
    #     c123_235 = sym_schema('1,/(6,L,N),/Z,', reliability=1.0)
    #
    #     chains = [Chain([s_1_2, c2_6ln])]
    #     c123_235.action.controller.update(chains)
    #
    #     # mock is used to directly set the pending schema
    #     mock_ss = SchemaSelection(
    #         select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
    #         value_strategies=[primitive_value_evaluation_strategy]
    #     )
    #
    #     sd = mock_ss.select(schemas=[s_1_2, c_1_3j, c123_235], state=sym_state('1'))
    #     self.assertEqual(s_1_2, sd.selected)


class TestEpsilonGreedy(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.eps_always_explore = EpsilonGreedyExploratoryStrategy(epsilon=1.0)
        self.eps_never_explore = EpsilonGreedyExploratoryStrategy(epsilon=0.0)
        self.eps_even_chance = EpsilonGreedyExploratoryStrategy(epsilon=0.5)

        self.schema = sym_schema('1,2/A/3,4')

        # schema with a composite action
        self.ca_schema = sym_schema('1,2/S1,/5,7')

        self.schemas = [self.schema,
                        self.ca_schema,
                        sym_schema('1,2/A/3,5'),
                        sym_schema('1,2/A/3,6')]

    def test_init(self):
        epsilon = 0.1
        eps_greedy = EpsilonGreedyExploratoryStrategy(epsilon)
        self.assertEqual(epsilon, eps_greedy.epsilon)

    def test_epsilon_setter(self):
        eps_greedy = EpsilonGreedyExploratoryStrategy(0.5)

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
        eps_greedy = EpsilonGreedyExploratoryStrategy(epsilon=epsilon, decay_strategy=decay_strategy)

        prev_epsilon = epsilon
        for _ in range(100):
            _ = eps_greedy(schemas=self.schemas)
            self.assertEqual(decay_strategy.decay(prev_epsilon), eps_greedy.epsilon)
            prev_epsilon = eps_greedy.epsilon

    def test_epsilon_decay_to_minimum(self):
        epsilon = 1.0
        minimum = 0.5

        decay_strategy = GeometricDecayStrategy(rate=0.99, minimum=minimum)
        eps_greedy = EpsilonGreedyExploratoryStrategy(epsilon=epsilon, decay_strategy=decay_strategy)

        for _ in range(100):
            _ = eps_greedy(schemas=self.schemas)

        self.assertEqual(minimum, eps_greedy.epsilon)

    def test_pending_bypass(self):
        # test: epsilon greedy values SHOULD all be zero when a pending schema is provided
        expected = np.zeros_like(self.schemas)
        actual = self.eps_always_explore(self.schemas, pending=self.schema)

        self.assertTrue(np.array_equal(expected, actual))
