from collections import defaultdict
from collections import deque
from unittest import TestCase

import numpy as np

from schema_mechanism.core import Chain
from schema_mechanism.core import ItemPool
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import PendingDetails
from schema_mechanism.modules import PendingStatus
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.share import GlobalParams
from schema_mechanism.strategies.evaluation import NoOpEvaluationStrategy
from schema_mechanism.strategies.evaluation import PrimitiveValueEvaluationStrategy
from schema_mechanism.strategies.match import AbsoluteDiffMatchStrategy
from schema_mechanism.strategies.selection import RandomizeBestSelectionStrategy
from test_share.test_classes import MockSchema
from test_share.test_classes import MockSchemaSelection
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup


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

    def test_init(self):
        # test defaults
        ss = SchemaSelection()

        # test: a default selection strategy should be assigned if not explicitly requested
        self.assertIsNotNone(ss.select_strategy)

        # test: if not explicitly set, value strategies should be set to NoOpEvaluationStrategies
        self.assertIsInstance(ss.evaluation_strategy, NoOpEvaluationStrategy)

        select_strategy = RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1))
        evaluation_strategy = PrimitiveValueEvaluationStrategy()

        ss = SchemaSelection(
            select_strategy=select_strategy,
            evaluation_strategy=evaluation_strategy
        )

        # test: initializer should set select strategy to the requested strategy
        self.assertEqual(select_strategy, ss.select_strategy)

        # test: initializer should set exploratory evaluation strategy to the requested strategy
        self.assertEqual(evaluation_strategy, ss.evaluation_strategy)

    def test_select_no_applicable_schemas(self):
        ss = SchemaSelection()

        # Empty or None list of applicable schemas should return a ValueError
        self.assertRaises(ValueError, lambda: ss.select(schemas=[], state=self.selection_state))

        # noinspection PyTypeChecker
        self.assertRaises(ValueError, lambda: ss.select(schemas=None, state=self.selection_state))

    def test_select_single_applicable_schema(self):
        ss = SchemaSelection()

        # given a single applicable schema, select should return that schema
        schema = sym_schema('1,2/A1/3,4')
        sd = ss.select(schemas=[schema], state=self.selection_state)
        self.assertEqual(schema, sd.selected)

    def test_select_primitive_action_schema(self):
        # primitive value-based selections
        ##################################
        ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(1.0)),
            evaluation_strategy=PrimitiveValueEvaluationStrategy()
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

    def test_select_composite_action_schema(self):
        # testing selection with composite action schema (scenario: composite action schema wins)
        ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.1)),
            evaluation_strategy=PrimitiveValueEvaluationStrategy(),
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
        actual_values = ss.evaluation_strategy(applicable)

        # sanity check:
        for exp, act in zip(expected_values, actual_values):
            self.assertAlmostEqual(exp, act)

        # test: pending schema should be selected
        sd = ss.select(applicable, state=sym_state('1,2'))

        # test: the pending schema should be set as the composite action schema
        self.assertIs(schema_with_ca, ss.pending_schema)

        # test: a schema should have been selected from the composite action's component schemas
        self.assertIn(sd.selected, ss.pending_schema.action.controller.components)

    def test_select_composite_action_with_pending_schema(self):
        # testing selection with composite action schema (scenario: composite action schema loses)
        ss = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            evaluation_strategy=PrimitiveValueEvaluationStrategy()
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
        actual_values = ss.evaluation_strategy(applicable)

        # sanity check:
        for exp, act in zip(expected_values, actual_values):
            self.assertAlmostEqual(exp, act)

        # test: pending schema should NOT be selected
        sd = ss.select(applicable, state=sym_state('1,2'))

        # test: terminated pending list in selection details should be empty
        self.assertEqual(0, len(sd.terminated_pending))

        self.assertIs(self.s1_c12_r5, sd.selected)
        self.assertIs(None, ss.pending_schema)

    def test_repeated_select_from_pending_schema_components_until_completion(self):
        # testing the selection of a pending schema's components to completion

        # mock is used to directly set the pending schema
        mock_ss = MockSchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            evaluation_strategy=PrimitiveValueEvaluationStrategy(),
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

    def test_select_with_aborted_pending_schema(self):
        # testing the abortion of a pending schema (selection state leads to no applicable components)

        # mock is used to directly set the pending schema
        mock_ss = MockSchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            evaluation_strategy=PrimitiveValueEvaluationStrategy(),
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

    def test_select_with_interrupted_pending_schema_by_alternate_primitive_action_schema(self):
        # testing the interruption of a pending schema (alternative schema with much higher value)
        pending = self.sca24_c12_r35

        # mock is used to directly set the pending schema
        mock_ss = MockSchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            evaluation_strategy=PrimitiveValueEvaluationStrategy(),
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

    def test_select_with_interrupted_pending_schema_by_alternate_composite_action_schema(self):
        # testing the interruption of a pending schema for another composite action (alternative has higher value)
        pending = self.sca24_c12_r136

        # mock is used to directly set the pending schema
        mock_ss = MockSchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            evaluation_strategy=PrimitiveValueEvaluationStrategy(),
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

    def test_select_with_completed_pending_schema(self):
        # testing completed pending schema followed by immediate selection of another composite action schema
        pending = self.sca24_c12_r136

        # mock is used to directly set the pending schema
        mock_ss = MockSchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(0.01)),
            evaluation_strategy=PrimitiveValueEvaluationStrategy(),
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
