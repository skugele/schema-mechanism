from unittest import TestCase
from unittest.mock import ANY
from unittest.mock import MagicMock

import numpy as np

from schema_mechanism.core import Action
from schema_mechanism.core import Chain
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaTree
from schema_mechanism.func_api import actions
from schema_mechanism.func_api import primitive_schemas
from schema_mechanism.func_api import sym_asserts
from schema_mechanism.func_api import sym_item_assert
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.modules import create_context_spin_off
from schema_mechanism.modules import create_result_spin_off
from schema_mechanism.share import GlobalParams
from test_share.test_classes import MockSchema
from test_share.test_func import common_test_setup


class TestSchemaMemory(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        # allows direct setting of reliability (only reliable schemas are eligible for chaining)
        GlobalParams().set('schema_type', MockSchema)
        GlobalParams().set('backward_chains_update_frequency', 1.0)

        # always create composite actions for novel results
        GlobalParams().set('composite_action_min_baseline_advantage', -np.inf)

        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A2/')

        # composite action schemas
        self.s101 = sym_schema('/101,/')
        self.s102 = sym_schema('/102,/')

        self.tree = SchemaTree((s1, s2, self.s101, self.s102))

        # composite action result spin-off
        self.s101_r100 = sym_schema('/101,/100,')
        self.tree.add_result_spin_offs(self.s101, (self.s101_r100,))

        # composite action context spin-off
        self.s101_1_r100 = sym_schema('1,/101,/100,', reliability=1.0)
        self.tree.add_context_spin_offs(self.s101_r100, (self.s101_1_r100,))

        s1_1 = sym_schema('1,/A1/')
        s1_2 = sym_schema('2,/A1/')
        s1_3 = sym_schema('3,/A1/')
        s1_4 = sym_schema('4,/A1/')
        self.tree.add_context_spin_offs(s1, (s1_1, s1_2, s1_3, s1_4))

        s2_1 = sym_schema('1,/A2/')
        s2_2 = sym_schema('2,/A2/')
        self.tree.add_context_spin_offs(s2, (s2_1, s2_2))

        s1_1_1 = sym_schema('1,2/A1/')
        s1_1_2 = sym_schema('1,3/A1/')
        s1_1_3 = sym_schema('1,4/A1/')
        s1_1_4 = sym_schema('1,5/A1/')
        self.tree.add_context_spin_offs(s1_1, (s1_1_1, s1_1_2, s1_1_3, s1_1_4))

        s1_1_2_1 = sym_schema('1,3,5/A1/')
        s1_1_2_2 = sym_schema('1,3,6/A1/')
        s1_1_2_3 = sym_schema('1,3,7/A1/')
        self.tree.add_context_spin_offs(s1_1_2, (s1_1_2_1, s1_1_2_2, s1_1_2_3))

        s1_1_2_1_1 = sym_schema('1,3,5,7/A1/')
        self.tree.add_context_spin_offs(s1_1_2_1, (s1_1_2_1_1,))

        s1_1_2_3_1 = sym_schema('1,3,7,~9/A1/')
        self.tree.add_context_spin_offs(s1_1_2_3, (s1_1_2_3_1,))

        s1_r101 = sym_schema('/A1/101,')
        self.tree.add_result_spin_offs(s1, (s1_r101,))

        s1_1_r101 = sym_schema('1,/A1/101,', reliability=1.0)
        self.tree.add_context_spin_offs(s1_r101, (s1_1_r101,))

        self.no_context_schemas = {
            s1, s2, self.s101, self.s101_r100, self.s102
        }

        self.schemas = {
            s1, s2, s1_1, s1_2, s1_3, s1_4, s2_1, s2_2, s1_1_1, s1_1_2, s1_1_3, s1_1_4, s1_1_2_1, s1_1_2_2, s1_1_2_3,
            s1_1_2_1_1, s1_1_2_3_1, s1_r101, s1_1_r101,
        }

        self.composite_schemas = {
            self.s101, self.s101_r100, self.s101_1_r100, self.s102
        }

        # initialize composite schema controllers
        self.s101.action.controller.update([Chain([s1_1_r101])])
        self.s102.action.controller.update([])

        # Tree contents:
        #
        # root
        # |-- 1
        # |   |-- 1,4
        # |   |-- 1,2
        # |   |-- 1,5
        # |   +-- 1,3
        # |       |-- 1,3,7
        # |       |   +-- 1,3,7,~9
        # |       |-- 1,3,5
        # |       |   +-- 1,3,5,7
        # |       +-- 1,3,6
        # |-- 3
        # |-- 2
        # |-- 4
        # |-- 100
        # +-- 102
        self.sm = SchemaMemory.from_tree(self.tree)

    def test_init(self):
        self.assertEqual(self.tree.n_schemas, len(self.sm))

        # invalid primitives should generate ValueErrors
        invalid_primitive_schema = sym_schema('1,2,3/A1/')
        self.assertRaises(ValueError, lambda: SchemaMemory(primitives=(invalid_primitive_schema,)))

    def test_from_tree(self):
        primitives = primitive_schemas(actions(5))
        tree = SchemaTree(primitives)

        s1_1 = create_context_spin_off(primitives[0], sym_item_assert('1'))
        s1_2 = create_context_spin_off(primitives[0], sym_item_assert('2'))
        tree.add_context_spin_offs(primitives[0], (s1_1, s1_2))

        s1_r1 = create_result_spin_off(primitives[0], sym_item_assert('3'))
        s1_r2 = create_result_spin_off(primitives[0], sym_item_assert('4'))
        tree.add_result_spin_offs(primitives[0], (s1_r1, s1_r2))

        sm = SchemaMemory.from_tree(tree)
        self.assertEqual(tree.n_schemas, len(sm))

        for schema in [*primitives, s1_1, s1_2, s1_r1, s1_r2]:
            self.assertIn(schema, sm)

    def test_contains(self):
        for schema in self.tree:
            self.assertIn(schema, self.sm)

        # shouldn't be in SchemaMemory
        s_not_found = sym_schema('1,2,3,4,5/A7/')
        self.assertNotIn(s_not_found, self.sm)

    def test_schemas(self):
        pass

    def test_all_applicable(self):
        # case 1: only bare schemas without composite actions
        state = sym_state('')

        # /A1/ [applicable - matches all states]
        # /A1/101, [applicable - matches all states]
        # /A2/ [applicable - matches all states]
        # /101,/ [would be applicable, but no applicable controller components]
        # /101,/100, [would be applicable, but no applicable controller components]
        # /102,/ [would be applicable, but no controller components]
        nodes = self.sm.all_applicable(state)
        self.assertEqual(3, len(nodes))

        for node in nodes:
            self.assertTrue(node.context.is_satisfied(state))

            if node.action.is_composite():
                self.assertTrue(node.action.is_enabled(state=state))

        # case 2
        state = sym_state('3')

        # /A1/ [applicable - matches all states]
        # /A1/101, [applicable - matches all states]
        # /A2/ [applicable - matches all states]
        # 3,/A1/ [applicable - matches state == 3]
        nodes = self.sm.all_applicable(state)
        self.assertEqual(4, len(nodes))

        for node in nodes:
            self.assertTrue(node.context.is_satisfied(state))

            if node.action.is_composite():
                self.assertTrue(node.action.is_enabled(state=state))

        # case 3: includes nodes with context assertion that are negated
        state = sym_state('1,3,5,7')
        nodes = self.sm.all_applicable(state)
        self.assertEqual(16, len(nodes))

        for node in nodes:
            self.assertTrue(node.context.is_satisfied(state))

            if node.action.is_composite():
                self.assertTrue(node.action.is_enabled(state=state))

    def test_update_all(self):

        selection_state = sym_state('1,3,5,7,12')
        result_state = sym_state('10,11,12')

        selected_schema = sym_schema('1,3,5,7/A1/10,')

        all_schemas = set([s for s in self.sm])
        applicable_schemas = set(self.sm.all_applicable(selection_state))
        non_applicable_schemas = all_schemas - applicable_schemas

        selection_details = SchemaSelection.SelectionDetails(
            applicable=applicable_schemas,
            selected=selected_schema,
            selection_state=selection_state,
            terminated_pending=[],
            effective_value=1.0  # value fabricated. does not matter for test
        )

        for s in all_schemas:
            s.update = MagicMock()

        self.sm.update_all(selection_details, result_state)

        for s in applicable_schemas:
            s.update.assert_called()

            # test: activated SHOULD be True if schema's action equals selected schema's action
            if s.action == selected_schema.action:
                s.update.assert_called_with(activated=True, succeeded=ANY, s_prev=ANY,
                                            s_curr=ANY, new=ANY, lost=ANY, explained=ANY)

            # test: activated SHOULD be False if schema's action equals selected schema's action
            else:
                s.update.assert_called_with(activated=False, succeeded=ANY, s_prev=ANY,
                                            s_curr=ANY, new=ANY, lost=ANY, explained=ANY)

            # test: succeeded SHOULD be True if schema's result is satisfied and schema is activated
            if (s.is_activated(selected_schema, selection_state, applicable=True)
                    and s.result.is_satisfied(result_state)):
                s.update.assert_called_with(succeeded=True, activated=ANY, s_prev=ANY,
                                            s_curr=ANY, new=ANY, lost=ANY, explained=ANY)

            # test: succeeded SHOULD be False if schema's result is not satisfied or schema is not activated
            else:
                s.update.assert_called_with(succeeded=False, activated=ANY, s_prev=ANY,
                                            s_curr=ANY, new=ANY, lost=ANY, explained=ANY)

        # test: update SHOULD NOT be called for non-applicable schemas
        for s in non_applicable_schemas:
            s.update.assert_not_called()

    def test_receive_1(self):
        # create a context spin-off
        n_schemas = len(self.sm)

        self.assertNotIn(sym_schema('1,3,7,~9,11/A1/'), self.sm)
        self.sm.receive(
            source=sym_schema('1,3,7,~9/A1/'),
            spin_off_type=Schema.SpinOffType.CONTEXT,
            relevant_items=[sym_item_assert('11')]
        )
        self.assertIn(sym_schema('1,3,7,~9,11/A1/'), self.sm)
        self.assertEqual(n_schemas + 1, len(self.sm))

    def test_receive_2(self):
        # create a result spin-off
        n_schemas = len(self.sm)

        self.assertNotIn(sym_schema('/A1/1000,'), self.sm)

        # composite action for this result. (novel results are added as composite actions during result spin-off)
        self.assertNotIn(sym_schema('/1000,/'), self.sm)

        self.sm.receive(
            source=sym_schema('/A1/'),
            spin_off_type=Schema.SpinOffType.RESULT,
            relevant_items=[sym_item_assert('1000')]
        )

        self.assertIn(sym_schema('/A1/1000,'), self.sm)
        self.assertIn(sym_schema('/1000,/'), self.sm)

        self.assertEqual(n_schemas + 2, len(self.sm))

    def test_receive_3(self):
        # create multiple context spin-offs
        n_schemas = len(self.sm)

        self.assertNotIn(sym_schema('1,3,7,~9,11/A1/'), self.sm)
        self.assertNotIn(sym_schema('1,3,7,~9,13/A1/'), self.sm)
        self.sm.receive(
            source=sym_schema('1,3,7,~9/A1/'),
            spin_off_type=Schema.SpinOffType.CONTEXT,
            relevant_items=sym_asserts('11,13')
        )
        self.assertIn(sym_schema('1,3,7,~9,11/A1/'), self.sm)
        self.assertIn(sym_schema('1,3,7,~9,13/A1/'), self.sm)
        self.assertEqual(n_schemas + 2, len(self.sm))

    def test_receive_4(self):
        # create multiple result spin-offs
        n_schemas = len(self.sm)

        self.assertNotIn(sym_schema('/A1/1000,'), self.sm)
        self.assertNotIn(sym_schema('/A1/1001,'), self.sm)

        # composite actions for these results. (novel results are added as composite actions during result spin-off)
        self.assertNotIn(sym_schema('/1000,/'), self.sm)
        self.assertNotIn(sym_schema('/1001,/'), self.sm)

        self.sm.receive(
            source=sym_schema('/A1/'),
            spin_off_type=Schema.SpinOffType.RESULT,
            relevant_items=sym_asserts('1000,1001')
        )

        self.assertIn(sym_schema('/A1/1000,'), self.sm)
        self.assertIn(sym_schema('/A1/1001,'), self.sm)

        # new composite actions
        self.assertIn(sym_schema('/1000,/'), self.sm)
        self.assertIn(sym_schema('/1001,/'), self.sm)

        # 4 = 2 (new result spin-offs) + 2 (new basic composite action schemas)
        self.assertEqual(n_schemas + 4, len(self.sm))

    def test_receive_5(self):
        # create multiple context spin-offs, one of which already exists in schema memory
        n_schemas = len(self.sm)

        self.assertIn(sym_schema('1,3/A1/'), self.sm)
        self.assertNotIn(sym_schema('3,6/A1/'), self.sm)
        self.sm.receive(
            source=sym_schema('3,/A1/'),
            spin_off_type=Schema.SpinOffType.CONTEXT,
            relevant_items=sym_asserts('1,6')
        )
        self.assertIn(sym_schema('1,3/A1/'), self.sm)
        self.assertIn(sym_schema('3,6/A1/'), self.sm)

        # should only be one new schema
        self.assertEqual(n_schemas + 1, len(self.sm))

    def test_receive_6(self):
        # create multiple result spin-offs, one of which already exists in schema memory
        n_schemas = len(self.sm)

        # sanity check: schema '/A1/101/' should exist in tree, but not '/A1/1001'
        self.assertIn(sym_schema('/A1/101,'), self.sm)
        self.assertNotIn(sym_schema('/A1/1001,'), self.sm)

        # composite actions for these results. (novel results are added as composite actions during result spin-off)
        self.assertIn(sym_schema('/101,/'), self.sm)
        self.assertNotIn(sym_schema('/1001,/'), self.sm)

        self.sm.receive(
            source=sym_schema('/A1/'),
            spin_off_type=Schema.SpinOffType.RESULT,
            relevant_items=sym_asserts('101,1001')
        )

        # test: new result spin-off should have been added
        self.assertIn(sym_schema('/A1/1001,'), self.sm)

        # test: new bare schema with novel composite action should have been added
        self.assertIn(sym_schema('/1001,/'), self.sm)

        # test: adds 1 new result spin-off and 1 new bare schema for novel result
        self.assertEqual(n_schemas + 2, len(self.sm))

    def test_is_novel_result(self):
        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A2/')

        tree = SchemaTree((s1, s2))

        s1_a = sym_schema('/A1/A,')
        s1_b = sym_schema('/A1/B,')
        s1_c = sym_schema('/A1/C,')

        tree.add_result_spin_offs(s1, [s1_a, s1_b, s1_c])

        s2_e = sym_schema('/A2/E,')
        s2_not_f = sym_schema('/A2/~F,')

        tree.add_result_spin_offs(s2, [s2_e, s2_not_f])

        s1_c_a = sym_schema('C,/A1/A,')
        s1_cd_a = sym_schema('C,D/A1/A,')

        tree.add_context_spin_offs(s1_a, [s1_c_a])
        tree.add_context_spin_offs(s1_c_a, [s1_cd_a])

        s2_cd = sym_schema('/A2/(C,D),')
        tree.add_result_spin_offs(s2, [s2_cd])

        sm = SchemaMemory.from_tree(tree)

        known_results = {s.result for s in tree.root.schemas_satisfied_by}
        unknown_results = {sym_state_assert(f'{i},') for i in range(10)}

        # test: results already exist in SchemaMemory - should return False
        for r in known_results:
            self.assertFalse(sm.is_novel_result(r))

        # test: results do not exist in SchemaMemory - should return True
        for r in unknown_results:
            self.assertTrue(sm.is_novel_result(r))

        spin_offs = {Schema(action=Action('A2'), result=r) for r in unknown_results}
        tree.add_result_spin_offs(s2, spin_offs)

        sm = SchemaMemory.from_tree(tree)

        # test: after adding result to SchemaMemory, all these should return False
        for r in unknown_results:
            self.assertFalse(sm.is_novel_result(r))
