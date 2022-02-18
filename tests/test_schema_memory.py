from unittest import TestCase
from unittest.mock import ANY
from unittest.mock import MagicMock

from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import SchemaTree
from schema_mechanism.func_api import actions
from schema_mechanism.func_api import primitive_schemas
from schema_mechanism.func_api import sym_assert
from schema_mechanism.func_api import sym_asserts
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.modules import create_context_spin_off
from schema_mechanism.modules import create_result_spin_off
from test_share.test_func import common_test_setup


class TestSchemaMemory(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.tree = SchemaTree()

        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A2/')
        self.tree.add_primitives((s1, s2))

        s1_1 = sym_schema('1/A1/')
        s1_2 = sym_schema('2/A1/')
        s1_3 = sym_schema('3/A1/')
        s1_4 = sym_schema('4/A1/')
        self.tree.add_context_spin_offs(s1, (s1_1, s1_2, s1_3, s1_4))

        s2_1 = sym_schema('1/A2/')
        s2_2 = sym_schema('2/A2/')
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

        s1_1_r100 = sym_schema('1/A1/100')
        self.tree.add_result_spin_offs(s1_1, (s1_1_r100,))

        # Tree contents:
        #
        # root
        # |-- /A1/
        # |   |-- 2/A1/
        # |   |-- 3/A1/
        # |   |-- 4/A1/
        # |   +-- 1/A1/
        # |       |-- 1,3/A1/
        # |       |   |-- 1,7,3/A1/
        # |       |   |   +-- 1,7,3,~9/A1/
        # |       |   |-- 1,5,3/A1/
        # |       |   |   +-- 1,7,5,3/A1/
        # |       |   +-- 1,6,3/A1/
        # |       |-- 1,4/A1/
        # |       |-- 1,5/A1/
        # |       +-- 1,2/A1/
        # +-- /A2/
        #     |-- 1/A2/
        #     +-- 2/A2/
        self.sm = SchemaMemory.from_tree(self.tree)

    def test_init(self):
        self.assertEqual(self.tree.n_schemas, len(self.sm))

        # invalid primitives should generate ValueErrors
        invalid_primitive_schema = sym_schema('1,2,3/A1/')
        self.assertRaises(ValueError, lambda: SchemaMemory(primitives=(invalid_primitive_schema,)))

    def test_from_tree(self):
        tree = SchemaTree()

        primitives = primitive_schemas(actions(5))
        tree.add_primitives(primitives)

        s1_1 = create_context_spin_off(primitives[0], sym_assert('1'))
        s1_2 = create_context_spin_off(primitives[0], sym_assert('2'))
        tree.add_context_spin_offs(primitives[0], (s1_1, s1_2))

        s1_1r1 = create_result_spin_off(s1_1, sym_assert('3'))
        s1_1r2 = create_result_spin_off(s1_1, sym_assert('4'))
        tree.add_result_spin_offs(s1_1, (s1_1r1, s1_1r2))

        sm = SchemaMemory.from_tree(tree)
        self.assertEqual(tree.n_schemas, len(sm))

        for schema in [*primitives, s1_1, s1_2, s1_1r1, s1_1r2]:
            self.assertIn(schema, sm)

    def test_contains(self):
        for schema in self.tree:
            self.assertIn(schema, self.sm)

        # shouldn't be in SchemaMemory
        s_not_found = sym_schema('1,2,3,4,5/A7/')
        self.assertNotIn(s_not_found, self.sm)

    def test_all_applicable(self):
        # case 1: only no-context, action-only nodes
        state = sym_state('')
        nodes = self.sm.all_applicable(state)
        self.assertEqual(2, len(nodes))

        for node in nodes:
            self.assertTrue(node.context.is_satisfied(state))

        # case 2
        state = sym_state('3')
        nodes = self.sm.all_applicable(state)
        self.assertEqual(3, len(nodes))

        for node in nodes:
            self.assertTrue(node.context.is_satisfied(state))

        # case 3: includes nodes with context assertions that are negated
        state = sym_state('1,3,5,7')
        nodes = self.sm.all_applicable(state)
        self.assertEqual(12, len(nodes))

        for node in nodes:
            self.assertTrue(node.context.is_satisfied(state))

    def test_update_all(self):

        selection_state = sym_state('1,3,5,7,12')
        result_state = sym_state('10,11,12')

        selected_schema = sym_schema('1,3,5,7/A1/10')

        all_schemas = set(self.sm.schemas)
        applicable_schemas = set(self.sm.all_applicable(selection_state))
        non_applicable_schemas = all_schemas - applicable_schemas

        selection_details = SchemaSelection.SelectionDetails(applicable=applicable_schemas, selected=selected_schema)

        for s in all_schemas:
            s.update = MagicMock()

        self.sm.update_all(selection_details, selection_state, result_state)

        for s in applicable_schemas:
            s.update.assert_called()

            if s.action == selected_schema.action:
                s.update.assert_called_with(activated=True, s_prev=ANY, s_curr=ANY, new=ANY, lost=ANY)

            else:
                s.update.assert_called_with(activated=False, s_prev=ANY, s_curr=ANY, new=ANY, lost=ANY)

        for s in non_applicable_schemas:
            s.update.assert_not_called()

        self.assertEqual(self.sm.stats.n_updates, len(applicable_schemas))

    def test_receive_1(self):
        # create a context spin-off
        n_schemas = len(self.sm)

        self.assertNotIn(sym_schema('1,3,7,~9,11/A1/'), self.sm)
        self.sm.receive(
            source=sym_schema('1,3,7,~9/A1/'),
            spin_off_type=Schema.SpinOffType.CONTEXT,
            relevant_items=[sym_assert('11')]
        )
        self.assertIn(sym_schema('1,3,7,~9,11/A1/'), self.sm)
        self.assertEqual(n_schemas + 1, len(self.sm))

    def test_receive_2(self):
        # create a result spin-off
        n_schemas = len(self.sm)

        self.assertNotIn(sym_schema('1,3,7,~9/A1/100'), self.sm)
        self.sm.receive(
            source=sym_schema('1,3,7,~9/A1/'),
            spin_off_type=Schema.SpinOffType.RESULT,
            relevant_items=[sym_assert('100')]
        )
        self.assertIn(sym_schema('1,3,7,~9/A1/100'), self.sm)
        self.assertEqual(n_schemas + 1, len(self.sm))

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

        self.assertNotIn(sym_schema('1,3,7,~9/A1/100'), self.sm)
        self.assertNotIn(sym_schema('1,3,7,~9/A1/101'), self.sm)
        self.sm.receive(
            source=sym_schema('1,3,7,~9/A1/'),
            spin_off_type=Schema.SpinOffType.RESULT,
            relevant_items=sym_asserts('100,101')
        )
        self.assertIn(sym_schema('1,3,7,~9/A1/100'), self.sm)
        self.assertIn(sym_schema('1,3,7,~9/A1/101'), self.sm)
        self.assertEqual(n_schemas + 2, len(self.sm))

    def test_receive_5(self):
        # create multiple context spin-offs, one of which already exists in schema memory
        n_schemas = len(self.sm)

        self.assertIn(sym_schema('1,3/A1/'), self.sm)
        self.assertNotIn(sym_schema('3,6/A1/'), self.sm)
        self.sm.receive(
            source=sym_schema('3/A1/'),
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

        self.assertIn(sym_schema('1/A1/100'), self.sm)
        self.assertNotIn(sym_schema('1/A1/101'), self.sm)
        self.sm.receive(
            source=sym_schema('1/A1/'),
            spin_off_type=Schema.SpinOffType.RESULT,
            relevant_items=sym_asserts('100,101')
        )
        self.assertIn(sym_schema('1/A1/100'), self.sm)
        self.assertIn(sym_schema('1/A1/101'), self.sm)

        # should only be one new schema
        self.assertEqual(n_schemas + 1, len(self.sm))
