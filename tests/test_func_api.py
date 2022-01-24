import unittest

from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.data_structures import NULL_STATE_ASSERT
from schema_mechanism.data_structures import Schema
from schema_mechanism.data_structures import SchemaTreeNode
from schema_mechanism.data_structures import StateAssertion
from schema_mechanism.func_api import sym_assert
from schema_mechanism.func_api import sym_asserts
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_schema_tree_node
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert


class TestFunctionalApi(unittest.TestCase):
    def test_sym_state(self):
        state = sym_state('')
        self.assertEqual(0, len(state))

        # single state element
        state = sym_state('1')
        self.assertEqual(1, len(state))
        self.assertIn(1, state)

        # multiple state element
        state = sym_state('1,2,3,4,5')
        self.assertEqual(5, len(state))
        self.assertTrue(all({se in state for se in range(1, 6)}))

    def test_sym_assert(self):
        # non-negated item assertion
        ia = sym_assert('1')
        self.assertIsInstance(ia, ItemAssertion)
        self.assertEqual(sym_item('1'), ia.item)
        self.assertEqual(False, ia.negated)

        # negated item assertion
        ia = sym_assert('~1')
        self.assertIsInstance(ia, ItemAssertion)
        self.assertEqual(sym_item('1'), ia.item)
        self.assertEqual(True, ia.negated)

    def test_sym_asserts(self):
        # single item assertions
        ias = sym_asserts('1')
        self.assertEqual(1, len(ias))
        self.assertIn(sym_assert('1'), ias)

        # multiple, mixed item assertions
        ias = sym_asserts('~1,2,7')
        self.assertEqual(3, len(ias))
        self.assertIn(sym_assert('~1'), ias)
        self.assertIn(sym_assert('2'), ias)
        self.assertIn(sym_assert('7'), ias)

    def test_sym_state_assert(self):
        # empty state assertion
        sa = sym_state_assert('')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(0, len(sa))

        # single non-negated element state assertion
        sa = sym_state_assert('1')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(1, len(sa))
        self.assertIn(sym_assert('1'), sa)

        # single negated element state assertion
        sa = sym_state_assert('~1')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(1, len(sa))
        self.assertIn(sym_assert('~1'), sa)

        # multiple elements with mixed negation state assertion
        sa = sym_state_assert('~1,2,3,~4,5')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(5, len(sa))
        self.assertIn(sym_assert('~1'), sa)
        self.assertIn(sym_assert('2'), sa)
        self.assertIn(sym_assert('3'), sa)
        self.assertIn(sym_assert('~4'), sa)
        self.assertIn(sym_assert('5'), sa)

    def test_sym_schema_tree_node(self):
        # blank node should be allowed
        stn = sym_schema_tree_node('//', label='blank')
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertIs(None, stn.context)
        self.assertIs(None, stn.action)
        self.assertEqual('blank', stn.label)

        # action only node
        stn = sym_schema_tree_node('/A1/')
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertIs(None, stn.context)
        self.assertEqual(Action('A1'), stn.action)

        # context and action node
        stn = sym_schema_tree_node('1,2,~3/A1/')
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertEqual(sym_state_assert('1,2,~3'), stn.context)
        self.assertEqual(Action('A1'), stn.action)

        # should work even if a result is given
        stn = sym_schema_tree_node('1,2,~3/A1/4,5')
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertEqual(sym_state_assert('1,2,~3'), stn.context)
        self.assertEqual(Action('A1'), stn.action)

    def test_sym_schema(self):
        # no action should return error
        self.assertRaises(ValueError, lambda: sym_schema('//'))

        # primitive schema
        schema = sym_schema('/A1/')
        self.assertIsInstance(schema, Schema)
        self.assertIs(NULL_STATE_ASSERT, schema.context)
        self.assertIs(NULL_STATE_ASSERT, schema.result)
        self.assertEqual(Action('A1'), schema.action)

        # context only schema
        schema = sym_schema('1,~2/A1/')
        self.assertIsInstance(schema, Schema)
        self.assertIsNot(NULL_STATE_ASSERT, schema.context)
        self.assertIs(NULL_STATE_ASSERT, schema.result)
        self.assertEqual(Action('A1'), schema.action)
        self.assertIn(sym_assert('1'), schema.context)
        self.assertIn(sym_assert('~2'), schema.context)
        self.assertNotIn(sym_assert('3'), schema.context)

        # result only schema
        schema = sym_schema('/A1/1,~2')
        self.assertIsInstance(schema, Schema)
        self.assertIs(NULL_STATE_ASSERT, schema.context)
        self.assertIsNot(NULL_STATE_ASSERT, schema.result)
        self.assertEqual(Action('A1'), schema.action)
        self.assertIn(sym_assert('1'), schema.result)
        self.assertIn(sym_assert('~2'), schema.result)
        self.assertNotIn(sym_assert('3'), schema.result)

        # full schema
        schema = sym_schema('1,~2/A1/~3,4')
        self.assertIsInstance(schema, Schema)
        self.assertIsNot(NULL_STATE_ASSERT, schema.context)
        self.assertIsNot(NULL_STATE_ASSERT, schema.result)
        self.assertEqual(Action('A1'), schema.action)
        self.assertIn(sym_assert('1'), schema.context)
        self.assertIn(sym_assert('~2'), schema.context)
        self.assertNotIn(sym_assert('3'), schema.context)
        self.assertIn(sym_assert('~3'), schema.result)
        self.assertIn(sym_assert('4'), schema.result)
        self.assertNotIn(sym_assert('5'), schema.result)
