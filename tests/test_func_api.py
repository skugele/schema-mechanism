import unittest

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import ItemAssertion
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.core import StateAssertion
from schema_mechanism.func_api import sym_assert
from schema_mechanism.func_api import sym_asserts
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_schema_tree_node
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from test_share.test_func import common_test_setup


class TestFunctionalApi(unittest.TestCase):
    def setUp(self):
        common_test_setup()

    def test_sym_item(self):
        # state element only should give item with default primitive value
        item = sym_item('1')

        self.assertEqual('1', item.source)
        self.assertEqual(0.0, item.primitive_value)

        # state element and primitive value
        item = sym_item('2', primitive_value=1.0)

        self.assertEqual('2', item.source)
        self.assertEqual(1.0, item.primitive_value)

        # multiple calls with same state element should return same object
        item1 = sym_item('3', primitive_value=1.0)
        item2 = sym_item('3', primitive_value=1.0)

        self.assertIs(item1, item2)
        self.assertEqual('3', item2.source)
        self.assertEqual(1.0, item2.primitive_value)

    def test_sym_state(self):
        state = sym_state('')
        self.assertEqual(0, len(state))

        # single state element
        state = sym_state('1')
        self.assertEqual(1, len(state))
        self.assertIn('1', state)

        # multiple state element
        state = sym_state('1,2,3,4,5')
        self.assertEqual(5, len(state))
        self.assertTrue(all({str(se) in state for se in range(1, 6)}))

    def test_sym_assert(self):
        # non-negated item assertion
        ia = sym_assert('1')
        self.assertIsInstance(ia, ItemAssertion)
        self.assertEqual(sym_item('1'), ia.item)
        self.assertEqual(False, ia.is_negated)

        # negated item assertion
        ia = sym_assert('~1')
        self.assertIsInstance(ia, ItemAssertion)
        self.assertEqual(sym_item('1'), ia.item)
        self.assertEqual(True, ia.is_negated)

    def test_sym_asserts(self):
        # single item assertion
        ias = sym_asserts('1')
        self.assertEqual(1, len(ias))
        self.assertIn(sym_assert('1'), ias)

        # multiple, mixed item assertion
        ias = sym_asserts('~1,2,7')
        self.assertEqual(3, len(ias))
        self.assertIn(sym_assert('~1'), ias)
        self.assertIn(sym_assert('2'), ias)
        self.assertIn(sym_assert('7'), ias)

    def test_sym_state_assert(self):
        # test: single non-negated element state assertion
        sa = sym_assert('1,')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(1, len(sa))
        self.assertIn(sym_assert('1'), sa)

        # test: single negated element state assertion
        sa = sym_assert('~1,')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(1, len(sa))
        self.assertIn(sym_assert('~1'), sa)

        # test: multiple elements with mixed negation state assertion
        sa = sym_assert('~1,2,3,~4,5')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(5, len(sa))
        self.assertIn(sym_assert('~1'), sa)
        self.assertIn(sym_assert('2'), sa)
        self.assertIn(sym_assert('3'), sa)
        self.assertIn(sym_assert('~4'), sa)
        self.assertIn(sym_assert('5'), sa)

        # test: single composite item
        sa = sym_assert('(1,2,3),')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(1, len(sa))

        # test: single negated composite item
        sa = sym_assert('~(1,2,3),')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(1, len(sa))

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
        self.assertEqual(sym_assert('1,2,~3'), stn.context)
        self.assertEqual(Action('A1'), stn.action)

        # should work even if a result is given
        stn = sym_schema_tree_node('1,2,~3/A1/4,5')
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertEqual(sym_assert('1,2,~3'), stn.context)
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

    def test_sym_composite_item(self):
        self.assertIsInstance(sym_item('(1,2)'), CompositeItem)

        self.assertEqual(CompositeItem(sym_state_assert('1,2')), sym_item('(1,2)'))
        self.assertEqual(CompositeItem(sym_state_assert('1,~2')), sym_item('(1,~2)'))

        self.assertRaises(ValueError, lambda: sym_item(''))
        self.assertRaises(ValueError, lambda: sym_item('()'))
        self.assertRaises(ValueError, lambda: sym_item('(3)'))
