import unittest

from schema_mechanism.core import Action
from schema_mechanism.core import CompositeAction
from schema_mechanism.core import CompositeItem
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.core import StateAssertion
from schema_mechanism.core import SymbolicItem
from schema_mechanism.func_api import sym_assert
from schema_mechanism.func_api import sym_composite_item
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_items
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

    def test_sym_composite_item(self):
        # test: should give composite item with default primitive value if only given state elements
        item = sym_composite_item('(1,2)')

        self.assertIsInstance(item, CompositeItem)
        self.assertIn('1', item.source)
        self.assertEqual(0.0, item.primitive_value)

        # test: should give composite item with primitive value set if given state elements and a primitive value
        item = sym_composite_item('(1,3)', primitive_value=1.0)

        self.assertIsInstance(item, CompositeItem)
        self.assertEqual(frozenset(['1', '3']), item.source)
        self.assertEqual(1.0, item.primitive_value)

        # test: multiple calls with same state element should return the same object on each call
        item1 = sym_composite_item('(1,4)', primitive_value=-1.5)
        item2 = sym_composite_item('(1,4)', primitive_value=-1.5)

        self.assertIsInstance(item1, CompositeItem)
        self.assertIs(item1, item2)
        self.assertEqual(frozenset(['1', '4']), item2.source)
        self.assertEqual(-1.5, item2.primitive_value)

        # test: invalid string representations should return a ValueError
        self.assertRaises(ValueError, lambda: sym_composite_item(''))
        self.assertRaises(ValueError, lambda: sym_composite_item('()'))
        self.assertRaises(ValueError, lambda: sym_composite_item('(3)'))

    def test_sym_items(self):

        sources_lists = [
            # non-composite item sources
            ['1'],
            ['1', '2'],
            ['1', '2', '3'],

            # composite item sources
            ['(1,2)'],
            ['(1,~3)'],
            ['(1,2)', '(1,~3)'],
            ['(1,2)', '(1,~3)', '(3,4)'],

            # mixed non-composite & composite sources
            ['1', '(1,2)'],
            ['(2,3)', '4'],
            ['3', '(4,5)', '6'],
        ]

        for sources in sources_lists:

            # multiple sources should be delimited by semi-colons
            str_repr = ';'.join([s for s in sources])
            items = sym_items(str_repr)

            # test: the number of items returned should equal the number of sources
            self.assertEqual(len(sources), len(items))

            for item, source in zip(items, sources):
                expected_item_type = CompositeItem if ',' in source else SymbolicItem

                # test: the item type should match the expected item type for source string
                self.assertIsInstance(item, expected_item_type)

                # test: the returned item's source should match the expected source for the corresponding string element
                self.assertEqual(sym_item(source), item)

                # test: primitive value should be the default
                self.assertEqual(0.0, item.primitive_value)

        # test: sending items with primitive values should set their primitive values correctly
        primitive_values = [-1.7, 10.0]
        items = sym_items('X;(Y,Z)', primitive_values=primitive_values)

        for item, expected_primitive_value in zip(items, primitive_values):
            self.assertEqual(expected_primitive_value, item.primitive_value)

        # test: a ValueError should be raised if there is a mismatch between items and primitive values length
        self.assertRaises(ValueError, lambda: sym_items('1;(1,2);3', []))
        self.assertRaises(ValueError, lambda: sym_items('1;(1,2);3', [1.0]))
        self.assertRaises(ValueError, lambda: sym_items('1;(1,2);3', [1.0, 2.0]))
        self.assertRaises(ValueError, lambda: sym_items('1;(1,2);3', [1.0, 2.0, 3.0, 4.0]))

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

    def test_sym_state_assert(self):
        # test: single element state assertion
        sa = sym_state_assert('1,')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(1, len(sa))
        self.assertIn(sym_item('1'), sa)

        # test: multiple elements
        sa = sym_state_assert('1,2,3')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(3, len(sa))
        self.assertIn(sym_item('1'), sa)
        self.assertIn(sym_item('2'), sa)
        self.assertIn(sym_item('3'), sa)

        # test: single composite item
        sa = sym_state_assert('(1,2,3),')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(1, len(sa))
        self.assertIn(sym_item('(1,2,3)'), sa)

        # test: multiple item state assertion with optional trailing comma
        sa = sym_state_assert('1,2,3,')
        self.assertIsInstance(sa, StateAssertion)
        self.assertEqual(3, len(sa))
        self.assertIn(sym_item('1'), sa)
        self.assertIn(sym_item('2'), sa)
        self.assertIn(sym_item('3'), sa)

        # test: ValueError should be raised for invalid string representations
        self.assertRaises(ValueError, lambda: sym_state_assert(','))
        self.assertRaises(ValueError, lambda: sym_state_assert('X/A/Z'))
        self.assertRaises(ValueError, lambda: sym_state_assert('X,,'))
        self.assertRaises(ValueError, lambda: sym_state_assert('((X,Y,Z))'))

        # test: ValueError should be raise if string representation corresponds to a different object type
        self.assertRaises(ValueError, lambda: sym_state_assert('X,/A1/Z,'))

    def test_sym_schema_tree_node(self):
        # blank node should be allowed
        stn = sym_schema_tree_node('', label='blank')
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertIs(NULL_STATE_ASSERT, stn.context)
        self.assertEqual('blank', stn.label)

        # context and action node
        stn = sym_schema_tree_node('1,2,3')
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertEqual(sym_assert('1,2,3'), stn.context)

        # should work even if a result is given
        stn = sym_schema_tree_node('1,2,3')
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertEqual(sym_assert('1,2,3'), stn.context)

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
        schema = sym_schema('1,2/A1/')
        self.assertIsInstance(schema, Schema)
        self.assertIsNot(NULL_STATE_ASSERT, schema.context)
        self.assertIs(NULL_STATE_ASSERT, schema.result)
        self.assertEqual(Action('A1'), schema.action)
        self.assertEqual(sym_assert('1,2'), schema.context)
        self.assertIs(NULL_STATE_ASSERT, schema.result)

        # result only schema
        schema = sym_schema('/A1/1,2')
        self.assertIsInstance(schema, Schema)
        self.assertIs(NULL_STATE_ASSERT, schema.context)
        self.assertEqual(Action('A1'), schema.action)
        self.assertEqual(sym_assert('1,2'), schema.result)

        # full schema
        schema = sym_schema('1,2/A1/3,4')
        self.assertIsInstance(schema, Schema)
        self.assertEqual(sym_assert('1,2'), schema.context)
        self.assertEqual(Action('A1'), schema.action)
        self.assertEqual(sym_assert('3,4'), schema.result)

        # schema with composite item in result
        schema = sym_schema('/A1/(3,4),5')
        self.assertIsInstance(schema, Schema)
        self.assertIs(NULL_STATE_ASSERT, schema.context)
        self.assertEqual(Action('A1'), schema.action)
        self.assertEqual(sym_assert('(3,4),5'), schema.result)

        # test: verify that multiple calls with the same schema string return the same object
        for str_repr in ['/A1/', '1,/A1/', '1,2/A1/', '/A1/1,', '/A1/1,2', '1,/A1/3,', '1,2/A1/3,4', '/A1/(3,4),5']:
            self.assertIs(sym_schema(str_repr), sym_schema(str_repr))

        schema_1 = sym_schema('/play/')
        schema_2 = SchemaPool().get(SchemaUniqueKey(context=None, action=Action('play'), result=None))

        self.assertIs(schema_1, schema_2)

    def test_sym_schema_with_composite_actions(self):
        # primitive schemas with composite actions
        schema = sym_schema('/A,/')
        self.assertIsInstance(schema, Schema)
        self.assertIs(NULL_STATE_ASSERT, schema.context)
        self.assertIs(NULL_STATE_ASSERT, schema.result)
        self.assertEqual(CompositeAction(sym_state_assert('A,')), schema.action)

        schema = sym_schema('/(1,2),/')
        self.assertIsInstance(schema, Schema)
        self.assertIs(NULL_STATE_ASSERT, schema.context)
        self.assertIs(NULL_STATE_ASSERT, schema.result)
        self.assertEqual(CompositeAction(sym_state_assert('(1,2),')), schema.action)

        # context only schema with composite actions
        schema = sym_schema('1,2/(1,2),/')
        self.assertIsInstance(schema, Schema)
        self.assertEqual(sym_assert('1,2'), schema.context)
        self.assertEqual(CompositeAction(sym_state_assert('(1,2),')), schema.action)
        self.assertIs(NULL_STATE_ASSERT, schema.result)

        # result only schema
        schema = sym_schema('/(1,2),/1,2')
        self.assertIsInstance(schema, Schema)
        self.assertIs(NULL_STATE_ASSERT, schema.context)
        self.assertEqual(CompositeAction(sym_state_assert('(1,2),')), schema.action)
        self.assertEqual(sym_assert('1,2'), schema.result)

        # full schema
        schema = sym_schema('1,2/(1,2),/3,4')
        self.assertIsInstance(schema, Schema)
        self.assertEqual(sym_assert('1,2'), schema.context)
        self.assertEqual(CompositeAction(sym_state_assert('(1,2),')), schema.action)
        self.assertEqual(sym_assert('3,4'), schema.result)
