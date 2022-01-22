from itertools import chain
from random import sample
from time import time
from unittest import TestCase

import test_share
from schema_mechanism.data_structures import Action
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import Schema
from schema_mechanism.func_api import actions
from schema_mechanism.func_api import primitive_schemas
from schema_mechanism.func_api import sym_assert
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_schema_tree_node
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import SchemaTree
from schema_mechanism.modules import SchemaTreeNode
from schema_mechanism.modules import create_spin_off
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestSchemaTree(TestCase):
    def setUp(self) -> None:
        self.empty_tree = SchemaTree()

        # schemas used in tree
        self.s1 = sym_schema('/A1/')
        self.s2 = sym_schema('/A2/')

        self.s1_1 = sym_schema('1/A1/')
        self.s1_2 = sym_schema('2/A1/')
        self.s1_3 = sym_schema('3/A1/')
        self.s1_4 = sym_schema('4/A1/')

        self.s2_1 = sym_schema('1/A2/')
        self.s2_2 = sym_schema('2/A2/')

        self.s1_1_1 = sym_schema('1,2/A1/')
        self.s1_1_2 = sym_schema('1,3/A1/')
        self.s1_1_3 = sym_schema('1,4/A1/')
        self.s1_1_4 = sym_schema('1,5/A1/')

        self.s1_1_2_1 = sym_schema('1,3,5/A1/')
        self.s1_1_2_2 = sym_schema('1,3,6/A1/')
        self.s1_1_2_3 = sym_schema('1,3,7/A1/')

        self.s1_1_2_1_1 = sym_schema('1,3,5,7/A1/')
        self.s1_1_2_3_1 = sym_schema('1,3,7,9/A1/')

        self.tree = SchemaTree()
        self.tree.add(self.tree.root, (self.s1, self.s2), Schema.SpinOffType.CONTEXT)
        self.tree.add(self.s1, (self.s1_1, self.s1_2, self.s1_3, self.s1_4), Schema.SpinOffType.CONTEXT)
        self.tree.add(self.s2, (self.s2_1, self.s2_2), Schema.SpinOffType.CONTEXT)
        self.tree.add(self.s1_1, (self.s1_1_1, self.s1_1_2, self.s1_1_3, self.s1_1_4), Schema.SpinOffType.CONTEXT)
        self.tree.add(self.s1_1_2, (self.s1_1_2_1, self.s1_1_2_2, self.s1_1_2_3), Schema.SpinOffType.CONTEXT)
        self.tree.add(self.s1_1_2_1, (self.s1_1_2_1_1,), Schema.SpinOffType.CONTEXT)
        self.tree.add(self.s1_1_2_3, (self.s1_1_2_3_1,), Schema.SpinOffType.CONTEXT)

    def test_iter(self):
        for n in self.empty_tree:
            self.fail()

        count = 0
        for n in self.tree:
            count += 1
        self.assertEqual(len(self.tree), count)

    def test_add(self):
        # TODO: Need a test case with multiple adds of the same schema.

        # adding single primitive schema to tree
        s3 = sym_schema('/A3/')

        len_before_add = len(self.tree)
        self.tree.add(self.tree.root, (s3,))
        len_after_add = len(self.tree)

        self.assertEqual(len_before_add + 1, len_after_add)
        self.assertIn(s3, self.tree)
        self.assertIn(s3, self.tree.get(s3).schemas)

        # adding single context spin-off schema
        s2_3 = sym_schema('3/A2/')

        len_before_add = len(self.tree)
        self.tree.add(self.s2, (s2_3,), Schema.SpinOffType.CONTEXT)
        len_after_add = len(self.tree)

        self.assertEqual(len_before_add + 1, len_after_add)
        self.assertIn(s2_3, self.tree)
        self.assertIn(s2_3, self.tree.get(s2_3).schemas)

        # adding multiple context spin-off schemas
        s1_1_1_1 = sym_schema('1,2,3/A1/')
        s1_1_1_2 = sym_schema('1,2,4/A1/')

        len_before_add = len(self.tree)
        self.tree.add(self.s1_1_1, (s1_1_1_1, s1_1_1_2), Schema.SpinOffType.CONTEXT)
        len_after_add = len(self.tree)

        self.assertEqual(len_before_add + 2, len_after_add)
        self.assertIn(s1_1_1_1, self.tree)
        self.assertIn(s1_1_1_2, self.tree)
        self.assertIn(s1_1_1_1, self.tree.get(s1_1_1_1).schemas)
        self.assertIn(s1_1_1_2, self.tree.get(s1_1_1_2).schemas)

        # adding a result spin-off schema
        s3_r1 = sym_schema('/A3/1')

        len_before_add = len(self.tree)
        self.tree.add(s3, (s3_r1,), Schema.SpinOffType.RESULT)
        len_after_add = len(self.tree)

        self.assertEqual(len_before_add, len_after_add)
        self.assertIn(s3_r1, self.tree.get(s3).schemas)

        # sending empty set should result in a value error
        self.assertRaises(ValueError, lambda: self.tree.add(self.tree.root, schemas=[]))

        # adding same schema 2x should result in no changes to the tree
        len_before_add = len(self.tree)
        self.tree.add(self.s1_1_1, (s1_1_1_1,), Schema.SpinOffType.CONTEXT)
        len_after_add = len(self.tree)

        self.assertEqual(len_before_add, len_after_add)

    def test_is_valid_node_1(self):
        # all nodes in this tree should be valid
        for schema in self.tree:
            self.assertTrue(self.tree.is_valid_node(schema))

    def test_is_valid_node_2(self):
        # root node should be valid
        self.assertTrue(self.tree.is_valid_node(self.tree.root))

    def test_is_valid_node_3(self):
        # unattached schema should be invalid
        s_unattached = sym_schema_tree_node('7/A4/')
        self.assertFalse(self.tree.is_valid_node(s_unattached))

    def test_is_valid_node_4(self):
        # the children of root should only be primitive schemas
        s_illegal_parent = sym_schema('1/A3/')
        self.tree.add(self.tree.root, (s_illegal_parent,))
        s_illegal_parent_node = self.tree.get(s_illegal_parent)
        self.assertFalse(self.tree.is_valid_node(s_illegal_parent_node))

    def test_is_valid_node_5(self):
        # non-primitive nodes should have the same action as their parents
        s_illegal_action = sym_schema('1/A3/')
        self.tree.add(self.s1, (s_illegal_action,))
        s_illegal_action_node = self.tree.get(s_illegal_action)
        self.assertFalse(self.tree.is_valid_node(s_illegal_action_node))

    def test_is_valid_node_6(self):
        s1 = sym_schema('/A1/')
        s1_1 = sym_schema('1/A1/')
        s1_1_1 = sym_schema('1,2/A1/')

        tree = SchemaTree()
        tree.add(tree.root, (s1,))
        tree.add(s1, (s1_1,))
        tree.add(s1_1, (s1_1_1,))

        for n in tree:
            self.assertTrue(tree.is_valid_node(n))

        # children should have all of the context item assertions of their parents
        s_illegal_context = sym_schema('1,5,6/A1/')
        tree.add(s1_1_1, (s_illegal_context,))
        s_illegal_context_node = tree.get(s_illegal_context)
        self.assertFalse(tree.is_valid_node(s_illegal_context_node))

    def test_validate(self):
        # all nodes in tree should be valid
        invalid_nodes = self.tree.validate()
        self.assertEqual(0, len(invalid_nodes))

        # adding invalid nodes
        invalid_1 = sym_schema('1/A3/')
        invalid_2 = sym_schema('1,2,3,4,5/A1/')

        # invalid due to parent and child having different actions
        self.tree.add(self.s2, (invalid_1,))

        # invalid due to parent and child having inconsistent contexts
        self.tree.add(self.s1_1_2_1_1, (invalid_2,))

        invalid_nodes = self.tree.validate()
        self.assertEqual(2, len(invalid_nodes))
        self.assertTrue(all({self.tree.get(s) in invalid_nodes for s in [invalid_1, invalid_2]}))

    def test_contains(self):
        tree = SchemaTree()

        p_schemas = primitive_schemas(actions(5))
        tree.add(tree.root, schemas=p_schemas)
        self.assertTrue(all({s in tree for s in p_schemas}))

        # any other schema SHOULD NOT be contained in tree
        another = sym_schema('9/A7/')
        self.assertNotIn(another, tree)

        # after adding, another schema should be contained in tree
        tree.add(p_schemas[0], schemas=(another,))
        self.assertIn(another, tree)

        # contains should work for SchemaTreeNodes as well
        node_another = sym_schema_tree_node('9/A7/')
        self.assertIn(node_another, tree)

        node_missing = sym_schema_tree_node('1/A4/')
        self.assertNotIn(node_missing, tree)

    @test_share.string_test
    def test_str(self):
        print(self.tree)

    def test_find(self):

        # empty tree should return empty set
        app_schemas = self.empty_tree.find(state=sym_state('1'))
        self.assertEqual(0, len(app_schemas))

        # primitive schemas should be applicable to all states
        app_schemas = self.tree.find(state=sym_state(''))
        self.assertEqual(len(self.tree.root.children), len(app_schemas))

        # larger example
        ################

        # case 1:
        state = sym_state('1,3,5,7')
        app_schemas = self.tree.find(state)
        self.assertEqual(10, len(app_schemas))

        self.assertIn(self.s1, app_schemas)
        self.assertIn(self.s2, app_schemas)
        self.assertIn(self.s1_1, app_schemas)
        self.assertIn(self.s1_3, app_schemas)
        self.assertIn(self.s2_1, app_schemas)
        self.assertIn(self.s1_1_2, app_schemas)
        self.assertIn(self.s1_1_4, app_schemas)
        self.assertIn(self.s1_1_2_1, app_schemas)
        self.assertIn(self.s1_1_2_3, app_schemas)
        self.assertIn(self.s1_1_2_1_1, app_schemas)

        # case 2: only primitives
        state = sym_state('7,9')

        app_schemas = self.tree.find(state)
        self.assertEqual(len(self.tree.root.children), len(app_schemas))

        for schema in chain.from_iterable([c.schemas for c in self.tree.root.children]):
            self.assertIn(schema, app_schemas)

        # case 3: entire tree applicable
        state = sym_state('1,2,3,4,5,6,7,8,9')

        app_schemas = self.tree.find(state)
        self.assertEqual(len(self.tree), len(app_schemas))
        for schema in chain.from_iterable([c.schemas for c in self.tree.root.children]):
            self.assertIn(schema, app_schemas)

    @test_share.performance_test
    def test_performance(self):
        n_items = 1000
        n_iters = 1000

        pool = ItemPool()
        for state_element in range(n_items):
            pool.get(state_element)

        # create linear tree
        n_schemas = 100

        schema = sym_schema('/A1/')
        tree = SchemaTree()
        tree.add(tree.root, (schema,))

        for i in range(n_schemas - 1):
            spinoff = create_spin_off(schema, Schema.SpinOffType.CONTEXT, sym_assert(str(i)))
            tree.add(schema, (spinoff,))
            schema = spinoff

        self.assertEqual(n_schemas, len(tree))
        self.assertEqual(n_schemas, tree.height)

        elapsed_time = 0

        for _ in range(n_iters):
            start = time()
            _ = tree.find(state=sample(range(n_items), k=5))
            end = time()
            elapsed_time += end - start

        print(f'Time calling SchemaTree.find_applicable {n_iters:,} times (linear tree) {elapsed_time}s')

        n_schemas = 1000

        # create large (randomly balanced) tree
        schema = sym_schema('/A1/')
        tree = SchemaTree()
        tree.add(tree.root, (schema,))

        for state_element in range(n_schemas - 1):
            spinoff = create_spin_off(schema, Schema.SpinOffType.CONTEXT, sym_assert(str(state_element)))
            tree.add(schema, (spinoff,))
            schema = sample(list(chain.from_iterable([d.schemas for d in tree.root.descendants])), k=1)[0]

        self.assertEqual(n_schemas, len(tree))

        for _ in range(n_iters):
            start = time()
            _ = tree.find(state=sample(range(n_items), k=10))
            end = time()
            elapsed_time += end - start

        print(f'Time calling SchemaTree.find_applicable {n_iters:,} times (random tree) {elapsed_time}s')


class TestSchemaTreeNode(TestCase):
    def setUp(self) -> None:
        self.blank_stn = SchemaTreeNode()

        self.stn_1 = sym_schema_tree_node('1,2/A1/')
        self.stn_2 = sym_schema_tree_node('2,3/A1/')

    def test_init(self):
        stn = SchemaTreeNode()
        self.assertIs(None, stn.context)
        self.assertIs(None, stn.action)

        # action only node
        stn = SchemaTreeNode(action=Action('A1'))
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertIs(None, stn.context)
        self.assertEqual(Action('A1'), stn.action)

        # context and action node
        stn = SchemaTreeNode(context=sym_state_assert('1,2,~3'), action=Action('A1'))
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertEqual(sym_state_assert('1,2,~3'), stn.context)
        self.assertEqual(Action('A1'), stn.action)

    def test_equal(self):
        copy = sym_schema_tree_node('1,2/A1/')

        self.assertEqual(self.stn_1, self.stn_1)
        self.assertEqual(self.stn_1, copy)
        self.assertNotEqual(self.stn_1, self.stn_2)

        self.assertTrue(is_eq_reflexive(self.stn_1))
        self.assertTrue(is_eq_symmetric(x=self.stn_1, y=copy))
        self.assertTrue(is_eq_transitive(x=self.stn_1, y=copy, z=sym_schema_tree_node('1,2/A1/')))
        self.assertTrue(is_eq_consistent(x=self.stn_1, y=copy))
        self.assertTrue(is_eq_with_null_is_false(self.stn_1))

    def test_hash(self):
        copy = sym_schema_tree_node('1,2/A1/')

        self.assertIsInstance(hash(self.stn_1), int)
        self.assertTrue(is_hash_consistent(self.stn_1))
        self.assertTrue(is_hash_same_for_equal_objects(x=self.stn_1, y=copy))
