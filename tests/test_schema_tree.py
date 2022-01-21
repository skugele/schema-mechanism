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
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import SchemaTree
from schema_mechanism.modules import create_spin_off


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
        self.tree.add_all(parent=self.tree.root, children=(self.s1, self.s2))
        self.tree.add_all(parent=self.s1, children=(self.s1_1, self.s1_2, self.s1_3, self.s1_4))
        self.tree.add_all(parent=self.s2, children=(self.s2_1, self.s2_2))
        self.tree.add_all(parent=self.s1_1, children=(self.s1_1_1, self.s1_1_2, self.s1_1_3, self.s1_1_4))
        self.tree.add_all(parent=self.s1_1_2, children=(self.s1_1_2_1, self.s1_1_2_2, self.s1_1_2_3))
        self.tree.add_all(parent=self.s1_1_2_1, children=(self.s1_1_2_1_1,))
        self.tree.add_all(parent=self.s1_1_2_3, children=(self.s1_1_2_3_1,))

    def test_iter(self):
        for n in self.empty_tree:
            self.fail()

        count = 0
        for n in self.tree:
            count += 1

        self.assertEqual(len(self.tree), count)

    def test_add_all(self):
        # adding single primitive schema to tree
        s3 = sym_schema('/A3/')

        len_before_add = len(self.tree)
        self.tree.add_all(self.tree.root, (s3,))
        len_after_add = len(self.tree)

        self.assertEqual(len_before_add + 1, len_after_add)
        self.assertIn(s3, self.tree)
        self.assertIn(s3, self.tree.primitive_schemas)

        # adding single non-primitive schema to tree
        s2_3 = sym_schema('3/A2/')

        len_before_add = len(self.tree)
        self.tree.add_all(self.s2, (s2_3,))
        len_after_add = len(self.tree)

        self.assertEqual(len_before_add + 1, len_after_add)
        self.assertIn(s2_3, self.tree)

        # adding multiple non-primitive schemas
        s1_1_1_1 = sym_schema('1,2,3/A1/')
        s1_1_1_2 = sym_schema('1,2,4/A1/')

        len_before_add = len(self.tree)
        self.tree.add_all(self.s1_1_1, (s1_1_1_1, s1_1_1_2))
        len_after_add = len(self.tree)

        self.assertEqual(len_before_add + 2, len_after_add)
        self.assertIn(s1_1_1_1, self.tree)
        self.assertIn(s1_1_1_2, self.tree)

        # sending empty set should result in a value error
        self.assertRaises(ValueError, lambda: self.tree.add_all(parent=self.tree.root, children=[]))

    def test_primitive_schemas(self):
        self.assertEqual(2, len(self.tree.primitive_schemas))

        self.assertIn(self.s1, self.tree.primitive_schemas)
        self.assertIn(self.s2, self.tree.primitive_schemas)

        for ps in self.tree.primitive_schemas:
            self.assertTrue(1, ps.depth)
            self.assertEqual(self.tree.root, ps.parent)

    def test_is_valid_node_1(self):
        # all nodes in this tree should be valid
        for schema in self.tree:
            self.assertTrue(self.tree.is_valid_node(schema))

    def test_is_valid_node_2(self):
        # root node should be valid
        self.assertTrue(self.tree.is_valid_node(self.tree.root))

    def test_is_valid_node_3(self):
        # unattached schema should be invalid
        s_unattached = sym_schema('3/A1/')
        self.assertFalse(self.tree.is_valid_node(s_unattached))

    def test_is_valid_node_4(self):
        # the children of root should only be primitive schemas
        s_illegal_parent = sym_schema('1/A3/')
        self.tree.add_all(self.tree.root, (s_illegal_parent,))
        self.assertFalse(self.tree.is_valid_node(s_illegal_parent))

    def test_is_valid_node_5(self):
        # non-primitive nodes should have the same action as their parents
        s_illegal_action = sym_schema('1/A3/')
        self.tree.add_all(self.tree.primitive_schemas[0], (s_illegal_action,))
        self.assertFalse(self.tree.is_valid_node(s_illegal_action))

    def test_is_valid_node_6(self):
        s1 = sym_schema('/A1/')
        s1_1 = sym_schema('1/A1/')
        s1_1_1 = sym_schema('1,2/A1/')

        tree = SchemaTree()
        tree.add_all(tree.root, (s1,))
        tree.add_all(s1, (s1_1,))
        tree.add_all(s1_1, (s1_1_1,))

        for n in tree:
            self.assertTrue(tree.is_valid_node(n))

        # children should have all of the context item assertions of their parents
        s_illegal_context = sym_schema('1,5,6/A1/')
        tree.add_all(s1_1_1, (s_illegal_context,))
        self.assertFalse(tree.is_valid_node(s_illegal_context))

    def test_validate(self):
        # all nodes in tree should be valid
        invalid_nodes = self.tree.validate()
        self.assertEqual(0, len(invalid_nodes))

        # adding invalid nodes
        invalid_1 = sym_schema('1/A3/')
        invalid_2 = sym_schema('1,2,3,4,5/A1/')

        # invalid due to parent and child having different actions
        self.tree.add_all(self.s2, (invalid_1,))

        # invalid due to parent and child having inconsistent contexts
        self.tree.add_all(self.s1_1_2_1_1, (invalid_2,))

        invalid_nodes = self.tree.validate()
        self.assertEqual(2, len(invalid_nodes))
        self.assertIn(invalid_1, invalid_nodes)
        self.assertIn(invalid_2, invalid_nodes)

    def test_contains(self):
        p_schemas = primitive_schemas(actions(5))

        tree = SchemaTree()
        tree.add_all(parent=tree.root, children=p_schemas)
        self.assertTrue(all({s in tree for s in p_schemas}))

        # any other schema SHOULD NOT be contained in tree
        another = Schema(action=Action())
        self.assertNotIn(another, tree)

        # after adding, another schema should be contained in tree
        tree.add_all(parent=p_schemas[0], children=(another,))
        self.assertIn(another, tree)

    @test_share.string_test
    def test_str(self):
        print(self.tree)

    def test_find_applicable(self):

        # empty tree should return empty set
        app_schemas = self.empty_tree.find_applicable(state=sym_state('1'))
        self.assertEqual(0, len(app_schemas))

        # primitive schemas should be applicable to all states
        app_schemas = self.tree.find_applicable(state=sym_state(''))
        self.assertEqual(len(self.tree.primitive_schemas), len(app_schemas))

        # larger example
        ################

        # case 1:
        state = sym_state('1,3,5,7')
        app_schemas = self.tree.find_applicable(state)
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

        app_schemas = self.tree.find_applicable(state)
        self.assertEqual(len(self.tree.primitive_schemas), len(app_schemas))
        for s in self.tree.primitive_schemas:
            self.assertIn(s, app_schemas)

        # case 3: entire tree applicable
        state = sym_state('1,2,3,4,5,6,7,8,9')

        app_schemas = self.tree.find_applicable(state)
        self.assertEqual(len(self.tree), len(app_schemas))
        for s in self.tree:
            self.assertIn(s, app_schemas)

    @test_share.performance_test
    def test_performance(self):
        n_items = 1000
        n_iters = 1000

        pool = ItemPool()
        for state_element in range(n_items):
            pool.get(state_element)

        # create linear tree
        n_schemas = 100

        tree = SchemaTree()
        tree.add_all(parent=tree.root,
                     children=primitive_schemas((Action(),)))

        node = tree.primitive_schemas[0]
        for i in range(n_schemas - 1):
            spinoff = create_spin_off(node, Schema.SpinOffType.CONTEXT, sym_assert(str(i)))
            tree.add_all(node, (spinoff,))
            node = spinoff

        self.assertEqual(n_schemas, len(tree))
        self.assertEqual(n_schemas, tree.height)

        elapsed_time = 0

        for _ in range(n_iters):
            start = time()
            _ = tree.find_applicable(state=sample(range(n_items), k=5))
            end = time()
            elapsed_time += end - start

        print(f'Time calling SchemaTree.find_applicable {n_iters:,} times (linear tree) {elapsed_time}s')

        n_schemas = 1000

        # create large (randomly balanced) tree
        tree = SchemaTree()
        tree.add_all(parent=tree.root,
                     children=primitive_schemas((Action(),)))

        node = tree.primitive_schemas[0]
        for state_element in range(n_schemas - 1):
            spinoff = create_spin_off(node, Schema.SpinOffType.CONTEXT, sym_assert(str(state_element)))
            tree.add_all(node, (spinoff,))
            node = sample(tree.root.descendants, k=1)[0]

        self.assertEqual(n_schemas, len(tree))

        for _ in range(n_iters):
            start = time()
            _ = tree.find_applicable(state=sample(range(n_items), k=10))
            end = time()
            elapsed_time += end - start

        print(f'Time calling SchemaTree.find_applicable {n_iters:,} times (random tree) {elapsed_time}s')
