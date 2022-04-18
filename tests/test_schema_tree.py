import os
from copy import copy
from pathlib import Path
from random import sample
from tempfile import TemporaryDirectory
from time import time
from unittest import TestCase

import test_share
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import SchemaTree
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.func_api import actions
from schema_mechanism.func_api import primitive_schemas
from schema_mechanism.func_api import sym_item_assert
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_schema_tree_node
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import create_context_spin_off
from schema_mechanism.modules import create_result_spin_off
from schema_mechanism.persistence import deserialize
from schema_mechanism.persistence import serialize
from test_share.test_func import common_test_setup
from test_share.test_func import file_was_written
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestSchemaTree(TestCase):
    def setUp(self) -> None:
        # root
        # |-- 1
        # |   |-- 1,2
        # |   |-- 1,3
        # |   |   |-- 1,3,5
        # |   |   |   +-- 1,3,5,7
        # |   |   |-- 1,3,6
        # |   |   +-- 1,3,7
        # |   |       +-- 1,3,7,~9
        # |   |           |-- 1,11,3,7,~9
        # |   |           +-- 1,13,3,7,~9
        # |   +-- 1,4
        # |-- ~1
        # |   +-- ~1,~7
        # |       +-- ~1,~7,~9
        # |-- ~2
        # +-- 3
        #     +-- 3,6
        self.p1 = sym_schema('/A1/')
        self.p2 = sym_schema('/A2/')
        self.p3 = sym_schema('/A3/')

        self.tree = SchemaTree([self.p1, self.p2, self.p3])

        # NODE (ROOT) schemas
        self.p1_r1 = sym_schema('/A1/X,')
        self.p2_r1 = sym_schema('/A2/X,')
        self.p3_r1 = sym_schema('/A3/X,')

        self.tree.add_result_spin_offs(self.p1, [self.p1_r1])
        self.tree.add_result_spin_offs(self.p2, [self.p2_r1])
        self.tree.add_result_spin_offs(self.p3, [self.p3_r1])

        # NODE (1) schemas
        self.p1_r1_c1 = sym_schema('1,/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1, [self.p1_r1_c1])

        # NODE (1,2):
        self.p1_r1_c1_c1 = sym_schema('1,2/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1, [self.p1_r1_c1_c1])

        # NODE (1,3):
        self.p1_r1_c1_c2 = sym_schema('1,3/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1, [self.p1_r1_c1_c2])

        # NODE (1,4):
        self.p1_r1_c1_c3 = sym_schema('1,4/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1, [self.p1_r1_c1_c3])

        # NODE (1,3,5):
        self.p1_r1_c1_c2_c1 = sym_schema('1,3,5/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2, [self.p1_r1_c1_c2_c1])

        # NODE (1,3,6):
        self.p1_r1_c1_c2_c2 = sym_schema('1,3,6/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2, [self.p1_r1_c1_c2_c2])

        # NODE (1,3,7):
        self.p1_r1_c1_c2_c3 = sym_schema('1,3,7/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2, [self.p1_r1_c1_c2_c3])

        # NODE (1,3,5,7):
        self.p1_r1_c1_c2_c1_c1 = sym_schema('1,3,5,7/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2_c1, [self.p1_r1_c1_c2_c1_c1])

        # NODE (1,3,7,~9):
        self.p1_r1_c1_c2_c3_c1 = sym_schema('1,3,7,~9/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2_c3, [self.p1_r1_c1_c2_c3_c1])

        # NODE (1,3,7,~9,11):
        self.p1_r1_c1_c2_c3_c1_c1 = sym_schema('1,3,7,~9,11/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2_c3_c1, [self.p1_r1_c1_c2_c3_c1_c1])

        # NODE (1,3,7,~9,13):
        self.p1_r1_c1_c2_c3_c1_c2 = sym_schema('1,3,7,~9,13/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2_c3_c1, [self.p1_r1_c1_c2_c3_c1_c2])

        # NODE (~1):
        self.p3_r1_c1 = sym_schema('~1,/A3/X,')
        self.tree.add_context_spin_offs(self.p3_r1, [self.p3_r1_c1])

        # NODE (~1,~7):
        self.p3_r1_c1_c1 = sym_schema('~1,~7/A3/X,')
        self.tree.add_context_spin_offs(self.p3_r1_c1, [self.p3_r1_c1_c1])

        # NODE (~1,~7,~9):
        self.p3_r1_c1_c1_c1 = sym_schema('~1,~7,~9/A3/X,')
        self.tree.add_context_spin_offs(self.p3_r1_c1_c1, [self.p3_r1_c1_c1_c1])

        # NODE (~2):
        self.p2_r1_c1 = sym_schema('~2,/A2/X,')
        self.tree.add_context_spin_offs(self.p2_r1, [self.p2_r1_c1])

        # NODE (3):
        self.p1_r1_c2 = sym_schema('3,/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1, [self.p1_r1_c2])

        # NODE (3,1): [REDUNDANT WITH 1,3... Shouldn't result in a new schema or node in the self.tree]
        self.p1_r1_c2_c1 = sym_schema('3,1/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c2, [self.p1_r1_c2_c1])

        # NODE (3,6):
        self.p1_r1_c2_c2 = sym_schema('3,6/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c2, [self.p1_r1_c2_c2])

    def test_init(self):
        # test: empty list of primitives should raise a ValueError
        self.assertRaises(ValueError, lambda: SchemaTree([]))

        primitives = [sym_schema(f'/A{i}/') for i in range(5)]
        tree = SchemaTree(primitives)

        # test: primitives should be properly added to root node
        self.assertSetEqual(set(primitives), set(tree.root.schemas_satisfied_by))

        # test: n_schemas should be incremented to reflect the number of primitive schemas
        self.assertEqual(len(primitives), tree.n_schemas)

    def test_iter(self):
        # test: a primitives only tree's iterator should only return the root node
        tree = SchemaTree([sym_schema('/A1/'), sym_schema('/A2/')])
        for n in tree:
            self.assertIs(tree.root, n)

        # test: iterator should visit all nodes in tree
        count = 0
        for _ in self.tree:
            count += 1
        self.assertEqual(len(self.tree), count)

    def test_add_result_spin_offs(self):
        p1 = sym_schema('/A1/')
        p2 = sym_schema('/A2/')

        tree = SchemaTree(primitives=[p1, p2])

        # test: ValueError should be raised raised if collection of spin_offs is empty
        self.assertRaises(ValueError, lambda: tree.add_result_spin_offs(p1, spin_offs=[]))

        spin_offs = [sym_schema(f'/A1/{i},') for i in range(5)]

        # adding single result spinoff schema to tree
        len_before_add = len(tree)
        tree.add_result_spin_offs(p1, spin_offs)
        self.assertEqual(len_before_add + len(spin_offs), len(tree))

        for schema in spin_offs:
            self.assertIn(schema, tree)

        # test: ValueError should be raised if source and spin-offs have different actions
        self.assertRaises(ValueError, lambda: tree.add_result_spin_offs(p2, spin_offs))

    def test_add_context_spin_offs(self):
        p1 = sym_schema('/A1/')
        p2 = sym_schema('/A2/')

        tree = SchemaTree(primitives=[p1, p2])

        # test: ValueError should be raised if collection of spin_offs is empty
        self.assertRaises(ValueError, lambda: tree.add_context_spin_offs(p1, spin_offs=[]))

        # add a result spin-offs to serve as source
        s1 = sym_schema('/A1/X,')
        tree.add_result_spin_offs(p1, [s1])

        # context spin-offs
        spin_offs = [sym_schema(f'{i},/A1/X,') for i in range(5)]

        # adding single context spinoff schema to tree
        len_before_add = len(tree)
        tree.add_context_spin_offs(s1, spin_offs)

        self.assertEqual(len_before_add + len(spin_offs), len(tree))

        for schema in spin_offs:
            self.assertIn(schema, tree)

        # test: ValueError should be raised if source and spin-offs have different actions
        self.assertRaises(ValueError, lambda: tree.add_context_spin_offs(p2, spin_offs))

    def test_add_spin_offs(self):
        # more realistic example that interleaves result and context spin-offs
        # ---> in actual operation, result spin-offs always precede context spin-offs

        # primitive schemas
        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A2/')
        s3 = sym_schema('/A3/')

        tree = SchemaTree(primitives=(s1, s2, s3))

        # test: adding result spin-offs to primitives [ROUND 1]
        s1_r1 = sym_schema('/A1/101,')
        s1_r2 = sym_schema('/A1/102,')
        s2_r1 = sym_schema('/A2/101,')

        n_schemas_before_add = tree.n_schemas
        tree.add_result_spin_offs(s1, [s1_r1, s1_r2])
        tree.add_result_spin_offs(s2, [s2_r1])

        self.assertEqual(n_schemas_before_add + 3, tree.n_schemas)

        # test: primitives and result spin-offs of primitives should be added to root node
        for schema in [s1, s2, s3, s1_r1, s1_r2, s2_r1]:
            self.assertIn(schema, tree.root.schemas_satisfied_by)

        # test: adding context spin-offs to first round of result spin-offs [ROUND 2]
        s1_r1_c1 = sym_schema('1,/A1/101,')
        s1_r1_c2 = sym_schema('2,/A1/101,')
        s1_r2_c1 = sym_schema('1,/A1/102,')
        s2_r1_c1 = sym_schema('1,/A2/101,')
        s2_r1_c2 = sym_schema('2,/A2/101,')

        # test: tree nodes SHOULD NOT exist for context spin-offs before add
        for schema in [s1_r1_c1, s1_r1_c2, s1_r2_c1, s2_r1_c1, s2_r1_c2]:
            self.assertRaises(KeyError, lambda: tree.get(schema.context))

        n_schemas_before_add = tree.n_schemas
        tree.add_context_spin_offs(s1_r1, [s1_r1_c1, s1_r1_c2])
        tree.add_context_spin_offs(s1_r2, [s1_r2_c1])
        tree.add_context_spin_offs(s2_r1, [s2_r1_c1, s2_r1_c2])
        n_schemas_after_add = tree.n_schemas

        self.assertEqual(n_schemas_before_add + 5, n_schemas_after_add)

        # test: tree nodes SHOULD exist for context spin-offs after add
        try:
            for schema in [s1_r1_c1, s1_r1_c2, s1_r2_c1, s2_r1_c1, s2_r1_c2]:
                self.assertIsNotNone(tree.get(schema.context))
        except KeyError as e:
            self.fail(f'Unexpected exception: {str(e)}')

        # test: context spin-offs should be added to new nodes (not their sources nodes)
        for schema in [s1_r1_c1, s1_r1_c2, s1_r2_c1, s2_r1_c1, s2_r1_c2]:
            self.assertEqual(schema.context, tree.get(schema.context).context)

        # test: adding context spin-offs to 2nd round of context spin-offs [ROUND 3]
        s1_r1_c1_2 = sym_schema('1,2/A1/101,')
        s1_r2_c1_3 = sym_schema('1,3/A1/102,')
        s2_r1_c1_2 = sym_schema('1,2/A2/101,')

        # test: tree nodes SHOULD NOT exist for context spin-offs before add
        for schema in [s1_r1_c1_2, s1_r2_c1_3, s2_r1_c1_2]:
            self.assertRaises(KeyError, lambda: tree.get(schema.context))

        n_schemas_before_add = tree.n_schemas
        tree.add_context_spin_offs(s1_r1_c1, [s1_r1_c1_2, s1_r2_c1_3])
        tree.add_context_spin_offs(s2_r1_c1, [s2_r1_c1_2])
        n_schemas_after_add = tree.n_schemas

        self.assertEqual(n_schemas_before_add + 3, n_schemas_after_add)

        # test: context spin-offs should be added to new nodes (not their sources nodes)
        for schema in [s1_r1_c1_2, s1_r2_c1_3, s2_r1_c1_2]:
            self.assertEqual(schema.context, tree.get(schema.context).context)

    def test_add_primitives(self):
        p1 = sym_schema('/A1/')
        p2 = sym_schema('/A2/')

        tree = SchemaTree(primitives=[p1, p2])

        # test: primitive actions
        new_primitives = [sym_schema(f'/A{i}/') for i in range(5, 10)]

        for s in new_primitives:
            self.assertNotIn(s, tree)

        tree.add_primitives(new_primitives)

        for s in new_primitives:
            self.assertIn(s, tree)

        # test: composite actions
        new_primitives = [
            sym_schema(f'/1,2/'),
            sym_schema(f'/1,3/'),
            sym_schema(f'/2,3/'),
            sym_schema(f'/3,~4/'),
        ]
        tree.add_primitives(new_primitives)

        for s in new_primitives:
            self.assertIn(s, tree)

    def test_is_valid_node_1(self):
        # sanity check: all nodes in the test tree should be valid
        for schema in self.tree:
            self.assertTrue(self.tree.is_valid_node(schema))

    def test_is_valid_node_2(self):
        # test: unattached tree node should be invalid
        s_unattached = sym_schema_tree_node('7,')
        self.assertFalse(self.tree.is_valid_node(s_unattached))

    def test_is_valid_node_3(self):
        # test: composite results must also exist as contexts in tree

        p1 = sym_schema('/A1/')
        p2_r1 = sym_schema('/A1/1,')
        s1 = sym_schema('1,/A1/1,')
        s2 = sym_schema('1,2/A1/1,')
        s3 = sym_schema('/A1/(1,2),')

        tree = SchemaTree(primitives=(p1,))
        tree.add_result_spin_offs(p1, [p2_r1])

        tree.add_context_spin_offs(p2_r1, [s1])
        tree.add_context_spin_offs(s1, [s2])

        self.assertTrue(tree.is_valid_node(tree.get(s1.context)))
        self.assertTrue(tree.is_valid_node(tree.get(s2.context)))

        # test: this should be a valid composite result spin-off
        tree.add_context_spin_offs(p1, [s3])
        self.assertTrue(tree.is_valid_node(tree.get(s3.context)))

    def test_is_valid_node_4(self):
        # test: children should have all of the context item assertion of their parents
        p1 = sym_schema('/A1/')
        p1_r1 = sym_schema('/A1/1,')
        s1 = sym_schema('3,/A1/1,')

        tree = SchemaTree(primitives=(p1,))
        tree.add_result_spin_offs(p1, [p1_r1])
        tree.add_context_spin_offs(p1_r1, [s1])

        # adding this schema should result in an invalid node
        s_illegal_context = sym_schema('1,2/A1/1,')
        tree.add_context_spin_offs(s1, [s_illegal_context])

        self.assertFalse(tree.is_valid_node(tree.get(s_illegal_context.context)))

    def test_is_valid_node_5(self):
        # test: nodes should have a depth equal to the number of item assertions in their context
        p1 = sym_schema('/A1/')
        p1_r1 = sym_schema('/A1/1,')
        s1 = sym_schema('1,/A1/1,')
        s2 = sym_schema('1,2/A1/1,')

        # should result in invalid node due to depth
        s3 = sym_schema('1,2,3,4/A1/1,')

        tree = SchemaTree(primitives=(p1,))
        tree.add_result_spin_offs(p1, [p1_r1])
        tree.add_context_spin_offs(p1_r1, [s1])
        tree.add_context_spin_offs(s1, [s2])

        # adding this schema should result in an invalid node
        tree.add_context_spin_offs(s2, [s3])

        self.assertFalse(tree.is_valid_node(tree.get(s3.context)))

    def test_validate(self):
        # all nodes in tree should be valid
        invalid_nodes = self.tree.validate()
        self.assertEqual(0, len(invalid_nodes))

        # adding invalid nodes
        invalid_1 = sym_schema('1,2,5/A1/X,')
        invalid_2 = sym_schema('~1,~6,~9/A3/X,')

        # invalid due to parent and child having inconsistent contexts
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2, (invalid_1,))
        self.tree.add_context_spin_offs(self.p3_r1_c1_c1_c1, (invalid_2,))

        invalid_nodes = self.tree.validate()
        self.assertEqual(2, len(invalid_nodes))
        self.assertTrue(all({self.tree.get(s.context) in invalid_nodes for s in [invalid_1, invalid_2]}))

    def test_contains(self):
        self.assertTrue(all({s in self.tree for s in self.tree}))

        # any other schema SHOULD NOT be contained in tree
        another = sym_schema('ZZ,/A1/X,')
        self.assertNotIn(another, self.tree)

        # after adding, another schema should be contained in tree
        self.tree.add_context_spin_offs(self.p1, spin_offs=(another,))
        self.assertIn(another, self.tree)

        # contains should work for SchemaTreeNodes as well
        node_another = sym_schema_tree_node('ZZ,')
        self.assertIn(node_another, self.tree)

        node_missing = sym_schema_tree_node('1,19,')
        self.assertNotIn(node_missing, self.tree)

    @test_share.string_test
    def test_str(self):
        print(self.tree)

    def test_find_all_satisfied(self):

        # case 1:
        nodes = self.tree.find_all_satisfied(state=sym_state('1,3,7'))
        expected_nodes = [
            self.tree.root,
            sym_schema_tree_node('1,'),
            sym_schema_tree_node('~2,'),
            sym_schema_tree_node('3,'),
            sym_schema_tree_node('1,3'),
            sym_schema_tree_node('1,3,7'),
            sym_schema_tree_node('1,3,7,~9'),
        ]

        self.assertSetEqual(set(expected_nodes), set(nodes))

        # case 2:
        nodes = self.tree.find_all_satisfied(state=sym_state('7,9'))
        expected_nodes = [
            self.tree.root,
            sym_schema_tree_node('~1,'),
            sym_schema_tree_node('~2,'),
        ]

        self.assertSetEqual(set(expected_nodes), set(nodes))

        # case 3:
        nodes = self.tree.find_all_satisfied(state=sym_state('2'))
        expected_nodes = [
            self.tree.root,
            sym_schema_tree_node('~1,'),
            sym_schema_tree_node('~1,~7,'),
            sym_schema_tree_node('~1,~7,~9'),
        ]

        self.assertSetEqual(set(expected_nodes), set(nodes))

    def test_find_all_would_satisfy(self):
        # case 1:
        nodes = self.tree.find_all_would_satisfy(assertion=sym_state_assert('1,3'))
        expected_nodes = [
            sym_schema_tree_node('1,3'),
            sym_schema_tree_node('1,3,5'),
            sym_schema_tree_node('1,3,6'),
            sym_schema_tree_node('1,3,7'),
            sym_schema_tree_node('1,3,5,7'),
            sym_schema_tree_node('1,3,7,~9'),
            sym_schema_tree_node('1,3,7,~9,11'),
            sym_schema_tree_node('1,3,7,~9,13'),
        ]

        self.assertSetEqual(set(expected_nodes), set(nodes))

        # case 2:
        nodes = self.tree.find_all_would_satisfy(assertion=sym_state_assert('8,9'))
        expected_nodes = []

        self.assertSetEqual(set(expected_nodes), set(nodes))

        # case 3:
        nodes = self.tree.find_all_would_satisfy(assertion=sym_state_assert('3,6'))
        expected_nodes = [
            sym_schema_tree_node('3,6'),
            sym_schema_tree_node('1,3,6'),
        ]

        self.assertSetEqual(set(expected_nodes), set(nodes))

    def test_serialize(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-schema-tree-serialize.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            serialize(self.tree, path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            recovered: SchemaTree = deserialize(path)

            self.assertEqual(self.tree, recovered)

    @test_share.performance_test
    def test_performance_1(self):
        n_iterations = 1000
        n_actions = 10

        tree = SchemaTree(primitive_schemas(actions(n_actions)))

        elapsed_time = 0
        schema = sample(list(tree.root.schemas_satisfied_by), k=1)[0]
        for state_element in range(n_iterations):
            spinoff = create_context_spin_off(schema, sym_item_assert(str(state_element)))

            start = time()
            tree.add_context_spin_offs(schema, (spinoff,))
            end = time()
            elapsed_time += end - start

            node = sample(tree.root.descendants, k=1)[0]
            schema = sample(list(node.schemas_satisfied_by), k=1)[0]

        print(f'Time calling SchemaTree.add_context_spin_offs {n_iterations:,} times {elapsed_time}s')

    @test_share.performance_test
    def test_performance_2(self):
        n_iterations = 1000
        n_actions = 10

        tree = SchemaTree(primitive_schemas(actions(n_actions)))

        elapsed_time = 0
        schema = sample(list(tree.root.schemas_satisfied_by), k=1)[0]
        for state_element in range(100, 100 + n_iterations):
            spinoff = create_result_spin_off(schema, sym_item_assert(str(state_element)))

            start = time()
            self.tree.add_result_spin_offs(schema, (spinoff,))
            end = time()
            elapsed_time += end - start

            schema = sample(list(tree.root.schemas_satisfied_by), k=1)[0]

        print(f'Time calling SchemaTree.add_result_spin_offs {n_iterations:,} times {elapsed_time}s')


class TestSchemaTreeNode(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.blank_stn = SchemaTreeNode()

        self.stn_1 = sym_schema_tree_node('1,2')
        self.stn_2 = sym_schema_tree_node('2,3')

    def test_init(self):
        stn = SchemaTreeNode()
        self.assertIs(NULL_STATE_ASSERT, stn.context)

        # action only node
        stn = SchemaTreeNode()
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertIs(NULL_STATE_ASSERT, stn.context)

        # context and action node
        stn = SchemaTreeNode(context=sym_state_assert('1,2,~3'))
        self.assertIsInstance(stn, SchemaTreeNode)
        self.assertEqual(sym_state_assert('1,2,~3'), stn.context)

    def test_schemas_satisfied_by(self):
        # test: setter/getter
        new_value = {sym_schema('1,2/A1/3,4'), sym_schema('1,2/A1/3,5')}

        # test: should be able to directly assign to set of schemas
        self.stn_1.schemas_satisfied_by = new_value
        self.assertSetEqual(new_value, self.stn_1.schemas_satisfied_by)

        # test: should be able to add values to set
        new_element = sym_schema('1,2/A2/3,4')
        self.stn_1.schemas_satisfied_by.add(new_element)
        self.assertIn(new_element, self.stn_1.schemas_satisfied_by)

        # test: previously added values should be retained
        for v in new_value:
            self.assertIn(v, self.stn_1.schemas_satisfied_by)

    def test_schemas_would_satisfy(self):
        # test: setter/getter
        new_value = {sym_schema('1,2/A1/3,4'), sym_schema('1,2/A1/3,5')}

        # test: should be able to directly assign to set of schemas
        self.stn_1.schemas_would_satisfy = new_value
        self.assertSetEqual(new_value, self.stn_1.schemas_would_satisfy)

        # test: should be able to add values to set
        new_element = sym_schema('1,2/A2/3,4')
        self.stn_1.schemas_would_satisfy.add(new_element)
        self.assertIn(new_element, self.stn_1.schemas_would_satisfy)

        # test: previously added values should be retained
        for v in new_value:
            self.assertIn(v, self.stn_1.schemas_would_satisfy)

    def test_equal(self):
        copy_ = copy(self.stn_1)

        self.assertEqual(self.stn_1, self.stn_1)
        self.assertEqual(self.stn_1, copy_)
        self.assertNotEqual(self.stn_1, self.stn_2)

        self.assertTrue(is_eq_reflexive(self.stn_1))
        self.assertTrue(is_eq_symmetric(x=self.stn_1, y=copy_))
        self.assertTrue(is_eq_transitive(x=self.stn_1, y=copy_, z=copy(copy_)))
        self.assertTrue(is_eq_consistent(x=self.stn_1, y=copy_))
        self.assertTrue(is_eq_with_null_is_false(self.stn_1))

    def test_hash(self):
        copy_ = copy(sym_schema_tree_node('1,2'))

        self.assertIsInstance(hash(self.stn_1), int)
        self.assertTrue(is_hash_consistent(self.stn_1))
        self.assertTrue(is_hash_same_for_equal_objects(x=self.stn_1, y=copy_))
