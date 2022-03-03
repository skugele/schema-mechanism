from itertools import chain
from random import sample
from time import time
from unittest import TestCase

import test_share
from schema_mechanism.core import Action
from schema_mechanism.core import SchemaResultTree
from schema_mechanism.core import SchemaResultTreeNode
from schema_mechanism.func_api import actions
from schema_mechanism.func_api import primitive_schemas
from schema_mechanism.func_api import sym_item_assert
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_schema_result_tree_node
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.modules import create_context_spin_off
from schema_mechanism.modules import create_result_spin_off
from test_share.test_func import common_test_setup
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestSchemaResultTree(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.empty_tree = SchemaResultTree()

        # schemas used in tree
        self.s1 = sym_schema('/A1/')
        self.s2 = sym_schema('/A2/')

        self.s1_1 = sym_schema('/A1/1,')
        self.s1_2 = sym_schema('/A1/2,')
        self.s1_3 = sym_schema('/A1/3,')
        self.s1_4 = sym_schema('/A1/4,')

        self.s2_1 = sym_schema('/A2/1,')
        self.s2_2 = sym_schema('/A2/2,')

        self.s1_1_1 = sym_schema('/A1/1,2')
        self.s1_1_2 = sym_schema('/A1/1,3')
        self.s1_1_3 = sym_schema('/A1/1,4')
        self.s1_1_4 = sym_schema('/A1/1,5')

        self.s2_1_1 = sym_schema('/A2/1,3')
        self.s2_1_2 = sym_schema('/A2/1,4')

        self.s1_1_2_1 = sym_schema('/A1/1,3,5')
        self.s1_1_2_2 = sym_schema('/A1/1,3,6')
        self.s1_1_2_3 = sym_schema('/A1/1,3,7')

        self.s1_1_2_1_1 = sym_schema('/A1/1,3,5,7')
        self.s1_1_2_3_1 = sym_schema('/A1/1,3,7,~9')

        self.s1_1_2_3_1_1 = sym_schema('/A1/1,3,7,~9,11')
        self.s1_1_2_3_1_2 = sym_schema('/A1/1,3,7,~9,13')

        self.tree = SchemaResultTree()
        self.tree.add_primitives((self.s1, self.s2))
        self.tree.add_result_spin_offs(self.s1, (self.s1_1, self.s1_2, self.s1_3, self.s1_4))
        self.tree.add_result_spin_offs(self.s2, (self.s2_1, self.s2_2))
        self.tree.add_result_spin_offs(self.s2_1, (self.s2_1_1, self.s2_1_2))
        self.tree.add_result_spin_offs(self.s1_1, (self.s1_1_1, self.s1_1_2, self.s1_1_3, self.s1_1_4))
        self.tree.add_result_spin_offs(self.s1_1_2, (self.s1_1_2_1, self.s1_1_2_2, self.s1_1_2_3))
        self.tree.add_result_spin_offs(self.s1_1_2_1, (self.s1_1_2_1_1,))
        self.tree.add_result_spin_offs(self.s1_1_2_3, (self.s1_1_2_3_1,))
        self.tree.add_result_spin_offs(self.s1_1_2_3_1, (self.s1_1_2_3_1_1, self.s1_1_2_3_1_2))

    def test_contains_all(self):
        # test: search should return schemas from matching nodes and their descendants
        nodes = self.tree.contains_all(sym_state_assert('1,4'))
        self.assertSetEqual(set(nodes), {self.s1_1_3, self.s2_1_2})

        nodes = self.tree.contains_all(sym_state_assert('1,3'))
        self.assertSetEqual(set(nodes), {self.s1_1_2, self.s2_1_1, self.s1_1_2_1, self.s1_1_2_2, self.s1_1_2_3,
                                         self.s1_1_2_1_1, self.s1_1_2_3_1, self.s1_1_2_3_1_1, self.s1_1_2_3_1_2})

        # test: if leaf nodes matched, only the schemas from those leaf nodes should be returned
        nodes = self.tree.contains_all(sym_state_assert('1,3,7,~9,13'))
        self.assertSetEqual(set(nodes), {self.s1_1_2_3_1_2})

        # test: empty collection should be returned if no match
        nodes = self.tree.contains_all(sym_state_assert('N1,N2'))
        self.assertSetEqual(set(nodes), set())

    def test_iter(self):
        for _ in self.empty_tree:
            self.fail()

        count = 0
        for _ in self.tree:
            count += 1
        self.assertEqual(len(self.tree), count)

    def test_add_primitives(self):
        tree = SchemaResultTree()

        # ValueError raised if list of primitives is empty
        self.assertRaises(ValueError, lambda: tree.add_primitives(primitives=[]))

        primitives = primitive_schemas(actions(5))

        # adding single primitive schema to tree
        len_before_add = len(tree)
        tree.add_primitives(primitives[0:1])
        len_after_add = len(tree)

        self.assertEqual(len_before_add + 1, len_after_add)
        self.assertIn(primitives[0], tree)
        self.assertIn(primitives[0], tree.get(primitives[0]).schemas)

        # adding multiple primitive schema to tree
        len_before_add = len(tree)
        tree.add_primitives(primitives[1:])
        len_after_add = len(tree)

        self.assertEqual(len_before_add + len(primitives[1:]), len_after_add)

        for p in primitives:
            self.assertIn(p, tree)
            self.assertIn(p, tree.get(p).schemas)

    def test_add_context_spin_offs(self):
        # ValueError raised if list of context spin-offs is empty
        self.assertRaises(ValueError, lambda: self.tree.add_context_spin_offs(self.s1, spin_offs=[]))

        source = self.s1_1_2_3_1
        spin_offs = [create_context_spin_off(source, sym_item_assert(f'{i}')) for i in range(10, 15)]

        # test: len of tree before should equal len of tree after adding single context spinoff schema to tree
        schemas_before_add = self.tree.n_schemas
        len_before_add = len(self.tree)
        self.tree.add_context_spin_offs(source, spin_offs[0:1])
        schemas_after_add = self.tree.n_schemas
        len_after_add = len(self.tree)

        self.assertEqual(schemas_before_add + 1, schemas_after_add)
        self.assertEqual(len_before_add, len_after_add)

        self.assertIn(spin_offs[0], self.tree)
        self.assertIn(spin_offs[0], self.tree.get(spin_offs[0]).schemas)

        # adding multiple context spin-off schemas to tree
        schemas_before_add = self.tree.n_schemas
        len_before_add = len(self.tree)
        self.tree.add_context_spin_offs(source, spin_offs[1:])
        schemas_after_add = self.tree.n_schemas
        len_after_add = len(self.tree)

        self.assertEqual(schemas_before_add + len(spin_offs[1:]), schemas_after_add)
        self.assertEqual(len_before_add, len_after_add)

        for p in spin_offs:
            self.assertIn(p, self.tree)
            self.assertIn(p, self.tree.get(p).schemas)

    def test_add_result_spin_offs(self):
        # ValueError raised if list of result spin_offs is empty
        self.assertRaises(ValueError, lambda: self.tree.add_result_spin_offs(self.s1, spin_offs=[]))

        source = self.s2
        spin_offs = [create_result_spin_off(source, sym_item_assert(f'{i}')) for i in range(100, 105)]

        # adding single result spinoff schema to tree
        schemas_before_add = self.tree.n_schemas
        len_before_add = len(self.tree)
        self.tree.add_result_spin_offs(source, spin_offs[0:1])
        schemas_after_add = self.tree.n_schemas
        len_after_add = len(self.tree)

        # test: the number of tree nodes and schemas SHOULD increase by one for each result spin-off added
        self.assertEqual(schemas_before_add + 1, schemas_after_add)
        self.assertEqual(len_before_add + 1, len_after_add)

        self.assertIn(spin_offs[0], self.tree)
        self.assertIn(spin_offs[0], self.tree.get(spin_offs[0]).schemas)

        # adding multiple result spin-off schemas to tree
        schemas_before_add = self.tree.n_schemas
        len_before_add = len(self.tree)
        self.tree.add_result_spin_offs(source, spin_offs[1:])
        schemas_after_add = self.tree.n_schemas
        len_after_add = len(self.tree)

        # test: the number of tree nodes and schemas SHOULD increase by the number of result spin-offs added
        self.assertEqual(schemas_before_add + len(spin_offs[1:]), schemas_after_add)
        self.assertEqual(len_before_add + len(spin_offs[1:]), len_after_add)

        for p in spin_offs:
            self.assertIn(p, self.tree)
            self.assertIn(p, self.tree.get(p).schemas)

    def test_add_spin_offs(self):
        # more realistic example that interleaves result and context spin-offs
        # ---> in actual operation, result spin-offs always precede context spin-offs

        # primitive schemas
        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A2/')
        s3 = sym_schema('/A3/')

        tree = SchemaResultTree(primitives=(s1, s2, s3))

        # test: adding result spin-offs to primitives [ROUND 1]
        s1_r1 = sym_schema('/A1/101,')
        s1_r2 = sym_schema('/A1/102,')
        s2_r1 = sym_schema('/A2/101,')

        n_schemas_before_add = tree.n_schemas
        tree.add_result_spin_offs(s1, [s1_r1, s1_r2])
        tree.add_result_spin_offs(s2, [s2_r1])
        n_schemas_after_add = tree.n_schemas

        self.assertEqual(n_schemas_before_add + 3, n_schemas_after_add)

        # test: adding context spin-offs to first round of result spin-offs [ROUND 2]
        s1_r1_c1 = sym_schema('1,/A1/101,')
        s1_r1_c2 = sym_schema('2,/A1/101,')
        s1_r2_c1 = sym_schema('1,/A1/102,')
        s2_r1_c1 = sym_schema('1,/A2/101,')
        s2_r1_c2 = sym_schema('2,/A2/101,')

        n_schemas_before_add = tree.n_schemas
        tree.add_context_spin_offs(s1_r1, [s1_r1_c1, s1_r1_c2])
        tree.add_context_spin_offs(s1_r2, [s1_r2_c1])
        tree.add_context_spin_offs(s2_r1, [s2_r1_c1, s2_r1_c2])
        n_schemas_after_add = tree.n_schemas

        self.assertEqual(n_schemas_before_add + 5, n_schemas_after_add)

        # test: adding context spin-offs to 2nd round of context spin-offs [ROUND 3]
        s1_r1_c1_2 = sym_schema('1,2/A1/101,')
        s1_r2_c1_3 = sym_schema('1,3/A1/102,')
        s2_r1_c1_2 = sym_schema('1,2/A2/101,')

        n_schemas_before_add = tree.n_schemas
        tree.add_context_spin_offs(s1_r1_c1, [s1_r1_c1_2, s1_r2_c1_3])
        tree.add_context_spin_offs(s2_r1_c1, [s2_r1_c1_2])
        n_schemas_after_add = tree.n_schemas

        self.assertEqual(n_schemas_before_add + 3, n_schemas_after_add)

    def test_is_valid_node_1(self):
        # all nodes in this tree should be valid
        for schema in self.tree:
            self.assertTrue(self.tree.is_valid_node(schema))

    def test_is_valid_node_2(self):
        # root node should be valid
        self.assertTrue(self.tree.is_valid_node(self.tree.root))

    def test_is_valid_node_3(self):
        # unattached schema should be invalid
        s_unattached = sym_schema_result_tree_node('/A4/7,')
        self.assertFalse(self.tree.is_valid_node(s_unattached))

    def test_is_valid_node_4(self):
        # the children of root should only be primitive schemas
        s_illegal_parent = sym_schema('/A3/1,')
        self.tree.add_primitives((s_illegal_parent,))
        s_illegal_parent_node = self.tree.get(s_illegal_parent)
        self.assertFalse(self.tree.is_valid_node(s_illegal_parent_node))

    def test_is_valid_node_5(self):
        # non-primitive nodes should have the same action as their parents
        s_illegal_action = sym_schema('/A3/1,')
        self.tree.add_result_spin_offs(self.s1, (s_illegal_action,))
        s_illegal_action_node = self.tree.get(s_illegal_action)
        self.assertFalse(self.tree.is_valid_node(s_illegal_action_node))

    def test_is_valid_node_6(self):
        s1 = sym_schema('/A1/')
        s1_1 = sym_schema('/A1/1,')
        s1_1_1 = sym_schema('/A1/1,2')

        tree = SchemaResultTree()
        tree.add_primitives((s1,))
        tree.add_result_spin_offs(s1, (s1_1,))
        tree.add_result_spin_offs(s1_1, (s1_1_1,))

        for n in tree:
            self.assertTrue(tree.is_valid_node(n))

        # children should have all of the context item assertion of their parents
        s_illegal_context = sym_schema('/A1/1,5,6')
        tree.add_result_spin_offs(s1_1_1, (s_illegal_context,))

        s_illegal_context_node = tree.get(s_illegal_context)
        self.assertFalse(tree.is_valid_node(s_illegal_context_node))

    def test_validate(self):
        # all nodes in tree should be valid
        invalid_nodes = self.tree.validate()
        self.assertEqual(0, len(invalid_nodes))

        # adding invalid nodes
        invalid_1 = sym_schema('/A3/1,')
        invalid_2 = sym_schema('/A1/1,2,3,4,5')

        # invalid due to parent and child having different actions
        self.tree.add_result_spin_offs(self.s2, (invalid_1,))

        # invalid due to parent and child having inconsistent contexts
        self.tree.add_result_spin_offs(self.s1_1_2_1_1, (invalid_2,))

        invalid_nodes = self.tree.validate()
        self.assertEqual(2, len(invalid_nodes))
        self.assertTrue(all({self.tree.get(s) in invalid_nodes for s in [invalid_1, invalid_2]}))

    def test_contains(self):
        tree = SchemaResultTree()

        p_1 = sym_schema('/A1/')
        p_2 = sym_schema('/A2/')

        p_schemas = [p_1, p_2]
        tree.add_primitives(p_schemas)
        self.assertTrue(all({s in tree for s in p_schemas}))

        # any other schema SHOULD NOT be contained in tree
        another = sym_schema('/A1/9,')
        self.assertNotIn(another, tree)

        # after adding, another schema should be contained in tree
        tree.add_result_spin_offs(p_1, spin_offs=(another,))
        self.assertIn(another, tree)

        # contains should work for SchemaResultTreeNodes as well
        node_another = sym_schema_result_tree_node('/A1/9,')
        self.assertIn(node_another, tree)

        node_missing = sym_schema_result_tree_node('/A2/1,')
        self.assertNotIn(node_missing, tree)

    @test_share.string_test
    def test_str(self):
        print(self.tree)

    def test_find_all_satisfied(self):
        # empty tree should return empty set
        nodes = self.empty_tree.find_all_satisfied(state=sym_state('1'))
        self.assertEqual(0, len(nodes))

        # primitive schemas should be applicable to all states
        nodes = self.tree.find_all_satisfied(state=sym_state(''))
        self.assertEqual(len(self.tree.root.children), len(nodes))

        # case 1:
        nodes = self.tree.find_all_satisfied(state=sym_state('1,3,5,7'))

        expected_nodes = {
            self.tree.get(schema) for schema in [
                self.s1, self.s2, self.s1_1, self.s1_3, self.s2_1, self.s1_1_2, self.s1_1_4, self.s2_1_1, self.s1_1_2_1,
                self.s1_1_2_3, self.s1_1_2_1_1, self.s1_1_2_3_1,
            ]
        }

        self.assertEqual(len(expected_nodes), len(nodes))
        self.assertTrue(all({n in nodes for n in expected_nodes}))

        # case 2: only primitives
        nodes = self.tree.find_all_satisfied(state=sym_state('7,9'))
        expected_nodes = {
            self.tree.get(schema) for schema in [self.s1, self.s2]}

        self.assertEqual(len(expected_nodes), len(nodes))
        self.assertTrue(all({n in nodes for n in expected_nodes}))

        # case 3: entire tree applicable
        nodes = self.tree.find_all_satisfied(state=sym_state('1,2,3,4,5,6,7,8,10,11,12,13'))

        self.assertEqual(len(self.tree), len(nodes))
        self.assertTrue(all({n in nodes for n in self.tree}))

    @test_share.performance_test
    def test_performance_1(self):
        n_iters = 1_000
        n_actions = 10

        tree = SchemaResultTree()
        tree.add_primitives(primitive_schemas(actions(n_actions)))

        elapsed_time = 0
        schema = sample(list(chain.from_iterable([c.schemas for c in tree.root.children])), k=1)[0]
        for state_element in range(n_iters):
            spinoff = create_context_spin_off(schema, sym_item_assert(str(state_element)))

            start = time()
            tree.add_context_spin_offs(schema, (spinoff,))
            end = time()
            elapsed_time += end - start

            schema = sample(list(chain.from_iterable([d.schemas for d in tree.root.descendants])), k=1)[0]

        print(f'Time calling SchemaResultTree.add_context_spinoffs {n_iters:,} times {elapsed_time}s')

    @test_share.performance_test
    def test_performance_2(self):
        n_iters = 1_000

        primitives = primitive_schemas(actions(100))
        tree = SchemaResultTree()
        tree.add_primitives(primitives)

        elapsed_time = 0
        schema = sample(primitives, k=1)[0]
        for state_element in range(n_iters):
            spinoff = create_result_spin_off(schema, sym_item_assert(str(state_element)))

            start = time()
            tree.add_result_spin_offs(schema, (spinoff,))
            end = time()
            elapsed_time += end - start

            schema = sample(primitives, k=1)[0]

        print(f'Time calling SchemaResultTree.add_result_spinoffs {n_iters:,} times {elapsed_time}s')


class TestSchemaResultTreeNode(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.blank_stn = SchemaResultTreeNode()

        self.stn_1 = sym_schema_result_tree_node('/A1/1,2')
        self.stn_2 = sym_schema_result_tree_node('/A1/2,3')

    def test_init(self):
        stn = SchemaResultTreeNode()
        self.assertIs(None, stn.result)
        self.assertIs(None, stn.action)

        # action only node
        stn = SchemaResultTreeNode(action=Action('A1'))
        self.assertIsInstance(stn, SchemaResultTreeNode)
        self.assertIs(None, stn.result)
        self.assertEqual(Action('A1'), stn.action)

        # result and action node
        stn = SchemaResultTreeNode(result=sym_state_assert('1,2,~3'), action=Action('A1'))
        self.assertIsInstance(stn, SchemaResultTreeNode)
        self.assertEqual(sym_state_assert('1,2,~3'), stn.result)
        self.assertEqual(Action('A1'), stn.action)

    def test_equal(self):
        copy = sym_schema_result_tree_node('/A1/1,2')

        self.assertEqual(self.stn_1, self.stn_1)
        self.assertEqual(self.stn_1, copy)
        self.assertNotEqual(self.stn_1, self.stn_2)

        self.assertTrue(is_eq_reflexive(self.stn_1))
        self.assertTrue(is_eq_symmetric(x=self.stn_1, y=copy))
        self.assertTrue(is_eq_transitive(x=self.stn_1, y=copy, z=sym_schema_result_tree_node('/A1/1,2')))
        self.assertTrue(is_eq_consistent(x=self.stn_1, y=copy))
        self.assertTrue(is_eq_with_null_is_false(self.stn_1))

    def test_hash(self):
        copy = sym_schema_result_tree_node('/A1/1,2')

        self.assertIsInstance(hash(self.stn_1), int)
        self.assertTrue(is_hash_consistent(self.stn_1))
        self.assertTrue(is_hash_same_for_equal_objects(x=self.stn_1, y=copy))
