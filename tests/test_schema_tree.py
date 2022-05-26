import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import test_share
from schema_mechanism.core import Action
from schema_mechanism.core import SchemaTree
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_schema_tree_node
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.serialization.json import deserialize
from schema_mechanism.serialization.json import serialize
from schema_mechanism.util import repr_str
from test_share.test_func import common_test_setup
from test_share.test_func import file_was_written
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestSchemaTree(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        # root
        # |-- 1
        # |   |-- 1,2
        # |   |-- 1,3
        # |   |   |-- 1,3,5
        # |   |   |   +-- 1,3,5,7
        # |   |   |-- 1,3,6
        # |   |   +-- 1,3,7
        # |   |       +-- 1,3,7,9
        # |   |           |-- 1,11,3,7,9
        # |   |           +-- 1,13,3,7,9
        # |   +-- 1,4
        # |-- 2
        # |   +-- 2,7
        # |       +-- 2,7,9
        # |-- 3
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

        # NODE (1,3,7,9):
        self.p1_r1_c1_c2_c3_c1 = sym_schema('1,3,7,9/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2_c3, [self.p1_r1_c1_c2_c3_c1])

        # NODE (1,3,7,9,11):
        self.p1_r1_c1_c2_c3_c1_c1 = sym_schema('1,3,7,9,11/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2_c3_c1, [self.p1_r1_c1_c2_c3_c1_c1])

        # NODE (1,3,7,9,13):
        self.p1_r1_c1_c2_c3_c1_c2 = sym_schema('1,3,7,9,13/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2_c3_c1, [self.p1_r1_c1_c2_c3_c1_c2])

        # NODE (2):
        self.p3_r1_c1 = sym_schema('2,/A3/X,')
        self.tree.add_context_spin_offs(self.p3_r1, [self.p3_r1_c1])

        # NODE (2,7):
        self.p3_r1_c1_c1 = sym_schema('2,7/A3/X,')
        self.tree.add_context_spin_offs(self.p3_r1_c1, [self.p3_r1_c1_c1])

        # NODE (2,7,9):
        self.p3_r1_c1_c1_c1 = sym_schema('2,7,9/A3/X,')
        self.tree.add_context_spin_offs(self.p3_r1_c1_c1, [self.p3_r1_c1_c1_c1])

        # NODE (3):
        self.p1_r1_c2 = sym_schema('3,/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1, [self.p1_r1_c2])

        # NODE (3,1): [REDUNDANT WITH 1,3... Shouldn't result in a new schema or node in the self.tree]
        self.p1_r1_c2_c1 = sym_schema('3,1/A1/X,')
        self.tree.add_context_spin_offs(self.p1_r1_c2, [self.p1_r1_c2_c1])

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

        tree = SchemaTree(schemas=[p1, p2])

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
        self.assertRaises(ValueError, lambda: tree.add_result_spin_offs(source=p2, spin_offs=spin_offs))

        # test: ValueError should be raised if spin-off contains a composite result without a corresponding tree node
        self.assertRaises(ValueError, lambda: tree.add_result_spin_offs(source=p1, spin_offs=[sym_schema('/A1/1,2,3')]))

    def test_add_context_spin_offs(self):
        p1 = sym_schema('/A1/')
        p2 = sym_schema('/A2/')

        tree = SchemaTree(schemas=[p1, p2])

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

        # test: ValueError should be raised if source node does not exist in tree
        self.assertRaises(ValueError, lambda: tree.add_context_spin_offs(
            source=sym_schema('X,Y/A2/B,C'), spin_offs=[sym_schema('X,Y,Z/A2/B,C')]))

    def test_add_spin_offs(self):
        # more realistic example that interleaves result and context spin-offs
        # ---> in actual operation, result spin-offs always precede context spin-offs

        # primitive schemas
        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A2/')
        s3 = sym_schema('/A3/')

        tree = SchemaTree(schemas=(s1, s2, s3))

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
            self.assertRaises(KeyError, lambda: tree.get(schema.context.flatten()))

        n_schemas_before_add = tree.n_schemas
        tree.add_context_spin_offs(s1_r1, [s1_r1_c1, s1_r1_c2])
        tree.add_context_spin_offs(s1_r2, [s1_r2_c1])
        tree.add_context_spin_offs(s2_r1, [s2_r1_c1, s2_r1_c2])
        n_schemas_after_add = tree.n_schemas

        self.assertEqual(n_schemas_before_add + 5, n_schemas_after_add)

        # test: tree nodes SHOULD exist for context spin-offs after add
        try:
            for schema in [s1_r1_c1, s1_r1_c2, s1_r2_c1, s2_r1_c1, s2_r1_c2]:
                self.assertIsNotNone(tree.get(schema.context.flatten()))
        except KeyError as e:
            self.fail(f'Unexpected exception: {str(e)}')

        # test: context spin-offs should be added to new nodes (not their sources nodes)
        for schema in [s1_r1_c1, s1_r1_c2, s1_r2_c1, s2_r1_c1, s2_r1_c2]:
            self.assertEqual(schema.context, tree.get(schema.context.flatten()).context)

        # test: adding context spin-offs to 2nd round of context spin-offs [ROUND 3]
        s1_r1_c1_2 = sym_schema('1,2/A1/101,')
        s1_r2_c1_3 = sym_schema('1,3/A1/102,')
        s2_r1_c1_2 = sym_schema('1,2/A2/101,')

        # test: tree nodes SHOULD NOT exist for context spin-offs before add
        for schema in [s1_r1_c1_2, s1_r2_c1_3, s2_r1_c1_2]:
            self.assertRaises(KeyError, lambda: tree.get(schema.context.flatten()))

        n_schemas_before_add = tree.n_schemas
        tree.add_context_spin_offs(s1_r1_c1, [s1_r1_c1_2, s1_r2_c1_3])
        tree.add_context_spin_offs(s2_r1_c1, [s2_r1_c1_2])
        n_schemas_after_add = tree.n_schemas

        self.assertEqual(n_schemas_before_add + 3, n_schemas_after_add)

        # test: context spin-offs should be added to new nodes (not their sources nodes)
        for schema in [s1_r1_c1_2, s1_r2_c1_3, s2_r1_c1_2]:
            self.assertEqual(schema.context, tree.get(schema.context.flatten()).context)

    # noinspection PyTypeChecker
    def test_add_bare_schemas(self):
        p1 = sym_schema('/A1/')
        p2 = sym_schema('/A2/')

        tree = SchemaTree(schemas=[p1, p2])

        # test: primitive actions
        new_primitives = [sym_schema(f'/A{i}/') for i in range(5, 10)]

        for s in new_primitives:
            self.assertNotIn(s, tree)

        tree.add_bare_schemas(new_primitives)

        for s in new_primitives:
            self.assertIn(s, tree)

        # test: composite actions
        new_primitives = [
            sym_schema(f'/1,2/'),
            sym_schema(f'/1,3/'),
            sym_schema(f'/2,3/'),
            sym_schema(f'/3,4/'),
        ]
        tree.add_bare_schemas(new_primitives)

        for s in new_primitives:
            self.assertIn(s, tree)

        # test: a ValueError should be raised if argument is None or empty collection
        self.assertRaises(ValueError, lambda: tree.add_bare_schemas(schemas=[]))
        self.assertRaises(ValueError, lambda: tree.add_bare_schemas(schemas=None))

    def test_is_valid_node_check_default_test_tree_valid(self):
        # sanity check: all nodes in the test tree should be valid
        for node in self.tree:
            self.assertTrue(self.tree.is_valid_node(node))

    def test_is_valid_node_check_for_unattached_node(self):
        # test: unattached tree node should be invalid
        s_unattached = sym_schema_tree_node('7,')
        self.assertFalse(self.tree.is_valid_node(s_unattached))

        # test: should raise ValueError if raise_on_invalid=True
        self.assertRaises(
            ValueError,
            lambda: self.tree.is_valid_node(
                node=s_unattached,
                raise_on_invalid=True)
        )

    def test_is_valid_node_check_valid_composite_results(self):
        # test: composite results must also exist as contexts in tree
        p1 = sym_schema('/A1/')
        p2_r1 = sym_schema('/A1/1,')
        s1 = sym_schema('1,/A1/1,')
        s2 = sym_schema('1,2/A1/1,')
        s3 = sym_schema('/A1/(1,2),')

        tree = SchemaTree(schemas=(p1,))
        tree.add_result_spin_offs(p1, [p2_r1])

        tree.add_context_spin_offs(p2_r1, [s1])
        tree.add_context_spin_offs(s1, [s2])

        self.assertTrue(tree.is_valid_node(tree.get(s1.context.flatten())))
        self.assertTrue(tree.is_valid_node(tree.get(s2.context.flatten())))

        # test: this should be a valid composite result spin-off
        tree.add_context_spin_offs(p1, [s3])
        self.assertTrue(tree.is_valid_node(tree.get(s3.context.flatten())))

    def test_is_valid_node_check_children_have_all_parent_item_assertions(self):
        # test: children should have all of the context item assertion of their parents
        p1 = sym_schema('/A1/')
        p1_r1 = sym_schema('/A1/1,')
        s1 = sym_schema('3,/A1/1,')

        tree = SchemaTree(schemas=(p1,))
        tree.add_result_spin_offs(p1, [p1_r1])
        tree.add_context_spin_offs(p1_r1, [s1])

        # adding this schema should result in an invalid node
        s_illegal_context = sym_schema('1,2/A1/1,')
        tree.add_context_spin_offs(s1, [s_illegal_context])

        self.assertFalse(tree.is_valid_node(tree.get(s_illegal_context.context.flatten())))

        # test: should raise ValueError if raise_on_invalid=True
        self.assertRaises(
            ValueError,
            lambda: tree.is_valid_node(
                node=tree.get(s_illegal_context.context.flatten()),
                raise_on_invalid=True)
        )

    def test_is_valid_node_check_correct_depth(self):
        # test: nodes should have a depth equal to the number of item assertions in their context
        p1 = sym_schema('/A1/')
        p1_r1 = sym_schema('/A1/1,')
        s1 = sym_schema('1,/A1/1,')
        s2 = sym_schema('1,2/A1/1,')

        # should result in invalid node due to depth
        s3 = sym_schema('1,2,3,4/A1/1,')

        tree = SchemaTree(schemas=(p1,))
        tree.add_result_spin_offs(p1, [p1_r1])
        tree.add_context_spin_offs(p1_r1, [s1])
        tree.add_context_spin_offs(s1, [s2])

        # adding this schema should result in an invalid node
        tree.add_context_spin_offs(s2, [s3])

        self.assertFalse(tree.is_valid_node(tree.get(s3.context.flatten())))

        # test: should raise ValueError if raise_on_invalid=True
        self.assertRaises(
            ValueError,
            lambda: tree.is_valid_node(
                node=tree.get(s3.context.flatten()),
                raise_on_invalid=True)
        )

    def test_is_valid_node_check_exactly_one_additional_item_assertion_in_child_node(self):
        # test: nodes' contexts should contain exactly one item assertion not in parent's context
        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A1/1,')
        s3 = sym_schema('2,/A1/1,')
        s4 = sym_schema('2,3,4/A1/1,')

        tree = SchemaTree(schemas=[s1])
        tree.add_result_spin_offs(source=s1, spin_offs=[s2])
        tree.add_context_spin_offs(source=s2, spin_offs=[s3])

        # this add should have generated an invalid node
        tree.add_context_spin_offs(source=s3, spin_offs=[s4])

        self.assertFalse(tree.is_valid_node(tree.get(s4.context.flatten())))

        # test: should raise ValueError if raise_on_invalid=True
        self.assertRaises(
            ValueError,
            lambda: tree.is_valid_node(
                node=tree.get(s4.context.flatten()),
                raise_on_invalid=True)
        )

    def test_is_valid_node_check_item_assertions_identical_across_all_schemas_in_node(self):
        # test: contexts should be same across all schemas in node's schemas_satisfied_by and equal to node's context

        # scenario 1 - root node invalid
        s1 = sym_schema('/A1/')
        tree = SchemaTree(schemas=[s1])

        node = tree.get(s1.context.flatten())
        node.schemas_satisfied_by = [
            sym_schema('1,/A1/2,')
        ]

        self.assertFalse(tree.is_valid_node(node))

        # test: should raise ValueError if raise_on_invalid=True
        self.assertRaises(
            ValueError,
            lambda: tree.is_valid_node(
                node=node,
                raise_on_invalid=True)
        )

        # scenario 2 - non-root node invalid
        s1 = sym_schema('/A1/')

        tree = SchemaTree(schemas=[s1])

        s2 = sym_schema('/A1/1,')
        tree.add_result_spin_offs(source=s1, spin_offs=[s2])

        s3 = sym_schema('2,/A1/1,')
        tree.add_context_spin_offs(source=s2, spin_offs=[s3])

        node = tree.get(s3.context)

        # adding invalid node
        s4 = sym_schema('3,/A1/1,')
        node.schemas_satisfied_by = [s4]

        self.assertFalse(tree.is_valid_node(node))

        # test: should raise ValueError if raise_on_invalid=True
        self.assertRaises(
            ValueError,
            lambda: tree.is_valid_node(
                node=node,
                raise_on_invalid=True)
        )

        # scenario 3 - schemas have same context assertions, but are different from node's context
        s1 = sym_schema('/A1/')

        tree = SchemaTree(schemas=[s1])

        s2 = sym_schema('/A1/1,')
        tree.add_result_spin_offs(source=s1, spin_offs=[s2])

        s3 = sym_schema('/A1/2,')
        tree.add_result_spin_offs(source=s1, spin_offs=[s3])

        s4 = sym_schema('2,/A1/1,')
        s5 = sym_schema('2,/A1/1,2')

        node = tree.get(s3.context)
        node.schemas_satisfied_by.update({s4, s5})

        self.assertFalse(tree.is_valid_node(node))

        # test: should raise ValueError if raise_on_invalid=True
        self.assertRaises(
            ValueError,
            lambda: tree.is_valid_node(
                node=node,
                raise_on_invalid=True)
        )

    def test_is_valid_composite_results_must_exist_as_contexts_in_tree(self):
        s1 = sym_schema('/A1/')
        tree = SchemaTree(schemas=[s1])

        node = tree.get(s1.context)
        node.schemas_satisfied_by.add(sym_schema('/A1/1,2'))

        self.assertFalse(tree.is_valid_node(node))

        # test: should raise ValueError if raise_on_invalid=True
        self.assertRaises(
            ValueError,
            lambda: tree.is_valid_node(
                node=node,
                raise_on_invalid=True)
        )

    def test_is_valid_node_check_schemas_would_satisfy_have_satisfying_results(self):
        s1 = sym_schema('/A1/')
        tree = SchemaTree(schemas=[s1])

        s2 = sym_schema('/A1/1,')
        tree.add_result_spin_offs(source=s1, spin_offs=[s2])

        s3 = sym_schema('2,/A1/1,')
        tree.add_context_spin_offs(source=s2, spin_offs=[s3])

        node = tree.get(s3.context)

        # adds an invalid schema (does not satisfy the node's context)
        node.schemas_would_satisfy.add(sym_schema('/A1/1,'))

        self.assertFalse(tree.is_valid_node(node))

        # test: should raise ValueError if raise_on_invalid=True
        self.assertRaises(
            ValueError,
            lambda: tree.is_valid_node(
                node=node,
                raise_on_invalid=True)
        )

    # 6. results should be identical across all contained schemas, and equal to node's context
    #             if not node_context_asserts.issubset(schema_result_asserts):
    #                 if raise_on_invalid:
    #                     raise ValueError('invalid node: schemas in schemas_would_satisfy must have result = node context')
    #                 return False
    # SchemaTree.is_valid_node
    # 	4. test case where composite results do not exist as contexts in the tree
    # 	5. test case where results are not identical across all contained schemas, and equal to node's context

    def test_validate(self):
        # all nodes in tree should be valid
        invalid_nodes = self.tree.validate()
        self.assertEqual(0, len(invalid_nodes))

        # adding invalid nodes
        invalid_1 = sym_schema('1,2,5/A1/X,')
        invalid_2 = sym_schema('1,6,9/A3/X,')

        # invalid due to parent and child having inconsistent contexts
        self.tree.add_context_spin_offs(self.p1_r1_c1_c2, (invalid_1,))
        self.tree.add_context_spin_offs(self.p3_r1_c1_c1_c1, (invalid_2,))

        invalid_nodes = self.tree.validate()
        self.assertEqual(2, len(invalid_nodes))
        self.assertTrue(all({self.tree.get(s.context.flatten()) in invalid_nodes for s in [invalid_1, invalid_2]}))

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

        # test: False should be returned for non-schema, non-node arguments
        self.assertNotIn(None, self.tree)
        self.assertNotIn(Action(), self.tree)

    def test_height(self):
        # test: height of tree with only bare nodes should be zero
        schemas = [sym_schema('/A1/'), sym_schema('/A2/')]
        tree = SchemaTree(schemas=schemas)
        self.assertEqual(0, tree.height)

        # test: height of tree with only bare nodes and single-item-assertion result nodes should be 1
        tree.add_result_spin_offs(source=sym_schema('/A1/'), spin_offs=[sym_schema('/A1/1,')])
        tree.add_result_spin_offs(source=sym_schema('/A2/'), spin_offs=[sym_schema('/A2/2,')])

        self.assertEqual(1, tree.height)

        # test: height of tree should remain 1 after adding single-item-assertion context nodes
        tree.add_context_spin_offs(source=sym_schema('/A1/1,'), spin_offs=[sym_schema('3,/A1/1,')])
        tree.add_context_spin_offs(source=sym_schema('/A2/2,'), spin_offs=[sym_schema('4,/A2/2,')])

        self.assertEqual(1, tree.height)

        # test: height of tree should increase to 2 after adding two-item assertion in context nodes
        tree.add_context_spin_offs(source=sym_schema('3,/A1/1,'), spin_offs=[sym_schema('1,3,/A1/1,')])
        tree.add_context_spin_offs(source=sym_schema('4,/A2/2,'), spin_offs=[sym_schema('1,4,/A2/2,')])

        self.assertEqual(2, tree.height)

        # test: larger example using tree created in setUp (height should be 5)
        self.assertEqual(5, self.tree.height)

    def test_str(self):
        tree_str = str(self.tree)
        for node in self.tree:
            node_str = str(node)

            # test: string representations for nodes in tree SHOULD be sub-strings of tree string representation
            self.assertIn(node_str, tree_str)

        # test: string representations for nodes NOT in tree SHOULD NOT be in tree string
        node = sym_schema_tree_node('J,K,L')
        self.assertNotIn(str(node), tree_str)

    def test_eq(self):
        tree = self.tree
        other = SchemaTree(schemas=[sym_schema('/A1/'), sym_schema('/A2/')])

        self.assertTrue(satisfies_equality_checks(obj=tree, other=other, other_different_type=1.0))

    def test_find_all_satisfied(self):

        # case 1: blank state should only return root node
        nodes = self.tree.find_all_satisfied(state=sym_state(''))
        self.assertSetEqual({self.tree.root}, set(nodes))

        # case 2: state with all unknown elements should only return root node
        nodes = self.tree.find_all_satisfied(state=sym_state('UNK,NOPE'))
        self.assertSetEqual({self.tree.root}, set(nodes))

        # case 3: state with known elements that are only including in contexts with other elements should only return
        #       : root
        nodes = self.tree.find_all_satisfied(state=sym_state('9,11'))
        self.assertSetEqual({self.tree.root}, set(nodes))

        # case 4: single element that matches a child of root should return that matching node and root
        nodes = self.tree.find_all_satisfied(state=sym_state('3'))
        self.assertSetEqual({self.tree.root, sym_schema_tree_node('3,')}, set(nodes))

        # case 5: example that crosses multiple branches from root
        nodes = self.tree.find_all_satisfied(state=sym_state('1,2,3,7'))
        expected_nodes = [
            self.tree.root,
            sym_schema_tree_node('1,'),
            sym_schema_tree_node('1,2,'),
            sym_schema_tree_node('1,3'),
            sym_schema_tree_node('1,3,7'),
            sym_schema_tree_node('2,'),
            sym_schema_tree_node('2,7'),
            sym_schema_tree_node('3,'),
        ]
        self.assertSetEqual(set(expected_nodes), set(nodes))

    def test_find_all_would_satisfy(self):
        # case 1: state that satisfies an entire branch starting with a child of root
        nodes = self.tree.find_all_would_satisfy(assertion=sym_state_assert('1'))

        matched_ancestor_node = self.tree.get(sym_state_assert('1'))
        self.assertSetEqual({matched_ancestor_node, *matched_ancestor_node.descendants}, set(nodes))

        # case 2:
        nodes = self.tree.find_all_would_satisfy(assertion=sym_state_assert('8,9'))
        expected_nodes = []

        self.assertSetEqual(set(expected_nodes), set(nodes))

        # case 3:
        nodes = self.tree.find_all_would_satisfy(assertion=sym_state_assert('3,6'))
        expected_nodes = [
            sym_schema_tree_node('1,3,6'),
        ]

        self.assertSetEqual(set(expected_nodes), set(nodes))

    @test_share.disable_test
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


class TestSchemaTreeNode(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.node = sym_schema_tree_node('1,2')

    def test_init(self):
        context = sym_state_assert('1,2,3')
        label = 'custom label'

        node = SchemaTreeNode(
            context=context,
            label=label)

        # test: attribute values should be properly set by initializer
        self.assertEqual(context, node.context)
        self.assertEqual(label, node.label)

        # test: schemas satisfied-by and would-satisfy should be initialized to empty sets
        self.assertSetEqual(set(), node.schemas_satisfied_by)
        self.assertSetEqual(set(), node.schemas_would_satisfy)

    def test_schemas_satisfied_by(self):
        schemas_satisfied_by = {
            sym_schema('1,/A1/2,'),
            sym_schema('2,/A1/3,'),
            sym_schema('1,2/A1/3,4'),
            sym_schema('1,2/A1/5,'),
        }

        self.node.schemas_satisfied_by = schemas_satisfied_by
        self.assertSetEqual(schemas_satisfied_by, self.node.schemas_satisfied_by)

    def test_schemas_would_satisfy(self):
        schemas_would_satisfy = {
            sym_schema('1,/A1/1,2'),
            sym_schema('2,/A1/1,2,3'),
            sym_schema('3,4/A1/1,2'),
        }

        self.node.schemas_would_satisfy = schemas_would_satisfy
        self.assertSetEqual(schemas_would_satisfy, self.node.schemas_would_satisfy)

    def test_equal(self):
        node = sym_schema_tree_node('1,2,3')
        other = sym_schema_tree_node('4,5,6')

        self.assertTrue(satisfies_equality_checks(obj=node, other=other, other_different_type=1.0))

    def test_repr(self):
        node = sym_schema_tree_node('1,2,3')
        attr_values = {'context': node.context,
                       'label': node.label}
        self.assertEqual(repr_str(node, attr_values), repr(node))

    def test_hash(self):
        node = sym_schema_tree_node('1,2,3')
        self.assertTrue(satisfies_hash_checks(obj=node))
