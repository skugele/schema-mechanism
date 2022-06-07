import itertools
from collections import Counter
from typing import Any
from unittest import TestCase

from schema_mechanism.core import Action
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaTree
from schema_mechanism.core import SchemaTreeNode
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_schema_tree_node
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.serialization.json.decoders import decode
from schema_mechanism.serialization.json.encoders import encode
from schema_mechanism.util import repr_str
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestSchemaTree(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        # root
        # |-- X
        # |-- Y
        # |-- Z
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
        # +-- 3
        self.p1 = sym_schema('/A1/')
        self.p2 = sym_schema('/A2/')
        self.p3 = sym_schema('/A3/')

        self.tree = SchemaTree()
        self.tree.add([self.p1, self.p2, self.p3])

        # NODE (ROOT) schemas
        self.p1_r1 = sym_schema('/A1/X,')
        self.p2_r1 = sym_schema('/A2/Y,')
        self.p3_r1 = sym_schema('/A3/Z,')

        self.tree.add(source=self.p1, schemas=[self.p1_r1])
        self.tree.add(source=self.p2, schemas=[self.p2_r1])
        self.tree.add(source=self.p3, schemas=[self.p3_r1])

        # NODE (1) schemas
        self.p1_r1_c1 = sym_schema('1,/A1/X,')
        self.tree.add(source=self.p1_r1, schemas=[self.p1_r1_c1])

        # NODE (1,2):
        self.p1_r1_c1_c1 = sym_schema('1,2/A1/X,')
        self.tree.add(source=self.p1_r1_c1, schemas=[self.p1_r1_c1_c1])

        # NODE (1,3):
        self.p1_r1_c1_c2 = sym_schema('1,3/A1/X,')
        self.tree.add(source=self.p1_r1_c1, schemas=[self.p1_r1_c1_c2])

        # NODE (1,4):
        self.p1_r1_c1_c3 = sym_schema('1,4/A1/X,')
        self.tree.add(source=self.p1_r1_c1, schemas=[self.p1_r1_c1_c3])

        # NODE (1,3,5):
        self.p1_r1_c1_c2_c1 = sym_schema('1,3,5/A1/Y,')
        self.tree.add(source=self.p1_r1_c1_c2, schemas=[self.p1_r1_c1_c2_c1])

        # NODE (1,3,6):
        self.p1_r1_c1_c2_c2 = sym_schema('1,3,6/A1/Y,')
        self.tree.add(source=self.p1_r1_c1_c2, schemas=[self.p1_r1_c1_c2_c2])

        # NODE (1,3,7):
        self.p1_r1_c1_c2_c3 = sym_schema('1,3,7/A1/Y,')
        self.tree.add(source=self.p1_r1_c1_c2, schemas=[self.p1_r1_c1_c2_c3])

        # NODE (1,3,5,7):
        self.p1_r1_c1_c2_c1_c1 = sym_schema('1,3,5,7/A1/Z,')
        self.tree.add(source=self.p1_r1_c1_c2_c1, schemas=[self.p1_r1_c1_c2_c1_c1])

        # NODE (1,3,7,9):
        self.p1_r1_c1_c2_c3_c1 = sym_schema('1,3,7,9/A1/Z,')
        self.tree.add(source=self.p1_r1_c1_c2_c3, schemas=[self.p1_r1_c1_c2_c3_c1])

        # NODE (1,3,7,9,11):
        self.p1_r1_c1_c2_c3_c1_c1 = sym_schema('1,3,7,9,11/A1/Z,')
        self.tree.add(source=self.p1_r1_c1_c2_c3_c1, schemas=[self.p1_r1_c1_c2_c3_c1_c1])

        # NODE (1,3,7,9,13):
        self.p1_r1_c1_c2_c3_c1_c2 = sym_schema('1,3,7,9,13/A1/Z,')
        self.tree.add(source=self.p1_r1_c1_c2_c3_c1, schemas=[self.p1_r1_c1_c2_c3_c1_c2])

        # NODE (2):
        self.p3_r1_c1 = sym_schema('2,/A3/1,3')
        self.tree.add(source=self.p3_r1, schemas=[self.p3_r1_c1])

        # NODE (2,7):
        self.p3_r1_c1_c1 = sym_schema('2,7/A3/1,3')
        self.tree.add(source=self.p3_r1_c1, schemas=[self.p3_r1_c1_c1])

        # NODE (2,7,9):
        self.p3_r1_c1_c1_c1 = sym_schema('2,7,9/A3/1,3,5,7')
        self.tree.add(source=self.p3_r1_c1_c1, schemas=[self.p3_r1_c1_c1_c1])

        # NODE (3):
        self.p1_r1_c2 = sym_schema('3,/A1/1,3,7,9')
        self.tree.add(source=self.p1_r1, schemas=[self.p1_r1_c2])

        # NODE (3,1): [REDUNDANT WITH 1,3... Shouldn't result in a new schema or node in the self.tree]
        self.p1_r1_c2_c1 = sym_schema('3,1/A1/X,')
        self.tree.add(source=self.p1_r1_c2, schemas=[self.p1_r1_c2_c1])

        self.schemas = {
            self.p1,
            self.p2,
            self.p3,
            self.p1_r1,
            self.p2_r1,
            self.p3_r1,
            self.p1_r1_c1,
            self.p1_r1_c1_c1,
            self.p1_r1_c1_c2,
            self.p1_r1_c1_c3,
            self.p1_r1_c1_c2_c1,
            self.p1_r1_c1_c2_c2,
            self.p1_r1_c1_c2_c3,
            self.p1_r1_c1_c2_c1_c1,
            self.p1_r1_c1_c2_c3_c1,
            self.p1_r1_c1_c2_c3_c1_c1,
            self.p1_r1_c1_c2_c3_c1_c2,
            self.p3_r1_c1,
            self.p3_r1_c1_c1,
            self.p3_r1_c1_c1_c1,
            self.p1_r1_c2
        }

    def test_init(self):

        # test: no-argument initialization of schema tree should result in a single node (root)
        tree = SchemaTree()

        # test: tree should initially have no schemas
        self.assertEqual(0, len(tree))

        # test: tree should have a root node
        self.assertIsNotNone(tree.root)

        root_node = sym_schema_tree_node('', label='test root')
        tree = SchemaTree(root=root_node)

        # test: when root is passed to initializer it should be set properly as root
        self.assertEqual(root_node, tree.root)

        # test: root should be properly set in the nodes map
        self.assertIn(root_node, tree.nodes_map.values())

        # test: when root and nodes map is passed to initializer, both should be set properly
        tree = SchemaTree(
            root=self.tree.root,
            nodes_map=self.tree.nodes_map
        )

        self.assertEqual(self.tree.root, tree.root)
        self.assertDictEqual(self.tree.nodes_map, tree.nodes_map)
        self.assertEqual(len(self.tree), len(tree))

        # test: tree initialized with nodes map without root should raise a ValueError
        self.assertRaises(ValueError, lambda: SchemaTree(nodes_map=self.tree.nodes_map))

    def test_iter(self):
        # test: iterating over an empty tree should stop immediately
        tree = SchemaTree()
        self.assertRaises(StopIteration, lambda: next(iter(tree)))

        # test: iterator should visit all nodes in tree
        counter = Counter([schema for schema in self.tree])

        self.assertSetEqual(set(counter.keys()), set(self.schemas))
        for count in counter.values():
            self.assertEqual(1, count)

    def test_len(self):
        # test: empty tree should return 0
        tree = SchemaTree()
        self.assertEqual(0, len(tree))

        # test: tree with single schema should return 1
        tree = SchemaTree()
        tree.add(schemas=[self.p1])
        self.assertEqual(1, len(tree))

        # test: length on larger tree
        self.assertEqual(len(self.schemas), len(self.tree))

    def test_contains(self):
        self.assertTrue(all({schema in self.tree for schema in self.tree}))

        # any other schema SHOULD NOT be contained in tree
        another = sym_schema('ZZ,/A1/X,')
        self.assertNotIn(another, self.tree)

        # after adding, another schema should be contained in tree
        self.tree.add(source=self.p1, schemas=(another,))
        self.assertIn(another, self.tree)

        # test: False should be returned for non-schema arguments
        self.assertNotIn(None, self.tree)
        self.assertNotIn(Action(), self.tree)

    def test_get(self):
        # test: calling get with null assertion should return tree's root
        self.assertIs(self.tree.root, self.tree.get(NULL_STATE_ASSERT))

        # test: lookup should performed as if composite items in state assertion were flattened into non-composite items
        node = self.tree.get(sym_state_assert('1,3,7,9,13'))

        self.assertEqual(node, self.tree.get(sym_state_assert('(1,3),7,9,13')))
        self.assertEqual(node, self.tree.get(sym_state_assert('(1,3,7),9,13')))
        self.assertEqual(node, self.tree.get(sym_state_assert('(1,3,7,9),13')))
        self.assertEqual(node, self.tree.get(sym_state_assert('(1,3,7,9,13)')))
        self.assertEqual(node, self.tree.get(sym_state_assert('(1,3,7),(9,13)')))
        self.assertEqual(node, self.tree.get(sym_state_assert('(1,3),(7,9),13')))

        # test: a node should be returned for the context and result assertions for each schema in tree
        for schema in self.tree:
            context_node = self.tree.get(schema.context)
            result_node = self.tree.get(schema.result)

            # test: the node corresponding to a schema's context should contain the schema in schemas_satisfied_by
            self.assertEqual(schema.context, context_node.context)
            self.assertIn(schema, context_node.schemas_satisfied_by)

            # test: the node corresponding to a schema's result should contain the schema in schemas_would_satisfy
            self.assertEqual(schema.result, result_node.context)
            self.assertIn(schema, result_node.schemas_would_satisfy)

    def test_set(self):
        tree = SchemaTree()

        node = sym_schema_tree_node('1,2,3,4')
        node.schemas_satisfied_by.add(sym_schema('1,2,3,4/A1/X,'))
        node.schemas_satisfied_by.add(sym_schema('A,B/A2/1,2,3,4'))

        tree.set(node)

        # test: set should add node to node map
        self.assertIn(node.context, tree.nodes_map)
        self.assertIn(node, tree.nodes_map.values())

        # test: after set, get on node's context should return node
        self.assertEqual(node, tree.get(node.context))

        # test: multiple sets with the same assertion should replace the earlier node
        new_node = sym_schema_tree_node('1,2,3,4')
        new_node.schemas_satisfied_by.add(sym_schema('1,2,3,4/A1/Y,Z'))
        new_node.schemas_satisfied_by.add(sym_schema('C,D/A2/1,2,3,4'))

        tree.set(new_node)

        self.assertNotEqual(node, tree.get(sym_state_assert('1,2,3,4')))
        self.assertEqual(new_node, tree.get(sym_state_assert('1,2,3,4')))

        # test: set should performed as if composite items in state assertion were flattened into non-composite items
        another_node = sym_schema_tree_node('(1,2),(3,4)')
        another_node.schemas_satisfied_by.add(sym_schema('1,2,3,4/A1/J,K'))
        another_node.schemas_satisfied_by.add(sym_schema('J,K/A2/1,2,3,4'))

        tree.set(another_node)

        self.assertEqual(another_node, tree.get(sym_state_assert('1,2,3,4')))

    # noinspection PyTypeChecker
    def test_add_raised_exceptions(self):
        tree = SchemaTree()

        # test: sending empty schema collection or None to add should raise a ValueError
        self.assertRaises(ValueError, lambda: tree.add(schemas=None))
        self.assertRaises(ValueError, lambda: tree.add(schemas=list()))
        self.assertRaises(ValueError, lambda: tree.add(schemas=set()))

        # test: ValueError should be raised if source node does not exist in tree
        self.assertRaises(ValueError, lambda: tree.add(source=sym_schema('X,Y/A2/B,C'),
                                                       schemas=[sym_schema('X,Y,Z/A2/B,C')]))

        # test: ValueError should be raised if spin-off contains a composite result without a corresponding tree node
        self.assertRaises(ValueError, lambda: tree.add(source=sym_schema('/A1/'), schemas=[sym_schema('/A1/1,2,3')]))

    def _assert_schema_added_to_tree(self, schema: Schema, tree: SchemaTree):
        # test: schema should be contained in tree after add
        self.assertIn(schema, tree)

        # test: schema should have been added to schemas_satisfied_by for context assertion's node
        self.assertIn(schema, tree.get(schema.context).schemas_satisfied_by)

        # test: schema should have been added to schemas_would_satisfy for result assertion's node
        self.assertIn(schema, tree.get(schema.result).schemas_would_satisfy)

    def test_add_bare_schemas_individually(self):
        bare_schemas = [
            sym_schema('/A1/'),
            sym_schema('/A2/'),
            sym_schema('/A3/'),
        ]

        tree = SchemaTree()
        for count, schema in enumerate(bare_schemas, start=1):
            # sanity check: schema should not be in tree before add
            self.assertNotIn(schema, tree)

            tree.add(schemas=[schema])

            # test: schema should be in tree after add
            self.assertIn(schema, tree)

            # test: tree's schema count should have been increased
            self.assertEqual(count, len(tree))

        # test: the set of tree nodes should now equal to the set of bare nodes after add
        self.assertSetEqual(set(bare_schemas), set(tree))

        for schema in bare_schemas:
            self._assert_schema_added_to_tree(schema, tree)

    def test_add_bare_schemas_collectively(self):
        bare_schemas = [
            sym_schema('/A1/'),
            sym_schema('/A2/'),
            sym_schema('/A3/'),
        ]

        tree = SchemaTree()

        # sanity check: tree's schemas should match the empty set
        self.assertSetEqual(set(), set(tree))

        tree.add(schemas=bare_schemas)

        self.assertEqual(len(bare_schemas), len(tree))

        # test: the set of tree nodes should now equal to the set of bare nodes after add
        self.assertSetEqual(set(bare_schemas), set(tree))

        # test: bare schemas should be in root node's schemas_satisfied_by and schemas_would_satisfy
        for schema in bare_schemas:
            self._assert_schema_added_to_tree(schema, tree)

    def test_add_result_spin_offs_individually(self):
        bare_schemas = [
            sym_schema('/A1/'),
            sym_schema('/A2/'),
            sym_schema('/A3/'),
        ]

        tree = SchemaTree()
        tree.add(schemas=bare_schemas)

        result_spin_offs = [sym_schema(f'/A1/{i},') for i in range(5)]

        # adding single result spinoff at a time to tree
        for schema in result_spin_offs:
            tree.add(source=sym_schema('/A1/'), schemas=[schema])

            # test: result spin-off should be contained in tree after add
            self.assertIn(schema, tree)

            # test: result spin-off should have been added to schemas_satisfied_by for context assertion's node
            self.assertIn(schema, tree.get(schema.context).schemas_satisfied_by)

            # test: result spin-off should have been added to schemas_would_satisfy for result assertion's node
            self.assertIn(schema, tree.get(schema.result).schemas_would_satisfy)

        self.assertEqual(len(bare_schemas) + len(result_spin_offs), len(tree))

        for schema in result_spin_offs:
            self._assert_schema_added_to_tree(schema, tree)

    def test_add_result_spin_offs_collectively(self):
        bare_schemas = [
            sym_schema('/A1/'),
            sym_schema('/A2/'),
            sym_schema('/A3/'),
        ]

        tree = SchemaTree()
        tree.add(schemas=bare_schemas)

        result_spin_offs = [sym_schema(f'/A1/{i},') for i in range(5)]

        tree.add(source=sym_schema('/A1/'), schemas=result_spin_offs)
        self.assertEqual(len(bare_schemas) + len(result_spin_offs), len(tree))

        for schema in result_spin_offs:
            self._assert_schema_added_to_tree(schema, tree)

    def test_add_context_spin_offs_individually(self):
        bare_schemas = [
            sym_schema('/A1/'),
            sym_schema('/A2/'),
            sym_schema('/A3/'),
        ]

        tree = SchemaTree()
        tree.add(schemas=bare_schemas)

        # add a result spin-offs to serve as source
        tree.add(source=sym_schema('/A1/'), schemas=[sym_schema('/A1/X,')])

        # context spin-offs
        context_spin_offs = [sym_schema(f'{i},/A1/X,') for i in range(5)]

        # adding single context spinoff schema to tree
        for schema in context_spin_offs:
            tree.add(source=sym_schema('/A1/X,'), schemas=[schema])

        self.assertEqual(len(bare_schemas) + len(context_spin_offs) + 1, len(tree))

        for schema in context_spin_offs:
            self._assert_schema_added_to_tree(schema, tree)

    def test_add_context_spin_offs_collectively(self):
        bare_schemas = [
            sym_schema('/A1/'),
            sym_schema('/A2/'),
            sym_schema('/A3/'),
        ]

        tree = SchemaTree()
        tree.add(schemas=bare_schemas)

        # add a result spin-offs to serve as source
        s1 = sym_schema('/A1/X,')
        tree.add(source=sym_schema('/A1/'), schemas=[s1])

        # context spin-offs
        context_spin_offs = [sym_schema(f'{i},/A1/X,') for i in range(5)]

        # adding single context spinoff schema to tree
        tree.add(source=s1, schemas=context_spin_offs)

        self.assertEqual(len(bare_schemas) + len(context_spin_offs) + 1, len(tree))

        for schema in context_spin_offs:
            self.assertIn(schema, tree)

    def test_add_spin_offs(self):
        # more realistic example that interleaves result and context spin-offs
        # ---> in actual operation, result spin-offs always precede context spin-offs

        # primitive schemas
        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A2/')
        s3 = sym_schema('/A3/')

        tree = SchemaTree()
        tree.add(schemas=(s1, s2, s3))

        # test: adding result spin-offs to primitives [ROUND 1]
        s1_r1 = sym_schema('/A1/101,')
        s1_r2 = sym_schema('/A1/102,')
        s2_r1 = sym_schema('/A2/101,')

        tree.add(source=s1, schemas=[s1_r1, s1_r2])
        tree.add(source=s2, schemas=[s2_r1])

        # test: primitives and result spin-offs of primitives should be added to root node
        for schema in [s1, s2, s3, s1_r1, s1_r2, s2_r1]:
            self.assertIn(schema, tree.root.schemas_satisfied_by)

        # test: adding context spin-offs to first round of result spin-offs [ROUND 2]
        s1_r1_c1 = sym_schema('1,/A1/101,')
        s1_r1_c2 = sym_schema('2,/A1/101,')
        s1_r2_c1 = sym_schema('1,/A1/102,')
        s2_r1_c1 = sym_schema('1,/A2/101,')
        s2_r1_c2 = sym_schema('2,/A2/101,')

        # test: schemas SHOULD NOT exist in tree before add
        for schema in [s1_r1_c1, s1_r1_c2, s1_r2_c1, s2_r1_c1, s2_r1_c2]:
            self.assertNotIn(schema, tree)

        tree.add(source=s1_r1, schemas=[s1_r1_c1, s1_r1_c2])
        tree.add(source=s1_r2, schemas=[s1_r2_c1])
        tree.add(source=s2_r1, schemas=[s2_r1_c1, s2_r1_c2])

        # test: schemas SHOULD exist in tree after add
        for schema in [s1_r1_c1, s1_r1_c2, s1_r2_c1, s2_r1_c1, s2_r1_c2]:
            self.assertIn(schema, tree)

        # test: context spin-offs should be added to new nodes (not their sources nodes)
        for schema in [s1_r1_c1, s1_r1_c2, s1_r2_c1, s2_r1_c1, s2_r1_c2]:
            self.assertIn(schema.context, tree.nodes_map)

        # test: adding context spin-offs to 2nd round of context spin-offs [ROUND 3]
        s1_r1_c1_2 = sym_schema('1,2/A1/101,')
        s1_r2_c1_3 = sym_schema('1,3/A1/102,')
        s2_r1_c1_2 = sym_schema('1,2/A2/101,')

        # test: tree nodes SHOULD NOT exist for context spin-offs before add
        for schema in [s1_r1_c1_2, s1_r2_c1_3, s2_r1_c1_2]:
            self.assertNotIn(schema, tree)

        tree.add(source=s1_r1_c1, schemas=[s1_r1_c1_2, s1_r2_c1_3])
        tree.add(source=s2_r1_c1, schemas=[s2_r1_c1_2])

        # test: tree nodes SHOULD exist for context spin-offs after add
        for schema in [s1_r1_c1_2, s1_r2_c1_3, s2_r1_c1_2]:
            self.assertIn(schema, tree)

    # noinspection PyTypeChecker
    def test_add_bare_schemas(self):
        p1 = sym_schema('/A1/')
        p2 = sym_schema('/A2/')

        tree = SchemaTree()
        tree.add(schemas=[p1, p2])

        # test: primitive actions
        new_primitives = [sym_schema(f'/A{i}/') for i in range(5, 10)]

        for s in new_primitives:
            self.assertNotIn(s, tree)

        tree.add(new_primitives)

        for s in new_primitives:
            self.assertIn(s, tree)

        # test: composite actions
        new_primitives = [
            sym_schema(f'/1,2/'),
            sym_schema(f'/1,3/'),
            sym_schema(f'/2,3/'),
            sym_schema(f'/3,4/'),
        ]
        tree.add(new_primitives)

        for s in new_primitives:
            self.assertIn(s, tree)

        # test: a ValueError should be raised if argument is None or empty collection
        self.assertRaises(ValueError, lambda: tree.add(schemas=[]))
        self.assertRaises(ValueError, lambda: tree.add(schemas=None))

    def test_validate_on_test_tree(self):
        # sanity check: all nodes in the test tree should be valid
        try:
            self.tree.validate()
        except ValueError as e:
            self.fail(f'Validate raised unexpected ValueError: {e}')

    def test_validate_with_valid_composite_results(self):
        # test: composite results must also exist as contexts in tree
        p1 = sym_schema('/A1/')
        p2_r1 = sym_schema('/A1/1,')
        s1 = sym_schema('1,/A1/1,')
        s2 = sym_schema('1,2/A1/1,')
        s3 = sym_schema('/A1/(1,2),')

        tree = SchemaTree()
        tree.add(schemas=(p1,))
        tree.add(source=p1, schemas=[p2_r1])

        tree.add(source=p2_r1, schemas=[s1])
        tree.add(source=s1, schemas=[s2])
        tree.add(source=p1, schemas=[s3])

        try:
            tree.validate()
        except ValueError as e:
            self.fail(f'Validate raised unexpected ValueError: {e}')

    def test_validate_with_invalid_composite_results(self):
        # test: composite results must also exist as contexts in tree
        p1 = sym_schema('/A1/')
        p2_r1 = sym_schema('/A1/1,')
        s1 = sym_schema('1,/A1/1,')
        s3 = sym_schema('/A1/(1,2),')

        tree = SchemaTree()
        tree.add(schemas=(p1,))
        tree.add(source=p1, schemas=[p2_r1])

        tree.add(source=p2_r1, schemas=[s1])

        # forces the addition of an illegal composite item schema (the direct add would raise a ValueError)
        tree.root.schemas_satisfied_by.add(s3)

        # test: validate should raise ValueError
        self.assertRaises(ValueError, lambda: tree.validate())

    def test_validate_children_have_all_parent_item_assertions(self):
        # test: children should have all of the context item assertion of their parents
        p1 = sym_schema('/A1/')
        p1_r1 = sym_schema('/A1/1,')
        s1 = sym_schema('3,/A1/1,')

        tree = SchemaTree()
        tree.add(schemas=(p1,))
        tree.add(source=p1, schemas=[p1_r1])
        tree.add(source=p1_r1, schemas=[s1])

        # adding this schema should result in an invalid node
        s_illegal_context = sym_schema('1,2/A1/1,')
        tree.add(source=s1, schemas=[s_illegal_context])

        # test: validate should raise ValueError
        self.assertRaises(ValueError, lambda: tree.validate())

    def test_validate_node_check_correct_depth(self):
        # test: nodes should have a depth equal to the number of item assertions in their context
        p1 = sym_schema('/A1/')
        p1_r1 = sym_schema('/A1/1,')
        s1 = sym_schema('1,/A1/1,')
        s2 = sym_schema('1,2/A1/1,')

        # should result in invalid node due to depth
        s3 = sym_schema('1,2,3,4/A1/1,')

        tree = SchemaTree()
        tree.add(schemas=[p1])
        tree.add(source=p1, schemas=[p1_r1])
        tree.add(source=p1_r1, schemas=[s1])
        tree.add(source=s1, schemas=[s2])

        # adding this schema should result in an invalid node
        tree.add(source=s2, schemas=[s3])

        # test: validate should raise ValueError
        self.assertRaises(ValueError, lambda: tree.validate())

    def test_is_valid_node_check_exactly_one_additional_item_assertion_in_child_node(self):
        # test: nodes' contexts should contain exactly one item assertion not in parent's context
        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A1/1,')
        s3 = sym_schema('2,/A1/1,')
        s4 = sym_schema('2,3,4/A1/1,')

        tree = SchemaTree()
        tree.add(schemas=[s1])
        tree.add(source=s1, schemas=[s2])
        tree.add(source=s2, schemas=[s3])

        # this add should have generated an invalid node
        tree.add(source=s3, schemas=[s4])

        # test: validate should raise ValueError
        self.assertRaises(ValueError, lambda: tree.validate())

    def test_is_valid_node_check_item_assertions_identical_across_all_schemas_in_node(self):
        # test: contexts should be same across all schemas in node's schemas_satisfied_by and equal to node's context

        # scenario 1 - root node invalid
        s1 = sym_schema('/A1/')
        tree = SchemaTree()
        tree.add(schemas=[s1])

        node = tree.get(s1.context.flatten())
        node.schemas_satisfied_by = [
            sym_schema('1,/A1/2,')
        ]

        # test: validate should raise ValueError
        self.assertRaises(ValueError, lambda: tree.validate())

        # scenario 2 - non-root node invalid
        s1 = sym_schema('/A1/')

        tree = SchemaTree()
        tree.add(schemas=[s1])

        s2 = sym_schema('/A1/1,')
        tree.add(source=s1, schemas=[s2])

        s3 = sym_schema('2,/A1/1,')
        tree.add(source=s2, schemas=[s3])

        node = tree.get(s3.context)

        # adding invalid node
        s4 = sym_schema('3,/A1/1,')
        node.schemas_satisfied_by = [s4]

        # test: validate should raise ValueError
        self.assertRaises(ValueError, lambda: tree.validate())

        # scenario 3 - schemas have same context assertions, but are different from node's context
        s1 = sym_schema('/A1/')

        tree = SchemaTree()
        tree.add(schemas=[s1])

        s2 = sym_schema('/A1/1,')
        tree.add(source=s1, schemas=[s2])

        s3 = sym_schema('/A1/2,')
        tree.add(source=s1, schemas=[s3])

        s4 = sym_schema('2,/A1/1,')
        s5 = sym_schema('2,/A1/1,2')

        node = tree.get(s3.context)
        node.schemas_satisfied_by.update({s4, s5})

        # test: validate should raise ValueError
        self.assertRaises(ValueError, lambda: tree.validate())

    def test_is_valid_composite_results_must_exist_as_contexts_in_tree(self):
        s1 = sym_schema('/A1/')

        tree = SchemaTree()
        tree.add(schemas=[s1])

        node = tree.get(s1.context)
        node.schemas_satisfied_by.add(sym_schema('/A1/1,2'))

        # test: validate should raise ValueError
        self.assertRaises(ValueError, lambda: tree.validate())

    def test_is_valid_node_check_schemas_would_satisfy_have_satisfying_results(self):
        s1 = sym_schema('/A1/')

        tree = SchemaTree()
        tree.add(schemas=[s1])

        s2 = sym_schema('/A1/1,')
        tree.add(source=s1, schemas=[s2])

        s3 = sym_schema('2,/A1/1,')
        tree.add(source=s2, schemas=[s3])

        node = tree.get(s3.context)

        # adds an invalid schema (does not satisfy the node's context)
        node.schemas_would_satisfy.add(sym_schema('/A1/1,'))

        # test: validate should raise ValueError
        self.assertRaises(ValueError, lambda: tree.validate())

    def test_validate_with_inconsistent_contexts_between_source_and_schemas(self):
        # adding invalid nodes to test tree
        invalid_1 = sym_schema('1,2,5/A1/X,')
        invalid_2 = sym_schema('1,6,9/A3/X,')

        # invalid due to parent and child having inconsistent contexts
        self.tree.add(source=self.p1_r1_c1_c2, schemas=(invalid_1,))
        self.tree.add(source=self.p3_r1_c1_c1_c1, schemas=(invalid_2,))

        # test: validate should raise ValueError
        self.assertRaises(ValueError, lambda: self.tree.validate())

    def test_str(self):
        tree_str = str(self.tree)

        for node in self.tree.nodes_map.values():
            node_str = str(node)

            # test: string representations for nodes in tree SHOULD be sub-strings of tree string representation
            self.assertIn(node_str, tree_str)

        # test: string representations for nodes NOT in tree SHOULD NOT be in tree string
        node = sym_schema_tree_node('J,K,L')
        self.assertNotIn(str(node), tree_str)

    def test_eq(self):
        tree = self.tree

        other = SchemaTree()
        other.add(schemas=[sym_schema('/A1/'), sym_schema('/A2/')])

        self.assertTrue(satisfies_equality_checks(obj=tree, other=other, other_different_type=1.0))

    def test_find_all_satisfied(self):
        # case 1: blank state should only return root node's schemas_satisfied_by
        schemas = self.tree.find_all_satisfied(state=sym_state(''))
        self.assertSetEqual(self.tree.root.schemas_satisfied_by, set(schemas))

        # case 2: state with all unknown elements should only return root node's schemas_satisfied_by
        schemas = self.tree.find_all_satisfied(state=sym_state('UNK,NOPE'))
        self.assertSetEqual(self.tree.root.schemas_satisfied_by, set(schemas))

        # case 3: state with known elements that are only including in contexts with other elements should only return
        #       : root node's schemas_satisfied_by
        schemas = self.tree.find_all_satisfied(state=sym_state('9,11'))
        self.assertSetEqual(self.tree.root.schemas_satisfied_by, set(schemas))

        # case 4: single element that matches a child of root should return that matching node and root
        schemas = self.tree.find_all_satisfied(state=sym_state('3'))
        nodes = [
            self.tree.root,
            self.tree.get(sym_state_assert('3,'))
        ]
        expected_schemas = itertools.chain.from_iterable(node.schemas_satisfied_by for node in nodes)
        self.assertSetEqual(set(schemas), set(expected_schemas))

        # case 5: example that crosses multiple branches from root
        schemas = self.tree.find_all_satisfied(state=sym_state('1,2,3,7'))
        nodes = [
            self.tree.root,
            self.tree.get(sym_state_assert('1,')),
            self.tree.get(sym_state_assert('1,2,')),
            self.tree.get(sym_state_assert('1,3')),
            self.tree.get(sym_state_assert('1,3,7')),
            self.tree.get(sym_state_assert('2,')),
            self.tree.get(sym_state_assert('2,7')),
            self.tree.get(sym_state_assert('3,')),
        ]
        expected_schemas = itertools.chain.from_iterable(node.schemas_satisfied_by for node in nodes)
        self.assertSetEqual(set(schemas), set(expected_schemas))

    def test_find_all_would_satisfy(self):
        # case 1: state assertion that would be satisfied by an entire branch in tree starting with a child of root
        schemas = self.tree.find_all_would_satisfy(assertion=sym_state_assert('X'))
        expected_nodes = [
            self.tree.get(sym_state_assert('X')),
            *self.tree.get(sym_state_assert('X')).descendants,
        ]
        expected_schemas = itertools.chain.from_iterable(node.schemas_would_satisfy for node in expected_nodes)
        self.assertSetEqual(set(expected_schemas), set(schemas))

        # case 2: state assertion that would not be satisfied by any schemas in current tree
        schemas = self.tree.find_all_would_satisfy(assertion=sym_state_assert('X,1,'))
        self.assertSetEqual(set(), set(schemas))

        # case 3: state assertion that would be satisfied by schemas in a single leaf node
        schemas = self.tree.find_all_would_satisfy(assertion=sym_state_assert('1,3,5,7'))
        expected_nodes = [
            self.tree.get(sym_state_assert('1,3,5,7')),
        ]
        expected_schemas = itertools.chain.from_iterable(node.schemas_would_satisfy for node in expected_nodes)
        self.assertSetEqual(set(expected_schemas), set(schemas))

        # case 4: state assertion that would be satisfied by a branch starting in middle of tree
        schemas = self.tree.find_all_would_satisfy(assertion=sym_state_assert('1,3'))
        expected_nodes = [
            self.tree.get(sym_state_assert('1,3')),
            *self.tree.get(sym_state_assert('1,3')).descendants,
        ]
        expected_schemas = itertools.chain.from_iterable(node.schemas_would_satisfy for node in expected_nodes)
        self.assertSetEqual(set(expected_schemas), set(schemas))

    def test_is_novel_result(self):
        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A2/')

        tree = SchemaTree()
        tree.add(schemas=(s1, s2))

        s1_a = sym_schema('/A1/A,')
        s1_b = sym_schema('/A1/B,')
        s1_c = sym_schema('/A1/C,')

        tree.add(source=s1, schemas=[s1_a, s1_b, s1_c])

        s2_e = sym_schema('/A2/E,')

        tree.add(source=s2, schemas=[s2_e])

        s1_c_a = sym_schema('C,/A1/A,')
        s1_cd_a = sym_schema('C,D/A1/A,')

        tree.add(source=s1_a, schemas=[s1_c_a])
        tree.add(source=s1_c_a, schemas=[s1_cd_a])

        s2_cd = sym_schema('/A2/(C,D),')
        tree.add(source=s2, schemas=[s2_cd])

        known_results = {s.result for s in tree.root.schemas_satisfied_by}
        unknown_results = {sym_state_assert(f'{i},') for i in range(10)}

        # test: results already exist in SchemaMemory - should return False
        for r in known_results:
            self.assertFalse(tree.is_novel_result(r))

        # test: results do not exist in SchemaMemory - should return True
        for r in unknown_results:
            self.assertTrue(tree.is_novel_result(r))

        spin_offs = {Schema(action=Action('A2'), result=r) for r in unknown_results}
        tree.add(source=s2, schemas=spin_offs)

        # test: after adding result to SchemaMemory, all these should return False
        for r in unknown_results:
            self.assertFalse(tree.is_novel_result(r))

    def test_encode_and_decode(self):
        object_registry: dict[int, Any] = dict()

        encoded_obj = encode(self.tree, object_registry=object_registry)
        decoded_obj: SchemaTree = decode(encoded_obj, object_registry=object_registry)

        self.assertEqual(self.tree, decoded_obj)
        self.assertEqual(self.tree.root, decoded_obj.root)
        self.assertDictEqual(self.tree.nodes_map, decoded_obj.nodes_map)
        self.assertEqual(len(self.tree), len(decoded_obj))


class TestSchemaTreeNode(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.node = sym_schema_tree_node('1,2')

    def test_init(self):
        context = sym_state_assert('1,2,3')
        schemas_satisfied_by = {
            sym_schema('1,2,3/A1/X,'),
            sym_schema('1,2,3/A1/Y,'),
        }
        schemas_would_satisfy = {
            sym_schema('X,/A1/1,2,3'),
            sym_schema('Y,/A1/1,2,3'),
        }
        label = 'custom label'

        node = SchemaTreeNode(
            context=context,
            schemas_satisfied_by=schemas_satisfied_by,
            schemas_would_satisfy=schemas_would_satisfy,
            label=label)

        # test: attribute values should be properly set by initializer if explicitly provided
        self.assertEqual(context, node.context)
        self.assertSetEqual(schemas_satisfied_by, node.schemas_satisfied_by)
        self.assertSetEqual(schemas_would_satisfy, node.schemas_would_satisfy)
        self.assertEqual(label, node.label)

        # test: defaults should be set properly if values not explicitly provided to the initializer
        node = SchemaTreeNode()

        default_context = NULL_STATE_ASSERT
        default_schemas_satisfied_by = set()
        default_schemas_would_satisfy = set()
        default_label = None

        self.assertEqual(default_context, node.context)
        self.assertSetEqual(default_schemas_satisfied_by, node.schemas_satisfied_by)
        self.assertSetEqual(default_schemas_would_satisfy, node.schemas_would_satisfy)
        self.assertEqual(default_label, node.label)

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
        attr_values = {
            'context': node.context,
            'schemas_satisfied_by': node.schemas_satisfied_by,
            'schemas_would_satisfy': node.schemas_would_satisfy,
            'label': node.label
        }
        self.assertEqual(repr_str(node, attr_values), repr(node))

    def test_hash(self):
        node = sym_schema_tree_node('1,2,3')
        self.assertTrue(satisfies_hash_checks(obj=node))

    def test_encode_and_decode(self):
        parent_tree_node = SchemaTreeNode(
            context=sym_state_assert('1,'),
            schemas_satisfied_by={
                sym_schema('1,/A1/X,'),
                sym_schema('1,/A2/Y,'),
            },
            schemas_would_satisfy={
                sym_schema('A,/A1/1,'),
                sym_schema('B,/A2/1,'),
            },
            label='parent node',
        )

        schema_tree_node = SchemaTreeNode(
            context=sym_state_assert('1,2'),
            schemas_satisfied_by={
                sym_schema('1,2/A1/X,'),
                sym_schema('1,2/A2/Y,'),
            },
            schemas_would_satisfy={
                sym_schema('A,/A1/1,2'),
                sym_schema('B,/A2/1,2'),
            },
            label='tree node',
        )

        child_tree_node = SchemaTreeNode(
            context=sym_state_assert('1,2,3'),
            schemas_satisfied_by={
                sym_schema('1,2,3/A1/X,'),
                sym_schema('1,2,3/A2/Y,'),
            },
            schemas_would_satisfy={
                sym_schema('A,/A1/1,2,3'),
                sym_schema('B,/A2/1,2,3'),
            },
            label='child node',
        )

        parent_tree_node.children = (schema_tree_node,)

        schema_tree_node.parent = parent_tree_node
        schema_tree_node.children = (child_tree_node,)

        child_tree_node.parent = schema_tree_node

        object_registry: dict[int, Any] = dict()

        encoded_obj = encode(schema_tree_node, object_registry=object_registry)
        decoded_obj: SchemaTreeNode = decode(encoded_obj, object_registry=object_registry)

        # test: verify that schema tree node properties are correct
        self.assertEqual(schema_tree_node, decoded_obj)
        self.assertSetEqual(schema_tree_node.schemas_satisfied_by, decoded_obj.schemas_satisfied_by)
        self.assertSetEqual(schema_tree_node.schemas_would_satisfy, decoded_obj.schemas_would_satisfy)
        self.assertEqual(schema_tree_node.label, decoded_obj.label)

        # test: verify that NodeMixin properties are retained
        self.assertEqual(parent_tree_node, decoded_obj.parent)
        self.assertTupleEqual((child_tree_node,), decoded_obj.children)
