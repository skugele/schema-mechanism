import os
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import test_share
from schema_mechanism.core import Action
from schema_mechanism.core import ReadOnlySchemaPool
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.serialization.json import deserialize
from schema_mechanism.serialization.json import serialize
from test_share.test_classes import MockSchema
from test_share.test_func import common_test_setup
from test_share.test_func import file_was_written


class TestSchemaPool(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.schema_pool: SchemaPool = SchemaPool()
        self.schema_pool.clear()

    def test_singleton(self):
        # test case: verify singleton (all objects refer to the same instance in memory)
        self.assertIs(self.schema_pool, SchemaPool())

    def test_init(self):
        self.assertEqual(0, len(self.schema_pool))

    def test_get(self):
        # test case: verify correct item type and primitive value
        s1_key = SchemaUniqueKey(context=sym_state_assert('1,2'), action=Action('A1'), result=sym_state_assert('4,5'))
        s1 = self.schema_pool.get(s1_key, schema_type=MockSchema, reliability=0.75)

        self.assertIsInstance(s1, MockSchema)
        self.assertEqual(s1_key.context, s1.context)
        self.assertEqual(s1_key.action, s1.action)
        self.assertEqual(s1_key.result, s1.result)
        self.assertEqual(0.75, s1.reliability)

        # test case: duplicated state elements return identical instances
        s1_again = self.schema_pool.get(s1_key)
        self.assertIs(s1, s1_again)

        # test case: duplicate state elements do not increase pool size
        self.assertEqual(1, len(self.schema_pool))

        # test case: unique items increment pool size by one per item
        s2_key = SchemaUniqueKey(context=sym_state_assert('1,3'), action=Action('A2'), result=sym_state_assert('4,6'))
        s2 = self.schema_pool.get(s2_key, schema_type=MockSchema, reliability=0.75)

        self.assertEqual(2, len(self.schema_pool))
        self.assertNotEqual(s1, s2)

    def test_schemas(self):

        # schemas created via sym_schema are added to the SchemaPool internally
        schemas = {
            sym_schema('W,/A1/1,'),
            sym_schema('X,/A2/2,'),
            sym_schema('Y,/A3/3,'),
            sym_schema('Z,/A4/4,'),
        }

        schemas_in_pool = {schema for schema in SchemaPool().schemas}

        # test: collection of schemas returned from SchemaPool() should match the set that was registered with pool
        self.assertSetEqual(schemas, schemas_in_pool)

    def test_schemas_when_empty(self):
        SchemaPool().clear()

        # test: empty collection should be returned from SchemaPool().schemas when SchemaPool() is empty
        self.assertSetEqual(set(), {schema for schema in SchemaPool().schemas})

    def test_contains(self):
        for action_id in range(100):
            key = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                  action=Action(f'A{action_id}'),
                                  result=sym_state_assert('4,5'))
            _ = self.schema_pool.get(key)

        # sanity check
        self.assertEqual(100, len(self.schema_pool))

        for action_id in range(100):
            key = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                  action=Action(f'A{action_id}'),
                                  result=sym_state_assert('4,5'))
            self.assertIn(key, self.schema_pool)

        key_not_in_pool = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                          action=Action('A100'),
                                          result=sym_state_assert('4,5'))
        self.assertNotIn(key_not_in_pool, self.schema_pool)

    @test_share.disable_test
    def test_serialize(self):
        # add a few schemas to the pool before serialization
        n_schemas = 100
        for action_id in range(n_schemas):
            key = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                  action=Action(f'A{action_id}'),
                                  result=sym_state_assert('4,5'))
            _ = self.schema_pool.get(key)

        # sanity check: pool not empty
        self.assertEqual(n_schemas, len(self.schema_pool))

        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-schema-pool-serialize.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            serialize(self.schema_pool, path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            recovered = deserialize(path)

            self.assertEqual(n_schemas, len(recovered))
            self.assertListEqual(list(self.schema_pool.schemas), list(recovered.schemas))

    def test_iterator(self):
        n_schemas = 100
        pool = SchemaPool()

        # populate schemas in pool
        for action_id in range(n_schemas):
            key = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                  action=Action(f'A{action_id}'),
                                  result=sym_state_assert('4,5'))
            _ = pool.get(key)

        encountered = defaultdict(lambda: 0)
        for schema in pool:
            encountered[schema] += 1

        self.assertEqual(n_schemas, sum(encountered[k] for k in encountered.keys()))
        self.assertTrue(all({encountered[k] == 1 for k in encountered.keys()}))


class TestReadOnlySchemaPool(TestCase):
    def setUp(self) -> None:
        SchemaPool().clear()

    def test(self):
        n_schemas = 100
        pool = SchemaPool()

        # populate schemas in pool
        for action_id in range(n_schemas):
            key = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                  action=Action(f'A{action_id}'),
                                  result=sym_state_assert('4,5'))
            _ = pool.get(key)

        ro_pool = ReadOnlySchemaPool()

        # test that read only view shows all items
        self.assertEqual(n_schemas, len(ro_pool))

        # test that all items exist in read-only view
        for action_id in range(n_schemas):
            key = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                  action=Action(f'A{action_id}'),
                                  result=sym_state_assert('4,5'))
            schema = pool.get(key)

            self.assertIsNotNone(schema)

            self.assertEqual(key.context, schema.context)
            self.assertEqual(key.action, schema.action)
            self.assertEqual(key.result, schema.result)

        # test non-existent element returns None and DOES NOT add any new elements
        key_not_in_pool = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                          action=Action('A100'),
                                          result=sym_state_assert('4,5'))

        self.assertIsNone(ro_pool.get(key_not_in_pool))
        self.assertEqual(n_schemas, len(ro_pool))

        self.assertRaises(NotImplementedError, lambda: ro_pool.clear())
