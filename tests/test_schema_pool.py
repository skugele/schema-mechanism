from collections import defaultdict
from unittest import TestCase

from schema_mechanism.core import Action
from schema_mechanism.core import ReadOnlySchemaPool
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaUniqueKey
from schema_mechanism.func_api import sym_state_assert
from test_share.test_classes import MockSchema
from test_share.test_func import common_test_setup


class TestSchemaPool(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_singleton(self):
        # test case: verify singleton (all objects refer to the same instance in memory)
        self.assertIs(SchemaPool(), SchemaPool())

    def test_init(self):
        pool = SchemaPool()
        self.assertEqual(0, len(pool))

    def test_get(self):
        pool = SchemaPool()

        # test case: verify correct item type and primitive value
        s1_key = SchemaUniqueKey(context=sym_state_assert('1,2'), action=Action('A1'), result=sym_state_assert('4,5'))
        s1 = pool.get(s1_key, schema_type=MockSchema, reliability=0.75)

        self.assertIsInstance(s1, MockSchema)
        self.assertEqual(s1_key.context, s1.context)
        self.assertEqual(s1_key.action, s1.action)
        self.assertEqual(s1_key.result, s1.result)
        self.assertEqual(0.75, s1.reliability)

        # test case: duplicated state elements return identical instances
        s1_again = pool.get(s1_key)
        self.assertIs(s1, s1_again)

        # test case: duplicate state elements do not increase pool size
        self.assertEqual(1, len(pool))

        # test case: unique items increment pool size by one per item
        s2_key = SchemaUniqueKey(context=sym_state_assert('1,3'), action=Action('A2'), result=sym_state_assert('4,6'))
        s2 = pool.get(s2_key, schema_type=MockSchema, reliability=0.75)

        self.assertEqual(2, len(pool))
        self.assertNotEqual(s1, s2)

    def test_contains(self):
        pool = SchemaPool()

        for action_id in range(100):
            key = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                  action=Action(f'A{action_id}'),
                                  result=sym_state_assert('4,5'))
            _ = pool.get(key)

        # sanity check
        self.assertEqual(100, len(pool))

        for action_id in range(100):
            key = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                  action=Action(f'A{action_id}'),
                                  result=sym_state_assert('4,5'))
            self.assertIn(key, pool)

        key_not_in_pool = SchemaUniqueKey(context=sym_state_assert('1,2'),
                                          action=Action('A100'),
                                          result=sym_state_assert('4,5'))
        self.assertNotIn(key_not_in_pool, pool)

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
