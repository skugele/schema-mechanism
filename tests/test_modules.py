from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from schema_mechanism.core import Action
from schema_mechanism.core import ItemPool
from schema_mechanism.core import NULL_STATE_ASSERT
from schema_mechanism.core import Schema
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaSpinOffType
from schema_mechanism.core import SchemaTree
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.modules import create_spin_off
from schema_mechanism.modules import load
from schema_mechanism.modules import save
from test_share.test_func import common_test_setup


class TestModuleFunctions(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_spinoff_schema(self):
        # test bare schema spin-off
        ###########################

        s1 = sym_schema('/action/')

        # result spin-off from item assertion
        s2 = create_spin_off(schema=s1, spin_off_type=SchemaSpinOffType.RESULT, item=sym_item('1'))

        self.assertEqual(s1.action, s2.action)
        self.assertEqual(1, len(s2.result))
        self.assertTrue(sym_item('1') in s2.result)
        self.assertIs(NULL_STATE_ASSERT, s2.context)

        # result spin-off from item assertion with composite item
        s3 = create_spin_off(schema=s1, spin_off_type=SchemaSpinOffType.RESULT, item=sym_item('(1,2)'))

        self.assertEqual(s1.action, s3.action)
        self.assertEqual(1, len(s3.result))
        self.assertTrue(sym_item('(1,2)') in s3.result)
        self.assertIs(NULL_STATE_ASSERT, s3.context)

        s_non_prim_1 = sym_schema('/action/11,12')
        s_non_prim_2 = sym_schema('10,/action/(1,2),')

        # test: result spin-offs MUST originate from primitive schemas
        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_1,
                                                              spin_off_type=SchemaSpinOffType.RESULT,
                                                              item=sym_item('3')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_2,
                                                              spin_off_type=SchemaSpinOffType.RESULT,
                                                              item=sym_item('3')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_1,
                                                              spin_off_type=SchemaSpinOffType.RESULT,
                                                              item=sym_item('(3,4)')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_2,
                                                              spin_off_type=SchemaSpinOffType.RESULT,
                                                              item=sym_item('(3,4)')))

    def test_spin_off_for_composite_action_schemas(self):
        # test bare schema spin-off
        ###########################

        s1 = sym_schema('/A,B,C,1,2,3/')

        # result spin-off from item assertion
        s2 = create_spin_off(schema=s1, spin_off_type=SchemaSpinOffType.RESULT, item=sym_item('1'))

        self.assertEqual(s1.action, s2.action)
        self.assertEqual(1, len(s2.result))
        self.assertTrue(sym_item('1') in s2.result)
        self.assertIs(NULL_STATE_ASSERT, s2.context)

        # result spin-off from item assertion with composite item
        s3 = create_spin_off(schema=s1, spin_off_type=SchemaSpinOffType.RESULT, item=sym_item('(1,2)'))

        self.assertEqual(s1.action, s3.action)
        self.assertEqual(1, len(s3.result))
        self.assertTrue(sym_item('(1,2)') in s3.result)
        self.assertIs(NULL_STATE_ASSERT, s3.context)

        s_non_prim_1 = sym_schema('/action/11,12')
        s_non_prim_2 = sym_schema('10,/action/(1,2),')

        # test: result spin-offs MUST originate from primitive schemas (unless ER_INCREMENTAL_RESULTS enabled)
        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_1,
                                                              spin_off_type=SchemaSpinOffType.RESULT,
                                                              item=sym_item('3')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_2,
                                                              spin_off_type=SchemaSpinOffType.RESULT,
                                                              item=sym_item('3')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_1,
                                                              spin_off_type=SchemaSpinOffType.RESULT,
                                                              item=sym_item('(3,4)')))

        self.assertRaises(ValueError, lambda: create_spin_off(schema=s_non_prim_2,
                                                              spin_off_type=SchemaSpinOffType.RESULT,
                                                              item=sym_item('(3,4)')))

    def test_save_and_load(self):
        schema_collection = SchemaTree()

        bare_schemas = [Schema(Action(f'A{i}')) for i in range(10)]
        schema_collection.add(schemas=bare_schemas)
        schema_collection.add(
            schemas=[
                sym_schema(f'/A{i}/{i},') for i in range(10)
            ]
        )

        schema_memory = SchemaMemory(schema_collection=schema_collection)

        schema_selection = SchemaSelection()
        schema_mechanism = SchemaMechanism(
            schema_memory=schema_memory,
            schema_selection=schema_selection
        )

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            save(
                modules=[schema_mechanism],
                path=path,
            )

            # clear pools
            item_pool: ItemPool = ItemPool()
            schema_pool: SchemaPool = SchemaPool()

            n_items_in_pool_before_load = len(item_pool)
            n_schemas_in_pool_before_load = len(schema_pool)

            # clear pools to ensure that load restores pool contents
            item_pool.clear()
            schema_pool.clear()

            recovered_schema_mechanism = load(path=path)

            # test: the restored schemas mechanism should equal the previously saved schema mechanism
            self.assertEqual(schema_mechanism, recovered_schema_mechanism)

            # test: the number of items in the item pool should match number of items before save
            self.assertEqual(n_items_in_pool_before_load, len(item_pool))

            # test: the number of schemas in the schema pool should match number of schemas before save
            self.assertEqual(n_schemas_in_pool_before_load, len(schema_pool))
