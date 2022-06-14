from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import schema_mechanism.serialization.json.decoders
import schema_mechanism.serialization.json.encoders
import test_share
from schema_mechanism.core import Chain
from schema_mechanism.core import ItemPool
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaTree
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.serialization import create_manifest
from schema_mechanism.serialization import create_object_registry
from schema_mechanism.strategies.evaluation import CompositeEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultExploratoryEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultGoalPursuitEvaluationStrategy
from schema_mechanism.strategies.match import AbsoluteDiffMatchStrategy
from schema_mechanism.strategies.selection import RandomizeBestSelectionStrategy
from test_share.test_func import common_test_setup


class TestSchemaMechanism(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.bare_schemas = [sym_schema(f'/A{i}/') for i in range(10)]
        self.primitive_items = {
            sym_item('W', primitive_value=-1.0),
            sym_item('X', primitive_value=-1.0),
            sym_item('Y', primitive_value=0.0),
            sym_item('Z', primitive_value=1.0)
        }
        self.primitive_actions = {schema.action for schema in self.bare_schemas}

        self.schema_tree = SchemaTree()
        self.schema_tree.add(schemas=self.bare_schemas)

        self.schema_memory = SchemaMemory(schema_collection=self.schema_tree)
        self.schema_selection = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(1.0)),
            evaluation_strategy=CompositeEvaluationStrategy(
                strategies=[
                    DefaultGoalPursuitEvaluationStrategy(),
                    DefaultExploratoryEvaluationStrategy(),
                ]
            )
        )

        self.schema_mechanism: SchemaMechanism = SchemaMechanism(
            schema_memory=self.schema_memory,
            schema_selection=self.schema_selection
        )

    def test_init(self):
        schema_mechanism = SchemaMechanism(
            schema_memory=self.schema_memory,
            schema_selection=self.schema_selection,
        )

        # test: attributes should have been set properly in initializer
        self.assertEqual(self.schema_memory, schema_mechanism.schema_memory)
        self.assertEqual(self.schema_selection, schema_mechanism.schema_selection)

    @test_share.disable_test
    def test_serialize(self):
        item_pool: ItemPool = ItemPool()
        schema_pool: SchemaPool = SchemaPool()

        item_pool.clear()
        schema_pool.clear()

        with TemporaryDirectory() as tmp_dir:
            pass

    def test_save_and_load(self):
        # root
        # |-- 1
        # |-- 2
        # |-- 3
        # |   +-- 2,3
        # +-- 4
        schema_a1 = sym_schema('/A1/')
        schema_a2 = sym_schema('/A2/')
        schema_a1_r1 = sym_schema('/A1/1,')
        schema_a2_r2 = sym_schema('/A2/2,')
        schema_a1_c3_r1 = sym_schema('3,/A1/1,')
        schema_a2_c4_r2 = sym_schema('4,/A2/2,')
        schema_a1_c23_r1 = sym_schema('2,3/A1/1,')
        schema_a2_r23 = sym_schema('/A2/2,3')

        schema_ca2 = sym_schema('/2,/')
        schema_ca2.action.controller.update([
            Chain([schema_a2_c4_r2]),
        ])

        tree = SchemaTree()

        # add bare schemas
        tree.add(schemas=[schema_a1, schema_a2])

        # add non-composite action spin-offs
        tree.add(schemas=[schema_a1_r1], source=schema_a1)
        tree.add(schemas=[schema_a2_r2], source=schema_a2)
        tree.add(schemas=[schema_a1_c3_r1], source=schema_a1_r1)
        tree.add(schemas=[schema_a2_c4_r2], source=schema_a2_r2)
        tree.add(schemas=[schema_a1_c23_r1], source=schema_a1_c3_r1)
        tree.add(schemas=[schema_a2_r23], source=schema_a2_r2)

        # add composite action bare schema
        tree.add(schemas=[schema_ca2])

        schema_memory: SchemaMemory = SchemaMemory(schema_collection=tree)

        self.schema_mechanism: SchemaMechanism = SchemaMechanism(
            schema_memory=schema_memory,
            schema_selection=self.schema_selection,
        )

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)

            manifest = create_manifest()
            object_registry = create_object_registry()

            # sanity check: directory should initially contain no files
            files_in_temp_dir = [file for file in path.glob('**/*') if file.is_file()]
            self.assertTrue(len(files_in_temp_dir) == 0)

            self.schema_mechanism.save(
                path=path,
                manifest=manifest,
                overwrite=False,
                encoder=schema_mechanism.serialization.json.encoders.encode,
                object_registry=object_registry,
            )

            # test: directory should contain serialized file(s)
            files_in_temp_dir = [file for file in path.glob('**/*') if file.is_file()]
            self.assertTrue(len(files_in_temp_dir) > 0)

            restored_schema_mechanism = SchemaMechanism.load(
                manifest=manifest,
                decoder=schema_mechanism.serialization.json.decoders.decode,
                object_registry=object_registry
            )

            self.assertEqual(self.schema_mechanism, restored_schema_mechanism)

            # test: all restored schemas should have been added to the schema pool
            schema_pool = SchemaPool()
            for schema in restored_schema_mechanism.schema_memory:
                self.assertTrue(schema in schema_pool)
