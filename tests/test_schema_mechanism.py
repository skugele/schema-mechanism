import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

from schema_mechanism.core import GlobalStats
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.persistence import deserialize
from schema_mechanism.persistence import serialize
from schema_mechanism.share import GlobalParams
from schema_mechanism.strategies.evaluation import CompositeEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultExploratoryEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultGoalPursuitEvaluationStrategy
from schema_mechanism.strategies.match import AbsoluteDiffMatchStrategy
from schema_mechanism.strategies.selection import RandomizeBestSelectionStrategy
from test_share.test_func import common_test_setup
from test_share.test_func import file_was_written


class TestSchemaMechanism(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.bare_schemas = [sym_schema(f'/A{i}/') for i in range(10)]
        self.primitive_items = {
            sym_item('X', primitive_value=-1.0),
            sym_item('Y', primitive_value=0.0),
            sym_item('Z', primitive_value=1.0)
        }
        self.primitive_actions = {schema.action for schema in self.bare_schemas}
        self.schema_memory = SchemaMemory(self.bare_schemas)
        self.schema_selection = SchemaSelection(
            select_strategy=RandomizeBestSelectionStrategy(AbsoluteDiffMatchStrategy(1.0)),
            evaluation_strategy=CompositeEvaluationStrategy(
                strategies=[
                    DefaultGoalPursuitEvaluationStrategy(),
                    DefaultExploratoryEvaluationStrategy(),
                ]
            )
        )

        self.sm: SchemaMechanism = SchemaMechanism(
            items=self.primitive_items,
            schema_memory=self.schema_memory,
            schema_selection=self.schema_selection,
            global_params=GlobalParams(),
            global_stats=GlobalStats())

    def test_init(self):
        # test: attributes should have been set properly in initializer
        self.assertIs(self.schema_memory, self.sm.schema_memory)
        self.assertIs(self.schema_selection, self.sm.schema_selection)

        # test: references to global parameters and statistics should be set
        self.assertIsInstance(self.sm.params, GlobalParams)
        self.assertIsInstance(self.sm.stats, GlobalStats)

        # test: stats action trace should include all actions from built-in schemas
        self.assertSetEqual(self.primitive_actions, set(self.sm.stats.action_trace.keys()))

    def test_serialize(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-schema-memory-serialize.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            serialize(self.sm, path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            recovered: SchemaMechanism = deserialize(path)

            # TODO: Need more thorough equivalence tests between serialized and original objects
            # test: deserialized SchemaMemory object should be equivalent to the original
            self.assertEqual(recovered.schema_memory, self.sm.schema_memory)

            # test: deserialized schema memory should be registered as an observer of its schemas
            recovered.schema_memory.receive = Mock()

            for schema in recovered.schema_memory:
                schema.notify_all(message='message')

            self.assertEqual(len(recovered.schema_memory), recovered.schema_memory.receive.call_count)
