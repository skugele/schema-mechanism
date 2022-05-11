import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

from schema_mechanism.core import Action
from schema_mechanism.core import DelegatedValueHelper
from schema_mechanism.core import EligibilityTraceDelegatedValueHelper
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ItemPool
from schema_mechanism.core import default_action_trace
from schema_mechanism.core import default_delegated_value_helper
from schema_mechanism.core import default_global_params
from schema_mechanism.core import default_global_stats
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_schema
from schema_mechanism.modules import SchemaMechanism
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.persistence import deserialize
from schema_mechanism.persistence import serialize
from schema_mechanism.share import GlobalParams
from schema_mechanism.strategies.decay import ExponentialDecayStrategy
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.evaluation import CompositeEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultExploratoryEvaluationStrategy
from schema_mechanism.strategies.evaluation import DefaultGoalPursuitEvaluationStrategy
from schema_mechanism.strategies.match import AbsoluteDiffMatchStrategy
from schema_mechanism.strategies.selection import RandomizeBestSelectionStrategy
from schema_mechanism.strategies.trace import AccumulatingTrace
from schema_mechanism.strategies.trace import ReplacingTrace
from schema_mechanism.strategies.trace import Trace
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
        item_pool: ItemPool = ItemPool()
        item_pool.clear()

        global_params = GlobalParams()
        global_stats = GlobalStats()

        action_trace: Trace[Action] = ReplacingTrace(
            decay_strategy=GeometricDecayStrategy(rate=0.5)
        )

        delegated_value_helper: DelegatedValueHelper = EligibilityTraceDelegatedValueHelper(
            discount_factor=0.999,
            eligibility_trace=AccumulatingTrace(
                decay_strategy=ExponentialDecayStrategy(rate=0.017)
            )
        )

        schema_mechanism = SchemaMechanism(
            items=self.primitive_items,
            schema_memory=self.schema_memory,
            schema_selection=self.schema_selection,
            global_params=global_params,
            global_stats=global_stats,
            delegated_value_helper=delegated_value_helper,
            action_trace=action_trace
        )

        # test: attributes should have been set properly in initializer
        self.assertIs(self.schema_memory, schema_mechanism.schema_memory)
        self.assertIs(self.schema_selection, schema_mechanism.schema_selection)
        self.assertIs(delegated_value_helper, schema_mechanism.delegated_value_helper)
        self.assertIs(action_trace, schema_mechanism.action_trace)
        self.assertIs(global_stats, schema_mechanism.stats)
        self.assertIs(global_params, schema_mechanism.params)

        # test: verify that primitive items were added to the item pool
        item_pool = ItemPool()
        for item in self.primitive_items:
            self.assertIn(item.source, item_pool)

    def test_init_defaults(self):
        # test: defaults should be properly set if optional parameters not given
        schema_mechanism = SchemaMechanism(
            items=self.primitive_items,
            schema_memory=self.schema_memory,
            schema_selection=self.schema_selection
        )

        self.assertIs(default_delegated_value_helper, schema_mechanism.delegated_value_helper)
        self.assertIs(default_action_trace, schema_mechanism.action_trace)
        self.assertIs(default_global_stats, schema_mechanism.stats)
        self.assertIs(default_global_params, schema_mechanism.params)

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
