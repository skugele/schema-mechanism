from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase
from unittest.mock import ANY
from unittest.mock import MagicMock

import numpy as np

import schema_mechanism.serialization.json.decoders
import schema_mechanism.serialization.json.encoders
from schema_mechanism.core import Chain
from schema_mechanism.core import SchemaPool
from schema_mechanism.core import SchemaSpinOffType
from schema_mechanism.core import SchemaTree
from schema_mechanism.core import get_global_params
from schema_mechanism.func_api import actions
from schema_mechanism.func_api import primitive_schemas
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_items
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import SchemaMemory
from schema_mechanism.modules import SelectionDetails
from schema_mechanism.modules import create_context_spin_off
from schema_mechanism.modules import create_result_spin_off
from schema_mechanism.serialization import create_manifest
from test_share.test_func import common_test_setup


class TestSchemaMemory(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        params = get_global_params()
        params.set('composite_actions.update_frequency', 1.0)

        # always create composite actions for novel results
        params.set('composite_actions.min_baseline_advantage', -np.inf)

        s1 = sym_schema('/A1/')
        s2 = sym_schema('/A2/')

        # composite action schemas
        self.s101 = sym_schema('/101,/')
        self.s102 = sym_schema('/102,/')

        self.tree = SchemaTree()
        self.tree.add(schemas=(s1, s2, self.s101, self.s102))

        # composite action result spin-off
        self.s101_r100 = sym_schema('/101,/100,')
        self.tree.add(source=self.s101, schemas=(self.s101_r100,))

        # composite action context spin-off
        self.s101_1_r100 = sym_schema('1,/101,/100,', reliability=1.0)
        self.tree.add(source=self.s101_r100, schemas=(self.s101_1_r100,))

        s1_1 = sym_schema('1,/A1/')
        s1_2 = sym_schema('2,/A1/')
        s1_3 = sym_schema('3,/A1/')
        s1_4 = sym_schema('4,/A1/')
        self.tree.add(source=s1, schemas=(s1_1, s1_2, s1_3, s1_4))

        s2_1 = sym_schema('1,/A2/')
        s2_2 = sym_schema('2,/A2/')
        self.tree.add(source=s2, schemas=(s2_1, s2_2))

        s1_1_1 = sym_schema('1,2/A1/')
        s1_1_2 = sym_schema('1,3/A1/')
        s1_1_3 = sym_schema('1,4/A1/')
        s1_1_4 = sym_schema('1,5/A1/')
        self.tree.add(source=s1_1, schemas=(s1_1_1, s1_1_2, s1_1_3, s1_1_4))

        s1_1_2_1 = sym_schema('1,3,5/A1/')
        s1_1_2_2 = sym_schema('1,3,6/A1/')
        s1_1_2_3 = sym_schema('1,3,7/A1/')
        self.tree.add(source=s1_1_2, schemas=(s1_1_2_1, s1_1_2_2, s1_1_2_3))

        s1_1_2_1_1 = sym_schema('1,3,5,7/A1/')
        self.tree.add(source=s1_1_2_1, schemas=(s1_1_2_1_1,))

        s1_1_2_3_1 = sym_schema('1,3,7,9/A1/')
        self.tree.add(source=s1_1_2_3, schemas=(s1_1_2_3_1,))

        s1_r101 = sym_schema('/A1/101,')
        self.tree.add(source=s1, schemas=(s1_r101,))

        s1_1_r101 = sym_schema('1,/A1/101,', reliability=1.0)
        self.tree.add(source=s1_r101, schemas=(s1_1_r101,))

        self.bare_schemas = {
            s1, s2, self.s101, self.s102
        }

        self.no_context_schemas = {
            s1, s2, self.s101, self.s101_r100, self.s102
        }

        self.schemas = {
            s1, s2, s1_1, s1_2, s1_3, s1_4, s2_1, s2_2, s1_1_1, s1_1_2, s1_1_3, s1_1_4, s1_1_2_1, s1_1_2_2, s1_1_2_3,
            s1_1_2_1_1, s1_1_2_3_1, s1_r101, s1_1_r101,
        }

        self.actions = {schema.action for schema in self.schemas}

        self.non_bare_schemas = self.schemas.difference(self.bare_schemas)

        self.composite_schemas = {
            self.s101, self.s101_r100, self.s101_1_r100, self.s102
        }

        self.all_schemas = {*self.schemas, *self.composite_schemas}

        # initialize composite schema controllers
        self.s101.action.controller.update([Chain([s1_1_r101])])
        self.s102.action.controller.update([])

        # Tree contents:
        #
        # root
        # |-- 100
        # |-- 1
        # |   |-- 1,3
        # |   |   |-- 1,3,7
        # |   |   |   +-- 1,3,7,9
        # |   |   |-- 1,3,6
        # |   |   +-- 1,3,5
        # |   |       +-- 1,3,5,7
        # |   |-- 1,5
        # |   |-- 1,4
        # |   +-- 1,2
        # |-- 3
        # |-- 2
        # |-- 4
        # +-- 101
        self.schema_memory = SchemaMemory(self.tree)

    def test_init(self):
        self.assertEqual(len(self.tree), len(self.schema_memory))

        # test: one or more bare schemas should be allowed in initializer
        try:
            # test with a single bare schema
            tree = SchemaTree()
            tree.add(list(self.bare_schemas)[:1])

            sm = SchemaMemory(tree)
            self.assertEqual(1, len(sm))

            # test with multiple bare schemas
            tree = SchemaTree()
            tree.add(self.bare_schemas)

            sm = SchemaMemory(tree)
            self.assertEqual(len(self.bare_schemas), len(sm))
        except ValueError as e:
            self.fail(f'Caught unexpected ValueError: {e}')

    def test_init_from_tree(self):
        primitives = primitive_schemas(actions(5))
        tree = SchemaTree()
        tree.add(primitives)

        s1_1 = create_context_spin_off(primitives[0], sym_item('1'))
        s1_2 = create_context_spin_off(primitives[0], sym_item('2'))
        tree.add(source=primitives[0], schemas=(s1_1, s1_2))

        s1_r1 = create_result_spin_off(primitives[0], sym_item('3'))
        s1_r2 = create_result_spin_off(primitives[0], sym_item('4'))
        tree.add(source=primitives[0], schemas=(s1_r1, s1_r2))

        sm = SchemaMemory(tree)
        self.assertEqual(len(tree), len(sm))

        for schema in [*primitives, s1_1, s1_2, s1_r1, s1_r2]:
            self.assertIn(schema, sm)

    def test_contains(self):
        for schema in self.tree:
            self.assertIn(schema, self.schema_memory)

        # shouldn't be in SchemaMemory
        s_not_found = sym_schema('1,2,3,4,5/A7/')
        self.assertNotIn(s_not_found, self.schema_memory)

    def test_iter(self):
        iter_result = [schema for schema in self.schema_memory]

        self.assertEqual(len(self.all_schemas), len(iter_result))
        self.assertSetEqual(set(self.all_schemas), set(schema for schema in self.schema_memory))

    def test_all_applicable(self):
        # case 1: only bare schemas without composite actions
        state = sym_state('')

        # Info on matching:
        #
        # /A1/ [applicable - matches all states]
        # /A1/101, [applicable - matches all states]
        # /A2/ [applicable - matches all states]
        # /101,/ [would be applicable, BUT no applicable controller components]
        # /101,/100, [would be applicable, BUT no applicable controller components]
        # /102,/ [would be applicable, BUT no controller components]
        schemas = self.schema_memory.all_applicable(state)
        self.assertEqual(3, len(schemas))

        for schema in schemas:
            self.assertTrue(schema.context.is_satisfied(state))

            if schema.action.is_composite():
                self.assertTrue(schema.action.is_enabled(state=state))

        # case 2: single non-null context match plus null context matches
        state = sym_state('3')

        # Info on matching:
        #
        # /A1/ [applicable - matches all states]
        # /A1/101, [applicable - matches all states]
        # /A2/ [applicable - matches all states]
        # 3,/A1/ [applicable - matches state == 3]
        schemas = self.schema_memory.all_applicable(state)
        self.assertEqual(4, len(schemas))

        for schema in schemas:
            self.assertTrue(schema.context.is_satisfied(state))

            if schema.action.is_composite():
                self.assertTrue(schema.action.is_enabled(state=state))

        # case 3: entire tree matched with the exception of single composite action schema that has a controller with
        #       : no applicable components
        state = sym_state('1,2,3,4,5,6,7,8,9,10,100,101')
        schemas = self.schema_memory.all_applicable(state)

        # all schemas minus the not applicable composite action schema (/102,/)
        expected_schemas = self.all_schemas.difference({self.s102})
        self.assertSetEqual(expected_schemas, set(schemas))

        # schemas returned by all_applicable should have the following properties:
        #   (1) their contexts should be satisfied
        #   (2) no overriding conditions should be in effect
        #   (3) if the schema has a composite action, its action should be enabled (i.e., its controller has an
        #       applicable schema)
        for schema in schemas:
            # test: applicable schema's context should be satisfied
            self.assertTrue(schema.context.is_satisfied(state))

            # test: applicable schema's context should be satisfied
            if schema.action.is_composite():
                self.assertTrue(schema.action.is_enabled(state=state))

    def test_update_all(self):

        selection_state = sym_state('1,3,5,7,12')
        result_state = sym_state('10,11,12')

        selected_schema = sym_schema('1,3,5,7/A1/10,')

        all_schemas = set([s for s in self.schema_memory])
        applicable_schemas = set(self.schema_memory.all_applicable(selection_state))
        non_applicable_schemas = all_schemas - applicable_schemas

        selection_details = SelectionDetails(
            applicable=applicable_schemas,
            selected=selected_schema,
            selection_state=selection_state,
            terminated_pending=[],
            effective_value=1.0  # value fabricated. does not matter for test
        )

        for s in all_schemas:
            s.update = MagicMock()

        self.schema_memory.update_all(selection_details, result_state)

        for s in applicable_schemas:
            s.update.assert_called()

            # test: activated SHOULD be True if schema's action equals selected schema's action
            if s.action == selected_schema.action:
                s.update.assert_called_with(activated=True, succeeded=ANY, selection_state=ANY, new=ANY, lost=ANY,
                                            explained=ANY)

            # test: activated SHOULD be False if schema's action equals selected schema's action
            else:
                s.update.assert_called_with(activated=False, succeeded=ANY, selection_state=ANY, new=ANY, lost=ANY,
                                            explained=ANY)

            # test: succeeded SHOULD be True if schema's result is satisfied and schema is activated
            if (s.is_activated(selected_schema, selection_state, applicable=True)
                    and s.result.is_satisfied(result_state)):
                s.update.assert_called_with(succeeded=True, activated=ANY, selection_state=ANY, new=ANY, lost=ANY,
                                            explained=ANY)

            # test: succeeded SHOULD be False if schema's result is not satisfied or schema is not activated
            else:
                s.update.assert_called_with(succeeded=False, activated=ANY, selection_state=ANY, new=ANY, lost=ANY,
                                            explained=ANY)

        # test: update SHOULD NOT be called for non-applicable schemas
        for s in non_applicable_schemas:
            s.update.assert_not_called()

    def test_receive_1(self):
        # create a context spin-off
        n_schemas = len(self.schema_memory)

        self.assertNotIn(sym_schema('1,3,7,9,11/A1/'), self.schema_memory)
        self.schema_memory.receive(
            source=sym_schema('1,3,7,9/A1/'),
            spin_off_type=SchemaSpinOffType.CONTEXT,
            relevant_items=[sym_item('11')]
        )
        self.assertIn(sym_schema('1,3,7,9,11/A1/'), self.schema_memory)
        self.assertEqual(n_schemas + 1, len(self.schema_memory))

    def test_receive_2(self):
        # create a result spin-off
        n_schemas = len(self.schema_memory)

        self.assertNotIn(sym_schema('/A1/1000,'), self.schema_memory)

        # composite action for this result. (novel results are added as composite actions during result spin-off)
        self.assertNotIn(sym_schema('/1000,/'), self.schema_memory)

        self.schema_memory.receive(
            source=sym_schema('/A1/'),
            spin_off_type=SchemaSpinOffType.RESULT,
            relevant_items=[sym_item('1000')]
        )

        self.assertIn(sym_schema('/A1/1000,'), self.schema_memory)
        self.assertIn(sym_schema('/1000,/'), self.schema_memory)

        self.assertEqual(n_schemas + 2, len(self.schema_memory))

    def test_receive_3(self):
        # create multiple context spin-offs
        n_schemas = len(self.schema_memory)

        self.assertNotIn(sym_schema('1,3,7,9,11/A1/'), self.schema_memory)
        self.assertNotIn(sym_schema('1,3,7,9,13/A1/'), self.schema_memory)
        self.schema_memory.receive(
            source=sym_schema('1,3,7,9/A1/'),
            spin_off_type=SchemaSpinOffType.CONTEXT,
            relevant_items=sym_items('11;13')
        )
        self.assertIn(sym_schema('1,3,7,9,11/A1/'), self.schema_memory)
        self.assertIn(sym_schema('1,3,7,9,13/A1/'), self.schema_memory)
        self.assertEqual(n_schemas + 2, len(self.schema_memory))

    def test_receive_4(self):
        # create multiple result spin-offs
        n_schemas = len(self.schema_memory)

        self.assertNotIn(sym_schema('/A1/1000,'), self.schema_memory)
        self.assertNotIn(sym_schema('/A1/1001,'), self.schema_memory)

        # composite actions for these results. (novel results are added as composite actions during result spin-off)
        self.assertNotIn(sym_schema('/1000,/'), self.schema_memory)
        self.assertNotIn(sym_schema('/1001,/'), self.schema_memory)

        self.schema_memory.receive(
            source=sym_schema('/A1/'),
            spin_off_type=SchemaSpinOffType.RESULT,
            relevant_items=sym_items('1000;1001')
        )

        self.assertIn(sym_schema('/A1/1000,'), self.schema_memory)
        self.assertIn(sym_schema('/A1/1001,'), self.schema_memory)

        # new composite actions
        self.assertIn(sym_schema('/1000,/'), self.schema_memory)
        self.assertIn(sym_schema('/1001,/'), self.schema_memory)

        # 4 = 2 (new result spin-offs) + 2 (new basic composite action schemas)
        self.assertEqual(n_schemas + 4, len(self.schema_memory))

    def test_receive_5(self):
        # create multiple context spin-offs, one of which already exists in schema memory
        n_schemas = len(self.schema_memory)

        self.assertIn(sym_schema('1,3/A1/'), self.schema_memory)
        self.assertNotIn(sym_schema('3,6/A1/'), self.schema_memory)
        self.schema_memory.receive(
            source=sym_schema('3,/A1/'),
            spin_off_type=SchemaSpinOffType.CONTEXT,
            relevant_items=sym_items('1;6')
        )
        self.assertIn(sym_schema('1,3/A1/'), self.schema_memory)
        self.assertIn(sym_schema('3,6/A1/'), self.schema_memory)

        # should only be one new schema
        self.assertEqual(n_schemas + 1, len(self.schema_memory))

    def test_receive_6(self):
        # create multiple result spin-offs, one of which already exists in schema memory
        n_schemas = len(self.schema_memory)

        # sanity check: schema '/A1/101/' should exist in tree, but not '/A1/1001'
        self.assertIn(sym_schema('/A1/101,'), self.schema_memory)
        self.assertNotIn(sym_schema('/A1/1001,'), self.schema_memory)

        # composite actions for these results. (novel results are added as composite actions during result spin-off)
        self.assertIn(sym_schema('/101,/'), self.schema_memory)
        self.assertNotIn(sym_schema('/1001,/'), self.schema_memory)

        self.schema_memory.receive(
            source=sym_schema('/A1/'),
            spin_off_type=SchemaSpinOffType.RESULT,
            relevant_items=sym_items('101;1001')
        )

        # test: new result spin-off should have been added
        self.assertIn(sym_schema('/A1/1001,'), self.schema_memory)

        # test: new bare schema with novel composite action should have been added
        self.assertIn(sym_schema('/1001,/'), self.schema_memory)

        # test: adds 1 new result spin-off and 1 new bare schema for novel result
        self.assertEqual(n_schemas + 2, len(self.schema_memory))

    def test_save_and_load(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)

            object_registry: dict[str, Any] = dict()
            manifest = create_manifest()

            # sanity check: directory should initially contain no files
            files_in_temp_dir = [file for file in path.glob('**/*') if file.is_file()]
            self.assertTrue(len(files_in_temp_dir) == 0)

            self.schema_memory.save(
                path=path,
                manifest=manifest,
                object_registry=object_registry,
                overwrite=False,
                encoder=schema_mechanism.serialization.json.encoders.encode
            )

            # test: directory should contain serialized file(s)
            files_in_temp_dir = [file for file in path.glob('**/*') if file.is_file()]
            self.assertTrue(len(files_in_temp_dir) > 0)

            # test: returned manifest should contain an entry to each serialized sub-component
            manifest_for_schema_memory = manifest['objects']['SchemaMemory']

            self.assertIn('schema_collection', manifest_for_schema_memory)
            schema_collection_filepath = Path(manifest_for_schema_memory['schema_collection'])

            # test: a file should exist in the temporary directory for each sub-component
            self.assertIn(schema_collection_filepath, files_in_temp_dir)

            recovered_schema_memory = SchemaMemory.load(
                manifest=manifest,
                decoder=schema_mechanism.serialization.json.decoders.decode,
                object_registry=object_registry
            )

            self.assertEqual(self.schema_memory, recovered_schema_memory)

            schema_pool = SchemaPool()
            for schema in recovered_schema_memory:
                self.assertTrue(schema in schema_pool)
