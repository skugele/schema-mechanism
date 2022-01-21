from unittest import TestCase

from schema_mechanism.data_structures import Action
from schema_mechanism.func_api import sym_schema
from schema_mechanism.func_api import sym_state
from schema_mechanism.modules import SchemaMemory


class TestSchemaMemory(TestCase):
    def setUp(self) -> None:
        # FIXME: This is a hack to initialize SchemaMemory with build-in schemas. Need to revisit this once I
        # FIXME: can save and load SchemaMemory state.
        self.s1 = sym_schema('/A1/')
        self.s2 = sym_schema('/A2/')

        self.s1_1 = sym_schema('1/A1/')
        self.s1_2 = sym_schema('2/A1/')
        self.s1_3 = sym_schema('3/A1/')
        self.s1_4 = sym_schema('4/A1/')
        self.s1.children += (self.s1_1, self.s1_2, self.s1_3, self.s1_4)

        self.s2_1 = sym_schema('1/A2/')
        self.s2_2 = sym_schema('2/A2/')
        self.s2.children += (self.s2_1, self.s2_2)

        self.s1_1_1 = sym_schema('1,2/A1/')
        self.s1_1_2 = sym_schema('1,3/A1/')
        self.s1_1_3 = sym_schema('1,4/A1/')
        self.s1_1_4 = sym_schema('1,5/A1/')
        self.s1_1.children += (self.s1_1_1, self.s1_1_2, self.s1_1_3, self.s1_1_4)

        self.s1_1_2_1 = sym_schema('1,3,5/A1/')
        self.s1_1_2_2 = sym_schema('1,3,6/A1/')
        self.s1_1_2_3 = sym_schema('1,3,7/A1/')
        self.s1_1_2.children += (self.s1_1_2_1, self.s1_1_2_2, self.s1_1_2_3)

        self.s1_1_2_1_1 = sym_schema('1,3,5,7/A1/')
        self.s1_1_2_1.children += (self.s1_1_2_1_1,)

        self.s1_1_2_3_1 = sym_schema('1,3,7,9/A1/')
        self.s1_1_2_3.children += (self.s1_1_2_3_1,)

        self.all_schemas = [
            self.s1,
            self.s2,
            self.s1_1,
            self.s1_2,
            self.s1_3,
            self.s1_4,
            self.s2_1,
            self.s2_2,
            self.s1_1_1,
            self.s1_1_2,
            self.s1_1_3,
            self.s1_1_4,
            self.s1_1_2_1,
            self.s1_1_2_2,
            self.s1_1_2_3,
            self.s1_1_2_1_1,
            self.s1_1_2_3_1,
        ]

        self.sm = SchemaMemory(primitive_schemas=(self.s1, self.s2))

    def test_init(self):
        self.assertEqual(len(self.all_schemas), len(self.sm))

        # should be supplied with at least one primitive action
        self.assertRaises(ValueError, lambda: SchemaMemory(primitive_schemas=[]))

        invalid_primitive_schema = sym_schema('1,2,3/A1/')
        self.assertRaises(ValueError, lambda: SchemaMemory(primitive_schemas=(invalid_primitive_schema,)))

    def test_contains(self):
        for schema in self.all_schemas:
            self.assertIn(schema, self.sm)

        # shouldn't be in SchemaMemory
        s_not_found = sym_schema('1,2,3,4,5/A7/')
        self.assertNotIn(s_not_found, self.sm)

    def test_all_applicable(self):
        app_schemas = self.sm.all_applicable(state=sym_state('1,3,5,7'))
        self.assertEqual(10, len(app_schemas))

        self.assertIn(self.s1, app_schemas)
        self.assertIn(self.s2, app_schemas)
        self.assertIn(self.s1_1, app_schemas)
        self.assertIn(self.s1_3, app_schemas)
        self.assertIn(self.s2_1, app_schemas)
        self.assertIn(self.s1_1_2, app_schemas)
        self.assertIn(self.s1_1_4, app_schemas)
        self.assertIn(self.s1_1_2_1, app_schemas)
        self.assertIn(self.s1_1_2_3, app_schemas)
        self.assertIn(self.s1_1_2_1_1, app_schemas)

    def test_all_activated(self):
        act_schemas = self.sm.all_activated(action=Action('A1'), state=sym_state('1,3'))
        self.assertEqual(4, len(act_schemas))
        self.assertIn(self.s1, act_schemas)
        self.assertIn(self.s1_1, act_schemas)
        self.assertIn(self.s1_3, act_schemas)
        self.assertIn(self.s1_1_2, act_schemas)

        # primitive only
        act_schemas = self.sm.all_activated(action=Action('A2'), state=sym_state('11'))
        self.assertEqual(1, len(act_schemas))
        self.assertIn(self.s2, act_schemas)

    def test_update_all(self):
        pass

    def test_receive(self):
        pass
