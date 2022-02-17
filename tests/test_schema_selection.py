from collections import defaultdict
from unittest import TestCase

from schema_mechanism.data_structures import ItemPool
from schema_mechanism.func_api import sym_schema
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.modules import primitive_values
from test_share.test_func import common_test_setup


class TestSchemaSelection(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.ss = SchemaSelection()

        pool = ItemPool()

        self.i1 = pool.get('1', 0.0)
        self.i2 = pool.get('2', 0.0)
        self.i3 = pool.get('3', 0.95)
        self.i4 = pool.get('4', -1.0)
        self.i5 = pool.get('5', 2.0)
        self.i6 = pool.get('6', -3.0)

        self.s_prim = sym_schema('/A1/')  # total value = 0.0
        self.s1 = sym_schema('1,2/A1/3')  # total value = 0.95
        self.s2 = sym_schema('1,2/A1/4')  # total value = -1.0
        self.s3 = sym_schema('1,2/A1/5')  # total value = 2.0
        self.s4 = sym_schema('1,2/A1/3,4')  # total value = -0.05
        self.s5 = sym_schema('1,2/A1/3,4,5')  # total value = 1.95
        self.s6 = sym_schema('1,2/A1/3,4,5,6')  # total value = -1.05
        self.s7 = sym_schema('1,2/A1/3,4,5,~6')  # total value = 1.95

    def test_init(self):
        pass

    def test_primitive_values(self):
        # sanity checks
        ###############

        # None or empty list SHOULD return empty list
        self.assertEqual(0, len(primitive_values(schemas=[])))

        # noinspection PyTypeChecker
        self.assertEqual(0, len(primitive_values(schemas=None)))

        expected_values = [0.0, 0.95, -1.0, 2.0, -0.05, 1.95, -1.05, 1.95]
        actual_values = primitive_values(
            schemas=[self.s_prim, self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7])

        for exp, act in zip(expected_values, actual_values):
            self.assertAlmostEqual(exp, act)

    def test_delegated_values(self):
        pass

    def test_instrumental_values(self):
        pass

    def test_select_1(self):
        # sanity checks
        ###############

        # Empty or None list of applicable schemas should return a ValueError
        self.assertRaises(ValueError, lambda: self.ss.select(schemas=[]))

        # noinspection PyTypeChecker
        self.assertRaises(ValueError, lambda: self.ss.select(schemas=None))

        # Given a single applicable schema, select should return that schema
        schema = sym_schema('1,2/A1/3,4')
        sd = self.ss.select(schemas=[schema])
        self.assertEqual(schema, sd.selected)

    def test_select_2(self):
        # primitive value-based selections
        ##################################

        # selection between uneven schemas, all with non-negated items (s3 should win)
        sd = self.ss.select(schemas=[self.s1, self.s2, self.s3, self.s4])
        self.assertEqual(self.s3, sd.selected)

        # selection between uneven schemas, some with negated items (s7 should win)
        sd = self.ss.select(schemas=[self.s4, self.s6, self.s7])
        self.assertEqual(self.s7, sd.selected)

        # selection between multiple schemas with close values should be randomized
        selections = defaultdict(lambda: 0.0)
        applicable_schemas = [self.s3, self.s5, self.s7]
        for _ in range(100):
            sd = self.ss.select(applicable_schemas)
            selections[sd.selected] += 1

        self.assertEqual(len(applicable_schemas), len(selections.keys()))

    def test_notify_all(self):
        pass
