import unittest
from collections import defaultdict
from unittest import TestCase

import numpy as np

from schema_mechanism.core import GlobalParams
from schema_mechanism.core import ItemPool
from schema_mechanism.func_api import sym_schema
from schema_mechanism.modules import EpsilonGreedyExploratoryStrategy
from schema_mechanism.modules import SchemaSelection
from schema_mechanism.modules import primitive_values
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup


class TestSchemaSelection(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        GlobalParams()
        self.ss = SchemaSelection()

        pool = ItemPool()

        self.i1 = pool.get('1', item_type=MockSymbolicItem, primitive_value=0.0, avg_accessible_value=-3.0)
        self.i2 = pool.get('2', item_type=MockSymbolicItem, primitive_value=0.0, avg_accessible_value=2.0)
        self.i3 = pool.get('3', item_type=MockSymbolicItem, primitive_value=0.95, avg_accessible_value=-1.0)
        self.i4 = pool.get('4', item_type=MockSymbolicItem, primitive_value=-1.0, avg_accessible_value=0.95)
        self.i5 = pool.get('5', item_type=MockSymbolicItem, primitive_value=2.0, avg_accessible_value=0.0)
        self.i6 = pool.get('6', item_type=MockSymbolicItem, primitive_value=-3.0, avg_accessible_value=0.0)

        # note: negated items have zero primitive value
        self.s_prim = sym_schema('/A1/')  # total primitive value = 0.0; total delegated value = 0.0
        self.s1 = sym_schema('1,2/A1/3,')  # total primitive value = 0.95; total delegated value = 0.0
        self.s2 = sym_schema('1,2/A1/4,')  # total primitive value = -1.0; total delegated value = 0.0
        self.s3 = sym_schema('1,2/A1/5,')  # total primitive value = 2.0; total delegated value = 0.0
        self.s4 = sym_schema('1,2/A1/3,4')  # total primitive value = -0.05; total delegated value = 0.0
        self.s5 = sym_schema('1,2/A1/3,4,5')  # total primitive value = 1.95; total delegated value = 0.0
        self.s6 = sym_schema('1,2/A1/3,4,5,6')  # total primitive value = -1.05; total delegated value = 0.0
        self.s7 = sym_schema('1,2/A1/3,4,5,~6')  # total primitive value = 1.95; total delegated value = 0.0

        self.s8 = sym_schema('1,2/A1/(3,4),')  # total primitive value = -0.05; total delegated value = 0.0
        self.s9 = sym_schema('1,2/A1/(~5,6),')  # total primitive value = 1.95; total delegated value = 0.0

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
        self.fail()

    def test_instrumental_values(self):
        self.fail()

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


class TestEpsilonGreedy(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.eps_always_explore = EpsilonGreedyExploratoryStrategy(epsilon=1.0)
        self.eps_never_explore = EpsilonGreedyExploratoryStrategy(epsilon=0.0)
        self.eps_even_chance = EpsilonGreedyExploratoryStrategy(epsilon=0.5)

        self.schema = sym_schema('1,2/A/3,4')
        self.schemas = [sym_schema('1,2/A/3,4'), sym_schema('1,2/A/3,5'), sym_schema('1,2/A/3,6')]

    def test_init(self):
        epsilon = 0.1
        eps_greedy = EpsilonGreedyExploratoryStrategy(epsilon)
        self.assertEqual(epsilon, eps_greedy.epsilon)

    def test_epsilon_setter(self):
        eps_greedy = EpsilonGreedyExploratoryStrategy(0.5)

        eps_greedy.epsilon = 0.0
        self.assertEqual(0.0, eps_greedy.epsilon)

        eps_greedy.epsilon = 1.0
        self.assertEqual(1.0, eps_greedy.epsilon)

        try:
            eps_greedy.epsilon = -1.00001
            eps_greedy.epsilon = 1.00001
            self.fail('ValueError expected on illegal assignment')
        except ValueError:
            pass

    def test_call(self):
        # test: should return empty np.array given empty ndarray
        values = self.eps_never_explore([])
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(0, len(values))

        values = self.eps_always_explore([])
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(0, len(values))

        # test: non-empty schema array should return an np.array with a single np.inf on exploratory choice
        values = self.eps_always_explore([self.schema])
        self.assertEqual(1, len(values))
        self.assertEqual(1, np.count_nonzero(values == np.inf))
        self.assertIsInstance(values, np.ndarray)

        values = self.eps_always_explore(self.schemas)
        self.assertIsInstance(values, np.ndarray)
        self.assertEqual(len(self.schemas), len(values))
        self.assertEqual(1, np.count_nonzero(values == np.inf))

        # test: non-empty schema array should return an np.array with same length of zeros non-exploratory choice
        values = self.eps_never_explore([self.schema])
        self.assertIsInstance(values, np.ndarray)
        self.assertListEqual(list(np.array([0])), list(values))

        values = self.eps_never_explore(self.schemas)
        self.assertIsInstance(values, np.ndarray)
        self.assertListEqual(list(np.zeros_like(self.schemas)), list(values))
