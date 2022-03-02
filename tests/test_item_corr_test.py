import unittest
from time import time
from typing import Iterable

import numpy as np

import test_share
from schema_mechanism.core import BarnardExactCorrelationTest
from schema_mechanism.core import DrescherCorrelationTest
from schema_mechanism.core import FisherExactCorrelationTest
from schema_mechanism.core import GlobalParams
from schema_mechanism.core import ItemCorrelationTest
from test_share.test_func import common_test_setup


class TestItemCorrTest(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        class TestItemCorr(ItemCorrelationTest):
            def __init__(self, pcs: float, ncs: float):
                self._pcs = pcs
                self._ncs = ncs

            def positive_corr_statistic(self, table: Iterable) -> float:
                return self._pcs

            def negative_corr_statistic(self, table: Iterable) -> float:
                return self._ncs

        self.TestClass = TestItemCorr

        self.pcs = 0.95
        self.ncs = 0.95

        self.test_instance = self.TestClass(pcs=self.pcs, ncs=self.ncs)

    # noinspection PyTypeChecker
    def test_validate_data(self):
        # test: valid 2x2 table of ints should be accepted
        try:
            self.test_instance.validate_data([[1, 2], [3, 4]])
            self.test_instance.validate_data(np.array([[1, 2], [3, 4]]))
        except ValueError as e:
            self.fail(f'unexpected ValueError encountered: {str(e)}')

        # test: non-iterables should not be accepted
        self.assertRaises(ValueError, lambda: self.test_instance.validate_data(None))
        self.assertRaises(ValueError, lambda: self.test_instance.validate_data(1.0))

        # test: 2x2 tables with non-integer type should be not be accepted
        self.assertRaises(ValueError, lambda: self.test_instance.validate_data([[1.0, 2.0], [3.0, 4.0]]))
        self.assertRaises(ValueError, lambda: self.test_instance.validate_data([['W', 'X'], ['Y', 'Z']]))

        # test: iterables with incorrect dimensions should be not be accepted
        self.assertRaises(ValueError, lambda: self.test_instance.validate_data([1, 2]))
        self.assertRaises(ValueError, lambda: self.test_instance.validate_data([[1], [2]]))

    def test_positive_corr(self):
        # test: should return True when positive corr. statistic is >= threshold
        GlobalParams().set('positive_correlation_threshold', 0.95)
        self.assertTrue(self.test_instance.positive_corr([[1, 2], [3, 4]]))

        # test: should return True when positive corr. statistic is < threshold
        GlobalParams().set('positive_correlation_threshold', 0.951)
        self.assertFalse(self.test_instance.positive_corr([[1, 2], [3, 4]]))

    def test_negative_corr(self):
        # test: should return True when negative corr. statistic is >= threshold
        GlobalParams().set('negative_correlation_threshold', 0.95)
        self.assertTrue(self.test_instance.negative_corr([[1, 2], [3, 4]]))

        # test: should return True when negative corr. statistic is < threshold
        GlobalParams().set('negative_correlation_threshold', 0.951)
        self.assertFalse(self.test_instance.negative_corr([[1, 2], [3, 4]]))


class TestDrescherCorrTest(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.dct = DrescherCorrelationTest()

    def test_positive_corr_statistic(self):
        table = [[100, 1], [1, 20]]
        self.assertAlmostEqual(0.9541, self.dct.positive_corr_statistic(table), delta=1e-3)

        table = [[0, 0], [0, 0]]
        self.assertEqual(0.0, self.dct.positive_corr_statistic(table))

    def test_negative_corr_statistic(self):
        table = [[1, 100], [20, 1]]
        self.assertAlmostEqual(0.9541, self.dct.negative_corr_statistic(table), delta=1e-3)

        table = [[0, 0], [0, 0]]
        self.assertEqual(0.0, self.dct.negative_corr_statistic(table))

    @test_share.performance_test
    def test_performance(self):
        n_iter = 100
        elapsed_time = 0.0
        table = [[12, 7], [3, 8]]

        for n in range(n_iter):
            start = time()
            self.dct.positive_corr_statistic(table)
            end = time()
            elapsed_time += end - start

        print(f'Time for {n_iter} Drescher ratio correlation tests: {elapsed_time}s')


class TestBarnardExactTest(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.bet = BarnardExactCorrelationTest()

    def test_positive_corr_statistic(self):
        table = [[12, 7], [3, 8]]
        self.assertAlmostEqual(0.96593, self.bet.positive_corr_statistic(table), delta=1e-4)

        table = [[0, 0], [0, 0]]
        self.assertEqual(0.0, self.bet.positive_corr_statistic(table))

    def test_negative_corr_statistic(self):
        table = [[7, 12], [8, 3]]
        self.assertAlmostEqual(0.96593, self.bet.negative_corr_statistic(table), delta=1e-4)

        table = [[0, 0], [0, 0]]
        self.assertEqual(0.0, self.bet.negative_corr_statistic(table))

    @test_share.performance_test
    def test_performance(self):
        n_iter = 100
        elapsed_time = 0.0
        table = [[12, 7], [3, 8]]

        for n in range(n_iter):
            start = time()
            self.bet.positive_corr_statistic(table)
            end = time()
            elapsed_time += end - start

        print(f'Time for {n_iter} Barnard exact correlation tests: {elapsed_time}s')


class TestFisherExactTest(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.fet = FisherExactCorrelationTest()

    def test_positive_corr_statistic(self):
        table = [[12, 7], [3, 8]]
        self.assertAlmostEqual(0.936, self.fet.positive_corr_statistic(table), delta=1e-3)

        table = [[0, 0], [0, 0]]
        self.assertEqual(0.0, self.fet.positive_corr_statistic(table))

    def test_negative_corr_statistic(self):
        table = [[7, 12], [8, 3]]
        self.assertAlmostEqual(0.936, self.fet.negative_corr_statistic(table), delta=1e-3)

        table = [[0, 0], [0, 0]]
        self.assertEqual(0.0, self.fet.negative_corr_statistic(table))

    @test_share.performance_test
    def test_performance(self):
        n_iter = 100
        elapsed_time = 0.0
        table = [[12, 7], [3, 8]]

        for n in range(n_iter):
            start = time()
            self.fet.positive_corr_statistic(table)
            end = time()
            elapsed_time += end - start

        print(f'Time for {n_iter} Fisher exact correlation tests: {elapsed_time}s')
