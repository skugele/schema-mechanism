import time
import unittest
from typing import Iterable

import test_share
from schema_mechanism.core import ItemCorrelationTest
from schema_mechanism.stats import BarnardExactCorrelationTest
from schema_mechanism.stats import DrescherCorrelationTest
from schema_mechanism.stats import FisherExactCorrelationTest
from test_share.test_func import common_test_setup


class TestItemCorrelationTest(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        class TestItemCorrelation(ItemCorrelationTest):
            def __init__(self, pcs: float, ncs: float):
                self._pcs = pcs
                self._ncs = ncs

            def positive_corr_statistic(self, table: Iterable) -> float:
                return self._pcs

            def negative_corr_statistic(self, table: Iterable) -> float:
                return self._ncs

        self.TestClass = TestItemCorrelation

        self.pcs = 0.95
        self.ncs = 0.95

        self.test_instance = self.TestClass(pcs=self.pcs, ncs=self.ncs)


class TestDrescherCorrelationTest(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.dct = DrescherCorrelationTest()

    def test_positive_corr_statistic(self):
        table = (100, 1, 1, 20)
        self.assertAlmostEqual(0.9541, self.dct.positive_corr_statistic(table), delta=1e-3)

        table = (0, 0, 0, 0)
        self.assertEqual(0.0, self.dct.positive_corr_statistic(table))

    def test_negative_corr_statistic(self):
        table = (1, 100, 20, 1)
        self.assertAlmostEqual(0.9541, self.dct.negative_corr_statistic(table), delta=1e-3)

        table = (0, 0, 0, 0)
        self.assertEqual(0.0, self.dct.negative_corr_statistic(table))

    @test_share.performance_test
    def test_performance(self):
        n_iter = 100
        elapsed_time = 0.0
        table = (12, 7, 3, 8)

        for n in range(n_iter):
            start = time.perf_counter()
            self.dct.positive_corr_statistic(table)
            end = time.perf_counter()
            elapsed_time += end - start

        print(f'Time for {n_iter} Drescher ratio correlation tests: {elapsed_time}s')


class TestBarnardExactTest(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.bet = BarnardExactCorrelationTest()

    def test_positive_corr_statistic(self):
        table = (12, 7, 3, 8)
        self.assertAlmostEqual(0.96593, self.bet.positive_corr_statistic(table), delta=1e-4)

        table = (0, 0, 0, 0)
        self.assertEqual(0.0, self.bet.positive_corr_statistic(table))

    def test_negative_corr_statistic(self):
        table = (7, 12, 8, 3)
        self.assertAlmostEqual(0.96593, self.bet.negative_corr_statistic(table), delta=1e-4)

        table = (0, 0, 0, 0)
        self.assertEqual(0.0, self.bet.negative_corr_statistic(table))

    @test_share.performance_test
    def test_performance(self):
        n_iter = 100
        elapsed_time = 0.0
        table = (12, 7, 3, 8)

        for n in range(n_iter):
            start = time.perf_counter()
            self.bet.positive_corr_statistic(table)
            end = time.perf_counter()
            elapsed_time += end - start

        print(f'Time for {n_iter} Barnard exact correlation tests: {elapsed_time}s')


class TestFisherExactTest(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.fet = FisherExactCorrelationTest()

    def test_positive_corr_statistic(self):
        table = (12, 7, 3, 8)
        self.assertAlmostEqual(0.936, self.fet.positive_corr_statistic(table), delta=1e-3)

        table = (0, 0, 0, 0)
        self.assertEqual(0.0, self.fet.positive_corr_statistic(table))

    def test_negative_corr_statistic(self):
        table = (7, 12, 8, 3)
        self.assertAlmostEqual(0.936, self.fet.negative_corr_statistic(table), delta=1e-3)

        table = (0, 0, 0, 0)
        self.assertEqual(0.0, self.fet.negative_corr_statistic(table))

    @test_share.performance_test
    def test_performance(self):
        n_iter = 100
        elapsed_time = 0.0
        table = (12, 7, 3, 8)

        for n in range(n_iter):
            start = time.perf_counter()
            self.fet.positive_corr_statistic(table)
            end = time.perf_counter()
            elapsed_time += end - start

        print(f'Time for {n_iter} Fisher exact correlation tests: {elapsed_time}s')
