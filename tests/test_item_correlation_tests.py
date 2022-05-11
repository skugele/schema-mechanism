import itertools
import time
import unittest

import test_share
from schema_mechanism.strategies.correlation_test import BarnardExactCorrelationTest
from schema_mechanism.strategies.correlation_test import CorrelationOnEncounter
from schema_mechanism.strategies.correlation_test import DrescherCorrelationTest
from schema_mechanism.strategies.correlation_test import FisherExactCorrelationTest
from test_share.test_func import common_test_setup


class TestCorrelationOnEncounter(unittest.TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.correlation_test = CorrelationOnEncounter()

    def test_positive_corr_statistic(self):
        """ Input data has the form [[N(A,X), N(not A,X)], [N(A,not X), N(not A,not X)]],"""
        # test: positive correlation statistic should be 1.0 if positive value in top-left table cell N(A,X)
        for a_x, not_a_x, a_not_x, not_a_not_x in itertools.product(range(0, 5), range(0, 5), range(0, 5), range(0, 5)):
            table = (a_x, not_a_x, a_not_x, not_a_not_x)

            if a_x >= 1:
                self.assertEqual(1.0, self.correlation_test.positive_corr_statistic(table))
            else:
                self.assertEqual(0.0, self.correlation_test.positive_corr_statistic(table))

    def test_negative_corr_statistic(self):
        """ Input data has the form [[N(A,X), N(not A,X)], [N(A,not X), N(not A,not X)]],"""
        # test: negative correlation statistic should be 1.0 if positive value in top-left table cell N(not A,X)
        for a_x, not_a_x, a_not_x, not_a_not_x in itertools.product(range(0, 5), range(0, 5), range(0, 5), range(0, 5)):
            table = (a_x, not_a_x, a_not_x, not_a_not_x)

            if not_a_x >= 1:
                self.assertEqual(1.0, self.correlation_test.negative_corr_statistic(table))
            else:
                self.assertEqual(0.0, self.correlation_test.negative_corr_statistic(table))


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
