from copy import copy
from time import time
from unittest import TestCase

import numpy as np

import test_share
from schema_mechanism.core import ECItemStats
from schema_mechanism.core import ERItemStats
from schema_mechanism.core import ReadOnlyECItemStats
from schema_mechanism.core import ReadOnlyERItemStats
from schema_mechanism.core import SchemaStats
from schema_mechanism.share import GlobalParams
from schema_mechanism.strategies.correlation_test import BarnardExactCorrelationTest
from schema_mechanism.strategies.correlation_test import DrescherCorrelationTest
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestECItemStatistics(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.item_stats = ECItemStats()

    def test_init(self):
        self.assertEqual(self.item_stats.n_success_and_on, 0)
        self.assertEqual(self.item_stats.n_success_and_off, 0)
        self.assertEqual(self.item_stats.n_fail_and_on, 0)
        self.assertEqual(self.item_stats.n_fail_and_off, 0)

    def test_specificity_1(self):
        # test: initial specificity SHOULD be NAN
        self.assertIs(np.NAN, self.item_stats.specificity)

    def test_specificity_2(self):
        # test: if an item was never On its specificity SHOULD be 1.0
        self.item_stats.update(on=False, success=False, count=1000)
        self.assertEqual(1.0, self.item_stats.specificity)

    def test_specificity_3(self):
        # test: if an item is always On its specificity SHOULD be 0.0
        self.item_stats.update(on=True, success=False, count=1000)
        self.assertEqual(0.0, self.item_stats.specificity)

    def test_specificity_4(self):
        # test: an item that is On and Off equally should have assertion specificity of 0.5
        self.item_stats.update(on=False, success=False, count=500)
        self.item_stats.update(on=True, success=False, count=500)
        self.assertEqual(0.5, self.item_stats.specificity)

    def test_update(self):
        self.item_stats.update(on=True, success=True, count=32)
        self.item_stats.update(on=True, success=False, count=8)
        self.item_stats.update(on=False, success=True, count=2)
        self.item_stats.update(on=False, success=False, count=8)

        self.assertEqual(self.item_stats.n_on, 40)
        self.assertEqual(self.item_stats.n_off, 10)

        self.assertEqual(self.item_stats.n_success_and_on, 32)
        self.assertEqual(self.item_stats.n_success_and_off, 2)
        self.assertEqual(self.item_stats.n_fail_and_on, 8)
        self.assertEqual(self.item_stats.n_fail_and_off, 8)

    def test_drescher_correlation(self):
        GlobalParams().set('ext_context.correlation_test', DrescherCorrelationTest())

        self.item_stats.update(on=True, success=True, count=12)
        self.item_stats.update(on=True, success=False, count=7)
        self.item_stats.update(on=False, success=True, count=3)
        self.item_stats.update(on=False, success=False, count=8)

        # results include a +1 increment to each of the above statistics
        self.assertAlmostEqual(0.6984, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.3362, self.item_stats.negative_correlation_stat, delta=1e-3)

    def test_barnard_correlation_1(self):
        GlobalParams().set('ext_context.correlation_test', BarnardExactCorrelationTest())

        self.item_stats.update(on=True, success=True, count=12)
        self.item_stats.update(on=True, success=False, count=7)
        self.item_stats.update(on=False, success=True, count=3)
        self.item_stats.update(on=False, success=False, count=8)

        # results include a +1 increment to each of the above statistics
        self.assertAlmostEqual(0.966, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.0, self.item_stats.negative_correlation_stat, delta=1e-3)

    def test_barnard_correlation_2(self):
        GlobalParams().set('ext_context.correlation_test', BarnardExactCorrelationTest())

        self.item_stats.update(on=True, success=True, count=3)
        self.item_stats.update(on=True, success=False, count=8)
        self.item_stats.update(on=False, success=True, count=12)
        self.item_stats.update(on=False, success=False, count=7)

        # results include a +1 increment to each of the above statistics
        self.assertAlmostEqual(0.0, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.966, self.item_stats.negative_correlation_stat, delta=1e-3)

    def test_eq(self):
        self.item_stats.update(on=True, success=True)
        self.item_stats.update(on=False, success=False)

        self.assertTrue(satisfies_equality_checks(obj=self.item_stats, other=ECItemStats()))

    def test_hash(self):
        self.item_stats.update(on=True, success=True)
        self.item_stats.update(on=False, success=False)

        self.assertTrue(satisfies_hash_checks(obj=self.item_stats))

    @test_share.performance_test
    def test_performance_equal(self):
        self.item_stats.update(on=True, success=True)
        self.item_stats.update(on=False, success=False)

        other = ECItemStats()

        n_iterations = 100_000

        elapsed_time = 0
        for _ in range(n_iterations):
            start = time()
            _ = self.item_stats == other
            end = time()
            elapsed_time += end - start

        print(f'Time for {n_iterations:,} calls to ECItemStats.__eq__ comparing unequal objects: {elapsed_time}s')

        elapsed_time = 0
        for _ in range(n_iterations):
            start = time()
            _ = self.item_stats == copy(self.item_stats)
            end = time()
            elapsed_time += end - start

        print(f'Time for {n_iterations:,} calls to ECItemStats.__eq__ comparing equal objects: {elapsed_time}s')

    @test_share.performance_test
    def test_performance_hash(self):
        self.item_stats.update(on=True, success=True)
        self.item_stats.update(on=False, success=False)

        n_iterations = 1_000_000

        elapsed_time = 0
        for _ in range(n_iterations):
            start = time()
            _ = hash(self.item_stats)
            end = time()
            elapsed_time += end - start

        print(f'Time for {n_iterations:,} calls to ECItemStats.__hash__: {elapsed_time}s')


class TestERItemStatistics(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.item_stats = ERItemStats()

    def test_init(self):
        self.assertEqual(self.item_stats.n_on_and_activated, 0)
        self.assertEqual(self.item_stats.n_on_and_not_activated, 0)
        self.assertEqual(self.item_stats.n_off_and_activated, 0)
        self.assertEqual(self.item_stats.n_off_and_not_activated, 0)

    def test_update(self):
        self.item_stats.update(on=True, activated=True, count=32)
        self.item_stats.update(on=True, activated=False, count=2)
        self.item_stats.update(on=False, activated=True, count=8)
        self.item_stats.update(on=False, activated=False, count=8)

        self.assertEqual(self.item_stats.n_on, 34)
        self.assertEqual(self.item_stats.n_off, 16)

        self.assertEqual(self.item_stats.n_on_and_activated, 32)
        self.assertEqual(self.item_stats.n_on_and_not_activated, 2)
        self.assertEqual(self.item_stats.n_off_and_activated, 8)
        self.assertEqual(self.item_stats.n_off_and_not_activated, 8)

    def test_eq(self):
        self.item_stats.update(on=True, activated=True)
        self.item_stats.update(on=False, activated=False)

        self.assertTrue(satisfies_equality_checks(obj=self.item_stats, other=ECItemStats()))

    def test_hash(self):
        self.item_stats.update(on=True, activated=True)
        self.item_stats.update(on=False, activated=False)

        self.assertTrue(satisfies_hash_checks(obj=self.item_stats))

    def test_drescher_correlation(self):
        GlobalParams().set('ext_result.correlation_test', DrescherCorrelationTest())

        self.item_stats.update(on=True, activated=True, count=12)
        self.item_stats.update(on=False, activated=True, count=7)
        self.item_stats.update(on=True, activated=False, count=3)
        self.item_stats.update(on=False, activated=False, count=8)

        self.assertAlmostEqual(0.6984, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.3362, self.item_stats.negative_correlation_stat, delta=1e-3)

    def test_barnard_correlation_1(self):
        GlobalParams().set('ext_result.correlation_test', BarnardExactCorrelationTest())

        self.item_stats.update(on=True, activated=True, count=12)
        self.item_stats.update(on=False, activated=True, count=7)
        self.item_stats.update(on=True, activated=False, count=3)
        self.item_stats.update(on=False, activated=False, count=8)

        self.assertAlmostEqual(0.966, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.0, self.item_stats.negative_correlation_stat, delta=1e-3)

    def test_barnard_correlation_2(self):
        GlobalParams().set('ext_result.correlation_test', BarnardExactCorrelationTest())

        self.item_stats.update(on=True, activated=True, count=3)
        self.item_stats.update(on=False, activated=True, count=8)
        self.item_stats.update(on=True, activated=False, count=12)
        self.item_stats.update(on=False, activated=False, count=7)

        self.assertAlmostEqual(0.0, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.966, self.item_stats.negative_correlation_stat, delta=1e-3)

    @test_share.performance_test
    def test_performance_equal(self):
        self.item_stats.update(on=True, activated=True)
        self.item_stats.update(on=False, activated=False)

        other = ECItemStats()

        n_iterations = 100_000

        elapsed_time = 0
        for _ in range(n_iterations):
            start = time()
            _ = self.item_stats == other
            end = time()
            elapsed_time += end - start

        print(f'Time for {n_iterations:,} calls to ERItemStats.__eq__ comparing unequal objects: {elapsed_time}s')

        elapsed_time = 0
        for _ in range(n_iterations):
            start = time()
            _ = self.item_stats == copy(self.item_stats)
            end = time()
            elapsed_time += end - start

        print(f'Time for {n_iterations:,} calls to ERItemStats.__eq__ comparing equal objects: {elapsed_time}s')

    @test_share.performance_test
    def test_performance_hash(self):
        self.item_stats.update(on=True, activated=True)
        self.item_stats.update(on=False, activated=False)

        n_iterations = 1_000_000

        elapsed_time = 0
        for _ in range(n_iterations):
            start = time()
            _ = hash(self.item_stats)
            end = time()
            elapsed_time += end - start

        print(f'Time for {n_iterations:,} calls to ERItemStats.__hash__: {elapsed_time}s')


class TestSchemaStats(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.ss = SchemaStats()

    def test_init(self):
        self.assertEqual(self.ss.n, 0)
        self.assertEqual(self.ss.n_activated, 0)
        self.assertEqual(self.ss.n_not_activated, 0)
        self.assertEqual(self.ss.n_success, 0)
        self.assertEqual(self.ss.n_fail, 0)

    def test_update(self):
        self.ss.update(activated=True, success=True, count=1)
        self.assertEqual(1, self.ss.n)
        self.assertEqual(1, self.ss.n_activated)
        self.assertEqual(1, self.ss.n_success)
        self.assertEqual(0, self.ss.n_fail)
        self.assertEqual(0, self.ss.n_not_activated)

        self.ss.update(activated=True, success=False, count=1)
        self.assertEqual(2, self.ss.n)
        self.assertEqual(2, self.ss.n_activated)
        self.assertEqual(1, self.ss.n_success)
        self.assertEqual(1, self.ss.n_fail)
        self.assertEqual(0, self.ss.n_not_activated)

        self.ss.update(activated=False, success=True, count=1)
        self.assertEqual(3, self.ss.n)
        self.assertEqual(2, self.ss.n_activated)
        self.assertEqual(1, self.ss.n_success)  # must be activated for success or fail
        self.assertEqual(1, self.ss.n_fail)  # must be activated for success or fail
        self.assertEqual(1, self.ss.n_not_activated)

        self.ss.update(activated=False, success=False, count=1)
        self.assertEqual(4, self.ss.n)
        self.assertEqual(2, self.ss.n_activated)
        self.assertEqual(1, self.ss.n_success)
        self.assertEqual(1, self.ss.n_fail)
        self.assertEqual(2, self.ss.n_not_activated)

    def test_n(self):
        # should always be updated by count
        self.ss.update(activated=True, success=True, count=1)
        self.ss.update(activated=True, success=False, count=1)
        self.ss.update(activated=False, success=True, count=1)

        # test with count > 1
        self.ss.update(activated=False, success=False, count=2)
        self.assertEqual(5, self.ss.n)

    def test_activated(self):
        self.ss.update(activated=True, success=True, count=1)
        self.ss.update(activated=True, success=False, count=1)
        self.assertEqual(2, self.ss.n_activated)

        self.ss.update(activated=False, success=True, count=1)
        self.ss.update(activated=False, success=False, count=1)
        self.assertEqual(2, self.ss.n_activated)

    def test_success(self):
        # must be activated to increment success
        self.ss.update(activated=True, success=True, count=1)
        self.ss.update(activated=True, success=False, count=1)
        self.assertEqual(1, self.ss.n_success)

        self.ss.update(activated=False, success=True, count=1)
        self.ss.update(activated=False, success=False, count=1)
        self.assertEqual(1, self.ss.n_success)


class TestReadOnlyECItemStats(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_update(self):
        item_stats = ReadOnlyECItemStats()
        self.assertRaises(NotImplementedError, lambda: item_stats.update(True, True))


class TestReadOnlyERItemStats(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_update(self):
        item_stats = ReadOnlyERItemStats()
        self.assertRaises(NotImplementedError, lambda: item_stats.update(True, True))
