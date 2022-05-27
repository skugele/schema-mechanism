from copy import copy
from copy import deepcopy
from time import time
from unittest import TestCase

import numpy as np

import test_share
from schema_mechanism.core import ECItemStats
from schema_mechanism.core import ERItemStats
from schema_mechanism.core import FrozenECItemStats
from schema_mechanism.core import FrozenERItemStats
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import ItemPool
from schema_mechanism.core import ReadOnlyECItemStats
from schema_mechanism.core import ReadOnlyERItemStats
from schema_mechanism.core import ReadOnlySchemaStats
from schema_mechanism.core import SchemaStats
from schema_mechanism.core import get_global_params
from schema_mechanism.core import get_global_stats
from schema_mechanism.core import set_global_stats
from schema_mechanism.func_api import sym_state
from schema_mechanism.serialization.json.decoders import decode
from schema_mechanism.serialization.json.encoders import encode
from schema_mechanism.strategies.correlation_test import BarnardExactCorrelationTest
from schema_mechanism.strategies.correlation_test import DrescherCorrelationTest
from schema_mechanism.util import repr_str
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestGlobalStats(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.global_stats = GlobalStats()
        self.global_stats._n = 10
        self.global_stats._baseline_value = 0.156

    def test_init(self):

        # test: attributes should be properly set if explicitly given to initializer
        n = 10
        baseline_value = -0.56

        global_stats = GlobalStats(n=n, baseline_value=baseline_value)

        self.assertEqual(n, global_stats.n)
        self.assertEqual(baseline_value, global_stats.baseline_value)

        # test: defaults should be properly set if values not given to initializer
        default_n = 0
        default_baseline_value = 0.0

        global_stats = GlobalStats()

        self.assertEqual(default_n, global_stats.n)
        self.assertEqual(default_baseline_value, global_stats.baseline_value)

    def test_global_instance_get_and_set_functions(self):
        stats = GlobalStats()
        set_global_stats(stats)

        self.assertEqual(stats, get_global_stats())

    def test_equals(self):
        stats = GlobalStats()
        stats.baseline_value = 1.23
        stats.n = 4

        other = GlobalStats()
        other.baseline_value = 4.56
        other.n = 17

        self.assertTrue(satisfies_equality_checks(obj=stats, other=other, other_different_type=1.0))

    def test_update(self):
        # setting learning rate to 1.0 to simplify testing (baseline value update depends on learning rate)
        params = get_global_params()
        params.set('learning_rate', 1.0)

        pool: ItemPool = ItemPool()
        for source in ('1', '2', '3', '4'):
            _ = pool.get(source, primitive_value=10.0)

        stats = GlobalStats()

        n: int = stats.n

        for _ in range(10):
            stats.update(selection_state=sym_state('1'), result_state=(sym_state('2')))

            # test: the number of selection events (n) should be increment during each update
            self.assertEqual(n + 1, stats.n)
            n = stats.n

    def test_reset(self):
        stats = GlobalStats()
        stats.n = 100.0
        stats.baseline_value = 7.25

        # test: clear should reset values to zero
        stats.reset()
        self.assertEqual(0, stats.n)
        self.assertEqual(0.0, stats.baseline_value)

    def test_encode_and_decode(self):
        encoded_obj = encode(self.global_stats)
        decoded_obj: GlobalStats = decode(encoded_obj)

        self.assertEqual(self.global_stats, decoded_obj)


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
        self.item_stats.update(on=False, success=False, count=10)
        self.item_stats.update(on=True, success=False, count=10)
        self.assertEqual(0.5, self.item_stats.specificity)

    def test_thresholds(self):
        params = get_global_params()

        for positive_correlation_threshold in np.linspace(0.0, 1.0, num=25):
            for negative_correlation_threshold in np.linspace(0.0, 1.0, num=25):
                params.set('ext_context.positive_correlation_threshold', positive_correlation_threshold)
                params.set('ext_context.negative_correlation_threshold', negative_correlation_threshold)

                # test: positive correlation threshold should be equal to the global parameter value
                self.assertEqual(positive_correlation_threshold, self.item_stats.positive_correlation_threshold)

                # test: negative correlation threshold should be equal to the global parameter value
                self.assertEqual(negative_correlation_threshold, self.item_stats.negative_correlation_threshold)

    def test_update(self):
        self.item_stats.update(on=True, success=True, count=32)
        self.item_stats.update(on=True, success=False, count=8)
        self.item_stats.update(on=False, success=True, count=2)
        self.item_stats.update(on=False, success=False, count=8)

        self.assertEqual(self.item_stats.n_success_and_on, 32)
        self.assertEqual(self.item_stats.n_success_and_off, 2)
        self.assertEqual(self.item_stats.n_fail_and_on, 8)
        self.assertEqual(self.item_stats.n_fail_and_off, 8)
        self.assertEqual(self.item_stats.n_on, 40)
        self.assertEqual(self.item_stats.n_off, 10)

    def test_reset(self):
        self.item_stats.update(on=True, success=True, count=32)
        self.item_stats.update(on=True, success=False, count=8)
        self.item_stats.update(on=False, success=True, count=2)
        self.item_stats.update(on=False, success=False, count=8)

        self.item_stats.reset()

        self.assertEqual(self.item_stats.n, 0)
        self.assertEqual(self.item_stats.n_on, 0)
        self.assertEqual(self.item_stats.n_off, 0)
        self.assertEqual(self.item_stats.n_success_and_on, 0)
        self.assertEqual(self.item_stats.n_success_and_off, 0)
        self.assertEqual(self.item_stats.n_fail_and_on, 0)
        self.assertEqual(self.item_stats.n_fail_and_off, 0)

    def test_drescher_correlation(self):
        params = get_global_params()
        params.set('ext_context.correlation_test', DrescherCorrelationTest())

        self.item_stats.update(on=True, success=True, count=12)
        self.item_stats.update(on=True, success=False, count=7)
        self.item_stats.update(on=False, success=True, count=3)
        self.item_stats.update(on=False, success=False, count=8)

        # results include a +1 increment to each of the above statistics
        self.assertAlmostEqual(0.6984, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.3362, self.item_stats.negative_correlation_stat, delta=1e-3)

    def test_barnard_correlation_1(self):
        params = get_global_params()
        params.set('ext_context.correlation_test', BarnardExactCorrelationTest())

        self.item_stats.update(on=True, success=True, count=12)
        self.item_stats.update(on=True, success=False, count=7)
        self.item_stats.update(on=False, success=True, count=3)
        self.item_stats.update(on=False, success=False, count=8)

        # results include a +1 increment to each of the above statistics
        self.assertAlmostEqual(0.966, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.0, self.item_stats.negative_correlation_stat, delta=1e-3)

    def test_barnard_correlation_2(self):
        params = get_global_params()
        params.set('ext_context.correlation_test', BarnardExactCorrelationTest())

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

        self.assertTrue(satisfies_equality_checks(obj=self.item_stats, other=ECItemStats(), other_different_type=1.0))

    def test_hash(self):
        self.item_stats.update(on=True, success=True)
        self.item_stats.update(on=False, success=False)

        self.assertTrue(satisfies_hash_checks(obj=self.item_stats))

    def test_str(self):
        attr_values = (
            f'sc: {self.item_stats.positive_correlation_stat:.2}',
            f'fc: {self.item_stats.negative_correlation_stat:.2}',
        )

        expected_str = f'{self.item_stats.__class__.__name__}[{"; ".join(attr_values)}]'
        self.assertEqual(expected_str, str(self.item_stats))

    def test_repr(self):
        attr_values = {
            'success_corr': f'{self.item_stats.positive_correlation_stat:.2}',
            'failure_corr': f'{self.item_stats.negative_correlation_stat:.2}',
            'n_on': f'{self.item_stats.n_on:,}',
            'n_off': f'{self.item_stats.n_off:,}',
            'n_success_and_on': f'{self.item_stats.n_success_and_on:,}',
            'n_success_and_off': f'{self.item_stats.n_success_and_off:,}',
            'n_fail_and_on': f'{self.item_stats.n_fail_and_on:,}',
            'n_fail_and_off': f'{self.item_stats.n_fail_and_off:,}',
        }

        expected_repr = repr_str(self.item_stats, attr_values)
        self.assertEqual(expected_repr, repr(self.item_stats))

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
        self.assertEqual(self.item_stats.n_activated, 40)
        self.assertEqual(self.item_stats.n_not_activated, 10)

    def test_reset(self):
        self.item_stats.update(on=True, activated=True, count=32)
        self.item_stats.update(on=True, activated=False, count=2)
        self.item_stats.update(on=False, activated=True, count=8)
        self.item_stats.update(on=False, activated=False, count=8)

        self.item_stats.reset()

        self.assertEqual(self.item_stats.n, 0)
        self.assertEqual(self.item_stats.n_on, 0)
        self.assertEqual(self.item_stats.n_off, 0)
        self.assertEqual(self.item_stats.n_on_and_activated, 0)
        self.assertEqual(self.item_stats.n_on_and_not_activated, 0)
        self.assertEqual(self.item_stats.n_off_and_activated, 0)
        self.assertEqual(self.item_stats.n_off_and_not_activated, 0)
        self.assertEqual(self.item_stats.n_activated, 0)
        self.assertEqual(self.item_stats.n_not_activated, 0)

    def test_eq(self):
        self.item_stats.update(on=True, activated=True)
        self.item_stats.update(on=False, activated=False)

        self.assertTrue(satisfies_equality_checks(obj=self.item_stats, other=ECItemStats(), other_different_type=1.0))

    def test_hash(self):
        self.item_stats.update(on=True, activated=True)
        self.item_stats.update(on=False, activated=False)

        self.assertTrue(satisfies_hash_checks(obj=self.item_stats))

    def test_str(self):

        attr_values = (
            f'ptc: {self.item_stats.positive_correlation_stat:.2}',
            f'ntc: {self.item_stats.negative_correlation_stat:.2}',
        )

        expected_str = f'{self.item_stats.__class__.__name__}[{"; ".join(attr_values)}]'
        self.assertEqual(expected_str, str(self.item_stats))

    def test_repr(self):
        attr_values = {
            'positive_transition_corr': f'{self.item_stats.positive_correlation_stat:.2}',
            'negative_transition_corr': f'{self.item_stats.negative_correlation_stat:.2}',
            'n_on': f'{self.item_stats.n_on:,}',
            'n_off': f'{self.item_stats.n_off:,}',
            'n_on_and_activated': f'{self.item_stats.n_on_and_activated:,}',
            'n_on_and_not_activated': f'{self.item_stats.n_on_and_not_activated:,}',
            'n_off_and_activated': f'{self.item_stats.n_off_and_activated:,}',
            'n_off_and_not_activated': f'{self.item_stats._n_off_and_not_activated:,}',
        }

        expected_repr = repr_str(self.item_stats, attr_values)
        self.assertEqual(expected_repr, repr(self.item_stats))

    def test_drescher_correlation(self):
        params = get_global_params()
        params.set('ext_result.correlation_test', DrescherCorrelationTest())

        self.item_stats.update(on=True, activated=True, count=12)
        self.item_stats.update(on=False, activated=True, count=7)
        self.item_stats.update(on=True, activated=False, count=3)
        self.item_stats.update(on=False, activated=False, count=8)

        self.assertAlmostEqual(0.6984, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.3362, self.item_stats.negative_correlation_stat, delta=1e-3)

    def test_barnard_correlation_1(self):
        params = get_global_params()
        params.set('ext_result.correlation_test', BarnardExactCorrelationTest())

        self.item_stats.update(on=True, activated=True, count=12)
        self.item_stats.update(on=False, activated=True, count=7)
        self.item_stats.update(on=True, activated=False, count=3)
        self.item_stats.update(on=False, activated=False, count=8)

        self.assertAlmostEqual(0.966, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.0, self.item_stats.negative_correlation_stat, delta=1e-3)

    def test_barnard_correlation_2(self):
        params = get_global_params()
        params.set('ext_result.correlation_test', BarnardExactCorrelationTest())

        self.item_stats.update(on=True, activated=True, count=3)
        self.item_stats.update(on=False, activated=True, count=8)
        self.item_stats.update(on=True, activated=False, count=12)
        self.item_stats.update(on=False, activated=False, count=7)

        self.assertAlmostEqual(0.0, self.item_stats.positive_correlation_stat, delta=1e-3)
        self.assertAlmostEqual(0.966, self.item_stats.negative_correlation_stat, delta=1e-3)

    def test_thresholds(self):
        params = get_global_params()

        for positive_correlation_threshold in np.linspace(0.0, 1.0, num=25):
            for negative_correlation_threshold in np.linspace(0.0, 1.0, num=25):
                params.set('ext_result.positive_correlation_threshold', positive_correlation_threshold)
                params.set('ext_result.negative_correlation_threshold', negative_correlation_threshold)

                # test: positive correlation threshold should be equal to the global parameter value
                self.assertEqual(positive_correlation_threshold, self.item_stats.positive_correlation_threshold)

                # test: negative correlation threshold should be equal to the global parameter value
                self.assertEqual(negative_correlation_threshold, self.item_stats.negative_correlation_threshold)

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

        self.schema_stats = SchemaStats()

    def test_init(self):
        # test: initializer should set attribute values correctly when explicitly sent to initializer
        n = 125
        n_activated = 100
        n_success = 17

        schema_stats = SchemaStats(
            n=n,
            n_activated=n_activated,
            n_success=n_success,
        )

        self.assertEqual(n, schema_stats.n)
        self.assertEqual(n_activated, schema_stats.n_activated)
        self.assertEqual(n_success, schema_stats.n_success)

        schema_stats = SchemaStats()

        # test: initializer should set default values correctly
        n_default = 0
        n_activated_default = 0
        n_success_default = 0

        self.assertEqual(n_default, schema_stats.n)
        self.assertEqual(n_activated_default, schema_stats.n_activated)
        self.assertEqual(n_success_default, schema_stats.n_success)

    def test_update(self):
        self.schema_stats.update(activated=True, success=True, count=1)
        self.assertEqual(1, self.schema_stats.n)
        self.assertEqual(1, self.schema_stats.n_activated)
        self.assertEqual(1, self.schema_stats.n_success)
        self.assertEqual(0, self.schema_stats.n_fail)
        self.assertEqual(0, self.schema_stats.n_not_activated)

        self.schema_stats.update(activated=True, success=False, count=1)
        self.assertEqual(2, self.schema_stats.n)
        self.assertEqual(2, self.schema_stats.n_activated)
        self.assertEqual(1, self.schema_stats.n_success)
        self.assertEqual(1, self.schema_stats.n_fail)
        self.assertEqual(0, self.schema_stats.n_not_activated)

        self.schema_stats.update(activated=False, success=True, count=1)
        self.assertEqual(3, self.schema_stats.n)
        self.assertEqual(2, self.schema_stats.n_activated)
        self.assertEqual(1, self.schema_stats.n_success)  # must be activated for success or fail
        self.assertEqual(1, self.schema_stats.n_fail)  # must be activated for success or fail
        self.assertEqual(1, self.schema_stats.n_not_activated)

        self.schema_stats.update(activated=False, success=False, count=1)
        self.assertEqual(4, self.schema_stats.n)
        self.assertEqual(2, self.schema_stats.n_activated)
        self.assertEqual(1, self.schema_stats.n_success)
        self.assertEqual(1, self.schema_stats.n_fail)
        self.assertEqual(2, self.schema_stats.n_not_activated)

    def test_n(self):
        # should always be updated by count
        self.schema_stats.update(activated=True, success=True, count=1)
        self.schema_stats.update(activated=True, success=False, count=1)
        self.schema_stats.update(activated=False, success=True, count=1)

        # test with count > 1
        self.schema_stats.update(activated=False, success=False, count=2)
        self.assertEqual(5, self.schema_stats.n)

    def test_activated(self):
        self.schema_stats.update(activated=True, success=True, count=1)
        self.schema_stats.update(activated=True, success=False, count=1)
        self.assertEqual(2, self.schema_stats.n_activated)

        self.schema_stats.update(activated=False, success=True, count=1)
        self.schema_stats.update(activated=False, success=False, count=1)
        self.assertEqual(2, self.schema_stats.n_activated)

    def test_success(self):
        # must be activated to increment success
        self.schema_stats.update(activated=True, success=True, count=1)
        self.schema_stats.update(activated=True, success=False, count=1)
        self.assertEqual(1, self.schema_stats.n_success)

        self.schema_stats.update(activated=False, success=True, count=1)
        self.schema_stats.update(activated=False, success=False, count=1)
        self.assertEqual(1, self.schema_stats.n_success)

    def test_derived_values(self):
        schema_stats = SchemaStats(
            n=123,
            n_success=67,
            n_activated=95,
        )

        # test: n_not_activated should equal n_activated - n_success
        self.assertEqual(schema_stats.n_activated - schema_stats.n_success, schema_stats.n_fail)

        # test: n_fail should equal n - n_activated
        self.assertEqual(schema_stats.n - schema_stats.n_activated, schema_stats.n_not_activated)

    def test_encode_and_decode(self):
        schema_stats = SchemaStats(
            n=123,
            n_success=84,
            n_activated=5,
        )
        encoded_obj = encode(schema_stats)
        decoded_obj: SchemaStats = decode(encoded_obj)

        self.assertEqual(schema_stats, decoded_obj)


class TestReadOnlySchemaStats(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_update(self):
        schema_stats = ReadOnlySchemaStats()
        self.assertRaises(NotImplementedError, lambda: schema_stats.update(selection_state=sym_state('1,2,3'),
                                                                           result_state=sym_state('4,5,6')))


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


class TestFrozenECItemStats(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.item_stats = FrozenECItemStats()

    def test_update(self):
        # test: frozen item stats should not change after update
        item_stats_before_update = deepcopy(self.item_stats)

        self.item_stats.update(on=True, success=True)
        self.item_stats.update(on=True, success=False)
        self.item_stats.update(on=False, success=True)
        self.item_stats.update(on=False, success=False)

        item_stats_after_update = deepcopy(self.item_stats)

        self.assertEqual(item_stats_before_update, item_stats_after_update)

    def test_str(self):
        self.assertEqual('FROZEN', str(self.item_stats))


class TestFrozenERItemStats(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.item_stats = FrozenERItemStats()

    def test_update(self):
        item_stats_before_update = deepcopy(self.item_stats)

        self.item_stats.update(on=True, activated=True)
        self.item_stats.update(on=True, activated=False)
        self.item_stats.update(on=False, activated=True)
        self.item_stats.update(on=False, activated=False)

        item_stats_after_update = deepcopy(self.item_stats)

        self.assertEqual(item_stats_before_update, item_stats_after_update)

    def test_str(self):
        self.assertEqual('FROZEN', str(self.item_stats))
