from unittest import TestCase

import numpy as np

from schema_mechanism.strategies.scaling import ScalingStrategy
from schema_mechanism.strategies.scaling import SigmoidScalingStrategy
from test_share.test_func import common_test_setup


class TestCommon(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.DEFAULT_ARRAY_SIZE = 10
        self.ONES = np.ones(self.DEFAULT_ARRAY_SIZE)
        self.ZEROS = np.zeros(self.DEFAULT_ARRAY_SIZE)
        self.NEG_INFINITY = -np.inf * self.ONES

    def assert_implements_scaling_strategy_interface(self, strategy: ScalingStrategy):
        # test: strategy should implement the ScalingStrategy interface
        self.assertTrue(isinstance(strategy, ScalingStrategy))

    def assert_returns_array_of_correct_length(self, strategy: ScalingStrategy):
        # test: strategy should return a numpy array equal in length to the given values array
        for length in range(0, 10):
            values = strategy.scale(values=np.ones(length))
            self.assertIsInstance(values, np.ndarray)
            self.assertEqual(length, len(values))


class TestSigmoidScalingStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.strategy = SigmoidScalingStrategy()

    def test_init(self):
        range_scale = 10.6
        vertical_shift = 1.1
        intercept = 2.5

        # test: attributes should be set to given values when provided to initializer
        strategy = SigmoidScalingStrategy(
            range_scale=range_scale,
            vertical_shift=vertical_shift,
            intercept=intercept
        )

        self.assertEqual(range_scale, strategy.range_scale)
        self.assertEqual(vertical_shift, strategy.vertical_shift)
        self.assertEqual(intercept, strategy.intercept)

        # test: check default initial values
        strategy = SigmoidScalingStrategy()

        self.assertEqual(2.0, strategy.range_scale)
        self.assertEqual(0.5, strategy.vertical_shift)
        self.assertEqual(0.0, strategy.intercept)

    def test_implements_scaling_strategy_interface(self):
        self.assert_implements_scaling_strategy_interface(self.strategy)

    def test_returns_array_of_correct_length(self):
        self.assert_returns_array_of_correct_length(self.strategy)

    def test_intercept(self):
        # test: scaled values should always be zero at intercept
        for intercept in np.linspace(-5.0, 5.0):
            strategy = SigmoidScalingStrategy(intercept=intercept)
            values = intercept * np.ones(10)
            scaled_values = strategy.scale(values)
            np.testing.assert_array_equal(np.zeros_like(values), scaled_values)

    def test_range_scale(self):
        for range_scale in np.linspace(1.0, 100.0):
            strategy = SigmoidScalingStrategy(range_scale=range_scale)

            values = np.array(np.linspace(-50, 50))
            scaled_values = strategy.scale(values)

            # test: scaled value range should be between -range scale/2.0 and range scale/2.0
            self.assertTrue(all(scaled_values <= range_scale / 2.0))
            self.assertTrue(all(scaled_values >= -range_scale / 2.0))

    def test_scaling_values(self):
        # test: all scaled values for given inputs should match expected values
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        expected_scaled_values = np.array([
            0,
            -0.462117,
            -0.761594,
            -0.905148,
            -0.964028,
            -0.986614,
            -0.995055,
            -0.998178,
            -0.999329,
            -0.999753,
            -0.999909
        ])

        actual_scaled_values = self.strategy.scale(values=values)
        np.testing.assert_allclose(expected_scaled_values, actual_scaled_values, rtol=1e-05)
