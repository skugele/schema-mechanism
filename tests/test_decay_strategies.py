import warnings
from unittest import TestCase

import numpy as np

from schema_mechanism.strategies.decay import DecayStrategy
from schema_mechanism.strategies.decay import ExponentialDecayStrategy
from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.decay import LinearDecayStrategy
from test_share.test_func import common_test_setup


class TestCommon(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.DEFAULT_ARRAY_SIZE = 10
        self.ONES = np.ones(self.DEFAULT_ARRAY_SIZE)
        self.NEG_INFINITY = -np.inf * self.ONES

    def assert_implements_decay_strategy_protocol(self, strategy: DecayStrategy):
        # test: strategy should implement the DecayStrategy protocol
        self.assertTrue(isinstance(strategy, DecayStrategy))

    def assert_returns_array_of_correct_length(self, strategy: DecayStrategy):
        # test: strategy should return a numpy array equal in length to the given values array
        for length in range(0, 10):
            values = strategy.decay(values=np.ones(length))
            self.assertIsInstance(values, np.ndarray)
            self.assertEqual(length, len(values))

    def assert_expected_general_behavior_from_decay_step_size(self, strategy: DecayStrategy):
        # test: step size of zero should return the same values (try with different initial values)
        for initial in np.linspace(0.0, 10.0):
            input_values = self.ONES
            output_values = strategy.decay(values=input_values, step_size=0.0)

            np.testing.assert_array_equal(input_values, output_values)

        # test: step size less than zero should raise a ValueError
        self.assertRaises(ValueError, lambda: strategy.decay(values=self.ONES, step_size=-0.1))

        # test: fractional step sizes should be supported and decrease values
        try:
            for step_size in np.linspace(1e-5, 1.0, endpoint=False):
                input_values = self.ONES
                output_values = strategy.decay(values=input_values, step_size=step_size)

                np.testing.assert_array_less(output_values, input_values)
        except Exception as e:
            self.fail(f'Raised unexpected exception: {e}')

        # test: step sizes greater than 1.0 should be supported and decrease values
        try:
            for step_size in np.linspace(1.0, 100.0, endpoint=False):
                input_values = self.ONES
                output_values = strategy.decay(values=input_values, step_size=step_size)

                np.testing.assert_array_less(output_values, input_values)
        except Exception as e:
            self.fail(f'Raised unexpected exception: {e}')

    def assert_decay_of_negative_infinity_input_produces_negative_infinity_output(self, strategy: DecayStrategy):
        # test: input array with values of -np.inf is allowed and returns an array of elements with values of -np.inf
        output_values = strategy.decay(self.NEG_INFINITY)
        np.testing.assert_array_equal(self.NEG_INFINITY, output_values)

    def assert_decay_default_step_size_is_one(self, strategy: DecayStrategy):
        # test: decay's default step size should be 1.0
        expected_values = strategy.decay(values=self.ONES, step_size=1.0)
        actual_values = strategy.decay(values=self.ONES)

        np.testing.assert_array_equal(expected_values, actual_values)

    def assert_decay_continues_to_limit(self, strategy: DecayStrategy, limit: float):
        # used to suppress an overflow warning
        warnings.filterwarnings('ignore')

        termination_values = limit * np.ones(self.DEFAULT_ARRAY_SIZE)

        values = self.ONES
        for _ in range(100_000_000):
            # terminate early if decay reaches limit
            if np.alltrue(values <= termination_values):
                break

            values = strategy.decay(values=values)

        self.assertTrue(np.alltrue(values <= termination_values))

    def assert_all_common_functionality(self, strategy: DecayStrategy) -> None:
        self.assert_implements_decay_strategy_protocol(strategy)
        self.assert_returns_array_of_correct_length(strategy)
        self.assert_expected_general_behavior_from_decay_step_size(strategy)
        self.assert_decay_of_negative_infinity_input_produces_negative_infinity_output(strategy)
        self.assert_decay_default_step_size_is_one(strategy)


class TestLinearDecayStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.strategy = LinearDecayStrategy(rate=0.25)

    def test_init(self):
        minimum = 0.0
        rate = 10.0

        strategy = LinearDecayStrategy(minimum=minimum, rate=rate)
        self.assertEqual(minimum, strategy.minimum)
        self.assertEqual(rate, strategy.rate)

        # test: rate must be strictly positive
        try:
            _ = LinearDecayStrategy(rate=1e-10)
            _ = LinearDecayStrategy(rate=1e10)
        except ValueError as e:
            self.fail(f'Unexpected ValueError raised by initializer: {str(e)}')

        self.assertRaises(ValueError, lambda: LinearDecayStrategy(minimum=0.0, rate=-1e-5))
        self.assertRaises(ValueError, lambda: LinearDecayStrategy(minimum=0.0, rate=0.0))

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_decay_produces_expected_results(self):
        # test: expected values should be returned from different step sizes
        expected_values = [1.0 - step * self.strategy.rate for step in range(0, 10)]

        for step_size, expected_value in enumerate(expected_values):
            output_values = self.strategy.decay(self.ONES, step_size=step_size)
            np.testing.assert_array_almost_equal(output_values, expected_value * np.ones_like(output_values))

    def test_decay_to_limit(self):
        # test: decay with minimum should decrease to limit
        self.assert_decay_continues_to_limit(strategy=LinearDecayStrategy(rate=0.01), limit=0.0)
        self.assert_decay_continues_to_limit(strategy=LinearDecayStrategy(rate=10.0), limit=-1e5)


class TestGeometricDecayStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.strategy = GeometricDecayStrategy(rate=0.5)

    def test_init(self):
        rate = 0.5

        # test: attributes should be set to given values when provided to initializer
        strategy = GeometricDecayStrategy(rate=rate)

        self.assertEqual(rate, strategy.rate)

        # test: rate must be between 0.0 and 1.0 (exclusive)
        try:
            _ = GeometricDecayStrategy(rate=1e-10)
            _ = GeometricDecayStrategy(rate=1.0 - 1e-10)
        except ValueError as e:
            self.fail(f'Unexpected ValueError raised by initializer: {str(e)}')

        self.assertRaises(ValueError, lambda: GeometricDecayStrategy(rate=1.0 + 1e-10))
        self.assertRaises(ValueError, lambda: GeometricDecayStrategy(rate=1.0))
        self.assertRaises(ValueError, lambda: GeometricDecayStrategy(rate=0.0))
        self.assertRaises(ValueError, lambda: GeometricDecayStrategy(rate=-1e-10))

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_decay_produces_expected_results(self):
        # test: expected values should be returned from different step sizes
        expected_values = [self.strategy.rate ** step for step in range(0, 10)]

        for step_size, expected_value in enumerate(expected_values):
            output_values = self.strategy.decay(self.ONES, step_size=step_size)
            np.testing.assert_array_almost_equal(output_values, expected_value * np.ones_like(output_values))

    def test_decay_to_zero(self):
        self.assert_decay_continues_to_limit(self.strategy, limit=0.0)


class TestExponentialDecayStrategy(TestCommon):
    def setUp(self) -> None:
        super().setUp()

        self.strategy = ExponentialDecayStrategy(rate=0.5)

    def test_init(self):
        rate = 0.5
        initial = 1.0
        minimum = 0.0

        # test: attributes should be set to given values when provided to initializer
        strategy = ExponentialDecayStrategy(rate=rate, minimum=minimum, initial=initial)

        self.assertEqual(rate, strategy.rate)
        self.assertEqual(initial, strategy.initial)
        self.assertEqual(minimum, strategy.minimum)

        # test: default initial value should be 1.0; default minimum value should be -np.inf
        strategy = ExponentialDecayStrategy(rate=rate)
        self.assertEqual(1.0, strategy.initial)
        self.assertEqual(-np.inf, strategy.minimum)

        # test: rate must be strictly greater than 0.0
        try:
            _ = ExponentialDecayStrategy(rate=1.0e-10)
            _ = ExponentialDecayStrategy(rate=100.0)
        except ValueError as e:
            self.fail(f'Unexpected ValueError raised by initializer: {str(e)}')

        self.assertRaises(ValueError, lambda: ExponentialDecayStrategy(rate=-1.0e10))
        self.assertRaises(ValueError, lambda: ExponentialDecayStrategy(rate=0.0))

    def test_common(self):
        self.assert_all_common_functionality(self.strategy)

    def test_decay_produces_expected_results(self):
        # test: should produce values equal to 1.0
        strategy = ExponentialDecayStrategy(initial=1.0, rate=0.5)

        input_values = 4.0 / 3.0 * self.ONES
        output_values = strategy.decay(values=input_values, step_size=1.0)

        np.testing.assert_array_almost_equal(output_values, np.ones_like(output_values))

        # test: expected values from different step sizes
        strategy = ExponentialDecayStrategy(initial=1.0, rate=0.01)

        expected_values = [
            1.0,  # step 0
            0.99,  # step 1
            0.9799,  # step 2
            0.969699,  # step 3
            0.959396,  # step 4
            0.94899,  # step 5
            0.93848,  # step 6
            0.927865,  # step 7
            0.917143,  # step 8
            0.906315,  # step 9
            0.895378  # step 10
        ]

        for step_size, expected_value in enumerate(expected_values):
            output_values = strategy.decay(self.ONES, step_size=step_size)
            np.testing.assert_array_almost_equal(output_values, expected_value * np.ones_like(output_values))

    def test_decay_without_limit(self):
        self.assert_decay_continues_to_limit(ExponentialDecayStrategy(initial=1.0, rate=2.0), limit=-np.inf)

    def test_decay_to_minimum(self):
        for minimum in np.linspace(-100.0, 1.0):
            self.assert_decay_continues_to_limit(
                ExponentialDecayStrategy(rate=2.0, initial=1.0, minimum=minimum), limit=minimum)
