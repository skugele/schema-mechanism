from unittest import TestCase

from schema_mechanism.strategies.decay import GeometricDecayStrategy
from schema_mechanism.strategies.decay import LinearDecayStrategy
from test_share.test_func import common_test_setup


class TestLinearDecayStrategy(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_init(self):
        minimum = 0.0
        rate = 10.0

        strategy = LinearDecayStrategy(minimum=minimum, rate=rate)
        self.assertEqual(minimum, strategy.minimum)
        self.assertEqual(rate, strategy.rate)

        # test: rate must be strictly positive
        try:
            _ = LinearDecayStrategy(rate=0.000000000000001)
            _ = LinearDecayStrategy(rate=10_000_000_000)
        except ValueError as e:
            self.fail(f'Unexpected ValueError raised by initializer: {str(e)}')

        self.assertRaises(ValueError, lambda: LinearDecayStrategy(minimum=0.0, rate=-0.1))
        self.assertRaises(ValueError, lambda: LinearDecayStrategy(minimum=0.0, rate=0.0))

    def test_decay_without_minimum(self):
        # test: decay without minimum should decrease without bound
        strategy = LinearDecayStrategy(minimum=None, rate=1.0)
        self.assertEqual(-1_000_000_000_000, strategy.decay(value=0.0, count=1_000_000_000_000))

    def test_decay_with_minimum(self):
        # test: decay with minimum should decrease until minimum value reached, and then stay fixed at minimum
        strategy = LinearDecayStrategy(minimum=-1000, rate=1.0)
        self.assertEqual(-1000, strategy.decay(value=0.0, count=1_000_000_000_000))


class TestGeometricDecayStrategy(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_init(self):
        minimum = 0.0
        rate = 0.5

        strategy = GeometricDecayStrategy(minimum=minimum, rate=rate)
        self.assertEqual(minimum, strategy.minimum)
        self.assertEqual(rate, strategy.rate)

        # test: rate must be strictly greater than 0.0 and less than 1.0
        try:
            _ = GeometricDecayStrategy(rate=1e-10)
            _ = GeometricDecayStrategy(rate=1 - 1e-10)
        except ValueError as e:
            self.fail(f'Unexpected ValueError raised by initializer: {str(e)}')

        self.assertRaises(ValueError, lambda: GeometricDecayStrategy(minimum=0.0, rate=-0.1))
        self.assertRaises(ValueError, lambda: GeometricDecayStrategy(minimum=0.0, rate=0.0))
        self.assertRaises(ValueError, lambda: GeometricDecayStrategy(minimum=1.0, rate=0.0))

    def test_decay_without_minimum(self):
        # test: decay should decrease until convergence
        strategy = GeometricDecayStrategy(minimum=None, rate=0.5)
        self.assertAlmostEqual(0.0, strategy.decay(value=1024.0, count=1000))

    def test_decay_with_minimum(self):
        # test: decay with minimum should decrease until minimum value reached, and then stay fixed at minimum
        strategy = GeometricDecayStrategy(minimum=0.5, rate=0.5)
        self.assertAlmostEqual(0.5, strategy.decay(value=1024.0, count=1000))
