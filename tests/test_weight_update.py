from unittest import TestCase

import numpy as np

from schema_mechanism.strategies.weight_update import CyclicWeightUpdateStrategy
from schema_mechanism.strategies.weight_update import NoOpWeightUpdateStrategy
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks


class TestNoOpWeightUpdateStrategy(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_update(self):
        strategy = NoOpWeightUpdateStrategy()

        initial_weights = [
            np.array([0.0, 1.0]),
            np.array([0.25, 0.75]),
            np.array([0.5, 0.5]),
            np.array([0.75, 0.25]),
            np.array([0.0, 1.0]),
        ]

        for weights in initial_weights:
            new_weights = strategy.update(weights)

            # test: the updated weights must always equal 1.0
            np.testing.assert_array_equal(weights, new_weights)

    def test_equals(self):
        self.assertTrue(satisfies_equality_checks(NoOpWeightUpdateStrategy(), other_different_type='other'))


class TestOscillatoryWeightUpdateStrategy(TestCase):

    def test_init(self):
        step_size = 0.123

        strategy = CyclicWeightUpdateStrategy(step_size=step_size)
        self.assertEqual(step_size, strategy.step_size)

        # test: step sizes <= 0 and step sizes >= 1 should result in a ValueError
        self.assertRaises(ValueError, lambda: CyclicWeightUpdateStrategy(step_size=0.0))
        self.assertRaises(ValueError, lambda: CyclicWeightUpdateStrategy(step_size=1.0))
        self.assertRaises(ValueError, lambda: CyclicWeightUpdateStrategy(step_size=-0.1))
        self.assertRaises(ValueError, lambda: CyclicWeightUpdateStrategy(step_size=1.1))

    def test_update(self):
        weights = np.array([0.5, 0.5])

        strategy = CyclicWeightUpdateStrategy(step_size=0.01)

        weights_list = [weights]
        for _ in range(1000):
            new_weights = strategy.update(weights_list[-1])

            # test: the updated weights must always equal 1.0
            self.assertEqual(sum(new_weights), 1.0)

            # break from loop when new weights are equal to the initial weights
            if np.array_equal(new_weights, weights_list[0]):
                break

            weights_list.append(new_weights)

        # test: with step size of 0.01, the weights should return to their initial value after 100 updates
        self.assertEqual(len(weights_list), 100)

        # test: step_sizes that do not evenly divide the range [0,1] should be supported
        strategy = CyclicWeightUpdateStrategy(step_size=0.1367)
        for _ in range(1000):
            new_weights = strategy.update(weights_list[-1])

            # test: the updated weights must always equal 1.0
            self.assertEqual(sum(new_weights), 1.0)

        # test: weight array is currently limited to length of 2. Other lengths should generate ValueErrors
        self.assertRaises(ValueError, lambda: strategy.update(np.array([0.1, 0.8, 0.1])))

    def test_equals(self):
        strategy = CyclicWeightUpdateStrategy(step_size=0.123)
        other = CyclicWeightUpdateStrategy(step_size=0.25)

        satisfies_equality_checks(strategy, other_same_type=other, other_different_type=13)
