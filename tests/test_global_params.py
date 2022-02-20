import unittest

from schema_mechanism.core import GlobalOption
from schema_mechanism.core import GlobalParams
from test_share.test_func import common_test_setup


class TestGlobalParams(unittest.TestCase):
    def setUp(self):
        common_test_setup()

        self.gp = GlobalParams()
        self.gp.reset()

    def test_learning_rate(self):
        # test: default learning rate
        self.assertEqual(GlobalParams.DEFAULT_LEARN_RATE, self.gp.learn_rate)

        # test: setter should accept values in the range [0.0, 1.0]
        try:
            for value in [0.0, 0.5, 1.0]:
                self.gp.learn_rate = value
                self.assertEqual(value, self.gp.learn_rate)
        except ValueError as e:
            self.fail(str(e))

        # test: setter should raise ValueError for any values outside of range [0.0, 1.0]
        for value in [-0.1, 1.1]:
            with self.assertRaises(ValueError):
                self.gp.learn_rate = value

    def test_options(self):
        # test: default options
        self.assertSetEqual(GlobalParams.DEFAULT_OPTIONS, self.gp.options)

        # test: setter should accept valid combinations of OptionalEnhancements
        requested_options = {
            GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA,
            GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE
        }
        self.gp.options = requested_options
        self.assertSetEqual(requested_options, self.gp.options)

        requested_options = {
            GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA,
        }
        self.gp.options = requested_options
        self.assertSetEqual(requested_options, self.gp.options)

        # test: EC_DEFER_TO_MORE_SPECIFIC automatically added if missing when EC_SUPPRESS_SPECIFIC_ON_MULTIPLE requested
        requested_options = {
            GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE
        }
        expected_options = {
            GlobalOption.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA,
            GlobalOption.EC_MOST_SPECIFIC_ON_MULTIPLE
        }
        self.gp.options = requested_options
        self.assertSetEqual(expected_options, self.gp.options)

    def test_singleton(self):
        self.assertIs(self.gp, GlobalParams())
