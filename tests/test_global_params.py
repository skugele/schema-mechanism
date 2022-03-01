import unittest

from schema_mechanism.core import CompositeItem
from schema_mechanism.core import GlobalParams
from schema_mechanism.core import SupportedFeature
from schema_mechanism.core import SymbolicItem
from schema_mechanism.core import Verbosity
from schema_mechanism.core import display_message
from schema_mechanism.core import is_feature_enabled
from test_share.test_classes import MockCompositeItem
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup


class TestGlobalParams(unittest.TestCase):
    def setUp(self):
        common_test_setup()

        self.gp = GlobalParams()
        self.gp.reset()

    def test_singleton(self):
        self.assertIs(self.gp, GlobalParams())

    def test_reliability_threshold(self):
        key = 'reliability_threshold'

        # test: value should be the default before updates
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: values between 0.0 and 1.0 exclusive should be accepted and returned
        self.gp.set(key, 0.0)
        self.assertEqual(0.0, self.gp.get(key))

        self.gp.set(key, 1.0)
        self.assertEqual(1.0, self.gp.get(key))

        # test: values NOT between 0.0 and 1.0 exclusive should be rejected
        for illegal_value in [-0.001, 1.001]:
            try:
                self.gp.set(key, illegal_value)
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_verbosity(self):
        key = 'verbosity'

        # test: value should be the default before updates
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: instances of Verbosity enum should be accepted and returned
        for value in Verbosity:
            self.gp.set(key, value)
            self.assertEqual(value, self.gp.get(key))

        # test: values that are not derived from Item class should be rejected
        for illegal_value in ['DEBUG', 0]:
            try:
                self.gp.set(key, illegal_value)
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_rng_seed(self):
        key = 'rng_seed'

        # test: value should be the default before updates
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: positive integer values should be accepted and returned
        self.gp.set(key, 12345)
        self.assertEqual(12345, self.gp.get(key))

        # test: values that are not integers should be rejected
        for illegal_value in [0.1, 'str']:
            try:
                self.gp.set(key, illegal_value)
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_dv_trace_max_len(self):
        key = 'dv_trace_max_len'

        # test: value should be the default before updates
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: integer values greater than 0 should be accepted and returned
        self.gp.set(key, 1)
        self.assertEqual(1, self.gp.get(key))

        self.gp.set(key, 100)
        self.assertEqual(100, self.gp.get(key))

        # test: values NOT between 0 and 1 and non-integer values should be rejected
        for illegal_value in [-1, 0.5, 'str']:
            try:
                self.gp.set(key, illegal_value)
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_item_type(self):
        key = 'item_type'

        # test: value should be the default before updates
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: instances of CompositeItem should be accepted and returned
        self.gp.set(key, SymbolicItem)
        self.assertEqual(SymbolicItem, self.gp.get(key))

        self.gp.set(key, MockSymbolicItem)
        self.assertEqual(MockSymbolicItem, self.gp.get(key))

        # test: values that are not derived from Item class should be rejected
        for illegal_value in [int, str]:
            try:
                self.gp.set(key, illegal_value)
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_composite_item_type(self):
        key = 'composite_item_type'

        # test: value should be the default before updates
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: instances of CompositeItem should be accepted and returned
        self.gp.set(key, CompositeItem)
        self.assertEqual(CompositeItem, self.gp.get(key))

        self.gp.set(key, MockCompositeItem)
        self.assertEqual(MockCompositeItem, self.gp.get(key))

        # test: values that are not derived from CompositeItem class should be rejected
        for illegal_value in [SymbolicItem, MockSymbolicItem, str]:
            try:
                self.gp.set(key, illegal_value)
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_output_format(self):
        key = 'output_format'

        # test: value should be the default before updates
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: instances of format strings should be accepted and returned
        expected_format = 'format string: {message} {timestamp}'
        self.gp.set(key, expected_format)
        self.assertEqual(expected_format, self.gp.get(key))

        # test: None should be accepted, which has the effect of turning off logging
        self.gp.set(key, None)
        self.assertEqual(None, self.gp.get(key))

        try:
            display_message('test', Verbosity.DEBUG)
        except RuntimeError as e:
            self.fail(f'Unexpected exception: {str(e)}')

        # test: any non-string value that is not None should raise a ValueError
        for illegal_value in [0, -1.1, list(), dict()]:
            try:
                self.gp.set(key, illegal_value)
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_pos_corr_threshold(self):
        key = 'positive_correlation_threshold'

        # test: value should be the default before updates
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: values between 0.0 and 1.0 exclusive should be accepted and returned
        self.gp.set(key, 0.0)
        self.assertEqual(0.0, self.gp.get(key))

        self.gp.set(key, 1.0)
        self.assertEqual(1.0, self.gp.get(key))

        # test: values NOT between 0.0 and 1.0 exclusive should raise a ValueError
        for illegal_value in [-0.001, 1.001]:
            try:
                self.gp.set(key, illegal_value)
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_neg_corr_threshold(self):
        key = 'negative_correlation_threshold'

        # test: value should be the default before updates
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: values between 0.0 and 1.0 exclusive should be accepted and returned
        self.gp.set(key, 0.0)
        self.assertEqual(0.0, self.gp.get(key))

        self.gp.set(key, 1.0)
        self.assertEqual(1.0, self.gp.get(key))

        # test: values NOT between 0.0 and 1.0 exclusive should raise a ValueError
        for illegal_value in [-0.001, 1.001]:
            try:
                self.gp.set(key, illegal_value)
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_learning_rate(self):
        key = 'learning_rate'

        # test: default learning rate
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: setter should accept values in the range [0.0, 1.0]
        try:
            for value in [0.0, 0.5, 1.0]:
                self.gp.set(key, value)
                self.assertEqual(value, self.gp.get(key))
        except ValueError as e:
            self.fail(str(e))

        # test: setter should raise ValueError for any values outside of range [0.0, 1.0]
        for value in [-0.1, 1.1]:
            with self.assertRaises(ValueError):
                self.gp.set(key, value)

    def test_reset(self):
        # setting arbitrary non-default values
        self.gp.set('composite_item_type', MockCompositeItem)
        self.gp.set('dv_trace_max_len', 17)
        self.gp.set('item_type', MockSymbolicItem)
        self.gp.set('learning_rate', 0.00001)
        self.gp.set('negative_correlation_threshold', 0.186)
        self.gp.set('features', [SupportedFeature.ER_INCREMENTAL_RESULTS])
        self.gp.set('output_format', '{message}')
        self.gp.set('positive_correlation_threshold', 0.176)
        self.gp.set('reliability_threshold', 0.72)
        self.gp.set('rng_seed', 123456)

        # test: reset should set values to their defaults
        self.gp.reset()

        for key in self.gp.defaults.keys():
            self.assertEqual(self.gp.defaults.get(key), self.gp.get(key))

    def test_features(self):
        key = 'features'

        # test: default features
        self.assertSetEqual(self.gp.defaults[key], self.gp.get(key))

        # test: remove all features
        self.gp.set(key, set())
        self.assertSetEqual(set(), self.gp.get(key))

        # test: setter should accept valid combinations of supported features
        for feature in SupportedFeature:

            # EC_MOST_SPECIFIC_ON_MULTIPLE requires EC_DEFER_TO_MORE_SPECIFIC_SCHEMA
            if feature is SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE:
                self.gp.set(key, {feature, SupportedFeature.EC_DEFER_TO_MORE_SPECIFIC_SCHEMA})
            else:
                self.gp.set(key, {feature})
            self.assertIn(feature, self.gp.get(key))

        # test: setter should accept combinations of features
        features = set(v for v in SupportedFeature)
        self.gp.set(key, features)
        self.assertSetEqual(features, self.gp.get(key))

        # test: EC_DEFER_TO_MORE_SPECIFIC is required when EC_MOST_SPECIFIC_ON_MULTIPLE (should raise a ValueError)
        self.assertRaises(ValueError, lambda: self.gp.set(key, {SupportedFeature.EC_MOST_SPECIFIC_ON_MULTIPLE}))

    def test_is_enabled(self):
        features = {SupportedFeature.EC_POSITIVE_ASSERTIONS_ONLY, SupportedFeature.ER_INCREMENTAL_RESULTS}
        self.gp.set('features', features)

        for feature in SupportedFeature:
            if feature in features:
                self.assertTrue(is_feature_enabled(feature))
            else:
                self.assertFalse(is_feature_enabled(feature))
