import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from schema_mechanism.core import Item
from schema_mechanism.core import Schema
from schema_mechanism.core import SupportedFeature
from schema_mechanism.core import is_feature_enabled
from schema_mechanism.share import GlobalParams
from schema_mechanism.share import Verbosity
from schema_mechanism.share import display_message
from schema_mechanism.stats import BarnardExactCorrelationTest
from schema_mechanism.stats import DrescherCorrelationTest
from schema_mechanism.stats import FisherExactCorrelationTest
from test_share.test_func import common_test_setup
from test_share.test_func import file_was_written
from test_share.test_func import serialize_enforces_overwrite_protection


class TestGlobalParams(unittest.TestCase):
    def setUp(self):
        common_test_setup()

        self.gp: GlobalParams = GlobalParams()
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

    def test_max_reliability_penalty(self):
        key = 'goal_pursuit_strategy.reliability.max_penalty'

        # test: value should be the default before updates
        self.assertEqual(self.gp.defaults[key], self.gp.get(key))

        # test: float values greater than 0.0 should be accepted and returned
        self.gp.set(key, 0.001)
        self.assertEqual(0.001, self.gp.get(key))

        self.gp.set(key, 100)
        self.assertEqual(100, self.gp.get(key))

        # test: values less than or equal to 0.0 should be rejected
        for illegal_value in [0.0, -0.0001, -100]:
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

    def test_correlation_thresholds(self):
        # test: correlation thresholds
        ##############################
        keys = [
            'ext_context.positive_correlation_threshold',
            'ext_context.negative_correlation_threshold',
            'ext_result.positive_correlation_threshold',
            'ext_result.negative_correlation_threshold',
        ]

        for key in keys:
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

    def test_correlation_tests(self):
        # test: correlation tests
        #########################
        keys = [
            'ext_context.correlation_test',
            'ext_result.correlation_test',
        ]

        for key in keys:

            # test: subclasses of ItemCorrelationTest should be allowed
            for legal_value in [DrescherCorrelationTest, FisherExactCorrelationTest, BarnardExactCorrelationTest]:
                try:
                    self.gp.set(key, legal_value)
                    self.assertEqual(legal_value, self.gp.get(key))
                except ValueError as e:
                    self.fail(f'Received unexpected ValueError: {e}')

            # test: values that are NOT subclasses of ItemCorrelationTest should raise a ValueError
            for illegal_value in [Schema, Item, 1.0, 'bad value']:
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
        self.gp.set('learning_rate', 0.00001)
        self.gp.set('ext_context.negative_correlation_threshold', 0.186)
        self.gp.set('features', [SupportedFeature.ER_INCREMENTAL_RESULTS])
        self.gp.set('output_format', '{message}')
        self.gp.set('ext_result.positive_correlation_threshold', 0.176)
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

    def test_defaults(self):
        self.assertEqual('{timestamp} [{severity}]\t{message}', self.gp.defaults['output_format'])
        self.assertEqual(0.01, self.gp.defaults['learning_rate'])
        self.assertEqual(0.4, self.gp.defaults['schema_selection.weights.explore_weight'])
        self.assertEqual(0.6, self.gp.defaults['schema_selection.weights.goal_weight'])
        self.assertEqual(0.95, self.gp.defaults['ext_context.negative_correlation_threshold'])
        self.assertEqual(0.95, self.gp.defaults['ext_context.positive_correlation_threshold'])
        self.assertEqual(0.95, self.gp.defaults['ext_result.negative_correlation_threshold'])
        self.assertEqual(0.95, self.gp.defaults['ext_result.positive_correlation_threshold'])
        self.assertEqual(0.95, self.gp.defaults['reliability_threshold'])
        self.assertLessEqual(0, self.gp.defaults['rng_seed'])

    def test_save_and_load(self):
        self.assertTrue(serialize_enforces_overwrite_protection(self.gp))

        # sets a few non-default values
        non_default_params = {
            'learning_rate': 0.00001,
            'ext_context.negative_correlation_threshold': 0.186,
            'features': [SupportedFeature.ER_INCREMENTAL_RESULTS],
            'output_format': '{message}',
            'ext_result.positive_correlation_threshold': 0.176,
            'reliability_threshold': 0.72,
            'rng_seed': 123456,
        }

        for key, value in non_default_params.items():
            self.gp.set(key, value)

        expected_dict = dict()
        expected_dict.update(self.gp.defaults)
        expected_dict.update(non_default_params)

        # sanity check
        self.assertDictEqual(self.gp.params, expected_dict)

        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-global_params-save_and_load.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            self.gp.save(path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            # clear non-default parameters
            self.gp.reset()

            # sanity check: only defaults
            self.assertDictEqual(self.gp.params, self.gp.defaults)

            self.gp.load(path)

            # test: non-default global params SHOULD be restored after load
            self.assertDictEqual(self.gp.params, expected_dict)
