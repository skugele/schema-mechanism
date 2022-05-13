import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from schema_mechanism.core import SupportedFeature
from schema_mechanism.core import get_global_params
from schema_mechanism.core import is_feature_enabled
from schema_mechanism.core import set_global_params
from schema_mechanism.persistence import deserialize
from schema_mechanism.persistence import serialize
from schema_mechanism.share import GlobalParams
from test_share.test_func import common_test_setup
from test_share.test_func import file_was_written


class TestGlobalParams(unittest.TestCase):
    def setUp(self):
        common_test_setup()

        self.gp: GlobalParams = get_global_params()
        self.gp.reset()

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
        features = {SupportedFeature.ER_INCREMENTAL_RESULTS}
        self.gp.set('features', features)

        for feature in SupportedFeature:
            if feature in features:
                self.assertTrue(is_feature_enabled(feature))
            else:
                self.assertFalse(is_feature_enabled(feature))

    def test_defaults(self):
        self.assertEqual(0.01, self.gp.defaults['learning_rate'])
        self.assertEqual(0.95, self.gp.defaults['ext_context.negative_correlation_threshold'])
        self.assertEqual(0.95, self.gp.defaults['ext_context.positive_correlation_threshold'])
        self.assertEqual(0.95, self.gp.defaults['ext_result.negative_correlation_threshold'])
        self.assertEqual(0.95, self.gp.defaults['ext_result.positive_correlation_threshold'])
        self.assertEqual(0.95, self.gp.defaults['reliability_threshold'])
        self.assertLessEqual(0, self.gp.defaults['rng_seed'])

    def test_serialize(self):
        # sets a few non-default values
        non_default_params = {
            'learning_rate': 0.00001,
            'ext_context.negative_correlation_threshold': 0.186,
            'features': [SupportedFeature.ER_INCREMENTAL_RESULTS],
            'ext_result.positive_correlation_threshold': 0.176,
            'reliability_threshold': 0.72,
            'rng_seed': 123456,
        }

        params = GlobalParams()
        for key, value in non_default_params.items():
            params.set(key, value)

        expected_dict = dict()
        expected_dict.update(params.defaults)
        expected_dict.update(non_default_params)

        # sanity check
        self.assertDictEqual({k: v for k, v in params}, expected_dict)

        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-global_params-save_and_load.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            serialize(params, path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            recovered: GlobalParams = deserialize(path)

            # test: non-default global params SHOULD be restored after load
            self.assertEqual(params, recovered)

    def test_global_params_accessor_functions(self):
        params = GlobalParams()
        set_global_params(params)

        self.assertEqual(params, get_global_params())
