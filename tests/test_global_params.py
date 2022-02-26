import unittest

from schema_mechanism.core import CompositeItem
from schema_mechanism.core import GlobalOption
from schema_mechanism.core import GlobalParams
from schema_mechanism.core import SymbolicItem
from schema_mechanism.core import Verbosity
from schema_mechanism.core import display_message
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
        # test: value should be the default before updates
        self.assertEqual(GlobalParams().DEFAULT_RELIABILITY_THRESHOLD, self.gp.reliability_threshold)

        # test: values between 0.0 and 1.0 inclusive should be accepted and returned
        self.gp.reliability_threshold = 0.0
        self.assertEqual(0.0, self.gp.reliability_threshold)

        self.gp.reliability_threshold = 1.0
        self.assertEqual(1.0, self.gp.reliability_threshold)

        # test: values NOT between 0.0 and 1.0 inclusive should be rejected
        for illegal_value in [-0.001, 1.001]:
            try:
                self.gp.reliability_threshold = illegal_value
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_verbosity(self):
        # test: value should be the default before updates
        self.assertEqual(GlobalParams().DEFAULT_VERBOSITY, self.gp.verbosity)

        # test: instances of Verbosity enum should be accepted and returned
        for value in Verbosity:
            self.gp.verbosity = value
            self.assertEqual(value, self.gp.verbosity)

        # test: values that are not derived from Item class should be rejected
        for illegal_value in ['DEBUG', 0]:
            try:
                self.gp.verbosity = illegal_value
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_rng_seed(self):
        # test: value should be the default before updates
        self.assertEqual(GlobalParams().DEFAULT_RNG_SEED, self.gp.rng_seed)

        # test: positive integer values should be accepted and returned
        self.gp.rng_seed = 12345
        self.assertEqual(12345, self.gp.rng_seed)

        # test: 0 or None should internally generate a new seed
        for value in [0, None]:
            self.gp.rng_seed = value
            self.assertIsInstance(self.gp.rng_seed, int)
            self.assertGreater(self.gp.rng_seed, 0)

        # test: values that are not integers should be rejected
        for illegal_value in [0.1, 'str']:
            try:
                self.gp.rng_seed = illegal_value
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_dv_trace_max_len(self):
        # test: value should be the default before updates
        self.assertEqual(GlobalParams().DEFAULT_DV_TRACE_MAX_LEN, self.gp.dv_trace_max_len)

        # test: integer values greater than 0 should be accepted and returned
        self.gp.dv_trace_max_len = 1
        self.assertEqual(1, self.gp.dv_trace_max_len)

        self.gp.dv_trace_max_len = 100
        self.assertEqual(100, self.gp.dv_trace_max_len)

        # test: values NOT between 0 and 1 and non-integer values should be rejected
        for illegal_value in [-1, 0, 0.5, 'str']:
            try:
                self.gp.dv_trace_max_len = illegal_value
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_item_type(self):
        # test: value should be the default before updates
        self.assertEqual(GlobalParams().DEFAULT_ITEM_TYPE, self.gp.item_type)

        # test: instances of CompositeItem should be accepted and returned
        self.gp.item_type = SymbolicItem
        self.assertEqual(SymbolicItem, self.gp.item_type)

        self.gp.item_type = MockSymbolicItem
        self.assertEqual(MockSymbolicItem, self.gp.item_type)

        # test: values that are not derived from Item class should be rejected
        for illegal_value in [int, str]:
            try:
                self.gp.item_type = illegal_value
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_composite_item_type(self):
        # test: value should be the default before updates
        self.assertEqual(GlobalParams().DEFAULT_COMPOSITE_ITEM_TYPE, self.gp.composite_item_type)

        # test: instances of CompositeItem should be accepted and returned
        self.gp.composite_item_type = CompositeItem
        self.assertEqual(CompositeItem, self.gp.composite_item_type)

        self.gp.composite_item_type = MockCompositeItem
        self.assertEqual(MockCompositeItem, self.gp.composite_item_type)

        # test: values that are not derived from CompositeItem class should be rejected
        for illegal_value in [SymbolicItem, MockSymbolicItem, str]:
            try:
                self.gp.composite_item_type = illegal_value
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_output_format(self):
        # test: value should be the default before updates
        self.assertEqual(GlobalParams().DEFAULT_OUTPUT_FORMAT, self.gp.output_format)

        # test: instances of format strings should be accepted and returned
        expected_format = 'format string: {message} {timestamp}'
        self.gp.output_format = expected_format
        self.assertEqual(expected_format, self.gp.output_format)

        # test: None should be accepted, which has the effect of turning off logging
        self.gp.output_format = None
        self.assertEqual(None, self.gp.output_format)

        try:
            display_message('test', Verbosity.DEBUG)
        except Exception as e:
            self.fail('Unexpected exception: {e)}')

        # test: any non-string value that is not None should raise a ValueError
        for illegal_value in [0, -1.1, list(), dict()]:
            try:
                self.gp.output_format = illegal_value
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_pos_corr_threshold(self):
        # test: value should be the default before updates
        self.assertEqual(GlobalParams().DEFAULT_POS_CORR_THRESHOLD, self.gp.pos_corr_threshold)

        # test: values between 0.0 and 1.0 inclusive should be accepted and returned
        self.gp.pos_corr_threshold = 0.0
        self.assertEqual(0.0, self.gp.pos_corr_threshold)

        self.gp.pos_corr_threshold = 1.0
        self.assertEqual(1.0, self.gp.pos_corr_threshold)

        # test: values NOT between 0.0 and 1.0 inclusive should raise a ValueError
        for illegal_value in [-0.001, 1.001]:
            try:
                self.gp.pos_corr_threshold = illegal_value
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

    def test_neg_corr_threshold(self):
        # test: value should be the default before updates
        self.assertEqual(GlobalParams().DEFAULT_NEG_CORR_THRESHOLD, self.gp.neg_corr_threshold)

        # test: values between 0.0 and 1.0 inclusive should be accepted and returned
        self.gp.neg_corr_threshold = 0.0
        self.assertEqual(0.0, self.gp.neg_corr_threshold)

        self.gp.neg_corr_threshold = 1.0
        self.assertEqual(1.0, self.gp.neg_corr_threshold)

        # test: values NOT between 0.0 and 1.0 inclusive should raise a ValueError
        for illegal_value in [-0.001, 1.001]:
            try:
                self.gp.neg_corr_threshold = illegal_value
                self.fail('Did not raise expected ValueError on illegal value')
            except ValueError:
                pass

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

    def test_reset(self):
        # setting arbitrary non-default values
        self.gp.composite_item_type = MockCompositeItem
        self.gp.dv_trace_max_len = 17
        self.gp.item_type = MockSymbolicItem
        self.gp.learn_rate = 0.00001
        self.gp.neg_corr_threshold = 0.186
        self.gp.options = [GlobalOption.ER_INCREMENTAL_RESULTS]
        self.gp.output_format = '{message}'
        self.gp.pos_corr_threshold = 0.176
        self.gp.reliability_threshold = 0.72
        self.gp.rng_seed = 123456

        # test: reset should set values to their defaults
        self.gp.reset()

        self.assertEqual(GlobalParams().DEFAULT_COMPOSITE_ITEM_TYPE, self.gp.composite_item_type)
        self.assertEqual(GlobalParams().DEFAULT_DV_TRACE_MAX_LEN, self.gp.dv_trace_max_len)
        self.assertEqual(GlobalParams().DEFAULT_ITEM_TYPE, self.gp.item_type)
        self.assertEqual(GlobalParams().DEFAULT_LEARN_RATE, self.gp.learn_rate)
        self.assertEqual(GlobalParams().DEFAULT_NEG_CORR_THRESHOLD, self.gp.neg_corr_threshold)
        self.assertEqual(GlobalParams().DEFAULT_OPTIONS, self.gp.options)
        self.assertEqual(GlobalParams().DEFAULT_OUTPUT_FORMAT, self.gp.output_format)
        self.assertEqual(GlobalParams().DEFAULT_POS_CORR_THRESHOLD, self.gp.pos_corr_threshold)
        self.assertEqual(GlobalParams().DEFAULT_RELIABILITY_THRESHOLD, self.gp.reliability_threshold)
        self.assertEqual(GlobalParams().DEFAULT_RNG_SEED, self.gp.rng_seed)

    def test_options(self):
        # suppress warnings
        GlobalParams().verbosity = Verbosity.ERROR

        # test: default options
        self.assertSetEqual(GlobalParams.DEFAULT_OPTIONS, self.gp.options)

        # test: setter should accept valid GlobalOptions
        for option in GlobalOption:
            self.gp.options = {option}
            self.assertIn(option, self.gp.options)

        # test: setter should accept combinations of options
        requested_options = set(v for v in GlobalOption)
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

    def test_is_enabled(self):
        enabled_options = {GlobalOption.EC_POSITIVE_ASSERTIONS_ONLY, GlobalOption.ER_INCREMENTAL_RESULTS}
        self.gp.options = enabled_options
        self.assertSetEqual(enabled_options, self.gp.options)

        for opt in set(opt for opt in GlobalOption).difference(enabled_options):
            self.assertNotIn(opt, self.gp.options)
