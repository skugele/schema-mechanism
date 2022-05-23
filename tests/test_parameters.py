import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from schema_mechanism.core import SupportedFeature
from schema_mechanism.core import default_global_params
from schema_mechanism.core import get_global_params
from schema_mechanism.core import set_global_params
from schema_mechanism.parameters import GlobalParams
from schema_mechanism.persistence import deserialize
from schema_mechanism.persistence import serialize
from schema_mechanism.validate import AcceptAllValidator
from schema_mechanism.validate import WhiteListValidator
from test_share.test_func import common_test_setup
from test_share.test_func import file_was_written
from test_share.test_func import satisfies_equality_checks


class TestGlobalParams(TestCase):
    def setUp(self):
        common_test_setup()

        self.gp: GlobalParams = GlobalParams()

    def test_get_and_set(self):
        parameter_key = 'key'

        first_set_value = '1'
        first_set_validator = WhiteListValidator(accept_set={'1', '3'})
        self.gp.set(name=parameter_key, value=first_set_value, validator=first_set_validator)

        # test: new key should exist with expected value and validator
        self.assertIn(parameter_key, self.gp)
        self.assertEqual(first_set_value, self.gp.get(parameter_key))

        second_set_value = '2'
        second_set_validator = WhiteListValidator(accept_set={'2', '4'})
        self.gp.set(name=parameter_key, value=second_set_value, validator=second_set_validator)

        # test: value should have been updated
        self.assertEqual(second_set_value, self.gp.get(parameter_key))

        # test: KeyError should be raised if key goes not exist on get
        self.assertIs(None, self.gp.get('does not exist'))

    def test_parameters(self):
        expected_parameters = [
            'parameter_1',
            'parameter_2',
        ]

        for i, parameter in enumerate(expected_parameters):
            self.gp.set(parameter, i)

        parameters_encountered = []
        for parameter in self.gp.parameters:
            parameters_encountered.append(parameter)

        self.assertSetEqual(set(expected_parameters), set(parameters_encountered))

    def test_contains(self):
        expected_parameters = [
            'parameter_1',
            'parameter_2',
        ]

        for i, parameter in enumerate(expected_parameters):
            self.gp.set(parameter, i)

        # test: parameters previously set in global parameters should be contained in global parameters
        for parameter in expected_parameters:
            self.assertIn(parameter, self.gp)

        # test: parameters not previously set in global parameters should NOT be contained in global parameters
        self.assertNotIn('parameter_3', self.gp)

    def test_str(self):
        global_params = get_global_params()

        # test: all parameters must appear in the GlobalParams string representation
        for param, value in global_params:
            self.assertIn(f"{param} = \'{value}\'", str(global_params))

    def test_reset(self):
        params = GlobalParams()

        params.set('parameter_1', 1, validator=AcceptAllValidator())
        params.set('parameter_2', 2, validator=AcceptAllValidator())
        params.set('parameter_3', 3, validator=AcceptAllValidator())

        # sanity check: number of parameters and validators should be positive
        self.assertGreater(len(params.parameters), 0)
        self.assertGreater(len(params.validators), 0)

        params.reset()

        # test: after reset, parameters and validators should be cleared of values
        self.assertEqual(len(params.parameters), 0)
        self.assertEqual(len(params.validators), 0)

    def test_serialize(self):
        # sets a few non-default values
        parameter_values = {
            'learning_rate': 0.00001,
            'ext_context.negative_correlation_threshold': 0.186,
            'features': [SupportedFeature.ER_INCREMENTAL_RESULTS],
            'ext_result.positive_correlation_threshold': 0.176,
            'reliability_threshold': 0.72,
        }

        for key, value in parameter_values.items():
            self.gp.set(key, value)

        expected_dict = dict()
        expected_dict.update(parameter_values)

        # sanity check
        self.assertDictEqual({k: v for k, v in self.gp}, expected_dict)

        with TemporaryDirectory() as tmp_dir:
            path = Path(os.path.join(tmp_dir, 'test-file-global_params-save_and_load.sav'))

            # sanity check: file SHOULD NOT exist
            self.assertFalse(path.exists())

            serialize(self.gp, path)

            # test: file SHOULD exist after call to save
            self.assertTrue(file_was_written(path))

            recovered: GlobalParams = deserialize(path)

            # test: non-default global params SHOULD be restored after load
            self.assertEqual(self.gp, recovered)

    def test_equals(self):
        params = default_global_params
        other = GlobalParams()

        self.assertTrue(satisfies_equality_checks(obj=params, other=other, other_different_type=1.0))


class TestSharedFunctions(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_get_and_set_global_params(self):
        # test: default_global_params should be returned on get_global_params (without prior set_global_params)
        self.assertEqual(default_global_params, get_global_params())

        # test: set_global_params should update object returned by get_global_params
        new_global_params = GlobalParams()
        set_global_params(new_global_params)

        self.assertEqual(new_global_params, get_global_params())
        self.assertIsNot(default_global_params, get_global_params())
