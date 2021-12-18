from unittest import TestCase

import numpy as np

from schema_mechanism.data_structures import State


class TestState(TestCase):
    def test_init(self):
        discrete_values = ['1234', '1245']
        continuous_values = [np.random.rand(16), np.random.rand(16)]

        # check None as defaults
        s = State()

        self.assertIsNone(s.discrete_values)
        self.assertIsNone(s.continuous_values)

        # check discrete values
        s = State(discrete_values=discrete_values)

        self.assertListEqual(discrete_values, list(s.discrete_values))
        self.assertIsNone(s.continuous_values)

        # check continuous values
        s = State(continuous_values=continuous_values)
        self.assertListEqual(continuous_values, list(s.continuous_values))
        self.assertIsNone(s.discrete_values)

        # check setting both
        s = State(discrete_values=discrete_values,
                  continuous_values=continuous_values)

        self.assertListEqual(discrete_values, list(s.discrete_values))
        self.assertListEqual(continuous_values, list(s.continuous_values))
