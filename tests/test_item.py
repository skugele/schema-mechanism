from unittest import TestCase

import numpy as np

from schema_mechanism.data_structures import ContinuousItem
from schema_mechanism.data_structures import DiscreteItem
from schema_mechanism.data_structures import State
from schema_mechanism.util import get_orthogonal_vector


class TestDiscreteItem(TestCase):

    def test_init(self):
        i = DiscreteItem(value='1234')
        self.assertEqual('1234', i.value)

    def test_is_on(self):
        i = DiscreteItem(value='1234')

        # item expected to be ON for these states
        self.assertTrue(i.is_on(state=State(discrete_values=['1234'])))
        self.assertTrue(i.is_on(state=State(discrete_values=['123', '1234'])))
        self.assertTrue(i.is_on(state=State(discrete_values=['123', '1234'],
                                            continuous_values=[np.random.rand(16)])))

        # item expected to be OFF for these states
        self.assertFalse(i.is_on(state=State()))
        self.assertFalse(i.is_on(state=State(discrete_values=['123'])))
        self.assertFalse(i.is_on(state=State(discrete_values=['123', '4321'])))


class TestContinuousItem(TestCase):
    def test_init(self):
        test_value = np.random.rand(16)

        i = ContinuousItem(value=test_value)
        self.assertTrue(np.array_equal(test_value, i.value))

        # check default activation threshold
        self.assertEqual(0.99, ContinuousItem.DEFAULT_ACTIVATION_THRESHOLD)

        # check default precision
        self.assertEqual(2, ContinuousItem.DEFAULT_PRECISION)

        # test that the value is immutable
        try:
            i.value[0] = 10.0
            self.fail('ContinuousItem\'s value should be immutable')
        except ValueError:
            pass

    def test_is_on(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])

        i = ContinuousItem(value=v1)

        # item expected to be ON for these states
        self.assertTrue(i.is_on(state=State(continuous_values=[v1]), threshold=1.0))
        self.assertTrue(i.is_on(state=State(continuous_values=[v1, v2]), threshold=1.0))

        # item expected to be OFF for these states
        self.assertFalse(i.is_on(state=State(continuous_values=[v2]), threshold=1.0))
        self.assertFalse(i.is_on(state=State(), threshold=1.0))

        # check custom threshold
        self.assertTrue(i.is_on(state=State(continuous_values=[v2]), threshold=0.0))

        # check custom precision
        self.assertFalse(i.is_on(state=State(continuous_values=[np.array([0.5, 0.5])]), threshold=1.0))
        self.assertTrue(i.is_on(state=State(continuous_values=[np.array([0.5, 0.5])]), threshold=1.0, precision=0))


class TestItem(TestCase):

    def test_polymorph(self):
        v1 = np.random.randn(16)
        v2 = get_orthogonal_vector(v1)

        state = State(discrete_values=['1234', '4321'], continuous_values=[v1, v2])

        items = [
            DiscreteItem('1234'),
            DiscreteItem('4321'),
            ContinuousItem(v1),
            ContinuousItem(v2),
        ]

        self.assertTrue(all(map(lambda i: i.is_on(state), items)))
