from unittest import TestCase

import numpy as np

from schema_mechanism.data_structures import ContinuousItem
from schema_mechanism.data_structures import DiscreteItem
from schema_mechanism.util import get_orthogonal_vector


class TestDiscreteItem(TestCase):

    def test_init(self):
        i = DiscreteItem(state_element='1234')
        self.assertEqual('1234', i.state_element)

    def test_is_on(self):
        i = DiscreteItem(state_element='1234')

        # item expected to be ON for these states
        self.assertTrue(i.is_on(state=['1234']))
        self.assertTrue(i.is_on(state=['123', '1234']))
        self.assertTrue(i.is_on(state=['123', '1234', np.random.rand(16)]))

        # item expected to be OFF for these states
        self.assertFalse(i.is_on(state=[]))
        self.assertFalse(i.is_on(state=['123']))
        self.assertFalse(i.is_on(state=['123', '4321']))


class TestContinuousItem(TestCase):
    def test_init(self):
        test_value = np.random.rand(16)

        i = ContinuousItem(state_element=test_value)
        self.assertTrue(np.array_equal(test_value, i.state_element))

        # check default activation threshold
        self.assertEqual(0.99, ContinuousItem.DEFAULT_ACTIVATION_THRESHOLD)

        # check default precision
        self.assertEqual(2, ContinuousItem.DEFAULT_PRECISION)

        # test that the value is immutable
        try:
            i.state_element[0] = 10.0
            self.fail('ContinuousItem\'s value should be immutable')
        except ValueError:
            pass

    def test_is_on(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])

        i = ContinuousItem(state_element=v1)

        # item expected to be ON for these states
        self.assertTrue(i.is_on(state=[v1], threshold=1.0))
        self.assertTrue(i.is_on(state=[v1, v2], threshold=1.0))

        # item expected to be OFF for these states
        self.assertFalse(i.is_on(state=[v2], threshold=1.0))
        self.assertFalse(i.is_on(state=[], threshold=1.0))

        # check custom threshold
        self.assertTrue(i.is_on([v2], threshold=0.0))

        # check custom precision
        self.assertFalse(i.is_on(state=[np.array([0.5, 0.5])], threshold=1.0))
        self.assertTrue(i.is_on(state=[np.array([0.5, 0.5])], threshold=1.0, precision=0))


class TestItem(TestCase):

    def test_polymorph(self):
        v1 = np.random.randn(16)
        v2 = get_orthogonal_vector(v1)

        state = ['1234', '4321', v1, v2]

        items = [
            DiscreteItem('1234'),
            DiscreteItem('4321'),
            ContinuousItem(v1),
            ContinuousItem(v2),
        ]

        self.assertTrue(all(map(lambda i: i.is_on(state), items)))
