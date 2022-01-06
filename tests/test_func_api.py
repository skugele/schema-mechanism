import unittest

import numpy as np

from schema_mechanism.data_structures import ContinuousItem
from schema_mechanism.data_structures import DiscreteItem
from schema_mechanism.data_structures import ItemAssertion
from schema_mechanism.func_api import gen_assert


class TestFunctionalApi(unittest.TestCase):
    def test_gen_assert(self):
        # discrete item assertions
        v = '1234'
        ia = gen_assert(state_element=v, negated=True)
        self.assertIsInstance(ia, ItemAssertion)
        self.assertIsInstance(ia.item, DiscreteItem)
        self.assertEqual(v, ia.item.state_element)
        self.assertEqual(True, ia.negated)

        # continuous item assertions
        v = np.random.rand(16)
        ia = gen_assert(state_element=v, negated=True)
        self.assertIsInstance(ia, ItemAssertion)
        self.assertIsInstance(ia.item, ContinuousItem)
        self.assertTrue(np.array_equal(v, ia.item.state_element))
        self.assertEqual(True, ia.negated)

        # invalid item assertions
        self.assertRaises(TypeError, lambda x: gen_assert(state_element=1.0, negated=False))
