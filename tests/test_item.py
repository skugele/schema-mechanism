from random import randint
from random import sample
from time import time
from unittest import TestCase

import numpy as np

from schema_mechanism.data_structures import SymbolicItem


class TestDiscreteItem(TestCase):

    def test_init(self):
        i = SymbolicItem(state_element='1234')
        self.assertEqual('1234', i.state_element)

    def test_is_on(self):
        i = SymbolicItem(state_element='1234')

        # item expected to be ON for these states
        self.assertTrue(i.is_on(state=['1234']))
        self.assertTrue(i.is_on(state=['123', '1234']))

        # item expected to be OFF for these states
        self.assertFalse(i.is_on(state=[]))
        self.assertFalse(i.is_on(state=['123']))
        self.assertFalse(i.is_on(state=['123', '4321']))

    def test_equal(self):
        v1, v2 = '1234', '123'
        i = SymbolicItem(state_element=v1)

        self.assertEqual(i, SymbolicItem(state_element=v1))
        self.assertNotEqual(i, SymbolicItem(state_element=v2))

    def test_performance(self):
        start = time()

        state = sample(range(10_000), k=10)
        items = [SymbolicItem(randint(0, 10_000)) for _ in range(1_000_000)]
        map(lambda i: i.is_on(state), items)

        end = time()
        print(f'elapsed time: {end - start}')
