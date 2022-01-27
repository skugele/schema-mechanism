from random import sample
from time import time
from unittest import TestCase

import test_share
from schema_mechanism.data_structures import SymbolicItem
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


class TestSymbolicItem(TestCase):

    def setUp(self) -> None:
        self.item = SymbolicItem(state_element='1234')

    def test_init(self):
        self.assertEqual('1234', self.item.state_element)

    def test_is_on(self):
        # item expected to be ON for these states
        self.assertTrue(self.item.is_on(state=['1234']))
        self.assertTrue(self.item.is_on(state=['123', '1234']))

        # item expected to be OFF for these states
        self.assertFalse(self.item.is_on(state=[]))
        self.assertFalse(self.item.is_on(state=['123']))
        self.assertFalse(self.item.is_on(state=['123', '4321']))

    def test_copy(self):
        copy = self.item.copy()

        self.assertEqual(self.item, copy)
        self.assertIsNot(self.item, copy)

    def test_equal(self):
        copy = self.item.copy()
        other = SymbolicItem('123')

        self.assertEqual(self.item, self.item)
        self.assertEqual(self.item, copy)
        self.assertNotEqual(self.item, other)

        self.assertTrue(is_eq_reflexive(self.item))
        self.assertTrue(is_eq_symmetric(x=self.item, y=copy))
        self.assertTrue(is_eq_transitive(x=self.item, y=copy, z=copy.copy()))
        self.assertTrue(is_eq_consistent(x=self.item, y=copy))
        self.assertTrue(is_eq_with_null_is_false(self.item))

    def test_hash(self):
        self.assertIsInstance(hash(self.item), int)
        self.assertTrue(is_hash_consistent(self.item))
        self.assertTrue(is_hash_same_for_equal_objects(x=self.item, y=self.item.copy()))

    @test_share.performance_test
    def test_performance(self):
        n_items = 100_000
        n_runs = 100_000
        n_distinct_states = 100_000_000_000
        n_state_elements = 25

        items = [SymbolicItem(str(value)) for value in range(n_items)]
        state = sample(range(n_distinct_states), k=n_state_elements)

        start = time()
        for run in range(n_runs):
            _ = map(lambda i: self.item.is_on(state), items)
        end = time()

        # TODO: Need to add a test that includes an upper bound on the elapsed time
        print(f'Time for {n_runs * n_items:,} SymbolicItem.is_on calls: {end - start}s')
