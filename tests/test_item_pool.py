from collections import defaultdict
from random import randint
from random import sample
from time import time
from unittest import TestCase

from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.data_structures import ItemPool


class TestSharedItemPool(TestCase):
    def setUp(self) -> None:
        ItemPool().clear()

    def test_singleton(self):
        # test case: verify singleton (all objects refer to the same instance in memory)
        self.assertIs(ItemPool(), ItemPool())

    def test_init(self):
        pool = ItemPool()
        self.assertEqual(0, len(pool))

    def test_get(self):
        pool = ItemPool()

        # test case: verify correct item type
        item1 = pool.get(1, SymbolicItem)
        self.assertIsInstance(item1, SymbolicItem)
        self.assertEqual(1, item1.state_element)

        # test case: duplicated state elements return identical instances
        item1_again = pool.get(1, SymbolicItem)
        self.assertIs(item1, item1_again)

        # test case: duplicate state elements do not increase pool size
        self.assertEqual(1, len(pool))

        # test case: unique items increment pool size by one per item
        item2 = pool.get(2, SymbolicItem)
        self.assertEqual(2, len(pool))
        self.assertNotEqual(item1, item2)

        # test case: mixed state elements
        item3 = pool.get('item3', SymbolicItem)
        self.assertIsInstance(item3, SymbolicItem)
        self.assertTrue(all(item3 != other for other in [item1, item2]))

        item3_again = pool.get('item3', SymbolicItem)
        self.assertEqual(item3, item3_again)
        self.assertEqual(3, len(pool))

        item4 = pool.get('item4', SymbolicItem)
        self.assertIsInstance(item4, SymbolicItem)
        self.assertTrue(all(item4 != other for other in [item1, item2, item3]))

    def test_contains(self):
        pool = ItemPool()

        for value in range(100):
            _ = pool.get(value, SymbolicItem)

        self.assertEqual(100, len(pool))

        for value in range(100):
            self.assertIn(value, pool)

        self.assertNotIn(100, pool)

    def test_iterator(self):
        pool = ItemPool()

        for value in range(100):
            _ = pool.get(value, SymbolicItem)

        encountered = defaultdict(lambda: 0)
        for item in pool.items():
            self.assertIsInstance(item, SymbolicItem)
            encountered[item.state_element] += 1

        self.assertEqual(100, len(encountered))
        self.assertTrue(all(encountered[i] == 1 for i in range(100)))

    def test_performance(self):
        n_items = 100_000

        pool = ItemPool()

        elapsed_time = 0

        for element in range(n_items):
            start = time()
            pool.get(element, SymbolicItem)
            end = time()
            elapsed_time += end - start

        self.assertEqual(n_items, len(pool))
        print(f'Elapsed time for creating {n_items:,} new items in pool: {elapsed_time}s')

        elapsed_time = 0
        for element in range(n_items):
            start = time()
            pool.get(element, SymbolicItem)
            end = time()
            elapsed_time += end - start

        print(f'Elapsed time for retrieving {n_items:,} existing items in pool: {elapsed_time}s')

        n_runs = 10_000
        n_repeats = 100
        n_distinct_states = n_items
        n_state_elements = 10
        elapsed_time = 0
        state = sample(range(n_distinct_states), k=n_state_elements)

        pool.get.cache_clear()

        for _ in range(n_repeats):
            for _ in range(n_runs):
                for element in state:
                    start = time()
                    pool.get(element, SymbolicItem)
                    end = time()
                    elapsed_time += end - start

        n_invocations = n_state_elements * n_runs
        avg_elapsed_time = elapsed_time / n_repeats
        print(f'Elapsed time for retrieving {n_invocations:,} items (using cache): {avg_elapsed_time}s ')
