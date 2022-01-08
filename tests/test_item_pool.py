from collections import defaultdict
from random import randint
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
        pool = ItemPool()
        for value in range(1_000_000):
            _ = pool.get(randint(0, 1000), SymbolicItem)