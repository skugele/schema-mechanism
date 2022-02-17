from collections import defaultdict
from random import sample
from time import time
from unittest import TestCase

import test_share
from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import ReadOnlyItemPool
from schema_mechanism.data_structures import State
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.func_api import sym_state
from test_share.test_func import common_test_setup


class TestSharedItemPool(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_singleton(self):
        # test case: verify singleton (all objects refer to the same instance in memory)
        self.assertIs(ItemPool(), ItemPool())

    def test_init(self):
        pool = ItemPool()
        self.assertEqual(0, len(pool))

    def test_get(self):
        pool = ItemPool()

        # test case: verify correct item type and primitive value
        item1 = pool.get('1', primitive_value=1.0, item_type=SymbolicItem)
        self.assertIsInstance(item1, SymbolicItem)
        self.assertEqual('1', item1.state_element)
        self.assertEqual(1.0, item1.primitive_value)

        # test case: duplicated state elements return identical instances
        item1_again = pool.get('1')
        self.assertIs(item1, item1_again)

        # test case: duplicate state elements do not increase pool size
        self.assertEqual(1, len(pool))

        # test case: unique items increment pool size by one per item
        item2 = pool.get('2')
        self.assertEqual(2, len(pool))
        self.assertNotEqual(item1, item2)

        # test case: mixed state elements
        item3 = pool.get('item3')
        self.assertIsInstance(item3, SymbolicItem)
        self.assertTrue(all(item3 != other for other in [item1, item2]))

        item3_again = pool.get('item3')
        self.assertEqual(item3, item3_again)
        self.assertEqual(3, len(pool))

        item4 = pool.get('item4')
        self.assertIsInstance(item4, SymbolicItem)
        self.assertTrue(all(item4 != other for other in [item1, item2, item3]))

    def test_contains(self):
        pool = ItemPool()

        for value in range(100):
            _ = pool.get(str(value))

        self.assertEqual(100, len(pool))

        for value in range(100):
            self.assertIn(str(value), pool)

        self.assertNotIn('100', pool)

    def test_iterator(self):
        pool = ItemPool()

        for value in range(100):
            _ = pool.get(str(value))

        encountered = defaultdict(lambda: 0)
        for item in pool:
            self.assertIsInstance(item, SymbolicItem)
            encountered[item] += 1

        self.assertEqual(100, len(encountered))
        self.assertTrue(all({encountered[k] == 1 for k in encountered.keys()}))

    @test_share.performance_test
    def test_performance(self):
        n_items = 100_000

        pool = ItemPool()

        elapsed_time = 0

        for element in range(n_items):
            start = time()
            pool.get(str(element))
            end = time()
            elapsed_time += end - start

        self.assertEqual(n_items, len(pool))
        print(f'Time creating {n_items:,} new items in pool: {elapsed_time}s')

        elapsed_time = 0
        start = time()
        for _ in pool:
            pass
        end = time()
        elapsed_time += end - start

        print(f'Time iterating over {n_items:,} items in pool: {elapsed_time}s')

        n_repeats = 10_000
        n_distinct_states = n_items
        n_state_elements = 10
        elapsed_time = 0
        state = sample(range(n_distinct_states), k=n_state_elements)

        for _ in range(n_repeats):
            for element in state:
                start = time()
                pool.get(str(element))
                end = time()
                elapsed_time += end - start

        n_iters = n_repeats * n_state_elements
        print(f'Time retrieving {n_iters:,} items randomly from pool of {n_items:,} items : {elapsed_time}s ')


class TestReadOnlyItemPool(TestCase):
    def setUp(self) -> None:
        ItemPool().clear()

    def test(self):
        n_items = 10

        pool = ItemPool()
        for i in range(n_items):
            pool.get(str(i))

        self.assertEqual(n_items, len(pool))

        ro_pool = ReadOnlyItemPool()

        # test that read only view shows all items
        self.assertEqual(n_items, len(ro_pool))

        # test that all items exist in read-only view
        for i in range(n_items):
            item = pool.get(str(i))
            self.assertIsNotNone(item)
            self.assertEqual(str(i), item.state_element)

        # test non-existent element returns None and DOES NOT add any new elements
        self.assertIsNone(ro_pool.get('nope'))
        self.assertEqual(n_items, len(ro_pool))

        self.assertRaises(NotImplementedError, lambda: ro_pool.clear())


class TestItemPoolStateView(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test(self):
        pool = ItemPool()
        for i in range(100):
            _ = pool.get(str(i))

        state = sym_state('1,2,3')
        view = ItemPoolStateView(state=state)

        for i in pool:
            if i.state_element in state:
                self.assertTrue(view.is_on(i))
            else:
                self.assertFalse(view.is_on(i))

    @test_share.performance_test
    def test_performance(self):
        n_items = 100_000

        pool = ItemPool()
        for i in range(n_items):
            _ = pool.get(str(i))

        n_distinct_states = n_items
        n_state_elements = 10

        state = State(sample(range(n_distinct_states), k=n_state_elements))

        # test view creation time
        start = time()
        view = ItemPoolStateView(state=state)
        end = time()
        elapsed_time = end - start
        print(f'Time creating item pool view for {n_items:,} {elapsed_time}s')

        elapsed_time = 0
        for i in pool:
            start = time()
            view.is_on(i)
            end = time()
            elapsed_time += end - start

        print(f'Time retrieving state for {n_items:,} in view: {elapsed_time}s')
