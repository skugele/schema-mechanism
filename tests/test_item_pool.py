from collections import defaultdict
from unittest import TestCase

from schema_mechanism.core import CompositeItem
from schema_mechanism.core import ItemPool
from schema_mechanism.core import ReadOnlyItemPool
from schema_mechanism.core import SymbolicItem
from schema_mechanism.func_api import sym_state
from test_share.test_func import common_test_setup


class TestItemPool(TestCase):
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
        self.assertEqual('1', item1.source)
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

        # test: NotImplementedError should be raised when source is not StateElement or frozenset
        for invalid_source in [set(), list(), dict()]:
            self.assertRaises(NotImplementedError, lambda: pool.get(invalid_source))

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

        # populate SymbolicItems
        state_elements = frozenset(str(i) for i in range(10))
        for se in state_elements:
            _ = pool.get(se, item_type=SymbolicItem)

        # populate CompositeItems
        states = frozenset((
            sym_state('1,2'),
            sym_state('3,4,5')
        ))

        for source in states:
            _ = pool.get(source, item_type=CompositeItem)

        encountered = defaultdict(lambda: 0)
        for item in pool:
            encountered[item] += 1

        self.assertEqual(sum(encountered[k] for k in encountered.keys()), len(states) + len(state_elements))
        self.assertTrue(all({encountered[k] == 1 for k in encountered.keys()}))


class TestReadOnlyItemPool(TestCase):
    def setUp(self) -> None:
        ItemPool().clear()

    def test(self):
        n_items = 10

        pool = ItemPool()
        for i in range(n_items):
            _ = pool.get(str(i))

        self.assertEqual(n_items, len(pool))

        ro_pool = ReadOnlyItemPool()

        # test that read only view shows all items
        self.assertEqual(n_items, len(ro_pool))

        # test that all items exist in read-only view
        for i in range(n_items):
            item = pool.get(str(i))
            self.assertIsNotNone(item)
            self.assertEqual(str(i), item.source)

        # test non-existent element returns None and DOES NOT add any new elements
        self.assertIsNone(ro_pool.get('nope'))
        self.assertEqual(n_items, len(ro_pool))

        self.assertRaises(NotImplementedError, lambda: ro_pool.clear())
