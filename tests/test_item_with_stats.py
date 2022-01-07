from unittest import TestCase

from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.data_structures import Item
from schema_mechanism.data_structures import ItemStatistics
from schema_mechanism.data_structures import ItemStatisticsDecorator


class TestItemWithStats(TestCase):
    def test(self):
        item = ItemStatisticsDecorator(item=SymbolicItem('1234'))

        # verify class is properly wrapped
        self.assertEqual('1234', item.state_element)
        self.assertTrue(item.is_on(state=['1234']))
        self.assertTrue(item.is_off(state=['4321']))
        self.assertTrue(isinstance(item, Item))

        # verify an associated statistics object exists
        self.assertTrue(isinstance(item.stats, ItemStatistics))
