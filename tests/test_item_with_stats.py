from unittest import TestCase

from schema_mechanism.data_structures import ItemStatisticsDecorator, DiscreteItem, State, Item, ItemStatistics


class TestItemWithStats(TestCase):
    def test(self):
        item = ItemStatisticsDecorator(item=DiscreteItem('1234', negated=True))

        # verify class is properly wrapped
        self.assertEqual('1234', item.value)
        self.assertEqual(True, item.negated)
        self.assertTrue(item.is_on(state=State(discrete_values=['1234'])))
        self.assertTrue(item.is_off(state=State(discrete_values=['4321'])))
        self.assertFalse(item.is_satisfied(state=State(discrete_values=['1234'])))
        self.assertTrue(isinstance(item, Item))

        # verify an associated statistics object exists
        self.assertTrue(isinstance(item.stats, ItemStatistics))
