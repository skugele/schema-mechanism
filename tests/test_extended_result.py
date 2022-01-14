from random import sample
from unittest import TestCase

from schema_mechanism.data_structures import ItemPool
from schema_mechanism.data_structures import ItemPoolStateView
from schema_mechanism.data_structures import NULL_ER_ITEM_STATS
from schema_mechanism.data_structures import SchemaStats
from schema_mechanism.data_structures import SymbolicItem
from schema_mechanism.func_api import create_item
from test_share.test_classes import ExtendedResultTestWrapper
from test_share.test_classes import MockObserver


class TestExtendedResult(TestCase):
    N_ITEMS = 1000

    def setUp(self) -> None:
        pool = ItemPool()
        pool.clear()

        for i in range(self.N_ITEMS):
            pool.get(i, SymbolicItem)

        self.er = ExtendedResultTestWrapper(SchemaStats())

        # for convenience, register a single observer
        self.obs = MockObserver()
        self.er.register(self.obs)

    def test_init(self):
        for i in ItemPool():
            self.assertIs(NULL_ER_ITEM_STATS, self.er.stats[i])

    def test_update(self):
        state = sample(range(self.N_ITEMS), k=10)

        # update an item from this state assuming the action was taken
        new_item = create_item(state[0])
        self.er.update(new_item, on=True, activated=True)

        item_stats = self.er.stats[new_item]
        for i in ItemPool():
            if i != new_item:
                self.assertIs(NULL_ER_ITEM_STATS, self.er.stats[i])
            else:
                self.assertIsNot(NULL_ER_ITEM_STATS, item_stats)

                for value in [item_stats.n_on,
                              item_stats.n_on_and_activated]:
                    self.assertEqual(1, value)

                for value in [item_stats.n_off,
                              item_stats.n_on_and_not_activated,
                              item_stats.n_off_and_activated,
                              item_stats.n_off_and_not_activated]:
                    self.assertEqual(0, value)

    def test_update_all(self):
        state = sample(range(self.N_ITEMS), k=10)
        view = ItemPoolStateView(state=state)

        # update all items in this state simulating case of action taken
        self.er.update_all(view, activated=True)

        # test that all items in the state have been updated
        for i in ItemPool():
            item_stats = self.er.stats[i]
            if i.state_element in state:
                self.assertEqual(1, item_stats.n_on)
                self.assertEqual(1, item_stats.n_on_and_activated)

                self.assertEqual(0, item_stats.n_off)
                self.assertEqual(0, item_stats.n_off_and_activated)

                self.assertEqual(0, item_stats.n_off_and_not_activated)
                self.assertEqual(0, item_stats.n_on_and_not_activated)
            else:
                self.assertEqual(1, item_stats.n_off)
                self.assertEqual(1, item_stats.n_off_and_activated)

                self.assertEqual(0, item_stats.n_on)
                self.assertEqual(0, item_stats.n_on_and_activated)

                self.assertEqual(0, item_stats.n_on_and_not_activated)
                self.assertEqual(0, item_stats.n_off_and_not_activated)

    def test_register_and_unregister(self):
        observer = MockObserver()
        self.er.register(observer)
        self.assertIn(observer, self.er.observers)

        self.er.unregister(observer)
        self.assertNotIn(observer, self.er.observers)

    def test_relevant_items(self):
        items = [create_item(i) for i in range(5)]

        i1 = items[0]

        # test updates that SHOULD NOT result in relevant item determination
        self.er.update(i1, on=True, activated=False)
        self.assertEqual(0, len(self.er.new_relevant_items))

        self.er.update(i1, on=False, activated=False, count=2)
        self.assertEqual(0, len(self.er.new_relevant_items))

        self.er.update(i1, on=False, activated=True)
        self.assertEqual(0, len(self.er.new_relevant_items))

        self.er.update(i1, on=True, activated=True)
        self.assertEqual(0, len(self.er.new_relevant_items))

        # test update that SHOULD result is a relevant item
        self.er.update(i1, on=True, activated=True)
        self.assertEqual(1, len(self.er.new_relevant_items))

        i1_stats = self.er.stats[i1]
        self.assertTrue(i1_stats.positive_transition_corr > self.er.POS_CORR_RELEVANCE_THRESHOLD)
        self.assertTrue(i1_stats.negative_transition_corr <= self.er.NEG_CORR_RELEVANCE_THRESHOLD)

        # verify only one relevant item
        self.assertEqual(1, len(self.er.relevant_items))
        self.assertIn(i1, self.er.relevant_items)

        # should add a 2nd relevant item
        i2 = items[1]

        self.er.update(i2, on=True, activated=True)
        self.er.update(i2, on=False, activated=False)

        self.assertEqual(2, len(self.er.relevant_items))
        self.assertIn(i2, self.er.relevant_items)

        # number of new relevant items SHOULD be reset to zero after notifying observers
        self.er.notify_all()
        self.assertEqual(0, len(self.er.new_relevant_items))
