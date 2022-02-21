from random import sample
from unittest import TestCase

from schema_mechanism.core import ExtendedResult
from schema_mechanism.core import GlobalParams
from schema_mechanism.core import ItemAssertion
from schema_mechanism.core import ItemPool
from schema_mechanism.core import NULL_ER_ITEM_STATS
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state_assert
from test_share.test_classes import MockObserver
from test_share.test_func import common_test_setup


class TestExtendedResult(TestCase):
    N_ITEMS = 1000

    def setUp(self) -> None:
        common_test_setup()

        # populate item pool
        self._item_pool = ItemPool()
        for i in range(self.N_ITEMS):
            _ = self._item_pool.get(str(i))

        self.result = sym_state_assert('100,101')
        self.er = ExtendedResult(result=self.result)

        # for convenience, register a single observer
        self.obs = MockObserver()
        self.er.register(self.obs)

    def test_init(self):
        for i in ItemPool():
            self.assertIs(NULL_ER_ITEM_STATS, self.er.stats[i])

        for ia in self.result:
            self.assertIn(ia.item, self.er.suppressed_items)

    def test_update_1(self):
        # testing updates for Items
        state = list(map(str, sample(range(self.N_ITEMS), k=10)))

        # update an item from this state assuming the action was taken
        new_item = sym_item(state[0])
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

    # def test_update_2(self):
    #     # testing updates for StateAssertions
    #     updated = [
    #         sym_state_assert('1,2,~3'),
    #         sym_state_assert('7,8'),
    #     ]
    #
    #     for sa in updated:
    #         self.er.update(sa, on=True, activated=True)
    #
    #     not_updated = [
    #         sym_state_assert('3,4,5'),
    #         sym_state_assert('~7,8'),
    #     ]
    #
    #     for sa in StateAssertionPool():
    #         item_stats = self.er.stats[sa]
    #         if sa in updated:
    #             self.assertIsNot(NULL_ER_ITEM_STATS, item_stats)
    #
    #             for value in [item_stats.n_on,
    #                           item_stats.n_on_and_activated]:
    #                 self.assertEqual(1, value)
    #
    #             for value in [item_stats.n_off,
    #                           item_stats.n_on_and_not_activated,
    #                           item_stats.n_off_and_activated,
    #                           item_stats.n_off_and_not_activated]:
    #                 self.assertEqual(0, value)
    #
    #         elif sa in not_updated:
    #             self.assertIs(NULL_ER_ITEM_STATS, item_stats)
    #
    # def test_update_all(self):
    #     # add state assertion to pool
    #
    #     updated_sa = [
    #         StateAssertionPool().get(sym_asserts('~0,3')),  # positive transition (on=True)
    #         StateAssertionPool().get(sym_asserts('~0,4')),  # positive transition (on=True)
    #         StateAssertionPool().get(sym_asserts('~0,5')),  # positive transition (on=True)
    #         StateAssertionPool().get(sym_asserts('3,4')),  # positive transition (on=True)
    #         StateAssertionPool().get(sym_asserts('4,5')),  # positive transition (on=True)
    #         StateAssertionPool().get(sym_asserts('~0,3,4')),  # positive transition (on=True)
    #         StateAssertionPool().get(sym_asserts('~0,4,5')),  # positive transition (on=True)
    #         StateAssertionPool().get(sym_asserts('~0,3,4,5')),  # positive transition (on=True)
    #     ]
    #
    #     updated_sa_off = [
    #         StateAssertionPool().get(sym_asserts('0,1')),  # negative transition (on=False)
    #     ]
    #
    #     not_updated_sa = [
    #         StateAssertionPool().get(sym_asserts('1,2')),
    #         StateAssertionPool().get(sym_asserts('0,1,2,3')),
    #         StateAssertionPool().get(sym_asserts('~1,2')),
    #         StateAssertionPool().get(sym_asserts('~0,1')),  # positive transition (on=True)
    #         StateAssertionPool().get(sym_asserts('1,2,3')),  # positive transition (on=True)
    #     ]
    #
    #     s_prev = sym_state('0,1,2')
    #     s_curr = sym_state('1,2,3,4,5')
    #
    #     new = new_state(s_prev=s_prev, s_curr=s_curr)
    #     lost = lost_state(s_prev=s_prev, s_curr=s_curr)
    #
    #     # update all items in this state simulating case of action taken
    #     self.er.update_all(activated=True, s_prev=s_prev, s_curr=s_curr, new=new, lost=lost)
    #
    #     # test that all items in the state have been updated
    #     for se in new:
    #         item_stats = self.er.stats[self._item_pool.get(se)]
    #         self.assertEqual(1, item_stats.n_on)
    #         self.assertEqual(1, item_stats.n_on_and_activated)
    #
    #         self.assertEqual(0, item_stats.n_off)
    #         self.assertEqual(0, item_stats.n_off_and_activated)
    #
    #         self.assertEqual(0, item_stats.n_off_and_not_activated)
    #         self.assertEqual(0, item_stats.n_on_and_not_activated)
    #
    #     for se in lost:
    #         item_stats = self.er.stats[self._item_pool.get(se)]
    #         self.assertEqual(1, item_stats.n_off)
    #         self.assertEqual(1, item_stats.n_off_and_activated)
    #
    #         self.assertEqual(0, item_stats.n_on)
    #         self.assertEqual(0, item_stats.n_on_and_activated)
    #
    #         self.assertEqual(0, item_stats.n_on_and_not_activated)
    #         self.assertEqual(0, item_stats.n_off_and_not_activated)
    #
    #     for sa in updated_sa_on:
    #         item_stats = self.er.stats[sa]
    #         self.assertIsNot(NULL_ER_ITEM_STATS, item_stats)
    #
    #         self.assertEqual(1, item_stats.n_on)
    #         self.assertEqual(1, item_stats.n_on_and_activated)
    #
    #         self.assertEqual(0, item_stats.n_off)
    #         self.assertEqual(0, item_stats.n_off_and_activated)
    #
    #         self.assertEqual(0, item_stats.n_off_and_not_activated)
    #         self.assertEqual(0, item_stats.n_on_and_not_activated)
    #
    #     for sa in updated_sa_off:
    #         item_stats = self.er.stats[sa]
    #         self.assertIsNot(NULL_ER_ITEM_STATS, item_stats)
    #
    #         self.assertEqual(1, item_stats.n_off)
    #         self.assertEqual(1, item_stats.n_off_and_activated)
    #
    #         self.assertEqual(0, item_stats.n_on)
    #         self.assertEqual(0, item_stats.n_on_and_activated)
    #
    #         self.assertEqual(0, item_stats.n_on_and_not_activated)
    #         self.assertEqual(0, item_stats.n_off_and_not_activated)
    #
    #     for sa in not_updated_sa:
    #         self.assertIs(NULL_ER_ITEM_STATS, self.er.stats[sa])

    def test_register_and_unregister(self):
        observer = MockObserver()
        self.er.register(observer)
        self.assertIn(observer, self.er.observers)

        self.er.unregister(observer)
        self.assertNotIn(observer, self.er.observers)

    def test_relevant_items_1(self):
        items = [sym_item(str(i)) for i in range(5)]

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
        self.assertTrue(i1_stats.positive_transition_corr > GlobalParams().pos_corr_threshold)
        self.assertTrue(i1_stats.negative_transition_corr <= GlobalParams().neg_corr_threshold)

        # verify only one relevant item
        self.assertEqual(1, len(self.er.relevant_items))
        self.assertIn(ItemAssertion(i1), self.er.relevant_items)

        # should add a 2nd relevant item
        i2 = items[1]

        self.er.update(i2, on=True, activated=True)
        self.er.update(i2, on=False, activated=False)

        self.assertEqual(2, len(self.er.relevant_items))
        self.assertIn(ItemAssertion(i2), self.er.relevant_items)

        # number of new relevant items SHOULD be reset to zero after notifying observers
        self.er.notify_all()
        self.assertEqual(0, len(self.er.new_relevant_items))

    def test_relevant_items_2(self):
        i1 = sym_item('100')

        self.assertIn(i1, self.er.suppressed_items)

        # test updates that SHOULD NOT result in relevant item determination
        self.er.update(i1, on=True, activated=True, count=10)
        self.er.update(i1, on=False, activated=False, count=10)

        i1_stats = self.er.stats[i1]
        self.assertTrue(i1_stats.positive_transition_corr > GlobalParams().pos_corr_threshold)
        self.assertTrue(i1_stats.negative_transition_corr <= GlobalParams().neg_corr_threshold)

        # verify suppressed item NOT in relevant items list
        self.assertEqual(0, len(self.er.relevant_items))
        self.assertNotIn(ItemAssertion(i1), self.er.relevant_items)
