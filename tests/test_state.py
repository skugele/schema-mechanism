from copy import copy
from unittest import TestCase

from schema_mechanism.core import CompositeItem
from schema_mechanism.core import GlobalStats
from schema_mechanism.core import Item
from schema_mechanism.core import ItemPool
from schema_mechanism.core import composite_items
from schema_mechanism.core import held_state
from schema_mechanism.core import lost_state
from schema_mechanism.core import new_state
from schema_mechanism.core import non_composite_items
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_items
from schema_mechanism.func_api import sym_state
from test_share.test_classes import MockSymbolicItem
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestState(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        # adds items to item pool
        _ = ItemPool().get('1', primitive_value=-1.0, delegated_value=3.0, item_type=MockSymbolicItem)
        _ = ItemPool().get('2', primitive_value=0.0, delegated_value=1.0, item_type=MockSymbolicItem)
        _ = ItemPool().get('3', primitive_value=1.0, delegated_value=-3.0, item_type=MockSymbolicItem)
        _ = ItemPool().get('4', primitive_value=-1.0, delegated_value=0.0, item_type=MockSymbolicItem)
        _ = ItemPool().get('5', primitive_value=0.0, delegated_value=-1.0, item_type=MockSymbolicItem)

        self.s = ('1', '2', '3')
        self.s_copy = copy(self.s)
        self.s_copy_copy = copy(self.s_copy)

        self.s_disjoint = ('4', '5')
        self.s_conjoint = ('1', '2', '4')
        self.s_contained = (*self.s, 100)

        self.s_empty = tuple()

        GlobalStats(baseline_value=-1.0)

    def test_init(self):
        self.assertEqual(0, len(self.s_empty))
        self.assertSetEqual(frozenset([]), frozenset(self.s_empty))

        self.assertEqual(3, len(self.s))
        self.assertSetEqual(frozenset(['1', '2', '3']), frozenset(self.s))

    def test_eq(self):
        self.assertNotEqual(self.s, self.s_disjoint)
        self.assertNotEqual(self.s, self.s_conjoint)
        self.assertNotEqual(self.s, self.s_contained)

        self.assertTrue(satisfies_equality_checks(obj=self.s, other_different_type=1.0))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.s))

    def test_len(self):
        self.assertEqual(0, len(self.s_empty))
        self.assertEqual(3, len(self.s))

    def test_contains(self):
        for se in ['1', '2', '3']:
            self.assertIn(se, self.s)

        for se in ['4', '5', '6']:
            self.assertNotIn(se, self.s)
            self.assertNotIn(se, self.s_empty)


class TestStateFunctions(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.s_1 = sym_state('1,2,3,4,5')
        self.s_2 = sym_state('4,5,6,7,8')
        self.s_empty = sym_state('')
        self.s_none = None

        # add composite items to pool
        self.pool: ItemPool = ItemPool()

        self.items: list[Item] = [sym_item(str(i)) for i in range(10)]

        self.ci1: Item = self.pool.get(sym_state('1,2,3'), item_type=CompositeItem)  # On in s_1
        self.ci2: Item = self.pool.get(sym_state('4,5'), item_type=CompositeItem)  # On in s_1 and s_2
        self.ci3: Item = self.pool.get(sym_state('6,7,8'), item_type=CompositeItem)  # On in s_2

    def test_all_state_functions_with_both_arguments_empty_or_none(self):
        # tests for empty or None states
        ################################

        # test: no items should be returned when both states are either empty or None
        for state_function in (held_state, lost_state, new_state):
            self.assertEqual(0, len(state_function(s_prev=self.s_empty, s_curr=self.s_empty)))
            self.assertEqual(0, len(state_function(s_prev=self.s_none, s_curr=self.s_empty)))
            self.assertEqual(0, len(state_function(s_prev=self.s_empty, s_curr=self.s_none)))
            self.assertEqual(0, len(state_function(s_prev=self.s_none, s_curr=self.s_none)))

    def test_held_with_single_argument_empty_or_none(self):
        # test: no items should be returned when either state is None or empty
        self.assertEqual(0, len(held_state(s_prev=self.s_empty, s_curr=self.s_1)))
        self.assertEqual(0, len(held_state(s_prev=self.s_1, s_curr=self.s_empty)))
        self.assertEqual(0, len(held_state(s_prev=self.s_none, s_curr=self.s_1)))
        self.assertEqual(0, len(held_state(s_prev=self.s_1, s_curr=self.s_none)))

    def test_held_state_with_disjoint_current_and_previous_states(self):
        # test: no items should be returned when previous and current states share no common elements
        self.assertSetEqual(
            set(),
            set(non_composite_items(
                held_state(
                    s_prev=sym_state('4,5,6'),
                    s_curr=sym_state('1,2,3')))))

    def test_held_state_with_non_composite_items(self):
        # test: non-composite items that are On in both previous and current state should be returned
        self.assertSetEqual(
            {sym_item('3')},
            set(non_composite_items(
                held_state(
                    s_prev=sym_state('1,2,3'),
                    s_curr=sym_state('3,4,5')))))

        # test: non-composite items that are On in both previous and current state should be returned
        self.assertSetEqual(
            {sym_item('3')},
            set(non_composite_items(
                held_state(
                    s_prev=sym_state('3,4,5'),
                    s_curr=sym_state('1,2,3')))))

        # test: non-composite items that are On in both previous and current state should be returned
        self.assertSetEqual(
            set(sym_items('4;5')),
            set(non_composite_items(
                held_state(
                    s_prev=sym_state('3,4,5'),
                    s_curr=sym_state('4,5,6')))))

        # test: non-composite items that are On in both previous and current state should be returned
        self.assertSetEqual(
            set(sym_items('4;5')),
            set(non_composite_items(
                held_state(
                    s_prev=sym_state('4,5,6'),
                    s_curr=sym_state('3,4,5')))))

    def test_held_state_with_composite_items(self):
        # test: composite items that are On in both previous and current state should be returned (single item test)
        self.assertSetEqual(
            {self.ci1},
            set(composite_items(
                held_state(
                    s_prev=sym_state('1,2,3'),
                    s_curr=sym_state('1,2,3,4')))))

        # test: composite items that are On in both previous and current state should be returned (single item test)
        self.assertSetEqual(
            {self.ci1},
            set(composite_items(
                held_state(
                    s_prev=sym_state('1,2,3,4'),
                    s_curr=sym_state('1,2,3')))))

        # test: composite items that are On in both previous and current state should be returned (multiple item test)
        self.assertSetEqual(
            {self.ci1, self.ci2},
            set(composite_items(
                held_state(
                    s_prev=sym_state('1,2,3,4,5'),
                    s_curr=sym_state('1,2,3,4,5,6,7,8')))))

        # test: composite items that are On in both previous and current state should be returned (multiple item test)
        self.assertSetEqual(
            {self.ci1, self.ci2},
            set(composite_items(
                held_state(
                    s_prev=sym_state('1,2,3,4,5,6,7,8'),
                    s_curr=sym_state('1,2,3,4,5')))))

    def test_lost_with_single_argument_empty_or_none(self):
        # test: no items should be returned if PREVIOUS state is empty or none
        self.assertSetEqual(set(), lost_state(s_prev=self.s_empty, s_curr=self.s_1))
        self.assertSetEqual(set(), lost_state(s_prev=self.s_none, s_curr=self.s_1))

        # test: all items in previous state should be returned if CURRENT state is empty or none
        self.assertSetEqual(
            {*sym_items('1;2;3;4;5'), self.ci1, self.ci2},
            lost_state(s_prev=self.s_1, s_curr=self.s_empty)
        )

        self.assertSetEqual(
            {*sym_items('1;2;3;4;5'), self.ci1, self.ci2},
            lost_state(s_prev=self.s_1, s_curr=self.s_none)
        )

    def test_lost_state_with_disjoint_current_and_previous_states(self):
        # test: all items On in previous state should be returned if previous and current states share no elements
        expected = {*sym_items('4;5;6'), self.ci2}
        actual = set(lost_state(s_prev=sym_state('4,5,6'), s_curr=sym_state('1,2,3')))

        self.assertSetEqual(expected, actual)

    def test_lost_state_with_non_composite_items(self):
        # test: non-composite items that are On in previous state but not current state should be returned
        self.assertSetEqual(
            {sym_item('1')},
            set(non_composite_items(
                lost_state(
                    s_prev=sym_state('1,2,3'),
                    s_curr=sym_state('2,3,4')))))

        self.assertSetEqual(
            set(sym_items('4;5')),
            set(non_composite_items(
                lost_state(
                    s_prev=sym_state('3,4,5'),
                    s_curr=sym_state('1,2,3')))))

        # test: non-composite items that are On in both previous and current state SHOULD NOT be returned
        self.assertSetEqual(
            set(),
            set(non_composite_items(
                lost_state(
                    s_prev=sym_state('3,4,5'),
                    s_curr=sym_state('3,4,5')))))

    def test_lost_state_with_composite_items(self):
        # test: composite items that are On in previous state but not current state should be returned
        self.assertSetEqual(
            {self.ci1, self.ci2, self.ci3},
            set(composite_items(
                lost_state(
                    s_prev=sym_state('1,2,3,4,5,6,7,8'),
                    s_curr=sym_state('2,3,4')))))

        # test: composite items that are On in both previous and current state SHOULD NOT be returned
        self.assertSetEqual(
            set(),
            set(composite_items(
                lost_state(
                    s_prev=sym_state('1,2,3,4,5,6,7,8'),
                    s_curr=sym_state('1,2,3,4,5,6,7,8')))))

    def test_new_with_single_argument_empty_or_none(self):
        # test: no items should be returned if CURRENT state is empty or none
        self.assertSetEqual(set(), new_state(s_prev=self.s_1, s_curr=self.s_empty))
        self.assertSetEqual(set(), new_state(s_prev=self.s_1, s_curr=self.s_none))

        # test: all items in CURRENT state should be returned if PREVIOUS state is empty or none
        self.assertSetEqual(
            {*sym_items('1;2;3;4;5'), self.ci1, self.ci2},
            new_state(s_prev=self.s_empty, s_curr=self.s_1)
        )

        self.assertSetEqual(
            {*sym_items('1;2;3;4;5'), self.ci1, self.ci2},
            new_state(s_prev=self.s_empty, s_curr=self.s_1)
        )

    def test_new_state_with_disjoint_current_and_previous_states(self):
        # test: all items On in CURRENT state should be returned if previous and current states share no elements
        expected = {*sym_items('4;5;6'), self.ci2}
        actual = set(new_state(s_prev=sym_state('1,2,3'), s_curr=sym_state('4,5,6')))

        self.assertSetEqual(expected, actual)

    def test_new_state_with_non_composite_items(self):
        # test: non-composite items that are On in CURRENT state but not PREVIOUS state should be returned
        self.assertSetEqual(
            {sym_item('4')},
            set(non_composite_items(
                new_state(
                    s_prev=sym_state('1,2,3'),
                    s_curr=sym_state('2,3,4')))))

        self.assertSetEqual(
            set(sym_items('1;2')),
            set(non_composite_items(
                new_state(
                    s_prev=sym_state('3,4,5'),
                    s_curr=sym_state('1,2,3')))))

        # test: non-composite items that are On in both previous and current state SHOULD NOT be returned
        self.assertSetEqual(
            set(),
            set(non_composite_items(
                new_state(
                    s_prev=sym_state('3,4,5'),
                    s_curr=sym_state('3,4,5')))))

    def test_new_state_with_composite_items(self):
        # test: composite items that are On in CURRENT state but not PREVIOUS state should be returned
        self.assertSetEqual(
            {self.ci1, self.ci2, self.ci3},
            set(composite_items(
                new_state(
                    s_prev=sym_state('2,3,4'),
                    s_curr=sym_state('1,2,3,4,5,6,7,8')))))

        # test: composite items that are On in both previous and current state SHOULD NOT be returned
        self.assertSetEqual(
            set(),
            set(composite_items(
                new_state(
                    s_prev=sym_state('1,2,3,4,5,6,7,8'),
                    s_curr=sym_state('1,2,3,4,5,6,7,8')))))
