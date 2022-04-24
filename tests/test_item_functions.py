import itertools
from unittest import TestCase

from schema_mechanism.core import item_contained_in
from schema_mechanism.core import reduce_to_most_specific_items
from schema_mechanism.func_api import sym_item
from test_share.test_func import common_test_setup


class TestItemContainedIn(TestCase):
    def setUp(self) -> None:
        common_test_setup()

    def test_identical_items(self):
        items = [
            sym_item('A'),
            sym_item('(A,B)'),
            sym_item('(A,B,C)'),
            sym_item('(~A,~B)'),
            sym_item('(A,~B,~C)'),
        ]

        # test: identical items should return True
        for item in items:
            self.assertTrue(item_contained_in(item, item))

    def test_with_non_composite_item_arguments(self):
        # test: two non-composite items should return True if they are equal, and False otherwise
        self.assertTrue(item_contained_in(sym_item('A'), sym_item('A')))

        self.assertFalse(item_contained_in(sym_item('A'), sym_item('B')))
        self.assertFalse(item_contained_in(sym_item('B'), sym_item('A')))

    def test_with_mixed_composite_and_non_composite_item_arguments(self):
        # test: should return True if a non-composite element argument is a state element of a composite item
        self.assertTrue(item_contained_in(sym_item('A'), sym_item('(A,B)')))
        self.assertTrue(item_contained_in(sym_item('B'), sym_item('(A,B)')))
        self.assertTrue(item_contained_in(sym_item('C'), sym_item('(A,B,C)')))
        self.assertTrue(item_contained_in(sym_item('A'), sym_item('(A,~B)')))
        self.assertTrue(item_contained_in(sym_item('B'), sym_item('(~A,B,C)')))
        self.assertTrue(item_contained_in(sym_item('C'), sym_item('(~A,B,C)')))

        # test: should return False if a non-composite element argument is NOT a state element of a composite item
        self.assertFalse(item_contained_in(sym_item('C'), sym_item('(A,B)')))
        self.assertFalse(item_contained_in(sym_item('D'), sym_item('(A,B,C)')))
        self.assertFalse(item_contained_in(sym_item('B'), sym_item('(A,~B)')))
        self.assertFalse(item_contained_in(sym_item('C'), sym_item('(A,~B)')))
        self.assertFalse(item_contained_in(sym_item('A'), sym_item('(~A,B,C)')))
        self.assertFalse(item_contained_in(sym_item('D'), sym_item('(~A,B,C)')))

    def test_composite_first_arg_non_composite_second_arg(self):
        # test: should return False if 1st argument is composite but 2nd is non-composite
        self.assertFalse(item_contained_in(sym_item('(A,B)'), sym_item('A')))
        self.assertFalse(item_contained_in(sym_item('(A,~B)'), sym_item('A')))
        self.assertFalse(item_contained_in(sym_item('(~A,~B)'), sym_item('C')))

    def test_with_composite_item_arguments(self):
        # test: both positive and negative item assertions in first item should be contained in second
        self.assertTrue(item_contained_in(sym_item('(A,B)'), sym_item('(A,B,C)')))
        self.assertTrue(item_contained_in(sym_item('(A,C)'), sym_item('(A,B,C)')))
        self.assertTrue(item_contained_in(sym_item('(B,C)'), sym_item('(A,B,C)')))

        self.assertFalse(item_contained_in(sym_item('(D,E)'), sym_item('(A,B,C)')))
        self.assertFalse(item_contained_in(sym_item('(C,D)'), sym_item('(A,B,C)')))
        self.assertFalse(item_contained_in(sym_item('(A,B,C,D)'), sym_item('(A,B,C)')))

        # test: negated item assertions of first item should be contained in second
        self.assertTrue(item_contained_in(sym_item('(A,B)'), sym_item('(A,B,~C)')))
        self.assertTrue(item_contained_in(sym_item('(~A,~B)'), sym_item('(~A,~B,~C)')))
        self.assertTrue(item_contained_in(sym_item('(~A,~B)'), sym_item('(~A,~B,C)')))
        self.assertFalse(item_contained_in(sym_item('(A,~B)'), sym_item('(~A,B)')))

        self.assertFalse(item_contained_in(sym_item('(A,B,~C)'), sym_item('(A,B,C)')))
        self.assertFalse(item_contained_in(sym_item('(A,B,C)'), sym_item('(A,B,~C)')))
        self.assertFalse(item_contained_in(sym_item('(~A,~B)'), sym_item('(~A,~C)')))

        self.assertFalse(item_contained_in(sym_item('(~A,~B,~C)'), sym_item('(~A,~B)')))
        self.assertFalse(item_contained_in(sym_item('(~A,~B,~C)'), sym_item('(~A,~B,~D)')))


class TestReduceToMostSpecific(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.distinct_items = [
            sym_item('A', primitive_value=7.0),
            sym_item('B', primitive_value=-7.0),
            sym_item('C', primitive_value=0.0),
            sym_item('(D,E)', primitive_value=1.0),
            sym_item('(F,G,H)', primitive_value=-1.0),
        ]

    def test_empty(self):
        # test: an empty collection should return an empty collection
        for empty_collection in [list(), set(), tuple()]:
            self.assertEqual(0, len(list(reduce_to_most_specific_items(items=empty_collection))))

    def test_none(self):
        # test: none should return an empty collection
        self.assertEqual(0, len(list(reduce_to_most_specific_items(items=None))))

    def test_distinct_items(self):
        # test: collections of distinct items should be returned unchanged
        for size in range(1, len(self.distinct_items) + 1):
            for collection in itertools.combinations(self.distinct_items, r=size):
                result = list(reduce_to_most_specific_items(items=collection))

                self.assertEqual(size, len(result))
                self.assertSetEqual(set(collection), set(result))

    def test_non_distinct_items(self):
        # test: a collection of non-distinct, non-overlapping items should be returned as a collections of their
        #     : distinct elements
        for size in range(1, len(self.distinct_items) + 1):
            for collection in itertools.combinations_with_replacement(self.distinct_items, r=size):
                result = list(reduce_to_most_specific_items(items=collection))

                # test: verify that result contains distinct elements
                self.assertEqual(len(set(result)), len(result))

                # test: verify that all elements are original elements are included in result
                self.assertSetEqual(set(collection), set(result))

    def test_identical_overlapping_items(self):
        # test: identical items should be reduced to a single instance
        for item in [sym_item('A'), sym_item('(A,B)'), sym_item('(A,B,~C)')]:
            self.assertListEqual([item], list(reduce_to_most_specific_items([item] * 2)))
            self.assertListEqual([item], list(reduce_to_most_specific_items([item] * 3)))

    def test_overlapping_items_remove_non_composites_contained_in_composite(self):

        # test scenario 1: single non-composite and single composite
        items = {sym_item('A'), sym_item('(A,B)')}
        items_to_remove = {sym_item('A')}
        expected_result = items.difference(items_to_remove)

        self.assertSetEqual(expected_result, set(reduce_to_most_specific_items(items)))

        # test scenario 2: multiple non-composite and single composite
        items = {sym_item('A'), sym_item('B'), sym_item('(A,B)')}
        items_to_remove = {sym_item('A'), sym_item('B')}
        expected_result = items.difference(items_to_remove)

        self.assertSetEqual(expected_result, set(reduce_to_most_specific_items(items)))

        # test scenario 3: multiple non-composite and multiple (non-overlapping) composites
        items = {sym_item('A'), sym_item('B'), sym_item('(A,C)'), sym_item('(B,D)')}
        items_to_remove = {sym_item('A'), sym_item('B')}
        expected_result = items.difference(items_to_remove)

        self.assertSetEqual(expected_result, set(reduce_to_most_specific_items(items)))

        # test scenario 4: multiple non-composite and multiple composites (some overlapping)
        items = {sym_item('A'), sym_item('B'), sym_item('(A,C)'), sym_item('(B,D)'), sym_item('(B,C,D)')}
        items_to_remove = {sym_item('A'), sym_item('B'), sym_item('(B,D)')}
        expected_result = items.difference(items_to_remove)

        self.assertSetEqual(expected_result, set(reduce_to_most_specific_items(items)))

    def test_overlapping_items_with_negated_assertions(self):
        # test scenario 1a: single non-composite and single composite
        items = {sym_item('A'), sym_item('(A,~B)')}
        items_to_remove = {sym_item('A')}
        expected_result = items.difference(items_to_remove)

        self.assertSetEqual(expected_result, set(reduce_to_most_specific_items(items)))

        # test scenario 1b: single non-composite and single composite
        items = {sym_item('A'), sym_item('(B,~C)')}
        items_to_remove = {}
        expected_result = items.difference(items_to_remove)

        self.assertSetEqual(expected_result, set(reduce_to_most_specific_items(items)))

        # test scenario 1c: single non-composite and single composite
        items = {sym_item('A'), sym_item('(~B,~C)')}
        items_to_remove = {}
        expected_result = items.difference(items_to_remove)

        self.assertSetEqual(expected_result, set(reduce_to_most_specific_items(items)))

        # test scenario 2: multiple non-composite and single composite
        items = {sym_item('A'), sym_item('B'), sym_item('(~A,B)')}
        items_to_remove = {sym_item('B')}
        expected_result = items.difference(items_to_remove)

        self.assertSetEqual(expected_result, set(reduce_to_most_specific_items(items)))

        # test scenario 3: multiple non-composite and multiple (non-overlapping) composites
        items = {sym_item('A'), sym_item('B'), sym_item('(A,~C)'), sym_item('(B,~D)')}
        items_to_remove = {sym_item('A'), sym_item('B')}
        expected_result = items.difference(items_to_remove)

        self.assertSetEqual(expected_result, set(reduce_to_most_specific_items(items)))

        # test scenario 4: multiple non-composite and multiple composites (some overlapping)
        items = {sym_item('A'), sym_item('B'), sym_item('(A,~C)'), sym_item('(B,~D)'), sym_item('(B,~C,~D)')}
        items_to_remove = {sym_item('A'), sym_item('B'), sym_item('(B,~D)')}
        expected_result = items.difference(items_to_remove)

        self.assertSetEqual(expected_result, set(reduce_to_most_specific_items(items)))
