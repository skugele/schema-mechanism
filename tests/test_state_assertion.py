from unittest import TestCase

from schema_mechanism.core import StateAssertion
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from schema_mechanism.func_api import sym_state_assert
from schema_mechanism.util import repr_str
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestStateAssertion(TestCase):
    def setUp(self) -> None:
        common_test_setup()

        self.items = [
            sym_item('1'),
            sym_item('2'),
            sym_item('3'),
            sym_item('(1,2)'),
            sym_item('(3,4,5)'),
        ]

        self.items_not_in = [
            sym_item('A'),
            sym_item('B'),
            sym_item('C'),
            sym_item('(A,B)'),
            sym_item('(C,D,E)'),
        ]

        self.state_assertion = StateAssertion(items=self.items)

    def test_init(self):
        # test: empty StateAssertion (no item assertion) SHOULD be allowed
        self.assertEqual(0, len(StateAssertion()))

        # test: multiple item assertion SHOULD be allowed
        self.assertEqual(1, len(StateAssertion(items=(sym_item('1'),))))

        # test: multiple item assertion SHOULD be allowed
        self.assertEqual(len(self.items), len(self.state_assertion))

        for ia in self.state_assertion:
            self.assertIn(ia, self.state_assertion)

    def test_is_satisfied(self):
        state = sym_state('1,2')

        # test: an empty StateAssertion should always be satisfied
        state_assertion = StateAssertion()
        self.assertTrue(state_assertion.is_satisfied(state))

        # single discrete item
        #######################

        # test: the following are expected to be satisfied
        state_assertion = StateAssertion((sym_item('1'),))
        self.assertTrue(state_assertion.is_satisfied(state))

        state_assertion = StateAssertion((sym_item('2'),))
        self.assertTrue(state_assertion.is_satisfied(state))

        # test: the following are expected to NOT be satisfied
        state_assertion = StateAssertion((sym_item('3'),))
        self.assertFalse(state_assertion.is_satisfied(state))

        # multiple discrete items (all must be matched)
        ###############################################

        # test: the following are expected to be satisfied
        state_assertion = StateAssertion((sym_item('1'), sym_item('2')))
        self.assertTrue(state_assertion.is_satisfied(state))

        # test: the following are expected to NOT be satisfied
        state_assertion = StateAssertion((sym_item('1'), sym_item('3')))
        self.assertFalse(state_assertion.is_satisfied(state))

        state_assertion = StateAssertion((sym_item('1'), sym_item('2'), sym_item('3')))
        self.assertFalse(state_assertion.is_satisfied(state))

    def test_is_contained(self):
        # test: the following are expected to be contained in the state assertion
        for item in self.items:
            self.assertIn(item, self.state_assertion)

        # test: the following are expected to NOT be contained in the state assertion
        for item in self.items_not_in:
            self.assertNotIn(item, self.state_assertion)

    def test_flatten(self):
        item_assertions = {
            sym_item('1'),
            sym_item('2'),
            sym_item('5'),
        }

        composite_item_assertions = {
            sym_item('(1,2)'),
            sym_item('(3,4)'),
        }

        # test: already flattened structures should return item unchanged
        sa = StateAssertion(items=item_assertions)
        self.assertSetEqual(item_assertions, set(sa.flatten()))

        # test: composite items should be flattened into sets of non-composite items
        sa = StateAssertion(items=composite_item_assertions)

        expected_item_asserts = {sym_item(str(i)) for i in range(1, 5)}
        self.assertSetEqual(expected_item_asserts, set(sa.flatten()))

        # test: overlapping composite items should only retain a single instance of their intersecting elements
        overlapping_composite_item_assertions = {
            sym_item('(1,2)'),
            sym_item('(2,3)'),
            sym_item('(1,3)'),
        }

        sa = StateAssertion(items=overlapping_composite_item_assertions)

        expected_item_asserts = {sym_item(str(i)) for i in range(1, 4)}
        self.assertSetEqual(expected_item_asserts, set(sa.flatten()))

        # test: composite and non-composite item assertions should be flattened into a set of non-composite assertions
        mixed_non_overlapping_assertions = {
            sym_item('(1,2)'),
            sym_item('3'),
            sym_item('4'),
        }

        sa = StateAssertion(items=mixed_non_overlapping_assertions)

        expected_item_asserts = {sym_item(str(i)) for i in range(1, 5)}
        self.assertSetEqual(expected_item_asserts, set(sa.flatten()))

        # test: overlapping composite/non-composite item assertions should retain only unique, non-composite assertions
        # test: composite and non-composite item assertions should be flattened into a set of non-composite assertions
        mixed_overlapping_assertions = {
            sym_item('(1,2)'),
            sym_item('2'),
            sym_item('3'),
            sym_item('(2,3)'),
        }

        sa = StateAssertion(items=mixed_overlapping_assertions)

        expected_item_asserts = {sym_item(str(i)) for i in range(1, 4)}
        self.assertSetEqual(expected_item_asserts, set(sa.flatten()))

    def test_iterable(self):
        count = 0
        for _ in self.state_assertion:
            count += 1

        self.assertEqual(len(self.items), count)

    def test_len(self):
        self.assertEqual(0, len(StateAssertion()))
        self.assertEqual(len(self.items), len(self.state_assertion))

    def test_eq(self):
        self.assertTrue(
            satisfies_equality_checks(
                obj=self.state_assertion,
                other=sym_state_assert('4,5,6'),
                other_different_type=1.0)
        )

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.state_assertion))

    def test_equal_with_composite_items(self):
        self.assertTrue(
            satisfies_equality_checks(
                obj=sym_state_assert('(4,5,6),'),
                other=sym_state_assert('(7,8,9),'),
                other_different_type=1.0)
        )

    def test_str(self):
        expected_str = ','.join(sorted(map(str, self.state_assertion.items)))
        self.assertEqual(expected_str, str(self.state_assertion))

    def test_repr(self):
        attr_values = {'items': ','.join(sorted(map(str, self.state_assertion.items)))}

        expected_repr = repr_str(self.state_assertion, attr_values)
        self.assertEqual(expected_repr, repr(self.state_assertion))
