from unittest import TestCase

from schema_mechanism.core import ItemAssertion
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_item_assert
from schema_mechanism.func_api import sym_state
from test_share.test_func import common_test_setup
from test_share.test_func import satisfies_equality_checks
from test_share.test_func import satisfies_hash_checks


class TestItemAssertion(TestCase):

    def setUp(self) -> None:
        common_test_setup()

        self.item = sym_item('1234', primitive_value=1.0)
        self.ia = ItemAssertion(self.item)
        self.ia_neg = ItemAssertion(self.item, negated=True)

    def test_init(self):
        self.assertEqual(self.item, self.ia.item)

        # check default negated value is False
        self.assertEqual(False, self.ia.is_negated)

        # check setting of non-default negated value to True
        self.assertEqual(True, self.ia_neg.is_negated)

        # check setting of primitive value
        self.assertEqual(1.0, self.ia.item.primitive_value)

        # check immutability
        try:
            # noinspection PyPropertyAccess
            self.ia.is_negated = False
            self.fail('ItemAssertion is not immutable as expected: able to set negated directly')
        except AttributeError:
            pass

        try:
            # noinspection PyPropertyAccess
            self.ia.item = sym_item('illegal')
            self.fail('ItemAssertion is not immutable as expected: able to set negated directly')
        except AttributeError:
            pass

    def test_len(self):
        i_non_composite = sym_item_assert('1')
        i_composite_1 = sym_item_assert('(1,2)')
        i_composite_2 = sym_item_assert('(~1,~2)')
        i_composite_3 = sym_item_assert('(1,2,~3,4,5)')

        # test: non-composite item assertions should have a length of 1
        self.assertEqual(1, len(i_non_composite))

        # test: composite item assertion should have length equal to that of its state assertion
        self.assertEqual(2, len(i_composite_1))
        self.assertEqual(2, len(i_composite_2))
        self.assertEqual(5, len(i_composite_3))

    def test_is_satisfied(self):
        # not negated
        #############

        # item expected to be ON for these states
        self.assertTrue(self.ia.is_satisfied(sym_state('1234')))
        self.assertTrue(self.ia.is_satisfied(sym_state('123,1234')))

        # item expected to be OFF for these states
        self.assertFalse(self.ia.is_satisfied(sym_state('')))
        self.assertFalse(self.ia.is_satisfied(sym_state('123')))
        self.assertFalse(self.ia.is_satisfied(sym_state('123,4321')))

        # negated
        #########

        # item expected to be ON for these states
        self.assertFalse(self.ia_neg.is_satisfied(sym_state('1234')))
        self.assertFalse(self.ia_neg.is_satisfied(sym_state('123,1234')))

        # item expected to be OFF for these states
        self.assertTrue(self.ia_neg.is_satisfied(sym_state('')))
        self.assertTrue(self.ia_neg.is_satisfied(sym_state('123')))
        self.assertTrue(self.ia_neg.is_satisfied(sym_state('123,4321')))

    def test_eq(self):
        self.assertTrue(satisfies_equality_checks(obj=self.ia, other=ItemAssertion(sym_item('123'))))

    def test_hash(self):
        self.assertTrue(satisfies_hash_checks(obj=self.ia))
