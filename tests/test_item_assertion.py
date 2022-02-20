from unittest import TestCase

from schema_mechanism.core import ItemAssertion
from schema_mechanism.func_api import sym_item
from schema_mechanism.func_api import sym_state
from test_share.test_func import common_test_setup
from test_share.test_func import is_eq_consistent
from test_share.test_func import is_eq_reflexive
from test_share.test_func import is_eq_symmetric
from test_share.test_func import is_eq_transitive
from test_share.test_func import is_eq_with_null_is_false
from test_share.test_func import is_hash_consistent
from test_share.test_func import is_hash_same_for_equal_objects


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

    def test_copy(self):
        copy = self.ia.copy()

        self.assertEqual(self.ia, copy)
        self.assertNotEqual(self.ia_neg, copy)
        self.assertIsNot(self.ia, copy)

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

    def test_equal(self):
        copy = self.ia.copy()
        other = ItemAssertion(sym_item('123'))

        self.assertEqual(self.ia, copy)
        self.assertNotEqual(self.ia, other)

        self.assertTrue(is_eq_reflexive(self.ia))
        self.assertTrue(is_eq_symmetric(x=self.ia, y=copy))
        self.assertTrue(is_eq_transitive(x=self.ia, y=copy, z=copy.copy()))
        self.assertTrue(is_eq_consistent(x=self.ia, y=copy))
        self.assertTrue(is_eq_with_null_is_false(self.ia))

        # test equality operation between Item and ItemAssertion
        self.assertEqual(self.ia, self.ia.item)
        self.assertNotEqual(self.ia, other.item)

        self.assertTrue(is_eq_symmetric(x=self.ia, y=self.ia.item))
        self.assertTrue(is_eq_consistent(x=self.ia, y=self.ia.item))

    def test_hash(self):
        self.assertIsInstance(hash(self.ia), int)
        self.assertTrue(is_hash_consistent(self.ia))
        self.assertTrue(is_hash_same_for_equal_objects(x=self.ia, y=self.ia.copy()))
